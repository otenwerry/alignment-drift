"""Subscription-backed Claude Code, Codex, and OpenCode Go adapters.

This module can launch paid/included-quota agent calls, hence the ``exp_`` prefix.
Claude Code and Codex run their pinned native CLI in the task sandbox with the user's
saved subscription login. OpenCode Go keeps the existing native OpenCode/Inspect bridge,
but the host side of that bridge calls Go with ``OPENCODE_API_KEY``. Unsupported Go
models retain the API-backed production fallback.

The evaluated workspace never receives the host credential by reference.  A private
copy is written into the disposable container.  Subscription compose files restrict
container egress to an allow-listed proxy for Anthropic/OpenAI domains.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import os
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence


CLAUDE_KEYCHAIN_SERVICE = "Claude Code-credentials"
CLAUDE_AUTH_B64_ENV = "CLAUDE_SUBSCRIPTION_CREDENTIALS_JSON_B64"
CODEX_AUTH_B64_ENV = "CODEX_SUBSCRIPTION_AUTH_JSON_B64"
CODEX_AUTH_GZIP_B64_ENV = "CODEX_SUBSCRIPTION_AUTH_JSON_GZIP_B64"
OPENCODE_GO_API_KEY_ENV = "OPENCODE_API_KEY"
OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
MAX_CODEX_AUTH_JSON_BYTES = 1024 * 1024
SUBSCRIPTION_EVENT_MODEL_PREFIX = "subscription/"
CLAUDE_HIDDEN_SYSTEM_MARKER = (
    "[Claude Code native system prompt unavailable: the subscription CLI does not "
    "expose its complete native system prompt to the caller.]"
)


def _claude_security_settings() -> dict:
    """Keep native tools autonomous without exposing the copied OAuth login."""

    protected = ["//root/.claude/**", "//root/.claude.json", "//proc/**"]
    return {
        "permissions": {
            "allow": ["Bash", "Read", "Edit", "Write", "Glob", "Grep", "NotebookEdit"],
            "deny": [
                *(f"Read({path})" for path in protected),
                *(f"Edit({path})" for path in protected),
                *(f"Write({path})" for path in protected),
            ],
        },
        "sandbox": {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            # Docker is already the outer isolation boundary. This mode retains
            # filesystem enforcement while reusing the container's /proc mount.
            "enableWeakerNestedSandbox": True,
            "filesystem": {
                "denyRead": ["/root/.claude", "/root/.claude.json", "/proc"],
                "denyWrite": ["/root/.claude", "/root/.claude.json", "/proc"],
            },
            "credentials": {
                "files": [
                    {"path": "/root/.claude/.credentials.json", "mode": "deny"},
                    {"path": "/root/.claude.json", "mode": "deny"},
                ],
                "envVars": [
                    {"name": "CLAUDE_CODE_OAUTH_TOKEN", "mode": "deny"},
                ],
            },
            "network": {
                "allowedDomains": [],
                "strictAllowlist": True,
            },
        },
    }


def _codex_security_config(*, model: str, reasoning: bool) -> dict:
    """Config layer that keeps the native shell away from Codex's login."""

    return {
        "allow_login_shell": False,
        "analytics": {"enabled": False},
        "approval_policy": "never",
        "check_for_update_on_startup": False,
        "cli_auth_credentials_store": "file",
        "default_permissions": "subscription-workspace",
        "features.apps": False,
        "features.browser_use": False,
        "features.browser_use_external": False,
        "features.browser_use_full_cdp_access": False,
        "features.computer_use": False,
        "features.image_generation": False,
        "features.multi_agent": False,
        "features.plugin_sharing": False,
        "features.plugins": False,
        "features.remote_plugin": False,
        "permissions.subscription-workspace": {"extends": ":workspace"},
        "shell_environment_policy": {
            "inherit": "core",
            "ignore_default_excludes": False,
            "exclude": [
                "*TOKEN*",
                "*KEY*",
                "*SECRET*",
                "*AUTH*",
                "*PASSWORD*",
                "*CREDENTIAL*",
            ],
        },
        "model": model,
        "model_reasoning_effort": "medium" if reasoning else "minimal",
    }


def _codex_filesystem_deny_toml() -> str:
    # inspect-swe's small TOML encoder cannot quote slash-containing keys.
    return (
        "\n\n[permissions.subscription-workspace.filesystem]\n"
        '"/root/.codex/auth.json" = "deny"\n'
        '"/proc" = "deny"\n'
    )


def _codex_mcp_server_config(mcp: Any) -> dict[str, Any]:
    """Translate Inspect's HTTP MCP descriptor into Codex config.toml fields."""

    transport = getattr(mcp, "type", None)
    if transport != "http":
        raise RuntimeError(
            f"Codex subscription bridge requires streamable HTTP MCP, got {transport!r}"
        )
    url = getattr(mcp, "url", None)
    if not isinstance(url, str) or not url:
        raise RuntimeError("Codex subscription bridge MCP config is missing its URL")

    # Codex chooses streamable HTTP from the presence of `url`; Inspect's generic
    # `type = "http"` discriminator is not part of Codex's config schema.
    config: dict[str, Any] = {"url": url}
    headers = getattr(mcp, "headers", None)
    if headers:
        # The pinned inspect-swe TOML encoder cannot represent a nested header
        # table. The local check_time bridge does not use headers, so fail rather
        # than silently writing a config that drops them.
        raise RuntimeError("Codex subscription bridge does not support MCP headers")
    tools = getattr(mcp, "tools", "all")
    if tools != "all":
        config["enabled_tools"] = list(tools)
    return config


def _codex_exec_transport(
    binary: str,
    common: list[str],
    session_id: str | None,
    prompt: str,
) -> tuple[list[str], str]:
    """Keep the prompt off argv and send it through Codex's documented stdin path."""

    command = (
        [binary, "exec", *common, "-"]
        if session_id is None
        else [binary, "exec", "resume", *common, session_id, "-"]
    )
    return command, prompt


@dataclass
class NativeSessionRef:
    """Mutable native state filled by subscription CLI invocations."""

    value: str | None = None
    auth_source: str | None = None
    auth_seeded: bool = False
    last_invocation: dict = field(default_factory=dict)
    invocations: list[dict] = field(default_factory=list)


@dataclass
class ParsedInvocation:
    """Messages and accounting recovered from one native CLI invocation."""

    messages: list[Any] = field(default_factory=list)
    model_events: list[Any] = field(default_factory=list)
    output: Any | None = None
    usage: list[dict] = field(default_factory=list)
    rate_limits: list[dict] = field(default_factory=list)
    loss_events: list[dict] = field(default_factory=list)
    system_prompt_observed: bool = False
    native_version: str | None = None
    native_session_id: str | None = None
    # Model responses the CLI emitted without any usage record. Kept as a
    # queryable count so under-reported usage totals never pass silently.
    unmetered_model_calls: int = 0
    # Codex's own cumulative session usage from its last token_count record;
    # a stored cross-check for the summed per-response usage list.
    native_total_usage: dict | None = None
    # The CLI's own authoritative usage for THIS invocation (Claude result
    # modelUsage). Claude's per-record usage snapshots are partial (verified
    # live: they under-count output and repeat cache reads), so totals and the
    # official Inspect tally come from here when present.
    authoritative_usage: dict | None = None


def _host_home() -> Path:
    return Path.home()


def _valid_json_object(data: bytes, *, label: str) -> bytes:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return data


def _claude_keychain_bytes() -> bytes | None:
    if sys.platform != "darwin" or not Path("/usr/bin/security").is_file():
        return None
    result = subprocess.run(
        [
            "/usr/bin/security",
            "find-generic-password",
            "-s",
            CLAUDE_KEYCHAIN_SERVICE,
            "-w",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return _valid_json_object(
        result.stdout.strip(), label="Claude Code keychain credential"
    )


def _claude_file_bytes() -> bytes | None:
    path = _host_home() / ".claude" / ".credentials.json"
    if not path.is_file():
        return None
    return _valid_json_object(path.read_bytes(), label=str(path))


def _codex_file_bytes() -> bytes | None:
    path = _host_home() / ".codex" / "auth.json"
    if not path.is_file():
        return None
    return _valid_json_object(path.read_bytes(), label=str(path))


def subscription_auth_source(scaffold: str) -> str | None:
    """Return a non-secret description of the available subscription login."""

    if scaffold == "claude_code":
        if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            return "CLAUDE_CODE_OAUTH_TOKEN"
        if os.environ.get(CLAUDE_AUTH_B64_ENV):
            return CLAUDE_AUTH_B64_ENV
        if (_host_home() / ".claude" / ".credentials.json").is_file():
            return "host_claude_credentials_file"
        if _claude_keychain_bytes() is not None:
            return "macos_keychain"
        return None
    if scaffold == "codex":
        # The raw blob is the per-run override used for a freshly issued login.
        # Prefer it to the gzip blob, which is normally loaded from the project's
        # persisted .env snapshot and may have gone stale after token rotation.
        if os.environ.get(CODEX_AUTH_B64_ENV):
            return CODEX_AUTH_B64_ENV
        if os.environ.get("CODEX_ACCESS_TOKEN"):
            return "CODEX_ACCESS_TOKEN"
        if os.environ.get(CODEX_AUTH_GZIP_B64_ENV):
            return CODEX_AUTH_GZIP_B64_ENV
        if (_host_home() / ".codex" / "auth.json").is_file():
            return "host_codex_auth_file"
        return None
    if scaffold == "opencode_go":
        return (
            OPENCODE_GO_API_KEY_ENV
            if os.environ.get(OPENCODE_GO_API_KEY_ENV)
            else None
        )
    return "api_fallback"


def require_subscription_auth(scaffolds: Iterable[str]) -> dict[str, str]:
    """Fail before spend if a direct subscription scaffold has no usable login."""

    sources: dict[str, str] = {}
    missing: list[str] = []
    for scaffold in sorted(set(scaffolds)):
        source = subscription_auth_source(scaffold)
        if source is None:
            missing.append(scaffold)
        else:
            sources[scaffold] = source
    if missing:
        help_text = []
        if "claude_code" in missing:
            help_text.append(
                "Claude Code: sign in locally, or set CLAUDE_CODE_OAUTH_TOKEN "
                f"or {CLAUDE_AUTH_B64_ENV} (required on remote/AWS workers)"
            )
        if "codex" in missing:
            help_text.append(
                f"Codex: run `codex login`, or set CODEX_ACCESS_TOKEN / "
                f"{CODEX_AUTH_GZIP_B64_ENV} / {CODEX_AUTH_B64_ENV} "
                "(required on remote/AWS workers)"
            )
        if "opencode_go" in missing:
            help_text.append(
                "OpenCode Go: create a Go API key in the OpenCode console and set "
                f"{OPENCODE_GO_API_KEY_ENV} in .env (also required on remote/AWS "
                "workers)"
            )
        raise SystemExit(
            "--harness=subscription has no subscription login for "
            + ", ".join(missing)
            + ". "
            + "; ".join(help_text)
        )
    return sources


def require_opencode_go_auth(*, reasoning: bool) -> str:
    """Validate the currently supported Go execution contract before spend."""

    if not reasoning:
        raise SystemExit(
            "OpenCode Go subscription runs currently require --reasoning=yes. The "
            "Go gateway exposes these models as reasoning models but does not expose "
            "a reliable cross-model off switch through the Inspect bridge; refusing "
            "to silently run a different setting."
        )
    return require_subscription_auth(["opencode_go"])["opencode_go"]


def build_opencode_go_model(
    target_name: str,
    routed_slug: str,
    *,
    reasoning: bool,
):
    """Build the host-side model used by OpenCode's local Inspect bridge."""

    from inspect_ai.model import GenerateConfig, get_model

    from exp_target_harness import opencode_go_model_spec

    spec = opencode_go_model_spec(target_name, routed_slug)
    if spec is None:
        raise ValueError(f"{target_name!r} is not mapped to OpenCode Go")
    require_opencode_go_auth(reasoning=reasoning)
    api_key = os.environ[OPENCODE_GO_API_KEY_ENV]
    model_id = spec["model"]
    config = GenerateConfig(max_tokens=int(spec["output_tokens"]))
    if spec["protocol"] == "anthropic":
        model_name = f"anthropic/opencode-go/{model_id}"
        return get_model(
            model_name,
            role="target",
            config=config,
            base_url=OPENCODE_GO_BASE_URL,
            api_key=api_key,
        )
    return get_model(
        f"openai-api/opencode-go/{model_id}",
        role="target",
        config=config,
        base_url=OPENCODE_GO_BASE_URL,
        api_key=api_key,
        responses_api=False,
        strict_tools=False,
    )


async def _write_private_file(sbox: Any, path: str, data: bytes) -> None:
    parent = path.rsplit("/", 1)[0]
    mkdir = await sbox.exec(["mkdir", "-p", parent])
    if not mkdir.success:
        raise RuntimeError(
            f"could not create subscription credential directory {parent}"
        )
    await sbox.write_file(path, data)
    result = await sbox.exec(["chmod", "600", path])
    if not result.success:
        raise RuntimeError(f"could not protect subscription credential at {path}")


async def _seed_claude_auth(sbox: Any) -> tuple[dict[str, str], str]:
    """Install a disposable Claude subscription credential copy."""

    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if token:
        return {"CLAUDE_CODE_OAUTH_TOKEN": token}, "CLAUDE_CODE_OAUTH_TOKEN"
    encoded = os.environ.get(CLAUDE_AUTH_B64_ENV)
    if encoded:
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as error:
            raise RuntimeError(f"{CLAUDE_AUTH_B64_ENV} is not valid base64") from error
        data = _valid_json_object(data, label=CLAUDE_AUTH_B64_ENV)
        await _write_private_file(sbox, "/root/.claude/.credentials.json", data)
        return {}, CLAUDE_AUTH_B64_ENV
    data = _claude_file_bytes()
    source = "host_claude_credentials_file"
    if data is None:
        data = _claude_keychain_bytes()
        source = "macos_keychain"
    if data is None:
        raise RuntimeError("Claude Code subscription credential disappeared after preflight")
    await _write_private_file(sbox, "/root/.claude/.credentials.json", data)
    return {}, source


def claude_credentials_b64() -> str | None:
    """Host Claude subscription login as one shippable base64 env value."""

    data = _claude_file_bytes()
    if data is None:
        data = _claude_keychain_bytes()
    if data is None:
        return None
    return base64.b64encode(data).decode("ascii")


async def _seed_codex_auth(sbox: Any, codex_binary: str) -> str:
    """Install a disposable Codex ChatGPT/access-token login copy."""

    # A raw blob is intentionally a per-run override. The gzip blob is the
    # persisted project snapshot loaded from .env, so checking it first would
    # silently discard a fresh credential supplied on the command line.
    encoded = os.environ.get(CODEX_AUTH_B64_ENV)
    if encoded:
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as error:
            raise RuntimeError(f"{CODEX_AUTH_B64_ENV} is not valid base64") from error
        data = _valid_json_object(data, label=CODEX_AUTH_B64_ENV)
        await _write_private_file(sbox, "/root/.codex/auth.json", data)
        return CODEX_AUTH_B64_ENV

    access_token = os.environ.get("CODEX_ACCESS_TOKEN")
    if access_token:
        result = await sbox.exec(
            [codex_binary, "login", "--with-access-token"],
            input=access_token,
            env={"CODEX_HOME": "/root/.codex"},
            timeout=60,
            concurrency=False,
        )
        if not result.success:
            raise RuntimeError(
                "Codex access-token login failed inside the sandbox: "
                + (result.stderr or result.stdout)[-500:]
            )
        return "CODEX_ACCESS_TOKEN"

    compressed = os.environ.get(CODEX_AUTH_GZIP_B64_ENV)
    if compressed:
        try:
            packed = base64.b64decode(compressed, validate=True)
            with gzip.GzipFile(fileobj=io.BytesIO(packed)) as compressed_file:
                data = compressed_file.read(MAX_CODEX_AUTH_JSON_BYTES + 1)
        except (ValueError, TypeError, OSError, EOFError) as error:
            raise RuntimeError(
                f"{CODEX_AUTH_GZIP_B64_ENV} is not valid gzip-base64"
            ) from error
        if len(data) > MAX_CODEX_AUTH_JSON_BYTES:
            raise RuntimeError(
                f"{CODEX_AUTH_GZIP_B64_ENV} expands beyond "
                f"{MAX_CODEX_AUTH_JSON_BYTES} bytes"
            )
        data = _valid_json_object(data, label=CODEX_AUTH_GZIP_B64_ENV)
        await _write_private_file(sbox, "/root/.codex/auth.json", data)
        return CODEX_AUTH_GZIP_B64_ENV

    data = _codex_file_bytes()
    if data is None:
        raise RuntimeError("Codex subscription credential disappeared after preflight")
    await _write_private_file(sbox, "/root/.codex/auth.json", data)
    return "host_codex_auth_file"


def _json_lines(text: str) -> list[dict]:
    records: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"native CLI emitted invalid JSONL at line {line_number}"
            ) from error
        if isinstance(value, dict):
            records.append(value)
    return records


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces = []
        for item in value:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                pieces.append(str(item.get("text") or item.get("content") or ""))
            else:
                pieces.append(str(item))
        return "\n".join(piece for piece in pieces if piece)
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"input": value}
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    return {"input": value}


def _usage_record(raw: dict | None, *, openai_total_input: bool) -> dict:
    reported = isinstance(raw, dict) and any(
        isinstance(raw.get(key), (int, float))
        for key in (
            "input_tokens", "input", "output_tokens", "output",
            "total_tokens", "cached_input_tokens", "cache_read_input_tokens",
        )
    )
    raw = raw if isinstance(raw, dict) else {}
    cache_read = int(
        raw.get("cache_read_input_tokens")
        or raw.get("cached_input_tokens")
        or raw.get("cache_read")
        or 0
    )
    cache_write = int(
        raw.get("cache_creation_input_tokens")
        or raw.get("cache_write_input_tokens")
        or raw.get("cache_write")
        or 0
    )
    reported_input = int(raw.get("input_tokens") or raw.get("input") or 0)
    ordinary_input = (
        max(0, reported_input - cache_read - cache_write)
        if openai_total_input
        else reported_input
    )
    output = int(raw.get("output_tokens") or raw.get("output") or 0)
    reasoning = raw.get("reasoning_output_tokens") or raw.get("reasoning_tokens")
    return {
        "reported": reported,
        "input": ordinary_input,
        "output": output,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "reasoning": int(reasoning) if isinstance(reasoning, (int, float)) else None,
        "total": int(raw.get("total_tokens") or ordinary_input + cache_read + cache_write + output),
    }


def _sum_usage_records(records: Sequence[dict]) -> dict:
    """Element-wise sum of usage records, reported only if any input was."""

    totals = {
        key: sum(int(record.get(key) or 0) for record in records)
        for key in ("input", "output", "cache_read", "cache_write", "reasoning", "total")
    }
    return {
        "reported": any(record.get("reported") for record in records),
        **totals,
    }


def _inspect_usage(record: dict):
    from inspect_ai.model import ModelUsage

    if not record.get("reported"):
        return None
    return ModelUsage(
        input_tokens=record["input"],
        output_tokens=record["output"],
        total_tokens=record["total"],
        input_tokens_cache_read=record["cache_read"],
        input_tokens_cache_write=record["cache_write"],
        reasoning_tokens=record.get("reasoning"),
        # Subscription quota is not a per-run dollar charge. Keep this absent so
        # the viewer does not mislabel the included usage as exact $0 API spend.
        total_cost=None,
    )


def _emit_model_event(
    *,
    routed_slug: str,
    input_messages: Sequence[Any],
    assistant: Any,
    usage: dict,
    reasoning_effort: str | None,
    record_usage: bool = True,
) -> Any:
    from inspect_ai.event import ModelEvent
    from inspect_ai.log import transcript
    from inspect_ai.model import GenerateConfig, ModelOutput

    output = ModelOutput.from_message(
        assistant,
        stop_reason="tool_calls" if assistant.tool_calls else "stop",
    )
    output.model = SUBSCRIPTION_EVENT_MODEL_PREFIX + routed_slug
    output.usage = _inspect_usage(usage)
    event = ModelEvent(
        model=SUBSCRIPTION_EVENT_MODEL_PREFIX + routed_slug,
        role="target",
        input=list(input_messages),
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(reasoning_effort=reasoning_effort),
        output=output,
    )
    transcript()._event(event)
    if record_usage:
        _record_subscription_usage(routed_slug, usage)
    return event


_UNPRICED_SLUGS_REGISTERED: set[str] = set()


def _record_subscription_usage(routed_slug: str, usage: dict) -> None:
    """Feed native CLI usage into Inspect's official model/role tallies.

    A hand-emitted ModelEvent alone never reaches ``log.stats.model_usage`` or
    ``sample.role_usage``; without this call the dead-target integrity check,
    the ``target_no_output`` sample check, and the runtime-accounting manifest
    all see zero target usage for every direct subscription run.
    """

    recorded = _inspect_usage(usage)
    if recorded is None:
        return
    from inspect_ai.model import ModelInfo, set_model_info

    # Private inspect_ai path; the exact inspect-swe/inspect_ai pin is asserted
    # at agent build time.
    from inspect_ai.model._model import record_and_check_model_usage

    slug = SUBSCRIPTION_EVENT_MODEL_PREFIX + routed_slug
    if slug not in _UNPRICED_SLUGS_REGISTERED:
        # Inspect's model-info lookup fuzzy-matches, so the subscription slug
        # would resolve to the underlying API model and stamp API-list dollars
        # onto included-quota usage. An exact cost-less registration wins over
        # the fuzzy path and keeps total_cost absent.
        set_model_info(slug, ModelInfo())
        _UNPRICED_SLUGS_REGISTERED.add(slug)
    record_and_check_model_usage(slug, recorded, role="target")


def parse_claude_stream(
    records: Sequence[dict],
    *,
    prior_messages: Sequence[Any],
    routed_slug: str,
    reasoning: bool,
    include_system_marker: bool,
) -> ParsedInvocation:
    """Translate Claude Code stream-json records into Inspect messages/events."""

    from inspect_ai._util.content import ContentReasoning, ContentText
    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool
    from inspect_ai.tool import ToolCall

    parsed = ParsedInvocation(messages=list(prior_messages))
    if include_system_marker and not any(
        getattr(message, "role", None) == "system" for message in parsed.messages
    ):
        parsed.messages.insert(0, ChatMessageSystem(content=CLAUDE_HIDDEN_SYSTEM_MARKER))

    # Claude Code streams one assistant record PER CONTENT BLOCK; records
    # sharing a message id are chunks of ONE provider response (a thinking-only
    # chunk is not an empty response). Each record repeats a partial usage
    # snapshot, so usage is taken once per response, from the last snapshot.
    pending_id: str | None = None
    pending_content: list[Any] = []
    pending_tool_calls: list[Any] = []
    pending_usage_raw: dict | None = None

    def flush_response() -> None:
        nonlocal pending_id, pending_usage_raw
        if pending_content or pending_tool_calls:
            assistant = ChatMessageAssistant(
                content=list(pending_content) or "",
                tool_calls=list(pending_tool_calls) or None,
                model=routed_slug,
                source="generate",
            )
            usage = _usage_record(pending_usage_raw, openai_total_input=False)
            event = _emit_model_event(
                routed_slug=routed_slug,
                input_messages=parsed.messages,
                assistant=assistant,
                usage=usage,
                reasoning_effort="medium" if reasoning else None,
                # Snapshots are partial; the official tally is recorded once
                # per invocation from the authoritative result usage.
                record_usage=False,
            )
            parsed.messages.append(assistant)
            parsed.model_events.append(event)
            if usage["reported"]:
                parsed.usage.append(usage)
            else:
                parsed.unmetered_model_calls += 1
            parsed.output = event.output
        pending_id = None
        pending_content.clear()
        pending_tool_calls.clear()
        pending_usage_raw = None

    for raw in records:
        record_type = raw.get("type")
        if record_type == "system":
            if raw.get("subtype") == "init":
                parsed.native_version = str(raw.get("claude_code_version") or "") or None
                parsed.native_session_id = (
                    str(raw.get("session_id") or "") or parsed.native_session_id
                )
            elif raw.get("subtype") == "compact_boundary":
                parsed.loss_events.append({
                    "kind": "context_compaction",
                    "event_uuid": raw.get("uuid"),
                    "source": "claude_code",
                    "tokens_before": (raw.get("compactMetadata") or {}).get("preTokens"),
                })
            continue
        if record_type == "rate_limit_event":
            parsed.rate_limits.append({
                "source": "claude_code_rate_limit_event",
                "rate_limit_info": raw.get("rate_limit_info"),
                "session_id": raw.get("session_id"),
                "event_uuid": raw.get("uuid"),
            })
            continue
        if record_type == "assistant":
            message = raw.get("message") or {}
            message_id = str(message.get("id") or "") or None
            if pending_id is not None and message_id != pending_id:
                flush_response()
            pending_id = message_id
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "text":
                    pending_content.append(
                        ContentText(text=str(block.get("text") or ""))
                    )
                elif kind == "thinking":
                    thinking = str(block.get("thinking") or "")
                    pending_content.append(ContentReasoning(
                        reasoning=thinking,
                        summary=thinking or None,
                        signature=block.get("signature"),
                    ))
                elif kind == "redacted_thinking":
                    pending_content.append(ContentReasoning(
                        reasoning="",
                        summary=None,
                        redacted=True,
                        internal={"redacted_data_present": bool(block.get("data"))},
                    ))
                elif kind == "tool_use":
                    pending_tool_calls.append(ToolCall(
                        id=str(block.get("id") or uuid.uuid4().hex),
                        function=str(block.get("name") or "unknown_tool"),
                        arguments=_arguments(block.get("input")),
                    ))
            if isinstance(message.get("usage"), dict):
                pending_usage_raw = message["usage"]
            if message_id is None:
                # No id to group by: treat this record as a complete response.
                flush_response()
            continue
        if record_type == "user":
            flush_response()
            for block in (raw.get("message") or {}).get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                parsed.messages.append(ChatMessageTool(
                    content=_text(block.get("content")),
                    tool_call_id=str(block.get("tool_use_id") or "") or None,
                    function=None,
                    tool_error=(
                        "Claude Code reported is_error=true"
                        if block.get("is_error") else None
                    ),
                ))
            continue
        if record_type == "result":
            flush_response()
            parsed.native_session_id = (
                str(raw.get("session_id") or "") or parsed.native_session_id
            )
            # total_cost_usd is an API-list equivalent for subscribers, not a bill.
            model_usage = raw.get("modelUsage") or raw.get("model_usage") or {}
            if not isinstance(model_usage, dict):
                model_usage = {}
            parsed.rate_limits.append({
                "source": "claude_code_result",
                "session_id": raw.get("session_id"),
                "num_turns": raw.get("num_turns"),
                "duration_ms": raw.get("duration_ms"),
                "duration_api_ms": raw.get("duration_api_ms"),
                "api_list_equivalent_usd": raw.get("total_cost_usd"),
                "model_usage": model_usage,
            })
            authoritative = _claude_result_usage(model_usage)
            if authoritative is not None:
                parsed.authoritative_usage = authoritative
    flush_response()
    return parsed


def _claude_result_usage(model_usage: dict) -> dict | None:
    """This invocation's authoritative usage from Claude's own result record.

    Covers every model the CLI used (helper models included), unlike the
    partial per-record stream snapshots.
    """

    by_model: dict[str, dict] = {}
    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    for model_name, raw in model_usage.items():
        if not isinstance(raw, dict):
            continue
        row = {
            "input": int(raw.get("inputTokens") or 0),
            "output": int(raw.get("outputTokens") or 0),
            "cache_read": int(raw.get("cacheReadInputTokens") or 0),
            "cache_write": int(raw.get("cacheCreationInputTokens") or 0),
        }
        by_model[str(model_name)] = row
        for key in totals:
            totals[key] += row[key]
    if not by_model:
        return None
    return {
        "reported": True,
        **totals,
        "reasoning": None,
        "total": sum(totals.values()),
        "source": "claude_code_result_model_usage",
        "by_model": by_model,
    }


def _codex_content_text(content: Any, *types: str) -> str:
    wanted = set(types)
    return "\n".join(
        str(block.get("text") or "")
        for block in (content or [])
        if isinstance(block, dict) and block.get("type") in wanted
        and block.get("text")
    )


def _codex_reasoning_text(payload: dict) -> str:
    summary = payload.get("summary") or []
    if isinstance(summary, str):
        return summary
    if isinstance(summary, list):
        return "\n".join(
            str(item.get("text") or item)
            for item in summary
            if item
        )
    return ""


def parse_codex_rollout(
    records: Sequence[dict],
    *,
    prior_messages: Sequence[Any],
    routed_slug: str,
    reasoning: bool,
    initial_session: bool,
) -> ParsedInvocation:
    """Translate newly appended Codex rollout records into Inspect messages/events."""

    from inspect_ai._util.content import ContentReasoning, ContentText
    from inspect_ai.model import (
        ChatMessageAssistant,
        ChatMessageSystem,
        ChatMessageTool,
        ChatMessageUser,
    )
    from inspect_ai.tool import ToolCall

    parsed = ParsedInvocation(messages=[] if initial_session else list(prior_messages))
    # One provider response spans several rollout items in this order (verified
    # against a real 0.146.1 rollout): reasoning, assistant message, tool calls,
    # tool OUTPUTS, and only then its single token_count. So the assistant
    # message must enter the transcript as soon as an output follows it (to keep
    # message order), while the accounting event stays open until the usage
    # arrives.
    pending_reasoning: list[ContentReasoning] = []
    pending_texts: list[ContentText] = []
    pending_tool_calls: list[ToolCall] = []
    materialized: tuple[list[Any], Any] | None = None

    def emit_materialized(usage_raw: dict | None) -> None:
        nonlocal materialized
        if materialized is None:
            return
        event_input, assistant = materialized
        materialized = None
        usage = _usage_record(usage_raw, openai_total_input=True)
        event = _emit_model_event(
            routed_slug=routed_slug,
            input_messages=event_input,
            assistant=assistant,
            usage=usage,
            # Codex subscription models do not expose a true no-reasoning setting.
            reasoning_effort="medium" if reasoning else "minimal",
        )
        parsed.model_events.append(event)
        if usage["reported"]:
            parsed.usage.append(usage)
        else:
            parsed.unmetered_model_calls += 1
        parsed.output = event.output

    def materialize_pending() -> None:
        nonlocal materialized
        if not (pending_reasoning or pending_texts or pending_tool_calls):
            return
        # A new response is starting while an earlier one still awaits its
        # token_count: emit the earlier one as unmetered instead of merging two
        # responses into one message.
        emit_materialized(None)
        assistant = ChatMessageAssistant(
            content=[*pending_reasoning, *pending_texts] or "",
            tool_calls=list(pending_tool_calls) or None,
            model=routed_slug,
            source="generate",
        )
        pending_reasoning.clear()
        pending_texts.clear()
        pending_tool_calls.clear()
        event_input = list(parsed.messages)
        parsed.messages.append(assistant)
        materialized = (event_input, assistant)

    for raw in records:
        outer_type = raw.get("type")
        payload = raw.get("payload") or {}
        payload_type = payload.get("type")
        if outer_type == "session_meta":
            parsed.native_version = str(payload.get("cli_version") or "") or None
            base = payload.get("base_instructions")
            base_text = (
                str(base.get("text") or "") if isinstance(base, dict) else str(base or "")
            )
            if initial_session and base_text:
                parsed.messages.append(ChatMessageSystem(content=base_text))
                parsed.system_prompt_observed = True
            continue
        if outer_type == "response_item" and payload_type == "reasoning":
            summary = _codex_reasoning_text(payload)
            pending_reasoning.append(ContentReasoning(
                reasoning=summary,
                summary=summary or None,
                redacted=not bool(summary),
                internal={"encrypted_content_present": bool(payload.get("encrypted_content"))},
            ))
            continue
        if outer_type == "response_item" and payload_type == "message":
            role = str(payload.get("role") or "")
            if role == "assistant":
                text = _codex_content_text(payload.get("content"), "output_text", "text")
                if text:
                    pending_texts.append(ContentText(text=text))
            elif initial_session and role in {"developer", "system"}:
                text = _codex_content_text(payload.get("content"), "input_text", "text")
                if text:
                    materialize_pending()
                    emit_materialized(None)
                    parsed.messages.append(ChatMessageSystem(
                        content=text,
                        # Codex sends its harness instructions under the OpenAI
                        # "developer" role; keep that identity on the record.
                        metadata=(
                            {"native_role": "developer"}
                            if role == "developer" else None
                        ),
                    ))
            elif initial_session and role == "user":
                text = _codex_content_text(payload.get("content"), "input_text", "text")
                if text:
                    materialize_pending()
                    emit_materialized(None)
                    injected = next(
                        (
                            tag
                            for tag in ("environment_context", "user_instructions")
                            if text.lstrip().startswith(f"<{tag}>")
                        ),
                        None,
                    )
                    parsed.messages.append(ChatMessageUser(
                        content=text,
                        # Genuinely user-role on the wire, but Codex injected
                        # the content itself; the controller never sent it.
                        metadata={"scaffold_injected": injected} if injected else None,
                    ))
            continue
        if outer_type == "response_item" and payload_type in {
            "function_call", "custom_tool_call", "local_shell_call"
        }:
            function = str(
                payload.get("name")
                or ("shell" if payload_type == "local_shell_call" else "unknown_tool")
            )
            call_id = str(payload.get("call_id") or payload.get("id") or uuid.uuid4().hex)
            raw_input = payload.get("arguments", payload.get("input"))
            if payload_type == "local_shell_call" and raw_input is None:
                raw_input = {"action": payload.get("action")}
            pending_tool_calls.append(ToolCall(
                id=call_id,
                function=function,
                arguments=_arguments(raw_input),
            ))
            continue
        if outer_type == "response_item" and payload_type in {
            "function_call_output", "custom_tool_call_output", "local_shell_call_output"
        }:
            # The response's token_count arrives AFTER its outputs, so only
            # place the assistant message; its accounting event stays open.
            materialize_pending()
            parsed.messages.append(ChatMessageTool(
                content=_text(payload.get("output")),
                tool_call_id=str(payload.get("call_id") or "") or None,
                function=None,
            ))
            continue
        if outer_type == "event_msg" and payload_type == "token_count":
            info = payload.get("info") or {}
            materialize_pending()
            emit_materialized(info.get("last_token_usage"))
            total = info.get("total_token_usage")
            if isinstance(total, dict):
                parsed.native_total_usage = _usage_record(
                    total, openai_total_input=True
                )
            rate_limits = payload.get("rate_limits")
            if isinstance(rate_limits, dict):
                parsed.rate_limits.append(rate_limits)
            continue
        if outer_type == "event_msg" and payload_type in {
            "context_compacted", "compaction"
        }:
            parsed.loss_events.append({
                "kind": "context_compaction",
                "event_uuid": payload.get("id") or raw.get("timestamp"),
                "source": "codex_cli",
            })

    # A future CLI may omit token_count on an otherwise successful final message.
    # Keep the response; the unmetered_model_calls counter records the gap.
    materialize_pending()
    emit_materialized(None)
    return parsed


async def _find_codex_rollout(sbox: Any, session_id: str) -> str:
    result = await sbox.exec([
        "find", "/root/.codex/sessions", "-type", "f",
        "-name", f"*{session_id}.jsonl", "-print",
    ])
    paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one Codex rollout for session {session_id}, found {paths}"
        )
    return paths[0]


def _subscription_env_base() -> dict[str, str]:
    # Compose supplies the allow-listed HTTPS proxy. These flags disable optional
    # analytics/update traffic and leave only model/auth requests.
    # Do not set CLAUDE_CODE_SUBPROCESS_ENV_SCRUB. In pinned Claude Code 2.1.220,
    # that hardening mode overrides enableWeakerNestedSandbox and makes bubblewrap
    # attempt a fresh /proc mount that unprivileged Docker rejects. Claude's
    # sandbox.credentials rules above remove its actual OAuth token instead.
    return {
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "DISABLE_AUTOUPDATER": "1",
        "IS_SANDBOX": "1",
        "NO_BROWSER": "1",
        "RUST_LOG": "warning",
    }


@asynccontextmanager
async def _subscription_tool_bridge(bridged_tools: Sequence[Any] | None):
    """Start Inspect's MCP bridge only when check_time must be exposed."""

    if not bridged_tools:
        yield SimpleNamespace(mcp_server_configs=[])
        return
    from inspect_ai.agent import AgentState, sandbox_agent_bridge

    async with sandbox_agent_bridge(
        AgentState(messages=[]), bridged_tools=bridged_tools
    ) as bridge:
        yield bridge


def build_subscription_agent(
    target_name: str,
    routed_slug: str,
    *,
    reasoning: bool,
    time_tool: Any | None = None,
    native_resume: dict | None = None,
):
    """Build a subscription agent, with API fallback for unsupported OpenCode models."""

    from inspect_ai.agent import AgentState, BridgedToolsSpec, agent, agent_with
    from inspect_ai.model import ChatMessageSystem
    from inspect_ai.util import ExecRemoteAwaitableOptions
    from inspect_ai.util import sandbox as sandbox_env
    from inspect_swe._codex_cli.config import codex_config_options
    from inspect_swe._codex_cli.agentbinary import codex_cli_binary_source
    from inspect_swe._claude_code.agentbinary import claude_code_binary_source
    from inspect_swe._util.agentbinary import ensure_agent_binary_installed
    from inspect_swe._util.toml import to_toml

    from exp_target_harness import (
        PRODUCTION_SCAFFOLD_VERSIONS,
        assert_inspect_swe_pin,
        build_production_agent,
        codex_subscription_model,
        opencode_go_model_spec,
        production_scaffold_for_target,
    )

    assert_inspect_swe_pin()
    scaffold = production_scaffold_for_target(target_name, routed_slug)
    if scaffold == "opencode":
        go_spec = opencode_go_model_spec(target_name, routed_slug)
        if go_spec is not None:
            go_model = build_opencode_go_model(
                target_name,
                routed_slug,
                reasoning=reasoning,
            )
            requested_model = f"opencode-go/{go_spec['model']}"
            return build_production_agent(
                target_name,
                routed_slug,
                time_tool=time_tool,
                native_resume=native_resume,
                opencode_model_override=requested_model,
                # OpenCode's provider sends the bare model id on both its
                # Anthropic and OpenAI-compatible request paths. Keep the full
                # spelling too so a provider-side change remains deterministic.
                opencode_model_aliases={
                    go_spec["model"]: go_model,
                    requested_model: go_model,
                },
            )
        return build_production_agent(
            target_name,
            routed_slug,
            time_tool=time_tool,
            native_resume=native_resume,
        )

    resume_session_id = (
        native_resume.get("native_session_id") if native_resume else None
    )
    if native_resume is not None and not resume_session_id:
        # Without this, Claude would --resume the literal string "None" and
        # Codex would silently start a fresh session while claiming a resume.
        raise RuntimeError(
            f"{scaffold} subscription continuation requires a native_session_id in "
            "the resume bundle; this bundle came from a run whose first CLI call "
            "never completed"
        )
    session_ref = NativeSessionRef(
        value=str(resume_session_id) if resume_session_id else None
    )
    bridged_tools = (
        [BridgedToolsSpec(name="environment", tools=[time_tool])]
        if time_tool is not None else None
    )

    @agent(name="Claude Code subscription" if scaffold == "claude_code" else "Codex subscription")
    def subscription_agent():
        async def execute(state: AgentState) -> AgentState:
            sbox = sandbox_env()
            session_id_change: dict | None = None
            # The bridge is used only for check_time MCP. Native model calls bypass it.
            async with _subscription_tool_bridge(bridged_tools) as tool_bridge:
                if scaffold == "claude_code":
                    binary = await ensure_agent_binary_installed(
                        claude_code_binary_source(),
                        PRODUCTION_SCAFFOLD_VERSIONS[scaffold],
                        None,
                        sbox,
                    )
                    if not session_ref.auth_seeded:
                        auth_env, auth_source = await _seed_claude_auth(sbox)
                        session_ref.auth_source = auth_source
                        session_ref.auth_seeded = True
                    else:
                        auth_source = session_ref.auth_source or "unknown"
                        auth_env = {}
                        if auth_source == "CLAUDE_CODE_OAUTH_TOKEN":
                            token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
                            if not token:
                                raise RuntimeError(
                                    "CLAUDE_CODE_OAUTH_TOKEN disappeared during the run"
                                )
                            auth_env["CLAUDE_CODE_OAUTH_TOKEN"] = token
                    if session_ref.value is None:
                        session_ref.value = str(uuid.uuid4())
                    has_assistant = any(
                        getattr(message, "role", None) == "assistant"
                        for message in state.messages
                    )
                    prompt = next(
                        (
                            message.text
                            for message in reversed(state.messages)
                            if getattr(message, "role", None) == "user"
                        ),
                        "",
                    )
                    cmd = [
                        binary,
                        "--print",
                        "--output-format",
                        "stream-json",
                        "--verbose",
                        "--safe-mode",
                        "--permission-mode",
                        "dontAsk",
                        "--settings",
                        json.dumps(
                            _claude_security_settings(),
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        "--tools",
                        "Bash,Read,Edit,Write,Glob,Grep,NotebookEdit",
                        "--no-chrome",
                        "--disable-slash-commands",
                        "--strict-mcp-config",
                        "--model",
                        routed_slug.split("/", 1)[-1],
                    ]
                    if reasoning:
                        cmd.extend(["--effort", "medium"])
                    if tool_bridge.mcp_server_configs:
                        from inspect_swe._claude_code.claude_code import resolve_mcp_servers

                        mcp_args, allowed = resolve_mcp_servers(tool_bridge.mcp_server_configs)
                        cmd.extend(mcp_args)
                        if allowed:
                            cmd.extend(["--allowed-tools", ",".join(allowed)])
                    cmd.extend(["--disallowed-tools", "WebSearch,WebFetch"])
                    if has_assistant:
                        cmd.extend(["--resume", session_ref.value])
                    else:
                        cmd.extend(["--session-id", session_ref.value])
                        system_texts = [
                            message.text for message in state.messages
                            if isinstance(message, ChatMessageSystem)
                            and message.text != CLAUDE_HIDDEN_SYSTEM_MARKER
                        ]
                        if system_texts:
                            cmd.extend(["--append-system-prompt", "\n\n".join(system_texts)])
                    cmd.extend(["--", prompt])
                    agent_env = _subscription_env_base() | auth_env
                    if not reasoning:
                        agent_env["CLAUDE_CODE_DISABLE_THINKING"] = "1"
                        agent_env["MAX_THINKING_TOKENS"] = "0"
                    result = await sbox.exec_remote(
                        cmd=cmd,
                        options=ExecRemoteAwaitableOptions(
                            cwd="/workspace",
                            env=agent_env,
                            concurrency=False,
                        ),
                        stream=False,
                    )
                    if not result.success:
                        # Both streams: CLIs put harmless notes on stderr and the
                        # real JSON error events on stdout.
                        raise RuntimeError(
                            f"Claude Code subscription agent failed ({result.returncode}); "
                            f"stderr tail: {(result.stderr or '')[-1000:]!r}; "
                            f"stdout tail: {(result.stdout or '')[-2000:]!r}"
                        )
                    parsed = parse_claude_stream(
                        _json_lines(result.stdout),
                        prior_messages=state.messages,
                        routed_slug=routed_slug,
                        reasoning=reasoning,
                        include_system_marker=not has_assistant,
                    )
                    if (
                        parsed.native_session_id
                        and parsed.native_session_id != session_ref.value
                    ):
                        # A resume can fork onto a new id; follow-up turns must
                        # target the id the CLI actually used.
                        session_id_change = {
                            "from": session_ref.value,
                            "to": parsed.native_session_id,
                        }
                        session_ref.value = parsed.native_session_id
                    # One official tally record per invocation, from Claude's
                    # own accounting (stream snapshots are partial).
                    _record_subscription_usage(
                        routed_slug,
                        parsed.authoritative_usage or _sum_usage_records(parsed.usage),
                    )
                else:
                    binary = await ensure_agent_binary_installed(
                        codex_cli_binary_source(),
                        PRODUCTION_SCAFFOLD_VERSIONS[scaffold],
                        None,
                        sbox,
                    )
                    if not session_ref.auth_seeded:
                        auth_source = await _seed_codex_auth(sbox, binary)
                        session_ref.auth_source = auth_source
                        session_ref.auth_seeded = True
                    else:
                        auth_source = session_ref.auth_source or "unknown"
                    subscription_model = codex_subscription_model(routed_slug)
                    config = _codex_security_config(
                        model=subscription_model,
                        reasoning=reasoning,
                    )
                    config.update(codex_config_options("disabled", True))
                    for mcp in tool_bridge.mcp_server_configs:
                        config[f"mcp_servers.{mcp.name}"] = (
                            _codex_mcp_server_config(mcp)
                        )
                    await sbox.exec(["mkdir", "-p", "/root/.codex"])
                    config_text = to_toml(config) + _codex_filesystem_deny_toml()
                    await sbox.write_file("/root/.codex/config.toml", config_text)
                    initial = session_ref.value is None
                    rollout_path = None
                    baseline = 0
                    if not initial:
                        rollout_path = await _find_codex_rollout(sbox, session_ref.value)
                        baseline = len((await sbox.read_file(rollout_path)).splitlines())
                    prompt = next(
                        (
                            message.text
                            for message in reversed(state.messages)
                            if getattr(message, "role", None) == "user"
                        ),
                        "",
                    )
                    common = [
                        "--json",
                        "--skip-git-repo-check",
                        "--ignore-rules",
                        "--strict-config",
                        "--model",
                        subscription_model,
                    ]
                    cmd, prompt_input = _codex_exec_transport(
                        binary,
                        common,
                        None if initial else session_ref.value,
                        prompt,
                    )
                    result = await sbox.exec_remote(
                        cmd=cmd,
                        options=ExecRemoteAwaitableOptions(
                            input=prompt_input,
                            cwd="/workspace",
                            env=_subscription_env_base() | {"CODEX_HOME": "/root/.codex"},
                            concurrency=False,
                        ),
                        stream=False,
                    )
                    if not result.success:
                        raise RuntimeError(
                            f"Codex subscription agent failed ({result.returncode}); "
                            f"stderr tail: {(result.stderr or '')[-1000:]!r}; "
                            f"stdout tail: {(result.stdout or '')[-2000:]!r}"
                        )
                    stdout_records = _json_lines(result.stdout)
                    if initial:
                        started = next(
                            (row for row in stdout_records if row.get("type") == "thread.started"),
                            None,
                        )
                        session_ref.value = str((started or {}).get("thread_id") or "") or None
                        if session_ref.value is None:
                            raise RuntimeError("Codex JSON stream did not report thread_id")
                        rollout_path = await _find_codex_rollout(sbox, session_ref.value)
                    assert rollout_path is not None
                    rollout_records = _json_lines(await sbox.read_file(rollout_path))
                    parsed = parse_codex_rollout(
                        rollout_records[baseline:],
                        prior_messages=state.messages,
                        routed_slug=routed_slug,
                        reasoning=reasoning,
                        initial_session=initial,
                    )

                state.messages = parsed.messages
                state.output = parsed.output
                if state.output is None:
                    raise RuntimeError(f"{scaffold} subscription CLI returned no agent output")
                invocation_record = {
                    "auth_source": auth_source,
                    "native_session_id": session_ref.value,
                    "native_session_id_changed": session_id_change,
                    "usage": parsed.usage,
                    "authoritative_usage": parsed.authoritative_usage,
                    "unmetered_model_calls": parsed.unmetered_model_calls,
                    "native_total_token_usage": parsed.native_total_usage,
                    "rate_limits": parsed.rate_limits,
                    "loss_events": parsed.loss_events,
                    "system_prompt_observed": parsed.system_prompt_observed,
                    "native_version": parsed.native_version,
                }
                session_ref.last_invocation = invocation_record
                session_ref.invocations.append(invocation_record)
                return state

        return execute

    return agent_with(subscription_agent(), name="subscription_agent"), session_ref


def session_id_value(value: str | NativeSessionRef | None) -> str | None:
    if isinstance(value, NativeSessionRef):
        return value.value
    return value


def record_subscription_native_version(harness: dict, native_version: str | None) -> None:
    """Record the CLI-reported version and refuse a mismatch with the exact pin."""

    if not native_version:
        # Codex resume invocations append no session_meta, so a missing version
        # can be legitimate. Store an explicit not-reported marker instead of
        # passing silently, without clobbering an earlier reported version.
        # (update_resolved_scaffold_version still enforces the installed pin.)
        harness.setdefault("native_reported_scaffold_version", None)
        harness.setdefault("native_reported_scaffold_version_matches_pin", None)
        return
    expected = harness.get("scaffold_version_selector")
    harness["native_reported_scaffold_version"] = native_version
    harness["native_reported_scaffold_version_matches_pin"] = (
        native_version == expected
    )
    if expected and native_version != expected:
        raise RuntimeError(
            f"native {harness.get('scaffold')} reported version {native_version}, "
            f"which does not match pin {expected}"
        )


def subscription_agent_record(session_ref: NativeSessionRef | None) -> dict:
    if not isinstance(session_ref, NativeSessionRef):
        return {}
    all_usage = [
        usage
        for invocation in session_ref.invocations
        for usage in invocation.get("usage") or []
    ]
    per_invocation_usage: list[dict] = []
    usage_sources: list[str] = []
    for invocation in session_ref.invocations:
        authoritative = invocation.get("authoritative_usage")
        if isinstance(authoritative, dict) and authoritative.get("reported"):
            per_invocation_usage.append(authoritative)
            usage_sources.append("cli_reported_result_usage")
        else:
            per_invocation_usage.append(
                _sum_usage_records(invocation.get("usage") or [])
            )
            usage_sources.append("per_call_stream_sum")
    usage_totals = {
        key: sum(int(usage.get(key) or 0) for usage in per_invocation_usage)
        for key in ("input", "output", "cache_read", "cache_write", "reasoning", "total")
    }
    native_versions = list(dict.fromkeys(
        str(invocation["native_version"])
        for invocation in session_ref.invocations
        if invocation.get("native_version")
    ))
    # Claude Code's result records report what the session WOULD have cost at
    # API list prices (per --print process, summed across invocations). This is
    # quota context for a subscription run, not a bill.
    api_equivalents = [
        entry["api_list_equivalent_usd"]
        for invocation in session_ref.invocations
        for entry in invocation.get("rate_limits") or []
        if isinstance(entry, dict)
        and entry.get("source") == "claude_code_result"
        and isinstance(entry.get("api_list_equivalent_usd"), (int, float))
    ]
    unmetered = sum(
        int(invocation.get("unmetered_model_calls") or 0)
        for invocation in session_ref.invocations
    )
    return {
        **session_ref.last_invocation,
        "invocation_count": len(session_ref.invocations),
        "model_call_count": len(all_usage),
        "unmetered_model_call_count": unmetered,
        "api_list_equivalent_usd_per_invocation": api_equivalents,
        "api_list_equivalent_usd_total": (
            sum(api_equivalents) if api_equivalents else None
        ),
        "system_prompt_observed": any(
            bool(invocation.get("system_prompt_observed"))
            for invocation in session_ref.invocations
        ),
        "native_versions": native_versions,
        "native_version": native_versions[-1] if native_versions else None,
        "usage_totals": usage_totals,
        "usage_totals_source": sorted(set(usage_sources)),
        "all_usage": all_usage,
        "all_rate_limits": [
            snapshot
            for invocation in session_ref.invocations
            for snapshot in invocation.get("rate_limits") or []
        ],
        "all_loss_events": [
            event
            for invocation in session_ref.invocations
            for event in invocation.get("loss_events") or []
        ],
    }
