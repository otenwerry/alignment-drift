"""Provider prompt-prefix caching for environment experiments.

This is deliberately separate from Inspect's response cache.  A provider prompt-cache
hit reuses only computation for an identical prompt prefix; every request still samples
a fresh response.  Inspect's response cache remains disabled.

Parallel epochs create a practical problem: identical requests can all reach a provider
before the first cache entry exists.  ``install_inspect_warmup`` wraps the paid Inspect
providers so one cache-sized prefix completes before matching requests fan out.  The
module stores only hashes of prompts and writes honest, per-run cache evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

from inspect_ai.model import ChatMessageBase, GenerateConfig


logger = logging.getLogger(__name__)
T = TypeVar("T")

# Conservative character proxy for the lowest common 1,024-token cache floor.
_MIN_CACHE_PROMPT_CHARS = 3_000


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def stable_key(namespace: str, *parts: Any) -> str:
    """Hash exact prefix inputs without retaining their potentially large text."""

    raw = json.dumps(
        [namespace, *(_jsonable(part) for part in parts)],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


class PromptCacheWarmupBarrier:
    """Allow one matching request to finish before its parallel peers start."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._warmed: set[str] = set()
        self.waited_requests = 0
        self.warmed_prefixes = 0
        self.failed_warmups = 0

    async def run(
        self,
        key: str,
        call: Callable[[], Awaitable[T]],
        *,
        succeeded: Callable[[T], bool] | None = None,
    ) -> T:
        if key in self._warmed:
            return await call()
        lock = self._locks.setdefault(key, asyncio.Lock())
        if lock.locked():
            self.waited_requests += 1
        async with lock:
            if key not in self._warmed:
                try:
                    result = await call()
                except BaseException:
                    self.failed_warmups += 1
                    raise
                if succeeded is None or succeeded(result):
                    self._warmed.add(key)
                    self.warmed_prefixes += 1
                else:
                    self.failed_warmups += 1
                return result
        # The lock is released before waiters make their own fresh calls.
        return await call()

    def stats(self) -> dict[str, int]:
        return {
            "warmed_prefixes": self.warmed_prefixes,
            "requests_held_for_warmup": self.waited_requests,
            "failed_warmups": self.failed_warmups,
        }


_barrier = PromptCacheWarmupBarrier()
_installed = False


def _cache_key_message(message: Any) -> Any:
    """Serialize a message without Inspect's provider-invisible bookkeeping ID."""

    if not isinstance(message, ChatMessageBase):
        return _jsonable(message)
    payload = message.model_dump(mode="json", exclude_none=False)
    payload.pop("id", None)
    return _jsonable(payload)


def _openrouter_session_id(model: Any, input_messages: Any) -> str | None:
    """Derive stable provider routing from the conversation opening."""

    if not isinstance(input_messages, (list, tuple)) or not input_messages:
        return None
    first_system = next(
        (message for message in input_messages if getattr(message, "role", None) == "system"),
        None,
    )
    first_non_system = next(
        (message for message in input_messages if getattr(message, "role", None) != "system"),
        None,
    )
    opening = [
        _cache_key_message(message)
        for message in (first_system, first_non_system)
        if message is not None
    ]
    if not opening:
        return None
    digest = stable_key(
        "environments-openrouter-session-v1",
        getattr(model, "model_name", None),
        getattr(model, "base_url", None),
        opening,
    )
    return f"environments-prefix-{digest[:48]}"


def _with_openrouter_session(
    model: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    input_messages = kwargs.get("input", args[0] if args else None)
    session_id = _openrouter_session_id(model, input_messages)
    if session_id is None:
        return args, kwargs

    config = kwargs.get("config", args[3] if len(args) > 3 else None) or GenerateConfig()
    if not isinstance(config, GenerateConfig):
        return args, kwargs
    extra_body = dict(config.extra_body or {})
    if "session_id" in extra_body:
        return args, kwargs
    extra_body["session_id"] = session_id
    routed_config = config.model_copy(update={"extra_body": extra_body}, deep=True)

    routed_args = list(args)
    routed_kwargs = dict(kwargs)
    if "config" in routed_kwargs:
        routed_kwargs["config"] = routed_config
    elif len(routed_args) > 3:
        routed_args[3] = routed_config
    else:
        routed_kwargs["config"] = routed_config
    return tuple(routed_args), routed_kwargs


def _request_key(model: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    input_messages = kwargs.get("input", args[0] if args else None)
    tools = kwargs.get("tools", args[1] if len(args) > 1 else None)
    tool_choice = kwargs.get("tool_choice", args[2] if len(args) > 2 else None)
    config = kwargs.get("config", args[3] if len(args) > 3 else None)
    if input_messages is None:
        return None

    # The final user message is usually the changing question. Providers can cache the
    # stable history before it, so group on that reusable prefix.
    message_list = list(input_messages) if isinstance(input_messages, (list, tuple)) else None
    stable_messages = message_list[:-1] if message_list else input_messages
    stable_payload = (
        [_cache_key_message(message) for message in stable_messages]
        if isinstance(stable_messages, list)
        else _cache_key_message(stable_messages)
    )
    payload = _jsonable([stable_payload, tools, tool_choice, config])
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if len(serialized) < _MIN_CACHE_PROMPT_CHARS:
        return None
    return stable_key(
        "environments-inspect-provider-request-v1",
        getattr(model, "model_name", None),
        getattr(model, "base_url", None),
        payload,
    )


def _succeeded(result: Any) -> bool:
    output = result[0] if isinstance(result, tuple) and result else result
    return not isinstance(output, BaseException)


def _patch_provider(provider: type, *, openrouter: bool = False) -> None:
    original = provider.generate
    if getattr(original, "_environments_prompt_cache_warmup", False):
        return

    async def generate(self, *args, **kwargs):
        key = _request_key(self, args, kwargs)
        if key is None:
            return await original(self, *args, **kwargs)
        call_args, call_kwargs = (
            _with_openrouter_session(self, args, kwargs)
            if openrouter
            else (args, kwargs)
        )
        routed_key = _request_key(self, call_args, call_kwargs)
        assert routed_key is not None
        return await _barrier.run(
            routed_key,
            lambda: original(self, *call_args, **call_kwargs),
            succeeded=_succeeded,
        )

    generate._environments_prompt_cache_warmup = True
    provider.generate = generate


def install_inspect_warmup() -> bool:
    """Install the warm-first barrier on Inspect's paid API providers."""

    global _installed
    if _installed:
        return True
    try:
        from inspect_ai.model._providers.anthropic import AnthropicAPI
        from inspect_ai.model._providers.openai import OpenAIAPI
        from inspect_ai.model._providers.openrouter import OpenRouterAPI
    except Exception as exc:
        logger.warning("provider prompt-cache warmup disabled: %s", exc)
        return False
    _patch_provider(AnthropicAPI)
    _patch_provider(OpenAIAPI)
    _patch_provider(OpenRouterAPI, openrouter=True)
    _installed = True
    return True


def _model_status(model: str, cache_read: int, cache_write: int) -> str:
    if cache_read > 0:
        return "verified_cache_read"
    if cache_write > 0:
        return "cache_write_only_no_reuse_observed"
    if model.startswith(("anthropic/", "openai/", "openrouter/")):
        return "no_cache_activity_observed"
    return "provider_cache_support_unknown"


def build_report(logs: list[Any]) -> dict[str, Any]:
    """Build provider-neutral evidence from completed Inspect log headers."""

    totals: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "total_cost_usd": 0.0,
        }
    )
    for log in logs:
        stats = getattr(log, "stats", None)
        for model, usage in (getattr(stats, "model_usage", None) or {}).items():
            row = totals[model]
            row["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
            row["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
            row["cache_read_input_tokens"] += getattr(usage, "input_tokens_cache_read", 0) or 0
            row["cache_write_input_tokens"] += getattr(usage, "input_tokens_cache_write", 0) or 0
            row["total_cost_usd"] += getattr(usage, "total_cost", 0) or 0

    models: dict[str, dict[str, Any]] = {}
    for model, raw in sorted(totals.items()):
        row = dict(raw)
        row["total_cost_usd"] = round(float(row["total_cost_usd"]), 8)
        row["status"] = _model_status(
            model,
            int(row["cache_read_input_tokens"]),
            int(row["cache_write_input_tokens"]),
        )
        prompt_tokens = (
            int(row["input_tokens"])
            + int(row["cache_read_input_tokens"])
            + int(row["cache_write_input_tokens"])
        )
        row["cache_read_fraction_of_prompt_tokens"] = (
            round(int(row["cache_read_input_tokens"]) / prompt_tokens, 6)
            if prompt_tokens
            else None
        )
        models[model] = row

    verified = [model for model, row in models.items() if row["status"] == "verified_cache_read"]
    return {
        "schema_version": "environments-prompt-cache-v1",
        "semantics": "provider prompt-prefix caching; every response is freshly generated",
        "inspect_response_cache_enabled": False,
        "warmup_barrier": {
            "installed": _installed,
            "key_version": "v1-ignore-provider-invisible-message-id",
            "openrouter_session_routing": "v1-stable-from-conversation-opening",
            **_barrier.stats(),
        },
        "models": models,
        "verified_models": verified,
        "unverified_models": sorted(set(models) - set(verified)),
    }


def write_report(run_dir: Path, logs: list[Any]) -> dict[str, Any]:
    report = build_report(logs)
    path = Path(run_dir) / "prompt_cache_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    read = sum(int(row["cache_read_input_tokens"]) for row in report["models"].values())
    write = sum(int(row["cache_write_input_tokens"]) for row in report["models"].values())
    print(
        f"[prompt cache] {read:,} tokens read, {write:,} written; "
        f"verified models={len(report['verified_models'])}, "
        f"unverified={len(report['unverified_models'])}; evidence -> {path}",
        flush=True,
    )
    return report
