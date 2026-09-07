"""Free tests for direct subscription transcript and accounting adapters."""

from __future__ import annotations

import asyncio
import base64
import gzip
import importlib.util
import inspect
import pathlib
import sys
import tomllib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.tool import MCPServerConfigHTTP
from inspect_swe._util.toml import to_toml


ENVIRONMENTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

import exp_subscription_harness as subscription  # noqa: E402
from cost_tracking import estimate_usage_cost  # noqa: E402
from exp_target_harness import (  # noqa: E402
    codex_subscription_model,
    opencode_go_model_spec,
    subscription_harness_metadata,
)


def _fake_emit(**kwargs):
    return SimpleNamespace(output=SimpleNamespace(message=kwargs["assistant"]))


def test_claude_stream_records_hidden_system_prompt_and_subscription_usage(
    monkeypatch,
) -> None:
    monkeypatch.setattr(subscription, "_emit_model_event", _fake_emit)
    parsed = subscription.parse_claude_stream(
        [
            {
                "type": "system",
                "subtype": "init",
                "claude_code_version": "2.1.220",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "answer"}],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 50,
                    },
                },
            },
            {
                "type": "result",
                "session_id": "session",
                "total_cost_usd": 0.123,
                "num_turns": 1,
            },
        ],
        prior_messages=[ChatMessageUser(content="question")],
        routed_slug="anthropic/claude-opus-4-6",
        reasoning=True,
        include_system_marker=True,
    )

    assert [message.role for message in parsed.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert "unavailable" in parsed.messages[0].text.lower()
    assert parsed.usage == [{
        "reported": True,
        "input": 100,
        "output": 20,
        "cache_read": 50,
        "cache_write": 0,
        "reasoning": None,
        "total": 170,
    }]
    assert parsed.rate_limits[0]["api_list_equivalent_usd"] == 0.123
    assert parsed.system_prompt_observed is False
    assert parsed.native_session_id == "session"
    assert parsed.unmetered_model_calls == 0


def test_claude_per_block_records_merge_into_one_response(monkeypatch) -> None:
    """Claude Code streams one record per content block; blocks sharing a
    message id are ONE response (a thinking-only chunk is not an empty
    response), and usage comes once per response from the last snapshot."""

    monkeypatch.setattr(subscription, "_emit_model_event", _fake_emit)
    parsed = subscription.parse_claude_stream(
        [
            {
                "type": "assistant",
                "message": {
                    "id": "msg_1",
                    "content": [{"type": "thinking", "thinking": "let me look"}],
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            },
            {
                "type": "assistant",
                "message": {
                    "id": "msg_1",
                    "content": [
                        {"type": "text", "text": "checking the file"},
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Read",
                            "input": {"file_path": "/workspace/a.py"},
                        },
                    ],
                    "usage": {"input_tokens": 3, "output_tokens": 20},
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": "print('hi')",
                    }],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "id": "msg_2",
                    "content": [{"type": "text", "text": "done"}],
                    "usage": {"input_tokens": 9, "output_tokens": 5},
                },
            },
            {
                "type": "result",
                "session_id": "session",
                "total_cost_usd": 0.5,
                "modelUsage": {
                    "claude-opus-4-6": {
                        "inputTokens": 12,
                        "outputTokens": 700,
                        "cacheReadInputTokens": 1000,
                        "cacheCreationInputTokens": 50,
                    },
                },
            },
        ],
        prior_messages=[ChatMessageUser(content="question")],
        routed_slug="anthropic/claude-opus-4-6",
        reasoning=True,
        include_system_marker=False,
    )

    assert [message.role for message in parsed.messages] == [
        "user", "assistant", "tool", "assistant",
    ]
    merged = parsed.messages[1]
    assert merged.tool_calls is not None and merged.tool_calls[0].id == "tool-1"
    assert "checking the file" in merged.text
    assert any(
        getattr(block, "reasoning", None) == "let me look"
        for block in merged.content
    )
    assert [usage["output"] for usage in parsed.usage] == [20, 5]
    assert parsed.unmetered_model_calls == 0
    assert parsed.authoritative_usage["output"] == 700
    assert parsed.authoritative_usage["cache_read"] == 1000
    assert parsed.authoritative_usage["by_model"]["claude-opus-4-6"]["output"] == 700


def test_codex_rollout_records_native_system_usage_and_quota(monkeypatch) -> None:
    monkeypatch.setattr(subscription, "_emit_model_event", _fake_emit)
    parsed = subscription.parse_codex_rollout(
        [
            {
                "type": "session_meta",
                "payload": {
                    "cli_version": "0.146.1",
                    "base_instructions": {"text": "native base"},
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "developer"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": "<environment_context>\n<cwd>/workspace</cwd>",
                    }],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "question"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"text": "reasoning summary"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 25,
                            "output_tokens": 20,
                            "reasoning_output_tokens": 5,
                            "total_tokens": 120,
                        }
                    },
                    "rate_limits": {
                        "primary": {"used_percent": 7.0, "window_minutes": 300}
                    },
                },
            },
        ],
        prior_messages=[],
        routed_slug="openai/gpt-5.6-sol",
        reasoning=True,
        initial_session=True,
    )

    assert [message.role for message in parsed.messages] == [
        "system",
        "system",
        "user",
        "user",
        "assistant",
    ]
    assert (parsed.messages[1].metadata or {}).get("native_role") == "developer"
    assert (parsed.messages[2].metadata or {}).get("scaffold_injected") == (
        "environment_context"
    )
    assert parsed.messages[3].metadata is None
    assert parsed.system_prompt_observed is True
    assert parsed.usage[0]["input"] == 75
    assert parsed.usage[0]["cache_read"] == 25
    assert parsed.usage[0]["reasoning"] == 5
    assert parsed.rate_limits[0]["primary"]["used_percent"] == 7.0


def test_codex_rollout_merges_one_turn_into_one_assistant_message(monkeypatch) -> None:
    """One response = reasoning + message + tool calls, with its outputs BEFORE
    its single token_count (real 0.146.1 rollout ordering, verified 2026-08-09).
    The usage arriving after the outputs must still attach to the response."""

    monkeypatch.setattr(subscription, "_emit_model_event", _fake_emit)
    parsed = subscription.parse_codex_rollout(
        [
            {
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"text": "thinking"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "running the check"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "call_id": "call-1",
                    "arguments": '{"command": "ls"}',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "call_id": "call-2",
                    "arguments": '{"command": "cat file.txt"}',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "file.txt",
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-2",
                    "output": "contents",
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 40,
                            "output_tokens": 10,
                        },
                        "total_token_usage": {
                            "input_tokens": 40,
                            "output_tokens": 10,
                        },
                    },
                },
            },
        ],
        prior_messages=[ChatMessageUser(content="question")],
        routed_slug="openai/gpt-5.6-sol",
        reasoning=True,
        initial_session=False,
    )

    assert [message.role for message in parsed.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assistant = parsed.messages[1]
    assert assistant.tool_calls is not None
    assert [call.id for call in assistant.tool_calls] == ["call-1", "call-2"]
    assert "running the check" in assistant.text
    assert len(parsed.usage) == 1
    assert parsed.usage[0]["output"] == 10
    assert parsed.unmetered_model_calls == 0
    assert parsed.native_total_usage["output"] == 10
    assert parsed.messages[2].tool_call_id == "call-1"
    assert parsed.messages[3].tool_call_id == "call-2"


def test_codex_rollout_counts_a_response_missing_its_token_count(monkeypatch) -> None:
    monkeypatch.setattr(subscription, "_emit_model_event", _fake_emit)
    parsed = subscription.parse_codex_rollout(
        [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "final answer"}],
                },
            },
        ],
        prior_messages=[ChatMessageUser(content="question")],
        routed_slug="openai/gpt-5.6-sol",
        reasoning=True,
        initial_session=False,
    )

    assert [message.role for message in parsed.messages] == ["user", "assistant"]
    assert parsed.usage == []
    assert parsed.unmetered_model_calls == 1


def test_subscription_metadata_and_cost_are_explicitly_not_metered() -> None:
    metadata = subscription_harness_metadata(
        "opus-4.6", "anthropic/claude-opus-4-6"
    )
    assert metadata["mode"] == "subscription"
    assert metadata["agent_billing"] == "subscription_included_usage"
    assert metadata["per_run_agent_cost_available"] is False
    assert metadata["native_system_prompt_observability"]["complete"] is False
    assert metadata["network_policy"]["container"] == "internal_only"

    priced = estimate_usage_cost(
        "subscription/anthropic/claude-opus-4-6",
        {"input": 100, "output": 10},
    )
    assert priced == {
        "cost_usd": None,
        "exact": False,
        "source": "subscription_not_metered",
    }

    go_metadata = subscription_harness_metadata(
        "qwen3.7-max", "openrouter/qwen/qwen3.7-max"
    )
    assert go_metadata["agent_billing"] == "subscription_included_usage"
    assert go_metadata["subscription_provider"] == "opencode_go"
    assert go_metadata["subscription_model_requested"] == (
        "opencode-go/qwen3.7-max"
    )
    assert go_metadata["credential_isolation"]["sandbox_copy"] is False
    assert go_metadata["network_policy"]["container"] == "none"
    assert estimate_usage_cost(
        "anthropic/opencode-go/qwen3.7-max",
        {"input": 100, "output": 10, "total_cost": 99.0},
    )["source"] == "subscription_not_metered"

    fallback = subscription_harness_metadata(
        "qwen3-32b", "openrouter/qwen/qwen3-32b"
    )
    assert fallback["agent_billing"] == "api_fallback"
    assert fallback["per_run_agent_cost_available"] is True


def test_opencode_go_catalog_mapping_is_explicit() -> None:
    assert opencode_go_model_spec(
        "qwen3.7-max", "openrouter/qwen/qwen3.7-max"
    ) == {
        "model": "qwen3.7-max",
        "protocol": "anthropic",
        "output_tokens": 65_536,
    }
    assert opencode_go_model_spec(
        "glm-5.2", "openrouter/z-ai/glm-5.2"
    ) == {
        "model": "glm-5.2",
        "protocol": "openai_compatible",
        "output_tokens": 131_072,
    }
    assert opencode_go_model_spec(
        "qwen3-32b", "openrouter/qwen/qwen3-32b"
    ) is None


def test_opencode_go_auth_uses_official_env_name_and_rejects_reasoning_off(
    monkeypatch,
) -> None:
    monkeypatch.setenv(subscription.OPENCODE_GO_API_KEY_ENV, "test-go-key")
    assert subscription.require_opencode_go_auth(reasoning=True) == (
        subscription.OPENCODE_GO_API_KEY_ENV
    )
    with pytest.raises(SystemExit, match="reasoning=yes"):
        subscription.require_opencode_go_auth(reasoning=False)


def test_opencode_go_model_stays_on_host_and_is_aliased_into_bridge(
    monkeypatch,
) -> None:
    host_model = object()
    monkeypatch.setattr(subscription, "build_opencode_go_model", lambda *a, **k: host_model)
    with (
        patch("exp_target_harness.assert_inspect_swe_pin"),
        patch(
            "exp_target_harness.build_production_agent",
            return_value=("go-agent", None),
        ) as build,
    ):
        agent, session_ref = subscription.build_subscription_agent(
            "qwen3.7-max",
            "openrouter/qwen/qwen3.7-max",
            reasoning=True,
        )

    assert agent == "go-agent" and session_ref is None
    kwargs = build.call_args.kwargs
    assert kwargs["opencode_model_override"] == "opencode-go/qwen3.7-max"
    assert kwargs["opencode_model_aliases"] == {
        "qwen3.7-max": host_model,
        "opencode-go/qwen3.7-max": host_model,
    }
    assert subscription.OPENCODE_GO_API_KEY_ENV not in kwargs


def test_opencode_go_host_model_uses_the_documented_wire_protocol(monkeypatch) -> None:
    monkeypatch.setenv(subscription.OPENCODE_GO_API_KEY_ENV, "test-go-key")
    with patch("inspect_ai.model.get_model", side_effect=["qwen", "glm"]) as get:
        assert subscription.build_opencode_go_model(
            "qwen3.7-max",
            "openrouter/qwen/qwen3.7-max",
            reasoning=True,
        ) == "qwen"
        assert subscription.build_opencode_go_model(
            "glm-5.2",
            "openrouter/z-ai/glm-5.2",
            reasoning=True,
        ) == "glm"

    qwen_call, glm_call = get.call_args_list
    assert qwen_call.args == ("anthropic/opencode-go/qwen3.7-max",)
    assert qwen_call.kwargs["role"] == "target"
    assert qwen_call.kwargs["config"].max_tokens == 65_536
    assert qwen_call.kwargs["base_url"] == subscription.OPENCODE_GO_BASE_URL
    assert qwen_call.kwargs["api_key"] == "test-go-key"
    assert glm_call.args == ("openai-api/opencode-go/glm-5.2",)
    assert glm_call.kwargs["role"] == "target"
    assert glm_call.kwargs["config"].max_tokens == 131_072
    assert glm_call.kwargs["base_url"] == subscription.OPENCODE_GO_BASE_URL
    assert glm_call.kwargs["api_key"] == "test-go-key"
    assert glm_call.kwargs["responses_api"] is False
    assert glm_call.kwargs["strict_tools"] is False


def test_opencode_go_host_model_providers_construct_without_other_api_keys(
    monkeypatch,
) -> None:
    monkeypatch.setenv(subscription.OPENCODE_GO_API_KEY_ENV, "test-go-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    qwen = subscription.build_opencode_go_model(
        "qwen3.7-max",
        "openrouter/qwen/qwen3.7-max",
        reasoning=True,
    )
    glm = subscription.build_opencode_go_model(
        "glm-5.2",
        "openrouter/z-ai/glm-5.2",
        reasoning=True,
    )

    assert qwen.api.service_model_name() == "qwen3.7-max"
    assert qwen.api.base_url == subscription.OPENCODE_GO_BASE_URL
    assert qwen.config.max_tokens == 65_536
    assert glm.api.service_model_name() == "glm-5.2"
    assert glm.api.base_url == subscription.OPENCODE_GO_BASE_URL
    assert glm.config.max_tokens == 131_072


def test_subscription_native_reported_version_must_match_pin() -> None:
    harness = {
        "scaffold": "codex",
        "scaffold_version_selector": "0.146.1",
    }
    subscription.record_subscription_native_version(harness, "0.146.1")
    assert harness["native_reported_scaffold_version_matches_pin"] is True

    try:
        subscription.record_subscription_native_version(harness, "0.999.0")
    except RuntimeError as error:
        assert "does not match pin" in str(error)
    else:
        raise AssertionError("a mismatched subscription CLI version was accepted")


def test_codex_subscription_model_drops_the_unsupported_snapshot_date() -> None:
    """ChatGPT-account Codex rejects dated API snapshot names (seen live 2026-08-09)."""

    assert codex_subscription_model("openai/gpt-5.5-2026-04-23") == "gpt-5.5"
    assert codex_subscription_model("openai/gpt-5.6-sol") == "gpt-5.6-sol"

    metadata = subscription_harness_metadata("gpt-5.5", "openai/gpt-5.5-2026-04-23")
    assert metadata["subscription_model_requested"] == "gpt-5.5"
    assert metadata["subscription_snapshot_pinning"] == (
        "chatgpt_account_serves_undated_model_snapshot_not_pinnable"
    )
    claude = subscription_harness_metadata("opus-4.6", "anthropic/claude-opus-4-6")
    assert "subscription_model_requested" not in claude


@pytest.mark.parametrize("session_id", [None, "session-123"])
def test_codex_large_prompt_uses_stdin_not_argv(session_id: str | None) -> None:
    prompt = "activity log line\n" * 20_000
    command, stdin = subscription._codex_exec_transport(
        "/usr/bin/codex",
        ["--json", "--model", "gpt-5.5"],
        session_id,
        prompt,
    )

    assert command[-1] == "-"
    assert prompt not in command
    assert sum(len(argument.encode()) + 1 for argument in command) < 1024
    assert stdin == prompt
    if session_id is None:
        assert command[:2] == ["/usr/bin/codex", "exec"]
        assert "resume" not in command
    else:
        assert command[:3] == ["/usr/bin/codex", "exec", "resume"]
        assert command[-2] == session_id


def test_codex_bridge_config_translates_inspect_mcp_schema() -> None:
    mcp = MCPServerConfigHTTP(
        name="environment",
        type="http",
        url="http://localhost:1234/mcp/environment",
        tools=["check_time"],
    )

    config_text = to_toml(
        {"mcp_servers.environment": subscription._codex_mcp_server_config(mcp)}
    )
    parsed = tomllib.loads(config_text)

    assert parsed["mcp_servers"]["environment"] == {
        "url": "http://localhost:1234/mcp/environment",
        "enabled_tools": ["check_time"],
    }
    assert "type =" not in config_text


def test_codex_bridge_config_rejects_legacy_sse() -> None:
    mcp = MCPServerConfigHTTP(
        name="environment",
        type="sse",
        url="http://localhost:1234/mcp/environment",
    )

    with pytest.raises(RuntimeError, match="requires streamable HTTP"):
        subscription._codex_mcp_server_config(mcp)


def test_codex_bridge_config_rejects_headers_the_toml_writer_would_drop() -> None:
    mcp = MCPServerConfigHTTP(
        name="environment",
        type="http",
        url="http://localhost:1234/mcp/environment",
        headers={"X-Bridge": "local"},
    )

    with pytest.raises(RuntimeError, match="does not support MCP headers"):
        subscription._codex_mcp_server_config(mcp)


def test_missing_native_version_is_flagged_not_silent() -> None:
    harness = {"scaffold": "codex", "scaffold_version_selector": "0.146.1"}
    subscription.record_subscription_native_version(harness, None)
    assert harness["native_reported_scaffold_version"] is None
    assert harness["native_reported_scaffold_version_matches_pin"] is None

    subscription.record_subscription_native_version(harness, "0.146.1")
    assert harness["native_reported_scaffold_version"] == "0.146.1"
    assert harness["native_reported_scaffold_version_matches_pin"] is True

    # A later version-less invocation (codex resume) must not clobber it.
    subscription.record_subscription_native_version(harness, None)
    assert harness["native_reported_scaffold_version"] == "0.146.1"
    assert harness["native_reported_scaffold_version_matches_pin"] is True


def test_subscription_agent_record_aggregates_dollars_and_unmetered_calls() -> None:
    session_ref = subscription.NativeSessionRef(value="session")
    session_ref.invocations = [
        {
            # Partial stream snapshots lose to the CLI's own reported usage.
            "usage": [{"reported": True, "input": 100, "output": 10,
                       "cache_read": 0, "cache_write": 0, "reasoning": 0,
                       "total": 110}],
            "authoritative_usage": {
                "reported": True, "input": 120, "output": 900,
                "cache_read": 5000, "cache_write": 40, "reasoning": None,
                "total": 6060,
            },
            "unmetered_model_calls": 0,
            "rate_limits": [{
                "source": "claude_code_result",
                "api_list_equivalent_usd": 0.10,
            }],
        },
        {
            "usage": [{"reported": True, "input": 7, "output": 3,
                       "cache_read": 0, "cache_write": 0, "reasoning": 0,
                       "total": 10}],
            "authoritative_usage": None,
            "unmetered_model_calls": 1,
            "rate_limits": [{
                "source": "claude_code_result",
                "api_list_equivalent_usd": 0.05,
            }],
        },
    ]
    session_ref.last_invocation = session_ref.invocations[-1]

    record = subscription.subscription_agent_record(session_ref)

    assert record["api_list_equivalent_usd_per_invocation"] == [0.10, 0.05]
    assert abs(record["api_list_equivalent_usd_total"] - 0.15) < 1e-9
    assert record["unmetered_model_call_count"] == 1
    assert record["model_call_count"] == 2
    assert record["usage_totals"]["output"] == 903
    assert record["usage_totals"]["input"] == 127
    assert record["usage_totals_source"] == [
        "cli_reported_result_usage", "per_call_stream_sum",
    ]


def test_subscription_resume_requires_a_native_session_id() -> None:
    try:
        subscription.build_subscription_agent(
            "opus-4.6",
            "anthropic/claude-opus-4-6",
            reasoning=True,
            native_resume={"native_session_id": None},
        )
    except RuntimeError as error:
        assert "native_session_id" in str(error)
    else:
        raise AssertionError(
            "a subscription resume without a native session id was accepted"
        )


def test_recorded_subscription_usage_reaches_inspect_tallies_unpriced(
    monkeypatch,
) -> None:
    recorded: dict[str, object] = {}

    def fake_record(model, usage, role=None):
        recorded.update(model=model, usage=usage, role=role)

    monkeypatch.setattr(
        "inspect_ai.model._model.record_and_check_model_usage", fake_record
    )
    usage = subscription._usage_record(
        {"input_tokens": 100, "output_tokens": 20}, openai_total_input=False
    )

    subscription._record_subscription_usage("anthropic/claude-opus-4-6", usage)

    assert recorded["model"] == "subscription/anthropic/claude-opus-4-6"
    assert recorded["role"] == "target"
    assert recorded["usage"].input_tokens == 100
    assert recorded["usage"].output_tokens == 20
    assert recorded["usage"].total_cost is None

    from inspect_ai.model import get_model_info

    info = get_model_info("subscription/anthropic/claude-opus-4-6")
    assert info is not None and info.cost is None


def test_unreported_usage_is_never_recorded_to_inspect(monkeypatch) -> None:
    def fail_record(model, usage, role=None):
        raise AssertionError("zero usage must not be recorded")

    monkeypatch.setattr(
        "inspect_ai.model._model.record_and_check_model_usage", fail_record
    )
    usage = subscription._usage_record(None, openai_total_input=False)
    subscription._record_subscription_usage("anthropic/claude-opus-4-6", usage)


def test_claude_b64_env_seeds_credentials_into_the_sandbox(monkeypatch) -> None:
    credentials = b'{"claudeAiOauth":{"accessToken":"placeholder"}}'
    encoded = base64.b64encode(credentials).decode()
    written: dict[str, object] = {}

    async def fake_write_private_file(sbox, path, data):
        written.update(sbox=sbox, path=path, data=data)

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv(subscription.CLAUDE_AUTH_B64_ENV, encoded)
    monkeypatch.setattr(subscription, "_write_private_file", fake_write_private_file)
    marker = object()

    env, source = asyncio.run(subscription._seed_claude_auth(marker))

    assert env == {}
    assert source == subscription.CLAUDE_AUTH_B64_ENV
    assert written == {
        "sbox": marker,
        "path": "/root/.claude/.credentials.json",
        "data": credentials,
    }
    assert (
        subscription.subscription_auth_source("claude_code")
        == subscription.CLAUDE_AUTH_B64_ENV
    )


def test_subscription_proxy_only_allows_provider_domain_suffixes() -> None:
    proxy_path = ENVIRONMENTS / "sandbox" / "subscription_proxy" / "proxy.py"
    spec = importlib.util.spec_from_file_location("subscription_proxy", proxy_path)
    assert spec is not None and spec.loader is not None
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)

    assert proxy.allowed_host("api.anthropic.com")
    assert proxy.allowed_host("chatgpt.com")
    assert not proxy.allowed_host("anthropic.com.example.org")
    assert not proxy.allowed_host("example.org")


def test_subscription_auth_preflight_accepts_alternatives_and_fails_loudly(
    monkeypatch,
) -> None:
    available = {"claude_code": "macos_keychain", "codex": None}
    monkeypatch.setattr(
        subscription,
        "subscription_auth_source",
        lambda scaffold: available.get(scaffold, "api_fallback"),
    )
    assert subscription.require_subscription_auth(["claude_code"]) == {
        "claude_code": "macos_keychain"
    }
    try:
        subscription.require_subscription_auth(["codex"])
    except SystemExit as error:
        assert "CODEX_ACCESS_TOKEN" in str(error)
    else:
        raise AssertionError("missing Codex subscription login was accepted")


def test_gzip_base64_codex_auth_is_restored_only_inside_sandbox(monkeypatch) -> None:
    auth_json = b'{"auth_mode":"chatgpt","tokens":{"access_token":"placeholder"}}'
    encoded = base64.b64encode(gzip.compress(auth_json)).decode()
    written: dict[str, object] = {}

    async def fake_write_private_file(sbox, path, data):
        written.update(sbox=sbox, path=path, data=data)

    monkeypatch.setenv(subscription.CODEX_AUTH_GZIP_B64_ENV, encoded)
    monkeypatch.setattr(subscription, "_write_private_file", fake_write_private_file)
    marker = object()

    source = asyncio.run(subscription._seed_codex_auth(marker, "/usr/bin/codex"))

    assert source == subscription.CODEX_AUTH_GZIP_B64_ENV
    assert written == {
        "sbox": marker,
        "path": "/root/.codex/auth.json",
        "data": auth_json,
    }


def test_raw_codex_auth_overrides_stale_persisted_gzip(monkeypatch) -> None:
    fresh = b'{"auth_mode":"chatgpt","tokens":{"access_token":"fresh"}}'
    stale = b'{"auth_mode":"chatgpt","tokens":{"access_token":"stale"}}'
    written: dict[str, object] = {}

    async def fake_write_private_file(sbox, path, data):
        written.update(sbox=sbox, path=path, data=data)

    monkeypatch.setenv(
        subscription.CODEX_AUTH_GZIP_B64_ENV,
        base64.b64encode(gzip.compress(stale)).decode(),
    )
    monkeypatch.setenv(
        subscription.CODEX_AUTH_B64_ENV,
        base64.b64encode(fresh).decode(),
    )
    monkeypatch.setattr(subscription, "_write_private_file", fake_write_private_file)
    marker = object()

    assert subscription.subscription_auth_source("codex") == (
        subscription.CODEX_AUTH_B64_ENV
    )
    source = asyncio.run(subscription._seed_codex_auth(marker, "/usr/bin/codex"))

    assert source == subscription.CODEX_AUTH_B64_ENV
    assert written == {
        "sbox": marker,
        "path": "/root/.codex/auth.json",
        "data": fresh,
    }


def test_subscription_native_tools_cannot_read_cli_credentials() -> None:
    claude = subscription._claude_security_settings()
    assert claude["sandbox"]["enabled"] is True
    assert claude["sandbox"]["failIfUnavailable"] is True
    assert claude["sandbox"]["allowUnsandboxedCommands"] is False
    assert claude["sandbox"]["enableWeakerNestedSandbox"] is True
    assert "/root/.claude" in claude["sandbox"]["filesystem"]["denyRead"]
    assert "Read(//root/.claude/**)" in claude["permissions"]["deny"]
    assert claude["sandbox"]["credentials"]["envVars"] == [
        {"name": "CLAUDE_CODE_OAUTH_TOKEN", "mode": "deny"}
    ]
    # Claude Code's general subprocess scrub forces its stronger /proc setup and
    # breaks nested bubblewrap. The credential-specific rules above remain active.
    assert "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB" not in (
        subscription._subscription_env_base()
    )

    codex = subscription._codex_security_config(
        model="gpt-5.6-sol", reasoning=True
    )
    assert codex["default_permissions"] == "subscription-workspace"
    assert codex["permissions.subscription-workspace"]["extends"] == ":workspace"
    assert codex["shell_environment_policy"]["inherit"] == "core"
    assert codex["allow_login_shell"] is False
    assert codex["cli_auth_credentials_store"] == "file"
    assert codex["features.apps"] is False
    assert codex["features.multi_agent"] is False
    filesystem = tomllib.loads(subscription._codex_filesystem_deny_toml())
    rules = filesystem["permissions"]["subscription-workspace"]["filesystem"]
    assert rules["/root/.codex/auth.json"] == "deny"
    assert rules["/proc"] == "deny"

    source = inspect.getsource(subscription.build_subscription_agent)
    assert "dangerously-skip-permissions" not in source
    assert "dangerously-bypass-approvals-and-sandbox" not in source


def test_subscription_images_install_required_nested_sandbox_helpers() -> None:
    for relative in (
        "sandbox/ml/Dockerfile.subscription",
        "sandbox/p_hacking/Dockerfile.subscription",
    ):
        dockerfile = (ENVIRONMENTS / relative).read_text()
        assert "bubblewrap" in dockerfile
        assert "ripgrep" in dockerfile
        assert "socat" in dockerfile
    for relative in (
        "sandbox/ml/compose.subscription.yaml",
        "sandbox/p_hacking/compose.subscription.yaml",
    ):
        compose = (ENVIRONMENTS / relative).read_text()
        assert "seccomp:unconfined" in compose
        assert "apparmor:unconfined" in compose
        assert "cap_add" not in compose
        assert "privileged:" not in compose
        assert "mats-environments-" in compose
