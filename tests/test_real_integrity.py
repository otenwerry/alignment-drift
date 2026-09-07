"""Free tests for shared pipeline/viewer integrity flags."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from real_integrity import (  # noqa: E402
    finalize_audit_integrity,
    integrity_sidecar_records,
    target_provider_events,
)


def ns(**values):
    return SimpleNamespace(**values)


def test_original_clock_limit_is_benchmark_only_even_with_usable_output() -> None:
    audit = {
        "real_env": {
            "protocol": {"ended_reason": "wall_clock_limit"},
            "artifacts": [{"path": "submission.csv"}],
            "compute": {"pipeline_exit_code": 1},
        },
        "real_ended_reason": "wall_clock_limit",
        "target_tools_mode": "real",
        "judgment": {"format": "legacy_numeric"},
        "target_provider_events": [],
        "tool_truncations": [],
        "compactions": [],
        "dead": False,
    }

    finalize_audit_integrity(audit)

    assert audit["integrity_status"] == "included"
    assert audit["mechanical_status"] == "benchmark_only"
    assert audit["benchmark_eligible"] is True
    assert audit["prefix_eligible"] is False
    assert [tag["code"] for tag in audit["status_tags"]] == [
        "ended_partway", "time_limit",
    ]
    assert [tag["severity"] for tag in audit["status_tags"]] == [
        "warning", "warning",
    ]
    assert {flag["code"] for flag in audit["flags"]} == {
        "pipeline_exit_nonzero", "wall_clock_limit",
    }


def test_abnormal_ended_reason_is_invalid_but_protocol_end_is_valid() -> None:
    def audit_for(reason: str) -> dict:
        return {
            "real_env": {
                "protocol": {"ended_reason": reason},
                "artifacts": [{"path": "submission.csv"}],
            },
            "real_ended_reason": reason,
            "target_tools_mode": "real",
            "judgment": {"format": "legacy_numeric"},
            "target_provider_events": [],
            "tool_truncations": [],
            "compactions": [],
            "dead": False,
        }

    interrupted = finalize_audit_integrity(audit_for("interrupted"))
    assert interrupted["integrity_status"] == "excluded"
    assert interrupted["mechanical_status"] == "invalid"
    assert {flag["code"] for flag in interrupted["flags"]} == {"ended_early"}
    assert interrupted["flags"][0]["label"] == "ended: interrupted"
    assert [tag["code"] for tag in interrupted["status_tags"]] == [
        "ended_partway", "technical_failure",
    ]

    normal = finalize_audit_integrity(audit_for("protocol_end"))
    assert normal["flags"] == []
    assert normal["mechanical_status"] == "valid"
    assert normal["status_tags"] == []


def test_continuation_clock_limit_is_valid_but_prefix_ineligible() -> None:
    audit = {
        "real_env": {
            "protocol": {"ended_reason": "wall_clock_limit"},
            "artifacts": [{"path": "submission.csv"}],
            "continuation": {"prefix": {"name": "source"}},
        },
        "target_tools_mode": "real",
        "judgment": {"format": "legacy_numeric"},
        "target_provider_events": [],
        "stored_integrity": {"issues": ["wall_clock_limit"]},
        "tool_truncations": [],
        "compactions": [],
        "dead": False,
    }

    finalize_audit_integrity(audit)

    assert audit["mechanical_status"] == "valid"
    assert audit["integrity_status"] == "included"
    assert audit["benchmark_eligible"] is True
    assert audit["prefix_eligible"] is False
    assert audit["integrity_issues"] == []
    assert audit["prefix_issues"] == ["wall_clock_limit"]
    assert [tag["code"] for tag in audit["status_tags"]] == [
        "ended_partway", "time_limit",
    ]
    assert [tag["severity"] for tag in audit["status_tags"]] == [
        "warning", "warning",
    ]


def test_multi_agent_clock_limit_retains_strict_completion_rule() -> None:
    audit = {
        "real_env": {
            "protocol": {"ended_reason": "wall_clock_limit"},
            "artifacts": [{"path": "submission.csv"}],
            "multi_agent": {"exposure": {"fully_delivered": True}},
        },
        "target_tools_mode": "real",
        "judgment": {"format": "legacy_numeric"},
        "target_provider_events": [],
        "tool_truncations": [],
        "compactions": [],
        "dead": False,
    }

    finalize_audit_integrity(audit)

    assert audit["mechanical_status"] == "invalid"
    assert audit["integrity_issues"] == ["wall_clock_limit"]
    assert audit["prefix_issues"] == []


def test_clock_limit_with_no_output_is_invalid_not_benchmark_only() -> None:
    audit = {
        "real_env": {
            "protocol": {"ended_reason": "wall_clock_limit"},
            "artifacts": [{"path": "submission.csv"}],
        },
        "target_tools_mode": "real",
        "judgment": {"format": "legacy_numeric"},
        "target_provider_events": [],
        "tool_truncations": [],
        "compactions": [],
        "dead": True,
    }

    finalize_audit_integrity(audit)

    assert audit["mechanical_status"] == "invalid"
    assert [tag["code"] for tag in audit["status_tags"]] == [
        "no_output", "time_limit",
    ]
    assert [tag["severity"] for tag in audit["status_tags"]] == [
        "error", "warning",
    ]


def test_unfinished_action_is_benchmark_only_for_original_and_valid_for_continuation() -> None:
    base = {
        "real_env": {
            "protocol": {"ended_reason": "protocol_end"},
            "artifacts": [{"path": "submission.csv"}],
        },
        "target_tools_mode": "real",
        "judgment": {"format": "legacy_numeric"},
        "messages": [{"role": "assistant", "tool_calls": [{"id": "call-1"}]}],
        "target_provider_events": [],
        "tool_truncations": [],
        "compactions": [],
        "dead": False,
    }

    original = finalize_audit_integrity(base)
    continuation = finalize_audit_integrity({
        **base,
        "real_env": {**base["real_env"], "continuation": {"prefix": {}}},
        "stored_integrity": {"issues": ["unfinished_action"]},
    })

    assert original["mechanical_status"] == "benchmark_only"
    assert continuation["mechanical_status"] == "valid"
    assert continuation["benchmark_eligible"] is True
    assert continuation["prefix_eligible"] is False
    assert continuation["integrity_issues"] == []
    assert continuation["prefix_issues"] == ["unfinished_action"]
    assert [tag["code"] for tag in original["status_tags"]] == [
        "ended_partway", "unfinished_action",
    ]


def test_pipeline_sidecar_records_continuation_timeout_as_valid_nonprefix() -> None:
    sample = ns(
        id="fraud_detection",
        epoch=3,
        events=[],
        messages=[ns(role="assistant", content="finished work", tool_calls=[])],
        role_usage={"target": ns(output_tokens=12)},
        metadata={"real_env": {
            "protocol": {"ended_reason": "wall_clock_limit"},
            "artifacts": [{"path": "submission.csv"}],
            "continuation": {"prefix": {"name": "source"}},
            "gates": [],
        }},
        scores={"environment_judge": ns(metadata={
            "environment_judge": {
                "result": {"environment_failure": {"value": False}},
                "post_validation": "passed",
            }
        })},
    )
    log = ns(
        eval=ns(
            task="continuation_test",
            model_roles={"target": ns(model="mock/target")},
        ),
        samples=[sample],
    )

    record = integrity_sidecar_records([log])[0]

    assert record["status"] == "included"
    assert record["mechanical_status"] == "valid"
    assert record["benchmark_eligible"] is True
    assert record["prefix_eligible"] is False
    assert record["issues"] == []
    assert record["prefix_issues"] == ["wall_clock_limit"]


def test_native_resume_failure_is_prefix_only_for_an_original() -> None:
    audit = {
        "real_env": {
            "protocol": {"ended_reason": "protocol_end"},
            "artifacts": [{"path": "submission.csv"}],
            "harness": {"mode": "production"},
            "native_resume_bundle": {"available": False},
        },
        "target_tools_mode": "real",
        "judgment": {"format": "legacy_numeric"},
        "messages": [{"role": "assistant", "content": "done"}],
        "target_provider_events": [],
        "tool_truncations": [],
        "compactions": [],
        "dead": False,
    }

    finalize_audit_integrity(audit)

    assert audit["mechanical_status"] == "benchmark_only"
    assert audit["prefix_issues"] == ["native_resume_bundle_unavailable"]
    assert [tag["code"] for tag in audit["status_tags"]] == ["technical_failure"]


def test_environment_failure_judgment_is_invalid() -> None:
    audit = {
        "real_env": {
            "protocol": {"ended_reason": "protocol_end"},
            "artifacts": [{"path": "submission.csv"}],
        },
        "target_tools_mode": "real",
        "judgment": {
            "format": "structured",
            "envelope": {"result": {}, "post_validation": "passed"},
            "dimensions": [{
                "key": "environment_failure", "status": "ok", "value": True,
            }],
        },
        "target_provider_events": [],
        "tool_truncations": [],
        "compactions": [],
        "dead": False,
    }

    finalize_audit_integrity(audit)

    assert audit["mechanical_status"] == "invalid"
    assert [tag["code"] for tag in audit["status_tags"]] == ["environment_failed"]


def test_every_judge_evidence_issue_becomes_a_specific_flag() -> None:
    caveats = [
        ("messages_excluded_by_family_policy", "judge messages omitted"),
        ("native_reasoning_excluded_by_policy", "judge reasoning omitted"),
        ("tool_calls_excluded_by_family_policy", "judge tool calls omitted"),
        ("tool_results_excluded_by_family_policy", "judge tool results omitted"),
        ("native_reasoning_not_fully_available_upstream", "reasoning incomplete"),
        ("changed_non_text_artifacts_not_copied", "non-text artifacts omitted"),
        ("target_scaffold_context_compacted", "context compacted"),
        ("target_scaffold_tool_output_pruned", "tool output pruned"),
        ("initial_task_snapshot_unavailable_upstream", "initial task unavailable"),
        ("submission_snapshots_unavailable_upstream", "submission snapshots unavailable"),
        ("final_artifacts_unavailable_upstream", "final artifacts unavailable"),
        ("prior_unrelated_task_omitted", "prior task omitted"),
    ]
    audit = {
        "real_env": {
            "protocol": {"ended_reason": "protocol_end"},
            "artifacts": {"files": []},
        },
        "judgment": {
            "format": "structured",
            "envelope": {
                "result": {},
                "post_validation": "passed",
                "evidence": {
                    "caveats": [
                        {
                            "code": code,
                            "description": f"detail for {code}",
                        }
                        for code, _ in caveats
                    ],
                    "artifacts": [{
                        "path": "/workspace/REPORT.md",
                        "snapshot": "final",
                        "truncated": True,
                    }],
                },
            },
        },
        "target_provider_events": [],
        "tool_truncations": [],
        "compactions": [],
        "dead": False,
    }

    finalize_audit_integrity(audit)

    labels = {flag["label"] for flag in audit["flags"]}
    assert labels == {
        *(label for _, label in caveats),
        "artifact incomplete ×1",
    }
    assert audit["integrity_status"] == "included"


def test_unrecovered_empty_target_response_is_stored_and_excluded() -> None:
    message = ns(content="", tool_calls=[], refusal=None)
    choice = ns(message=message, stop_reason="stop")
    event = ns(
        event="model", role="target", model="mock/target", input=["prompt"],
        tools=[], config={}, output=ns(choices=[choice], usage=None), error=None,
    )
    sample = ns(events=[event])

    events = target_provider_events(sample, "mock/target")
    audit = {
        "real_env": {"protocol": {}, "artifacts": []},
        "target_tools_mode": "real",
        "judgment": None,
        "target_provider_events": events,
        "tool_truncations": [],
        "compactions": [],
        "dead": False,
    }
    finalize_audit_integrity(audit)

    assert events == [{"kind": "empty_response", "attempt": 1, "stop_reason": "stop"}]
    assert "target_provider_empty_response" in audit["integrity_issues"]
    assert audit["integrity_status"] == "excluded"


def test_reasoning_only_target_response_is_recorded_but_not_blocking() -> None:
    message = ns(
        content=[ns(type="reasoning", reasoning="I am still working.")],
        tool_calls=[], refusal=None,
    )
    event = ns(
        event="model", role="target", model="mock/target", input=["prompt"],
        tools=[], config={}, output=ns(
            choices=[ns(message=message, stop_reason="stop")], usage=ns()
        ), error=None,
    )
    sample = ns(events=[event], metadata={})

    events = target_provider_events(sample, "mock/target")

    assert events == [{
        "kind": "reasoning_only_response",
        "attempt": 1,
        "stop_reason": "stop",
    }]


def test_empty_response_followed_by_another_target_call_is_nonblocking_retry() -> None:
    empty = ns(content="", tool_calls=[], refusal=None)
    answer = ns(content="done", tool_calls=[], refusal=None)
    events = [
        ns(
            event="model", role="target", model="mock/target", input=["prompt"],
            tools=[], config={}, output=ns(
                choices=[ns(message=empty, stop_reason="stop")], usage=ns()
            ), error=None,
        ),
        ns(
            event="model", role="target", model="mock/target", input=["prompt"],
            tools=[], config={}, output=ns(
                choices=[ns(message=answer, stop_reason="stop")], usage=ns()
            ), error=None,
        ),
    ]

    anomalies = target_provider_events(ns(events=events, metadata={}), "mock/target")

    assert anomalies == [{
        "kind": "empty_response_retried",
        "attempt": 1,
        "next_attempt": 2,
        "stop_reason": "stop",
    }]


def test_terminal_empty_from_interrupted_opencode_call_is_nonblocking() -> None:
    message = ns(content="", tool_calls=[], refusal=None)
    event = ns(
        event="model", role="target", model="mock/target", input=["prompt"],
        tools=[], config={}, output=ns(
            choices=[ns(message=message, stop_reason="stop")], usage=ns()
        ), error=None,
    )
    sample = ns(
        events=[event],
        metadata={"real_env": {
            "harness": {"scaffold": "opencode"},
            "protocol": {"ended_reason": "wall_clock_limit"},
        }},
    )

    anomalies = target_provider_events(sample, "mock/target")

    assert anomalies == [{
        "kind": "empty_response_interrupted",
        "attempt": 1,
        "stop_reason": "stop",
    }]


def test_current_provider_classification_supersedes_old_empty_sidecar_issue() -> None:
    audit = {
        "real_env": {
            "protocol": {"ended_reason": "wall_clock_limit"},
            "artifacts": {"files": []},
        },
        "target_tools_mode": "real",
        "judgment": {"format": "legacy_numeric"},
        "target_provider_events": [{
            "kind": "reasoning_only_response", "attempt": 1,
        }],
        "stored_integrity": {"issues": ["target_provider_empty_response"]},
        "tool_truncations": [],
        "compactions": [],
        "dead": False,
    }

    finalize_audit_integrity(audit)

    assert "target_provider_empty_response" not in audit["integrity_issues"]
    assert not any(
        flag["code"] == "target_provider_empty_response"
        for flag in audit["flags"]
    )


def test_pipeline_sidecar_records_per_sample_provider_failure() -> None:
    message = ns(content="", tool_calls=[], refusal=None)
    event = ns(
        event="model", role="target", model="mock/target", input=["prompt"],
        tools=[], config={}, output=ns(
            choices=[ns(message=message, stop_reason="content_filter")], usage=None
        ), error=None,
    )
    sample = ns(
        id="fraud_detection", epoch=3, events=[event], messages=[], role_usage={},
        metadata={"real_env": {
            "protocol": {"ended_reason": "cancelled"},
            "artifacts": [{"path": "submission.csv"}],
            "gates": [],
        }},
        scores={},
    )
    log = ns(
        eval=ns(task="real_audit_test", model_roles={"target": ns(model="mock/target")}),
        samples=[sample],
    )

    records = integrity_sidecar_records([log])

    assert records[0]["status"] == "excluded"
    assert records[0]["task"] == "real_audit_test"
    assert "target_provider_content_filter" in records[0]["issues"]
    assert records[0]["target_provider_events"][0]["kind"] == "content_filter"
