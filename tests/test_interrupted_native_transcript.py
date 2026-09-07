"""Free tests for conservative interrupted OpenCode transcript recovery."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from interrupted_native_transcript import (  # noqa: E402
    DEADLINE_RECOVERY_FORMAT,
    RECOVERY_FORMAT,
    recover_interrupted_opencode_messages,
    recover_predeadline_opencode_submission,
)
from env_viewer_load import sample_to_audit  # noqa: E402


def ns(**values):
    return SimpleNamespace(**values)


def message(role: str, content: object, *, message_id: str) -> SimpleNamespace:
    return ns(role=role, content=content, id=message_id, tool_calls=[])


def model_event(input_messages: list, output_message: object) -> SimpleNamespace:
    return ns(
        event="model",
        role="target",
        model="mock/glm",
        input=input_messages,
        output=ns(
            message=output_message,
            choices=[],
            stop_reason="stop",
            time=2.0,
            error=None,
        ),
        id="event-final",
        timestamp="2026-08-13T12:00:00Z",
        working_start=10.0,
        working_time=None,
        pending=None,
        error=None,
        retries=None,
    )


def test_recovers_newest_native_history_after_exact_quoted_user_boundary() -> None:
    system = message("system", "system", message_id="m0")
    first_user = message("user", "first task", message_id="m1")
    first_answer = message("assistant", "first answer", message_id="m2")
    followup = message("user", "Please finish.", message_id="m3")
    native_followup = message("user", '"Please finish."', message_id="native-u2")
    reasoning = message(
        "assistant",
        [ns(type="reasoning", reasoning="I should inspect the file.")],
        message_id="m4",
    )
    tool_result = message("tool", "file contents", message_id="m5")
    final = message("assistant", "Finished.", message_id="m6")
    event = model_event([native_followup, reasoning, tool_result], final)

    recovered, record = recover_interrupted_opencode_messages(
        [system, first_user, first_answer, followup],
        [event],
        target_model="mock/glm",
        applied_before_judging=True,
    )

    assert recovered == [
        system, first_user, first_answer, followup, reasoning, tool_result, final,
    ]
    assert record is not None
    assert record["format"] == RECOVERY_FORMAT
    assert record["matched_user_wrapper"] == "outer_double_quotes"
    assert record["added_message_count"] == 3
    assert record["terminal_output_included"] is True
    assert record["applied_before_judging"] is True
    assert record["complete"] is False


def test_omits_truly_empty_terminal_output_but_keeps_recorded_history() -> None:
    followup = message("user", "continue", message_id="u2")
    native_followup = message("user", '"continue"', message_id="native-u2")
    action = message("assistant", "working", message_id="a2")
    empty = message("assistant", "", message_id="empty")

    recovered, record = recover_interrupted_opencode_messages(
        [followup],
        [model_event([native_followup, action], empty)],
        applied_before_judging=False,
    )

    assert recovered == [followup, action]
    assert record is not None
    assert record["added_message_count"] == 1
    assert record["terminal_output_included"] is False


def test_collapses_only_exact_duplicate_inspect_message_ids() -> None:
    followup = message("user", "continue", message_id="u2")
    native_followup = message("user", '"continue"', message_id="native-u2")
    reasoning = message("assistant", "working", message_id="duplicate")

    recovered, record = recover_interrupted_opencode_messages(
        [followup],
        [model_event([native_followup, reasoning, reasoning], None)],
        applied_before_judging=True,
    )

    assert recovered == [followup, reasoning]
    assert record is not None
    assert record["message_id_normalization"] == {
        "policy": "collapse_exact_duplicate_inspect_message_ids",
        "exact_duplicate_messages_omitted": 1,
        "duplicate_message_ids": ["duplicate"],
        "conflicting_duplicate_ids": [],
    }


def test_rejects_conflicting_duplicate_inspect_message_ids() -> None:
    followup = message("user", "continue", message_id="u2")
    native_followup = message("user", '"continue"', message_id="native-u2")
    first = message("assistant", "first", message_id="duplicate")
    second = message("assistant", "second", message_id="duplicate")

    with pytest.raises(ValueError, match="non-identical messages"):
        recover_interrupted_opencode_messages(
            [followup],
            [model_event([native_followup, first, second], None)],
            applied_before_judging=True,
        )


def test_attachment_and_resolved_content_are_the_same_duplicate_message() -> None:
    followup = message("user", "continue", message_id="u2")
    resolved = message("assistant", "full output", message_id="duplicate")
    again = message("user", "again", message_id="u3")
    attached = message(
        "assistant", "attachment://aabbcc", message_id="duplicate"
    )

    recovered, record = recover_interrupted_opencode_messages(
        [followup, resolved, again],
        [model_event([
            message("user", '"again"', message_id="native-u3"),
            attached,
        ], None)],
        applied_before_judging=True,
        attachments={"aabbcc": "full output"},
    )

    assert recovered == [followup, resolved, again]
    assert record is None


def test_fails_closed_when_boundary_is_ambiguous_or_not_final_user() -> None:
    followup = message("user", "continue", message_id="u2")
    duplicate_a = message("user", '"continue"', message_id="native-a")
    duplicate_b = message("user", "continue", message_id="native-b")
    output = message("assistant", "answer", message_id="a2")

    ambiguous, ambiguous_record = recover_interrupted_opencode_messages(
        [followup],
        [model_event([duplicate_a, duplicate_b], output)],
        applied_before_judging=False,
    )
    complete, complete_record = recover_interrupted_opencode_messages(
        [followup, output],
        [model_event([duplicate_a], output)],
        applied_before_judging=False,
    )

    assert ambiguous == [followup]
    assert ambiguous_record is None
    assert complete == [followup, output]
    assert complete_record is None


def test_accepts_terminal_response_completed_before_deadline() -> None:
    followup = message("user", "continue", message_id="u2")
    native_followup = message("user", '"continue"', message_id="native-u2")
    final = message("assistant", "Finished.", message_id="a2")
    event = model_event([native_followup], final)

    recovered, transcript_record, deadline_record = (
        recover_predeadline_opencode_submission(
            [followup],
            [event],
            deadline_seconds_from_start=12.5,
            target_model="mock/glm",
            applied_before_judging=False,
        )
    )

    assert recovered == [followup, final]
    assert transcript_record is not None
    assert deadline_record == {
        "format": DEADLINE_RECOVERY_FORMAT,
        "accepted_as_submission": True,
        "rule": "terminal_model_response_completed_before_advertised_deadline",
        "scaffold": "opencode",
        "terminal_event_index": 0,
        "terminal_event_id": "event-final",
        "terminal_event_timestamp": "2026-08-13T12:00:00Z",
        "stop_reason": "stop",
        "response_characters": 9,
        "working_start_seconds": 10.0,
        "successful_model_call_seconds": 2.0,
        "response_completed_seconds_from_start": 12.0,
        "deadline_seconds_from_start": 12.5,
        "margin_before_deadline_seconds": 0.5,
        "timing_source": "event.working_start + output.time",
        "cli_exit_required_before_deadline": False,
    }


def test_rejects_terminal_response_that_completed_after_deadline() -> None:
    followup = message("user", "continue", message_id="u2")
    native_followup = message("user", '"continue"', message_id="native-u2")
    final = message("assistant", "Finished.", message_id="a2")

    _, transcript_record, deadline_record = recover_predeadline_opencode_submission(
        [followup],
        [model_event([native_followup], final)],
        deadline_seconds_from_start=11.9,
        applied_before_judging=False,
    )

    assert transcript_record is not None
    assert deadline_record is None


def test_rejects_pending_or_tool_call_terminal_output() -> None:
    followup = message("user", "continue", message_id="u2")
    native_followup = message("user", '"continue"', message_id="native-u2")
    final = message("assistant", "Finished.", message_id="a2")
    pending = model_event([native_followup], final)
    pending.pending = True
    tool_call = model_event([native_followup], final)
    final.tool_calls = [ns(function="keep_working")]

    for event in (pending, tool_call):
        _, transcript_record, deadline_record = (
            recover_predeadline_opencode_submission(
                [followup],
                [event],
                deadline_seconds_from_start=20.0,
                applied_before_judging=False,
            )
        )
        assert transcript_record is not None
        assert deadline_record is None


def test_viewer_marks_historical_judgment_that_missed_recovered_messages() -> None:
    followup = message("user", "continue", message_id="u2")
    native_followup = message("user", '"continue"', message_id="native-u2")
    action = message("assistant", "second-turn work", message_id="a2")
    final = message("assistant", "second-turn answer", message_id="a3")
    event = model_event([native_followup, action], final)
    event.working_start = 10.0
    event.output.usage = ns(
        input_tokens=100,
        input_tokens_cache_read=0,
        input_tokens_cache_write=0,
    )
    event.error = None
    event.tools = []
    event.config = {}
    sample = ns(
        id="seed", epoch=1, messages=[followup], events=[event], attachments={},
        metadata={"real_env": {
            "harness": {"mode": "production", "scaffold": "opencode"},
            "protocol": {"ended_reason": "wall_clock_limit"},
            "artifacts": {"files": []},
        }},
        scores={
            "audit_judge": ns(value={"behavior": 4}, metadata={}, explanation="")
        },
        role_usage={"target": ns(output_tokens=20)}, model_usage={}, started_at=None,
    )

    audit = sample_to_audit(
        mode="run", mode_mtime=0.0, task="task", run_metadata={
            "target_tools_mode": "real",
            "harness": "production",
        },
        roles={"target": ns(model="mock/glm")}, sample=sample,
    )

    assert audit["transcript_source"] == "reconstructed_interrupted_opencode_events"
    assert [item["text"] for item in audit["messages"]] == [
        "continue", "second-turn work", "second-turn answer",
    ]
    assert audit["judgment_transcript_coverage"] == {
        "complete": False,
        "stored_judgment_predates_reconstruction": True,
        "messages_seen_by_stored_judgment": 1,
        "messages_shown_after_reconstruction": 3,
        "recovered_messages_not_seen_by_stored_judgment": 2,
    }
    assert "judge_missing_recovered_messages" in audit["integrity_issues"]
