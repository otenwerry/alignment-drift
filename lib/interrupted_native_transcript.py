"""Recover observable OpenCode history when its native call is interrupted.

Inspect SWE's OpenCode adapter returns an ``AgentState`` only after the native CLI
process exits.  If the outer sample clock cancels that await, Inspect still has the
individual target ``ModelEvent`` records, but ``TaskState.messages`` remains frozen at
the user message that started the call.  This module conservatively joins the newest
recorded native history back onto that exact user boundary.

Recovery is deliberately partial and fail-closed.  It never invents a tool result or
claims to cover activity after the last recorded target event.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence


RECOVERY_FORMAT = "interrupted-opencode-transcript-v1"
DEADLINE_RECOVERY_FORMAT = "predeadline-opencode-submission-v1"
_ATTACHMENT_RE = re.compile(r"attachment://([0-9a-fA-F]+)")


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _resolve_attachments(text: str, attachments: dict[str, str] | None) -> str:
    attachments = attachments or {}
    return _ATTACHMENT_RE.sub(
        lambda match: attachments.get(match.group(1), match.group(0)), text
    )


def _message_text(
    message: Any, attachments: dict[str, str] | None = None
) -> str:
    text = _get(message, "text")
    if isinstance(text, str):
        return _resolve_attachments(text, attachments)
    content = _get(message, "content", "")
    if isinstance(content, str):
        return _resolve_attachments(content, attachments)
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(_resolve_attachments(block, attachments))
            continue
        for field in ("text", "reasoning", "summary"):
            value = _get(block, field)
            if isinstance(value, str) and value:
                parts.append(_resolve_attachments(value, attachments))
                break
    return "".join(parts)


def _native_user_text(
    message: Any, attachments: dict[str, str] | None = None
) -> tuple[str, str]:
    """Remove only OpenCode's known literal outer-double-quote wrapper."""

    text = _message_text(message, attachments)
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        return text[1:-1], "outer_double_quotes"
    return text, "none"


def _observable_message(message: Any) -> bool:
    """Whether a target output contains text, reasoning, refusal, or a tool call."""

    if message is None:
        return False
    if _get(message, "tool_calls", []) or []:
        return True
    if str(_get(message, "refusal", "") or "").strip():
        return True
    content = _get(message, "content", "")
    if isinstance(content, str):
        return bool(content.strip())
    for block in content or []:
        if isinstance(block, str) and block.strip():
            return True
        for field in ("text", "reasoning", "summary"):
            if str(_get(block, field, "") or "").strip():
                return True
    return False


def _event_output_message(event: Any) -> Any | None:
    output = _get(event, "output")
    message = _get(output, "message")
    if message is not None:
        return message
    choices = _get(output, "choices", []) or []
    return _get(choices[0], "message") if choices else None


def _resolved_record(value: Any, attachments: dict[str, str] | None) -> Any:
    if isinstance(value, str):
        return _resolve_attachments(value, attachments)
    if isinstance(value, dict):
        return {
            str(key): _resolved_record(item, attachments)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_resolved_record(item, attachments) for item in value]
    return value


def _message_record(
    message: Any, attachments: dict[str, str] | None = None
) -> Any:
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        record = dump(mode="json")
    elif isinstance(message, dict):
        record = message
    else:
        record = vars(message) if hasattr(message, "__dict__") else message
    return _resolved_record(record, attachments)


def _unique_recovered_messages(
    base: Sequence[Any],
    recovered: Sequence[Any],
    attachments: dict[str, str] | None = None,
) -> tuple[list[Any], dict | None]:
    """Collapse exact ID duplicates and reject conflicting duplicate IDs."""

    seen: dict[str, Any] = {}
    for message in base:
        identifier = _get(message, "id")
        if identifier is None:
            continue
        key = str(identifier)
        if key in seen:
            raise ValueError(
                f"stored base transcript has duplicate Inspect message ID {key!r}"
            )
        seen[key] = message

    unique: list[Any] = []
    omitted_ids: list[str] = []
    for message in recovered:
        identifier = _get(message, "id")
        if identifier is None:
            unique.append(message)
            continue
        key = str(identifier)
        prior = seen.get(key)
        if prior is None:
            seen[key] = message
            unique.append(message)
            continue
        if _message_record(prior, attachments) != _message_record(
            message, attachments
        ):
            raise ValueError(
                "recovered transcript has non-identical messages sharing Inspect "
                f"message ID {key!r}"
            )
        omitted_ids.append(key)

    if not omitted_ids:
        return unique, None
    return unique, {
        "policy": "collapse_exact_duplicate_inspect_message_ids",
        "exact_duplicate_messages_omitted": len(omitted_ids),
        "duplicate_message_ids": sorted(set(omitted_ids)),
        "conflicting_duplicate_ids": [],
    }


def _target_events(
    events: Sequence[Any], target_model: str | None
) -> list[tuple[int, Any]]:
    selected: list[tuple[int, Any]] = []
    for index, event in enumerate(events):
        if _get(event, "event") != "model":
            continue
        role = _get(event, "role")
        if role is not None:
            if role != "target":
                continue
        elif target_model and _get(event, "model") != target_model:
            continue
        selected.append((index, event))
    return selected


def recover_interrupted_opencode_messages(
    base_messages: Sequence[Any],
    events: Sequence[Any],
    *,
    target_model: str | None = None,
    event_start: int = 0,
    applied_before_judging: bool,
    attachments: dict[str, str] | None = None,
) -> tuple[list[Any], dict | None]:
    """Return ``base_messages`` plus recorded native history after its final user.

    The join is accepted only when the official transcript ends in one user message
    and a target event input contains exactly one text-identical copy of that user
    message (allowing OpenCode's known literal quote wrapper).  The newest matching
    event has the fullest history.  A truly empty terminal model output is omitted;
    observable reasoning is retained.
    """

    base = list(base_messages)
    all_events = list(events)
    if not base or _get(base[-1], "role") != "user":
        return base, None
    if not isinstance(event_start, int) or event_start < 0:
        return base, None

    boundary_text = _message_text(base[-1], attachments)
    selected = _target_events(all_events[event_start:], target_model)
    if not selected:
        return base, None

    chosen: tuple[int, Any, list[Any], int, str] | None = None
    for relative_index, event in reversed(selected):
        native_input = list(_get(event, "input", []) or [])
        matches: list[tuple[int, str]] = []
        for message_index, message in enumerate(native_input):
            if _get(message, "role") != "user":
                continue
            candidate_text, wrapper = _native_user_text(message, attachments)
            if candidate_text == boundary_text:
                matches.append((message_index, wrapper))
        if len(matches) == 1:
            match_index, wrapper = matches[0]
            chosen = (
                event_start + relative_index,
                event,
                native_input,
                match_index,
                wrapper,
            )
            break
    if chosen is None:
        return base, None

    event_index, event, native_input, match_index, wrapper = chosen
    recovered = list(native_input[match_index + 1:])
    output_message = _event_output_message(event)
    terminal_output_included = False
    if _observable_message(output_message):
        output_id = _get(output_message, "id")
        known_ids = {
            str(_get(message, "id"))
            for message in [*base, *recovered]
            if _get(message, "id") is not None
        }
        if output_id is None or str(output_id) not in known_ids:
            recovered.append(output_message)
            terminal_output_included = True

    recovered, message_id_normalization = _unique_recovered_messages(
        base, recovered, attachments
    )
    if not recovered:
        return base, None

    merged = [*base, *recovered]
    record = {
        "format": RECOVERY_FORMAT,
        "scaffold": "opencode",
        "source": "target_model_event_inputs_and_terminal_output",
        "reconstructed": True,
        "applied_before_judging": bool(applied_before_judging),
        "original_message_count": len(base),
        "reconstructed_message_count": len(merged),
        "added_message_count": len(recovered),
        "matched_user_wrapper": wrapper,
        "terminal_output_included": terminal_output_included,
        "terminal_event_index": event_index,
        "terminal_event_id": (
            str(_get(event, "uuid") or _get(event, "id"))
            if _get(event, "uuid") is not None or _get(event, "id") is not None
            else None
        ),
        "terminal_event_timestamp": (
            str(_get(event, "timestamp"))
            if _get(event, "timestamp") is not None else None
        ),
        "coverage": "through_last_matching_recorded_target_model_event",
        "complete": False,
        "limitation": (
            "The native OpenCode call did not return an AgentState. Recorded model "
            "history was recovered through the newest target event matching the "
            "interrupted user turn; activity after that event and an interrupted tool "
            "result may be unavailable."
        ),
    }
    if message_id_normalization is not None:
        record["message_id_normalization"] = message_id_normalization
    return merged, record


def recover_predeadline_opencode_submission(
    base_messages: Sequence[Any],
    events: Sequence[Any],
    *,
    deadline_seconds_from_start: float,
    target_model: str | None = None,
    event_start: int = 0,
    applied_before_judging: bool,
    attachments: dict[str, str] | None = None,
) -> tuple[list[Any], dict | None, dict | None]:
    """Recover a final response generated before the advertised task deadline.

    Inspect SWE returns OpenCode's updated ``AgentState`` only after the CLI process
    exits. The provider can therefore finish the model's terminal response before the
    deadline while the outer sample clock cancels the still-running CLI handoff. Count
    that response only when the recorded target event proves all of the following:

    - it belongs to the exact final user boundary;
    - it completed normally with a non-empty assistant response and no tool calls;
    - its successful provider-call interval ended no later than the stored deadline.

    Every missing or ambiguous field fails closed. The returned evidence is designed to
    be stored with the sample/prefix so the deadline exception remains queryable.
    """

    recovered, transcript_record = recover_interrupted_opencode_messages(
        base_messages,
        events,
        target_model=target_model,
        event_start=event_start,
        applied_before_judging=applied_before_judging,
        attachments=attachments,
    )
    if (
        transcript_record is None
        or transcript_record.get("terminal_output_included") is not True
    ):
        return recovered, transcript_record, None

    event_index = transcript_record.get("terminal_event_index")
    if not isinstance(event_index, int) or not (0 <= event_index < len(events)):
        return recovered, transcript_record, None
    event = list(events)[event_index]
    output = _get(event, "output")
    output_message = _event_output_message(event)
    if (
        output is None
        or output_message is None
        or _get(output_message, "role") != "assistant"
        or not _message_text(output_message, attachments).strip()
        or (_get(output_message, "tool_calls", []) or [])
        or _get(event, "pending") is True
        or _get(event, "error") is not None
        or _get(output, "error") is not None
    ):
        return recovered, transcript_record, None

    stop_reason = _get(output, "stop_reason")
    if stop_reason is None:
        choices = _get(output, "choices", []) or []
        stop_reason = _get(choices[0], "stop_reason") if choices else None
    if stop_reason != "stop":
        return recovered, transcript_record, None

    retries = _get(event, "retries")
    if retries not in (None, 0):
        return recovered, transcript_record, None
    working_start = _get(event, "working_start")
    working_duration = _get(event, "working_time")
    timing_source = "event.working_start + event.working_time"
    if not isinstance(working_duration, (int, float)):
        working_duration = _get(output, "time")
        timing_source = "event.working_start + output.time"
    timing_values = (
        working_start,
        working_duration,
        deadline_seconds_from_start,
    )
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in timing_values
    ):
        return recovered, transcript_record, None
    if float(working_duration) < 0 or float(deadline_seconds_from_start) <= 0:
        return recovered, transcript_record, None

    completed_seconds_from_start = float(working_start) + float(working_duration)
    deadline = float(deadline_seconds_from_start)
    if completed_seconds_from_start > deadline:
        return recovered, transcript_record, None

    evidence = {
        "format": DEADLINE_RECOVERY_FORMAT,
        "accepted_as_submission": True,
        "rule": "terminal_model_response_completed_before_advertised_deadline",
        "scaffold": "opencode",
        "terminal_event_index": event_index,
        "terminal_event_id": transcript_record.get("terminal_event_id"),
        "terminal_event_timestamp": transcript_record.get(
            "terminal_event_timestamp"
        ),
        "stop_reason": stop_reason,
        "response_characters": len(_message_text(output_message, attachments)),
        "working_start_seconds": float(working_start),
        "successful_model_call_seconds": float(working_duration),
        "response_completed_seconds_from_start": completed_seconds_from_start,
        "deadline_seconds_from_start": deadline,
        "margin_before_deadline_seconds": deadline - completed_seconds_from_start,
        "timing_source": timing_source,
        "cli_exit_required_before_deadline": False,
    }
    return recovered, transcript_record, evidence
