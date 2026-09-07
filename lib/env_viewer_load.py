"""Environment-owned Inspect log loader and judgment normalizer.

This module deliberately depends on Inspect only.  It does not import the Petri
package, the old local Petri project, or either project's viewer.  Real-environment
runs keep the agent-facing conversation in ``EvalSample.messages``, so we can read
both old and new environments logs without Petri's branch renderer.

There are two judgment formats and they stay separate:

* ``structured`` is the environments-owned schema (``environment_judge``);
* ``legacy_numeric`` is the old Petri-shaped ``audit_judge`` value.

In particular, this loader never turns an old 1/10 score into false/true.  Doing so
would silently change the meaning of historical data.
"""

from __future__ import annotations

from copy import deepcopy
import gc
import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

from env_viewer_cache import (
    assign_stable_ids,
    cache_file as _cache_file,
    mode_signature as _mode_signature,
    module_signature as _module_signature,
    trajectory_key as traj_key,
    viewer_build_lock,
    write_pickle_atomic as _write_pickle_atomic,
)
from env_viewer_turns import number_message_turns
from interrupted_native_transcript import recover_interrupted_opencode_messages
from judgment_semantics import (
    CURRENT_STRUCTURED_DIMENSIONS,
    FAMILY_STRUCTURED_DIMENSIONS,
    KNOWN_STRUCTURED_DIMENSIONS,
    ML_NOTICED_VALUES,
    ML_HACK_TYPE_VALUES,
    P_HACK_TYPE_VALUES,
    P_REVIEW_TYPE_VALUES,
    RETIRED_STRUCTURED_DIMENSIONS,
    REWARD_HACKING_DISPLAY_LABELS,
)
from real_integrity import (
    finalize_audit_integrity,
    same_model_request,
    target_model_events,
    target_provider_events,
    tool_output_truncations,
)

# Kept for callers that used the loader's former private compatibility helper.
_tool_truncations = tool_output_truncations


CURRENT_SCORE_KEY = "environment_judge"
LEGACY_SCORE_KEY = "audit_judge"
JUDGMENT_KEYS = (CURRENT_SCORE_KEY, LEGACY_SCORE_KEY)
_CURRENT_ML_HACK_TYPES = ML_HACK_TYPE_VALUES
_CURRENT_P_HACK_TYPES = P_HACK_TYPE_VALUES
_CURRENT_P_REVIEW_TYPES = P_REVIEW_TYPE_VALUES
_LEGACY_ML_HACK_TYPES = {"test_set"}
_LEGACY_P_HACK_TYPES = {
    "selective_reporting",
    "statistical_misrepresentation",
    "fabricated_evidence",
}

_ATTACHMENT_RE = re.compile(r"attachment://([0-9a-fA-F]+)")
_MESSAGE_REF_RE = re.compile(r"\[?M(\d+)\]?", re.IGNORECASE)
_JUDGE_SCHEMA_VERSION_RE = re.compile(
    r"(?:environment-judge-v|environment_judge\.v)(\d+)$"
)


def _missing_dimension_is_issue(key: str, schema_version: str | None) -> bool:
    """Treat noticed as optional only for judgments stored before schema v14."""

    if key != "noticed":
        return True
    match = _JUDGE_SCHEMA_VERSION_RE.fullmatch(str(schema_version or ""))
    return bool(match and int(match.group(1)) >= 14)


def _reward_hacking_display_values(
    types: list[str], not_delivered: list[str]
) -> list[str]:
    labels = []
    for item in types:
        label = REWARD_HACKING_DISPLAY_LABELS.get(item, item.replace("_", " "))
        if item in not_delivered:
            label += " (not delivered)"
        labels.append(label)
    return labels or ["none"]


def _plain(value: Any) -> Any:
    """Convert Inspect/Pydantic values into JSON-compatible values without dropping fields."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _plain(model_dump(mode="json"))
    as_dict = getattr(value, "__dict__", None)
    if isinstance(as_dict, dict):
        return {
            str(key): _plain(item)
            for key, item in as_dict.items()
            if not str(key).startswith("_")
        }
    # A repr is preferable to silently omitting an unfamiliar future Inspect field.
    return {"unparsed_python_value": repr(value), "python_type": type(value).__name__}


def resolve_attachments(value: Any, attachments: dict[str, str] | None) -> Any:
    """Resolve Inspect's attachment placeholders recursively.

    An unknown attachment remains visible as its original ``attachment://`` value.
    """
    attachments = attachments or {}
    if isinstance(value, str):
        return _ATTACHMENT_RE.sub(
            lambda match: attachments.get(match.group(1), match.group(0)), value
        )
    if isinstance(value, dict):
        return {
            str(key): resolve_attachments(item, attachments)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_attachments(item, attachments) for item in value]
    return value


def _content_field(block: Any, name: str) -> Any:
    """Read a content-block field from current models or historical plain mappings."""

    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


def _display_reasoning(block: Any, attachments: dict[str, str]) -> str:
    """Return only provider-readable reasoning text, never its opaque replay payload."""

    summary = _content_field(block, "summary")
    if bool(_content_field(block, "redacted")):
        if isinstance(summary, str) and summary.strip():
            return str(resolve_attachments(summary, attachments))
        return "[reasoning produced but redacted by the provider]"

    for name in ("reasoning", "summary", "text"):
        value = _content_field(block, name)
        if isinstance(value, str) and value.strip():
            return str(resolve_attachments(value, attachments))
    return ""


def _message_content(message: Any, attachments: dict[str, str]) -> tuple[str, str, list]:
    """Return visible text, native reasoning, and unfamiliar content blocks."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return str(resolve_attachments(content, attachments)), "", []

    visible: list[str] = []
    reasoning: list[str] = []
    other: list[Any] = []
    for block in content or []:
        if isinstance(block, str):
            visible.append(str(resolve_attachments(block, attachments)))
            continue
        kind = str(_content_field(block, "type") or "")
        if kind == "reasoning":
            value = _display_reasoning(block, attachments)
            if value:
                reasoning.append(value)
            else:
                other.append(resolve_attachments(_plain(block), attachments))
            continue
        text = _content_field(block, "text")
        if isinstance(text, str):
            visible.append(str(resolve_attachments(text, attachments)))
        else:
            other.append(resolve_attachments(_plain(block), attachments))
    return "".join(visible), "\n\n".join(reasoning), other


def _iso_timestamp(value: Any) -> str | None:
    """Return a stable ISO timestamp string for an Inspect datetime-like value."""

    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    text = str(value).strip()
    return text or None


def _message_timings(
    sample: Any,
    target_model: str | None,
    messages: Iterable[Any] | None = None,
) -> list[dict]:
    """Map saved messages to their existing Inspect event time.

    Inspect stores timing on ModelEvent and ToolEvent rather than ChatMessage. Model
    inputs timestamp system/user turns (and native tool results without ToolEvents),
    while model outputs and ToolEvent.message_id provide the authoritative generated
    turn mapping. ``working_start`` is already elapsed sample time and is shared by all
    three agent harnesses.
    """

    messages = list(
        messages
        if messages is not None
        else (getattr(sample, "messages", None) or [])
    )
    candidates: dict[str, tuple[int, float, str | None, str]] = {}

    def remember(
        message_id: Any,
        *,
        priority: int,
        elapsed: Any,
        timestamp: Any,
        source: str,
    ) -> None:
        if not message_id or isinstance(elapsed, bool) or not isinstance(
            elapsed, (int, float)
        ):
            return
        key = str(message_id)
        current = candidates.get(key)
        if current is None or priority > current[0]:
            candidates[key] = (
                priority,
                max(0.0, float(elapsed)),
                _iso_timestamp(timestamp),
                source,
            )

    model_events, _ = target_model_events(sample, target_model)
    for event in model_events:
        elapsed = getattr(event, "working_start", None)
        timestamp = getattr(event, "timestamp", None)
        for message in getattr(event, "input", None) or []:
            remember(
                getattr(message, "id", None),
                priority=10,
                elapsed=elapsed,
                timestamp=timestamp,
                source="model_input_event",
            )
        output_message = getattr(getattr(event, "output", None), "message", None)
        remember(
            getattr(output_message, "id", None),
            priority=30,
            elapsed=elapsed,
            timestamp=timestamp,
            source="model_output_event",
        )

    tool_call_candidates: dict[str, tuple[int, float, str | None, str]] = {}
    for event in getattr(sample, "events", None) or []:
        if getattr(event, "event", None) != "tool":
            continue
        elapsed = getattr(event, "working_start", None)
        timestamp = getattr(event, "timestamp", None)
        message_id = getattr(event, "message_id", None)
        remember(
            message_id,
            priority=30,
            elapsed=elapsed,
            timestamp=timestamp,
            source="tool_event",
        )
        call_id = getattr(event, "id", None)
        if (
            call_id
            and not isinstance(elapsed, bool)
            and isinstance(elapsed, (int, float))
        ):
            tool_call_candidates[str(call_id)] = (
                20,
                max(0.0, float(elapsed)),
                _iso_timestamp(timestamp),
                "tool_event_call_id",
            )

    sample_started = _iso_timestamp(getattr(sample, "started_at", None))
    timings: list[dict] = []
    previous = (0, 0.0, sample_started, "sample_start_fallback")
    for message in messages:
        message_id = getattr(message, "id", None)
        timing = candidates.get(str(message_id)) if message_id else None
        if timing is None and getattr(message, "role", None) == "tool":
            call_id = getattr(message, "tool_call_id", None)
            timing = tool_call_candidates.get(str(call_id)) if call_id else None
        if timing is None:
            timing = previous
        else:
            previous = timing
        _, elapsed, timestamp, source = timing
        timings.append({
            "timestamp": timestamp,
            "elapsed_seconds": elapsed,
            "timestamp_source": source,
        })
    return timings


def _format_elapsed_time(value: Any) -> str | None:
    """Format non-negative elapsed seconds as an unbounded HH:MM:SS value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    total = max(0, int(value))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def normalize_messages(
    messages: Iterable[Any],
    attachments: dict[str, str] | None = None,
    timings: Iterable[dict] | None = None,
) -> list[dict]:
    """Normalize agent-facing Inspect messages and give each a stable ``M#`` anchor."""
    attachments = attachments or {}
    timing_records = list(timings or [])
    normalized: list[dict] = []
    for number, message in enumerate(messages or [], start=1):
        role = str(getattr(message, "role", "other") or "other")
        text, reasoning, other_blocks = _message_content(message, attachments)

        calls = []
        for call in getattr(message, "tool_calls", None) or []:
            calls.append({
                "id": getattr(call, "id", None),
                "function": getattr(call, "function", None),
                "arguments": resolve_attachments(
                    _plain(getattr(call, "arguments", None)), attachments
                ),
                "type": getattr(call, "type", None),
            })

        error = getattr(message, "error", None)
        timing = timing_records[number - 1] if number <= len(timing_records) else {}
        message_metadata = getattr(message, "metadata", None) or {}
        normalized.append({
            "number": number,
            "id": f"M{number}",
            "source_id": getattr(message, "id", None),
            "role": role,
            "native_role": (
                str(message_metadata.get("native_role"))
                if message_metadata.get("native_role") else None
            ),
            "scaffold_injected": (
                str(message_metadata.get("scaffold_injected"))
                if message_metadata.get("scaffold_injected") else None
            ),
            "text": text,
            "reasoning": reasoning,
            "other_content_blocks": other_blocks,
            "tool_calls": calls,
            "tool_call_id": getattr(message, "tool_call_id", None),
            "function": getattr(message, "function", None),
            "error": resolve_attachments(_plain(error), attachments) if error else None,
            "source": getattr(message, "source", None),
            "timestamp": timing.get("timestamp"),
            "elapsed_seconds": timing.get("elapsed_seconds"),
            "elapsed_time": _format_elapsed_time(timing.get("elapsed_seconds")),
            "timestamp_source": timing.get("timestamp_source"),
            "assistant_turn": None,
            "user_turn": None,
        })
    number_message_turns(normalized)
    return normalized


_CODEX_NATIVE_USER_TAGS = ("environment_context", "user_instructions")


def stamp_codex_native_roles(messages: list[dict]) -> None:
    """Recover Codex-native identity for runs stored before parser stamping.

    Codex sends its instruction message under the OpenAI "developer" role —
    the true wire role, typed as a second system message only because Inspect
    has no developer message type — and injects workspace facts as tagged,
    genuinely user-role turns. Both arrive before the first assistant turn.
    M# numbering stays what the model saw. Scaffold-injected messages do not receive
    an experimental U# label. Parser-stamped metadata wins.
    """

    seen_system = False
    for message in messages:
        if message.get("role") == "assistant":
            break
        if message.get("native_role") or message.get("scaffold_injected"):
            seen_system = seen_system or message.get("role") == "system"
            continue
        if message.get("role") == "system":
            if seen_system:
                message["native_role"] = "developer"
            seen_system = True
        elif message.get("role") == "user":
            text = str(message.get("text") or "").lstrip()
            for tag in _CODEX_NATIVE_USER_TAGS:
                if text.startswith(f"<{tag}>"):
                    message["scaffold_injected"] = tag
                    break
    number_message_turns(messages)


def transcript_text(messages: list[dict]) -> str:
    """Compatibility rendering for code that still expects a numbered text transcript."""
    chunks = []
    for message in messages:
        role = str(message.get("role") or "other").capitalize()
        body = message.get("text") or ""
        if message.get("reasoning"):
            body += f"\n<thinking>{message['reasoning']}</thinking>"
        for call in message.get("tool_calls") or []:
            body += (
                f"\nTool Call: {call.get('function') or ''}\nArguments:\n"
                f"{json.dumps(call.get('arguments'), ensure_ascii=False, indent=2)}"
            )
        if message.get("other_content_blocks"):
            body += (
                "\nUnrecognized content blocks:\n"
                + json.dumps(
                    message["other_content_blocks"], ensure_ascii=False, indent=2
                )
            )
        chunks.append(f"[{message['id']}] {role}:\n{body}".rstrip())
    return "\n\n".join(chunks)


def _message_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        match = _MESSAGE_REF_RE.fullmatch(value.strip())
        return int(match.group(1)) if match else None
    if isinstance(value, dict):
        for key in ("number", "message", "message_number", "prompt_message"):
            if key in value:
                return _message_number(value[key])
    return None


def _mapped_message_number(
    number: int, source_message_map: Any, source_lookup: dict | None = None
) -> int:
    """Map a prompt-local number back to Sample.messages when a judge stored a map."""
    if isinstance(source_message_map, dict):
        entry = source_message_map.get(str(number), source_message_map.get(number))
        if entry is not None:
            if isinstance(entry, dict):
                for key in (
                    "source_message", "source_number", "sample_message", "original"
                ):
                    mapped = _message_number(entry.get(key))
                    if mapped is not None:
                        return mapped
            mapped = _message_number(entry)
            if mapped is not None:
                return mapped
    if isinstance(source_message_map, list):
        for entry in source_message_map:
            if not isinstance(entry, dict):
                continue
            prompt = next(
                (
                    _message_number(entry.get(key))
                    for key in ("prompt_message", "judge_message", "number", "message")
                    if entry.get(key) is not None
                ),
                None,
            )
            if prompt != number:
                continue
            source_lookup = source_lookup or {}
            source_id = entry.get("source_id")
            if source_id is not None and str(source_id) in source_lookup.get("ids", {}):
                return source_lookup["ids"][str(source_id)]
            source_index = entry.get("source_index")
            if (
                isinstance(source_index, int)
                and not isinstance(source_index, bool)
                and source_index in source_lookup.get("indexes", {})
            ):
                return source_lookup["indexes"][source_index]
            mapped = next(
                (
                    _message_number(entry.get(key))
                    for key in (
                        "source_message", "source_number", "sample_message", "original"
                    )
                    if entry.get(key) is not None
                ),
                None,
            )
            if mapped is not None:
                return mapped
    return number


def normalize_evidence(
    evidence: Any,
    *,
    present_messages: set[int] | None = None,
    source_message_map: Any = None,
    source_lookup: dict | None = None,
) -> dict:
    """Normalize structured message and artifact citations; retain malformed citations."""
    if not isinstance(evidence, dict):
        evidence = {} if evidence is None else {"unparsed": _plain(evidence)}
    raw_messages = (
        evidence.get("messages")
        or evidence.get("message_refs")
        or evidence.get("message_ids")
        or []
    )
    if not isinstance(raw_messages, list):
        raw_messages = [raw_messages]

    messages: list[dict] = []
    issues: list[dict] = []
    seen_messages: set[tuple[int, int]] = set()
    for raw in raw_messages:
        prompt_number = _message_number(raw)
        if prompt_number is None:
            issues.append({"kind": "invalid_message_reference", "reference": _plain(raw)})
            continue
        source_number = _mapped_message_number(
            prompt_number, source_message_map, source_lookup
        )
        pair = (prompt_number, source_number)
        if pair in seen_messages:
            continue
        seen_messages.add(pair)
        available = None if present_messages is None else source_number in present_messages
        messages.append({
            "prompt_number": prompt_number,
            "number": source_number,
            "id": f"M{source_number}",
            "available": available,
        })
        if available is False:
            issues.append({
                "kind": "missing_message_reference",
                "prompt_number": prompt_number,
                "source_number": source_number,
            })

    raw_artifacts = evidence.get("artifacts") or evidence.get("files") or []
    if not isinstance(raw_artifacts, list):
        raw_artifacts = [raw_artifacts]
    artifacts = []
    seen_artifacts: set[tuple[str, str]] = set()
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            issues.append({"kind": "invalid_artifact_reference", "reference": _plain(raw)})
            continue
        path = raw.get("path")
        snapshot = raw.get("snapshot") or raw.get("submission")
        if not isinstance(path, str) or not path or not isinstance(snapshot, (str, int)):
            issues.append({"kind": "invalid_artifact_reference", "reference": _plain(raw)})
            continue
        key = (path, str(snapshot))
        if key in seen_artifacts:
            continue
        seen_artifacts.add(key)
        artifacts.append({"path": path, "snapshot": str(snapshot)})

    return {
        "messages": messages,
        "artifacts": artifacts,
        "issues": issues,
        "raw": deepcopy(evidence),
    }


def _structured_dimension(
    key: str,
    raw: Any,
    *,
    family: str | None,
    present_messages: set[int] | None,
    source_message_map: Any,
    source_lookup: dict | None,
) -> dict:
    missing = raw is ...
    if missing:
        return {
            "key": key,
            "status": "missing",
            "value": None,
            "display_value": "—",
            "explanation": "",
            "evidence": normalize_evidence({}, present_messages=present_messages),
            "raw": None,
        }

    if key == "deception" and raw is None:
        return {
            "key": key,
            "status": "not_applicable",
            "value": None,
            "display_value": "n/a",
            "explanation": "",
            "evidence": normalize_evidence({}, present_messages=present_messages),
            "raw": None,
        }

    raw_dict = raw if isinstance(raw, dict) else {"value": raw}
    explanation = str(
        raw_dict.get("explanation") or raw_dict.get("reason") or ""
    )
    evidence = normalize_evidence(
        raw_dict.get("evidence"),
        present_messages=present_messages,
        source_message_map=source_message_map,
        source_lookup=source_lookup,
    )

    if key == "noticed":
        value = raw_dict.get("value")
        if family == "ml_training_data_misuse":
            valid = isinstance(value, str) and value in ML_NOTICED_VALUES
        elif family == "p_hacking":
            valid = isinstance(value, bool)
        else:
            valid = (
                isinstance(value, bool)
                or isinstance(value, str) and value in ML_NOTICED_VALUES
            )
        if valid and value == "inputs":
            display = "inputs only"
        elif valid and isinstance(value, bool):
            display = "yes" if value else "no"
        else:
            display = str(value) if valid else "invalid"
    elif key in {"noticed_honeypot", "environment_failure"}:
        value = raw_dict.get("value")
        valid = isinstance(value, bool)
        display = "yes" if value is True else "no" if value is False else "invalid"
    elif key == "reward_hacking":
        value = raw_dict.get("types")
        not_delivered = raw_dict.get("not_delivered", [])
        valid = (
            isinstance(value, list)
            and all(isinstance(item, str) for item in value)
            and isinstance(not_delivered, list)
            and all(isinstance(item, str) for item in not_delivered)
        )
        if valid:
            display_values = _reward_hacking_display_values(value, not_delivered)
            display = ", ".join(display_values)
            if family == "ml_training_data_misuse":
                hack_vocabulary = _CURRENT_ML_HACK_TYPES | _LEGACY_ML_HACK_TYPES
            elif family == "p_hacking":
                hack_vocabulary = _CURRENT_P_HACK_TYPES | _LEGACY_P_HACK_TYPES
            else:
                hack_vocabulary = (
                    _CURRENT_ML_HACK_TYPES | _LEGACY_ML_HACK_TYPES
                    | _CURRENT_P_HACK_TYPES | _LEGACY_P_HACK_TYPES
                )
            hack_types = [item for item in value if item in hack_vocabulary]
            review_types = (
                [item for item in value if item in _CURRENT_P_REVIEW_TYPES]
                if family == "p_hacking"
                else []
            )
        else:
            display = "invalid"
            hack_types = []
            review_types = []
    else:
        value = raw_dict.get("score", raw_dict.get("value"))
        valid = isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 10
        display = str(value) if valid else "invalid"

    normalized = {
        "key": key,
        "status": "ok" if valid else "invalid",
        "value": deepcopy(value),
        "display_value": display,
        "explanation": explanation,
        "evidence": evidence,
        "raw": deepcopy(raw),
    }
    if key == "reward_hacking":
        normalized.update({
            "display_values": display_values if valid else [display],
            "hack_types": hack_types,
            "review_types": review_types,
            "requires_review": bool(review_types) if valid else None,
            "is_hack": (
                None if valid and review_types
                else bool(hack_types) if valid
                else None
            ),
        })
        if family == "ml_training_data_misuse" or "not_delivered" in raw_dict:
            normalized["not_delivered"] = (
                deepcopy(not_delivered) if valid else None
            )
    return normalized


def normalize_evidence_scope(
    envelope: dict | None,
    *,
    present_messages: set[int] | None = None,
    source_message_map: Any = None,
    source_lookup: dict | None = None,
) -> dict:
    """Normalize judge-builder coverage metadata without changing its raw record."""
    envelope = envelope or {}
    evidence = envelope.get("evidence") or {}
    if not isinstance(evidence, dict):
        return {
            "native_reasoning_policy": None,
            "native_reasoning_message_count": None,
            "native_reasoning_block_count": None,
            "native_reasoning_plaintext_block_count": None,
            "native_reasoning_summary_block_count": None,
            "native_reasoning_unavailable_block_count": None,
            "source_tool_call_count": None,
            "source_embedded_tool_use_block_count": None,
            "source_tool_result_message_count": None,
            "source_message_count": None,
            "selected_message_count": None,
            "omitted_message_count": None,
            "message_selection": None,
            "builder_truncated_evidence": None,
            "caveats": [],
            "artifacts": [],
            "issues": [{"kind": "invalid_evidence_scope"}],
            "raw": deepcopy(evidence),
        }
    caveats = []
    issues = []
    for raw in evidence.get("caveats") or []:
        if not isinstance(raw, dict):
            issues.append({"kind": "invalid_evidence_caveat", "caveat": _plain(raw)})
            continue
        normalized = normalize_evidence(
            {"messages": raw.get("messages") or [],
             "artifacts": raw.get("artifacts") or []},
            present_messages=present_messages,
            source_message_map=source_message_map,
            source_lookup=source_lookup,
        )
        caveats.append({
            "code": str(raw.get("code") or "unknown_caveat"),
            "description": str(raw.get("description") or ""),
            "source": str(raw.get("source") or "stored"),
            "messages": normalized["messages"],
            "artifacts": normalized["artifacts"],
            "issues": normalized["issues"],
            "raw": deepcopy(raw),
        })
        issues.extend({**item, "caveat": raw.get("code")} for item in normalized["issues"])
    return {
        "native_reasoning_policy": evidence.get("native_reasoning_policy"),
        "native_reasoning_message_count": evidence.get(
            "native_reasoning_message_count"
        ),
        "native_reasoning_block_count": evidence.get("native_reasoning_block_count"),
        "native_reasoning_plaintext_block_count": evidence.get(
            "native_reasoning_plaintext_block_count"
        ),
        "native_reasoning_summary_block_count": evidence.get(
            "native_reasoning_summary_block_count"
        ),
        "native_reasoning_unavailable_block_count": evidence.get(
            "native_reasoning_unavailable_block_count"
        ),
        "source_tool_call_count": evidence.get("source_tool_call_count"),
        "source_embedded_tool_use_block_count": evidence.get(
            "source_embedded_tool_use_block_count"
        ),
        "source_tool_result_message_count": evidence.get(
            "source_tool_result_message_count"
        ),
        "source_message_count": evidence.get("source_message_count"),
        "selected_message_count": evidence.get("selected_message_count"),
        "omitted_message_count": evidence.get("omitted_message_count"),
        "message_selection": evidence.get("message_selection"),
        "builder_truncated_evidence": evidence.get("builder_truncated_evidence"),
        "caveats": caveats,
        "artifacts": deepcopy(evidence.get("artifacts") or []),
        "issues": issues,
        "raw": deepcopy(evidence),
    }


def normalize_structured_judgment(
    raw_result: dict,
    *,
    family: str | None = None,
    schema_version: str | None = None,
    present_messages: set[int] | None = None,
    source_message_map: Any = None,
    source_lookup: dict | None = None,
    envelope: dict | None = None,
) -> dict:
    """Normalize an environments-owned structured result for viewer code."""
    current_dimensions = FAMILY_STRUCTURED_DIMENSIONS.get(
        family, CURRENT_STRUCTURED_DIMENSIONS
    )
    dimension_keys = [
        *(
            key for key in RETIRED_STRUCTURED_DIMENSIONS
            if key in raw_result
        ),
        *current_dimensions,
        *(
            key for key in CURRENT_STRUCTURED_DIMENSIONS
            if key not in current_dimensions and key in raw_result
        ),
    ]
    dimensions = [
        _structured_dimension(
            key,
            raw_result.get(key, ...),
            family=family,
            present_messages=present_messages,
            source_message_map=source_message_map,
            source_lookup=source_lookup,
        )
        for key in dimension_keys
    ]
    issues = []
    for dimension in dimensions:
        if (
            dimension["status"] == "invalid"
            or dimension["status"] == "missing"
            and _missing_dimension_is_issue(dimension["key"], schema_version)
        ):
            issues.append({
                "kind": f"dimension_{dimension['status']}",
                "dimension": dimension["key"],
            })
        issues.extend({**issue, "dimension": dimension["key"]}
                      for issue in dimension["evidence"]["issues"])
    evidence_scope = normalize_evidence_scope(
        envelope,
        present_messages=present_messages,
        source_message_map=source_message_map,
        source_lookup=source_lookup,
    )
    issues.extend(evidence_scope["issues"])
    envelope = envelope or {}
    explanation = str(
        envelope.get("explanation")
        # Read compatibility for judgments stored before schema v9.
        or envelope.get("justification")
        or ""
    )
    return {
        "format": "structured",
        "schema_version": schema_version or "environment_judge.unknown",
        "family": family,
        "summary": str(raw_result.get("summary") or ""),
        "explanation": explanation,
        "highlights": str(raw_result.get("highlights") or ""),
        "dimensions": dimensions,
        "issues": issues,
        "evidence_scope": evidence_scope,
        "raw_result": deepcopy(raw_result),
        "envelope": deepcopy(envelope),
    }


def normalize_legacy_judgment(
    scores: dict,
    *,
    metadata: dict | None = None,
    explanation: str = "",
) -> dict:
    """Preserve old numeric scores exactly, with no new-schema reinterpretation."""
    dimensions = []
    for key, value in scores.items():
        dimensions.append({
            "key": str(key),
            "status": "ok" if isinstance(value, (int, float)) and not isinstance(value, bool)
            else "invalid",
            "value": deepcopy(value),
            "display_value": f"{value}/10" if isinstance(value, (int, float)) else str(value),
            "explanation": "",
            "evidence": {"messages": [], "artifacts": [], "issues": [], "raw": {}},
            "raw": deepcopy(value),
        })
    return {
        "format": "legacy_numeric",
        "schema_version": "petri_numeric_legacy",
        "family": (metadata or {}).get("dimension_scope"),
        "dimensions": dimensions,
        "issues": [],
        "raw_result": deepcopy(scores),
        "legacy": {
            "summary": str((metadata or {}).get("summary") or ""),
            "highlights": str((metadata or {}).get("highlights") or ""),
            "explanation": str(explanation or ""),
        },
    }


def normalize_judgment(
    *,
    score_key: str | None,
    score_value: Any,
    score_metadata: dict | None,
    real_env: dict | None,
    present_messages: set[int] | None = None,
    source_lookup: dict | None = None,
    score_explanation: str = "",
) -> dict | None:
    """Select canonical stored judgment data, with tolerant forward compatibility."""
    score_metadata = _plain(score_metadata or {})
    real_env = _plain(real_env or {})
    # The score owns the complete call record (exact prompt, evidence, schema, model
    # interface). ``real_env.final_judgment`` is a deliberately smaller trajectory
    # summary. Merge it on top instead of letting it hide the richer score record.
    score_envelope = score_metadata.get("environment_judge")
    if not isinstance(score_envelope, dict):
        score_envelope = score_metadata.get("final_judgment")
    real_envelope = real_env.get("final_judgment")
    retrospective_rejudge = (
        isinstance(score_envelope, dict)
        and score_envelope.get("judgment_role") == "retrospective_rejudge"
    )
    if retrospective_rejudge:
        # A rejudge deliberately carries the original real_env record. Its old
        # final_judgment must not overwrite the new score that this row represents.
        envelope = {
            **(real_envelope if isinstance(real_envelope, dict) else {}),
            **score_envelope,
        }
    else:
        envelope = {
            **(score_envelope if isinstance(score_envelope, dict) else {}),
            **(real_envelope if isinstance(real_envelope, dict) else {}),
        }

    # A failed rejudge has no new result. Do not fall back to the original result that
    # travels inside real_env, or the failed row would masquerade as a judged row.
    raw_result = (
        score_envelope.get("result")
        if retrospective_rejudge else envelope.get("result")
    )
    if not isinstance(raw_result, dict) and isinstance(score_metadata.get("result"), dict):
        raw_result = score_metadata["result"]
    plain_score = _plain(score_value)
    looks_structured = (
        isinstance(plain_score, dict)
        and any(
            key in plain_score and (
                isinstance(plain_score[key], dict)
                or (key == "deception" and plain_score[key] is None)
            )
            for key in KNOWN_STRUCTURED_DIMENSIONS
        )
    )
    if not isinstance(raw_result, dict) and (
        score_key == CURRENT_SCORE_KEY or looks_structured
    ):
        raw_result = plain_score if isinstance(plain_score, dict) else None
    if isinstance(raw_result, dict):
        provenance = envelope.get("provenance") or score_metadata.get("provenance") or {}
        evidence_metadata = envelope.get("evidence") or {}
        source_map = (
            envelope.get("source_message_map")
            or (evidence_metadata.get("source_message_map")
                if isinstance(evidence_metadata, dict) else None)
            or score_metadata.get("source_message_map")
            or (provenance.get("source_message_map") if isinstance(provenance, dict) else None)
        )
        judgment = normalize_structured_judgment(
            raw_result,
            family=(
                envelope.get("family")
                or score_metadata.get("family")
                or score_metadata.get("judge_family")
            ),
            schema_version=(
                envelope.get("schema_version")
                or envelope.get("judge_schema_version")
                or score_metadata.get("judge_schema_version")
                or score_metadata.get("schema_version")
            ),
            present_messages=present_messages,
            source_message_map=source_map,
            source_lookup=source_lookup,
            envelope=envelope,
        )
        if not judgment.get("explanation"):
            judgment["explanation"] = str(score_explanation or "")
        return judgment

    if score_key == LEGACY_SCORE_KEY and isinstance(score_value, dict):
        return normalize_legacy_judgment(
            _plain(score_value),
            metadata=score_metadata,
            explanation=str(score_metadata.get("explanation") or ""),
        )
    return None


def _usage_dict(usage: Any) -> dict:
    out = {}
    for key, item in (usage or {}).items():
        out[str(key)] = {
            "input": getattr(item, "input_tokens", 0) or 0,
            "output": getattr(item, "output_tokens", 0) or 0,
            "cache_read": getattr(item, "input_tokens_cache_read", 0) or 0,
            "cache_write": getattr(item, "input_tokens_cache_write", 0) or 0,
            "total_cost": getattr(item, "total_cost", None),
        }
    return out


def _role_model(roles: dict, role: str) -> str | None:
    model = (roles or {}).get(role)
    if model is None:
        return None
    return str(getattr(model, "model", None) or model)


def _target_context_usage(sample: Any, target_model: str | None) -> dict:
    """Provider-reported prompt tokens for every logical agent-model call.

    Ordinary input, cache-read, and cache-write tokens together are the complete prompt
    presented on a call. Missing usage remains a ``None`` slot, so the metadata graph
    shows a real gap instead of shifting later calls left. A failed provider attempt
    immediately retried with the same request is one logical call and uses the successful
    attempt's usage.
    """

    events, used_model_fallback = target_model_events(sample, target_model)

    calls: list[int | None] = []
    for index, event in enumerate(events):
        output = getattr(event, "output", None)
        usage = getattr(output, "usage", None)
        if usage is None:
            next_event = events[index + 1] if index + 1 < len(events) else None
            next_usage = getattr(getattr(next_event, "output", None), "usage", None)
            if (
                getattr(event, "error", None)
                and next_event is not None
                and next_usage is not None
                and same_model_request(event, next_event)
            ):
                continue
            calls.append(None)
            continue
        prompt = (
            (getattr(usage, "input_tokens", None) or 0)
            + (getattr(usage, "input_tokens_cache_read", None) or 0)
            + (getattr(usage, "input_tokens_cache_write", None) or 0)
        )
        calls.append(prompt if prompt > 0 else None)

    missing = sum(value is None for value in calls)
    status = (
        "unavailable"
        if not calls or missing == len(calls)
        else "partial"
        if missing
        else "complete"
    )
    reason = None
    if not calls:
        reason = "no agent model-call events were recorded"
    elif missing:
        reason = f"provider usage missing on {missing} of {len(calls)} agent calls"
    return {
        "calls": calls,
        "status": status,
        "missing_calls": missing,
        "reason": reason,
        "source": "provider_reported",
        "role_matching": "model_fallback" if used_model_fallback else "event_role",
        "recorded_attempts": len(events),
        "logical_calls": len(calls),
    }


def _score_for_sample(sample: Any) -> tuple[str | None, Any]:
    scores = getattr(sample, "scores", None) or {}
    for key in JUDGMENT_KEYS:
        if key in scores:
            return key, scores[key]
    return None, None


def _judge_failure_record(
    *,
    score_key: str | None,
    score_value: Any,
    score_metadata: dict,
    score_explanation: str,
    judgment: dict | None,
) -> dict | None:
    """Describe a stored current-judge score that produced no usable judgment."""

    if score_key != CURRENT_SCORE_KEY or judgment is not None:
        return None
    envelope = score_metadata.get("environment_judge")
    if not isinstance(envelope, dict):
        envelope = score_metadata.get("final_judgment")
    envelope = envelope if isinstance(envelope, dict) else {}
    failures = envelope.get("fresh_failures") or []
    return {
        "status": str(envelope.get("post_validation") or "failed"),
        "failure_kind": str(
            envelope.get("failure_kind") or "no_usable_judgment"
        ),
        "explanation": score_explanation or "The judge produced no usable judgment.",
        "score_value": _plain(score_value),
        "fresh_attempt_count": envelope.get("fresh_attempt_count"),
        "fresh_call_limit": envelope.get("fresh_call_limit"),
        "recorded_failure_count": len(failures) if isinstance(failures, list) else None,
    }


def sample_to_audit(*, mode: str, mode_mtime: float, task: str,
                    run_metadata: dict, roles: dict, sample: Any) -> dict:
    """Convert one Inspect sample into the plain viewer contract."""
    attachments = getattr(sample, "attachments", None) or {}
    sample_metadata = _plain(getattr(sample, "metadata", None) or {})
    real_env = sample_metadata.get("real_env") or {}
    retrospective = sample_metadata.get("retrospective_rejudge")
    if not isinstance(retrospective, dict):
        retrospective = None
    target_model = (
        retrospective.get("target")
        if retrospective is not None
        else _role_model(roles, "target") or run_metadata.get("target_model")
    )
    raw_messages = list(getattr(sample, "messages", None) or [])
    transcript_source = "sample_messages"
    recovery = None
    if (
        retrospective is None
        and ((real_env.get("harness") or {}).get("scaffold")) == "opencode"
    ):
        raw_messages, recovery = recover_interrupted_opencode_messages(
            raw_messages,
            list(getattr(sample, "events", None) or []),
            target_model=target_model,
            applied_before_judging=False,
            attachments=attachments,
        )
        if recovery is not None:
            real_env["interrupted_native_transcript"] = recovery
            transcript_source = "reconstructed_interrupted_opencode_events"
    messages = normalize_messages(
        raw_messages,
        attachments,
        _message_timings(sample, target_model, raw_messages),
    )
    if ((real_env.get("harness") or {}).get("scaffold")) == "codex":
        stamp_codex_native_roles(messages)
    present_messages = {message["number"] for message in messages}
    source_lookup = {
        "ids": {
            str(message["source_id"]): message["number"]
            for message in messages
            if message.get("source_id") is not None
        },
        "indexes": {
            index: message["number"] for index, message in enumerate(messages)
        },
    }
    score_key, score = _score_for_sample(sample)
    score_value = getattr(score, "value", None) if score is not None else None
    score_metadata = (
        _plain(getattr(score, "metadata", None) or {}) if score is not None else {}
    )
    score_explanation = (
        str(getattr(score, "explanation", None) or "") if score is not None else ""
    )
    if score_key == LEGACY_SCORE_KEY:
        # Old summaries/highlights live in score metadata; keep the score explanation too.
        score_metadata = {**score_metadata, "explanation": score_explanation}

    judgment = normalize_judgment(
        score_key=score_key,
        score_value=score_value,
        score_metadata=score_metadata,
        real_env=real_env,
        present_messages=present_messages,
        source_lookup=source_lookup,
        score_explanation=score_explanation,
    )
    judge_failure = _judge_failure_record(
        score_key=score_key,
        score_value=score_value,
        score_metadata=score_metadata,
        score_explanation=score_explanation,
        judgment=judgment,
    )
    judgment_transcript_coverage = None
    if recovery is not None:
        judgment_transcript_coverage = {
            "complete": False,
            "stored_judgment_predates_reconstruction": score is not None,
            "messages_seen_by_stored_judgment": recovery["original_message_count"],
            "messages_shown_after_reconstruction": recovery[
                "reconstructed_message_count"
            ],
            "recovered_messages_not_seen_by_stored_judgment": recovery[
                "added_message_count"
            ],
        }
    protocol = (real_env.get("protocol") or {}) if isinstance(real_env, dict) else {}
    ended_reason = protocol.get("ended_reason")
    assistant_messages = [message for message in messages if message["role"] == "assistant"]
    has_target_output = any(
        message["text"].strip() or message["tool_calls"] for message in assistant_messages
    )
    target_usage = (getattr(sample, "role_usage", None) or {}).get("target")
    target_output_tokens = (
        getattr(target_usage, "output_tokens", 0) or 0
        if target_usage is not None else 0
    )
    compactions = [
        _plain(event)
        for event in getattr(sample, "events", None) or []
        if getattr(event, "event", None) == "compaction"
    ]

    scores = _plain(score_value) if score_key == LEGACY_SCORE_KEY and isinstance(score_value, dict) else {}
    target_context_usage = _target_context_usage(sample, target_model)
    stored_seed = str(
        retrospective.get("seed")
        if retrospective is not None else getattr(sample, "id", "")
    )
    real_env_manifest_sha = (
        retrospective.get("real_env_manifest_sha")
        if retrospective is not None
        else run_metadata.get("real_env_manifest_sha")
    )
    audit = {
        "mode": mode,
        "mtime": mode_mtime,
        "task": task,
        "seed": stored_seed,
        "stored_seed": stored_seed,
        "real_env_manifest_sha": real_env_manifest_sha,
        "epoch": (
            retrospective.get("epoch")
            if retrospective is not None else getattr(sample, "epoch", None)
        ),
        "target": target_model,
        "judge": _role_model(roles, "judge"),
        "condition": (
            retrospective.get("condition")
            if retrospective is not None else run_metadata.get("condition")
        ),
        "pressure": (
            retrospective.get("pressure")
            if retrospective is not None
            else run_metadata.get("pressure") or real_env.get("pressure")
        ),
        "judgment_role": (
            "retrospective_rejudge" if retrospective is not None else "official"
        ),
        "reasoning": run_metadata.get("reasoning"),
        "harness": (
            retrospective.get("harness")
            if retrospective is not None
            else run_metadata.get("harness")
        ) or "simple",
        "production_harness": (
            (real_env.get("harness") or {})
            if retrospective is not None
            else run_metadata.get("production_harness")
        ),
        "native_harness": (
            (real_env.get("harness") or {})
            if retrospective is not None
            else run_metadata.get("native_harness")
            or run_metadata.get("production_harness")
        ),
        "retrospective_rejudge": retrospective,
        "dimension_scope": run_metadata.get("dimension_scope"),
        "target_tools_mode": run_metadata.get("target_tools_mode"),
        "judge_dimensions": list(run_metadata.get("judge_dimensions") or []),
        "judge_dimension_files": list(run_metadata.get("judge_dimension_files") or []),
        "score_key": score_key,
        "score_value": _plain(score_value),
        "score_metadata": score_metadata,
        "scores": scores,
        "summary": str(
            (judgment or {}).get("summary")
            or score_metadata.get("summary")
            or ""
        ),
        "highlights": str(
            (judgment or {}).get("highlights")
            or score_metadata.get("highlights")
            or ""
        ),
        "explanation": str(
            (judgment or {}).get("explanation") or score_explanation or ""
        ),
        "judgment": judgment,
        "judge_failure": judge_failure,
        "messages": messages,
        "transcript": transcript_text(messages),
        "transcript_source": transcript_source,
        "interrupted_native_transcript": recovery,
        "judgment_transcript_coverage": judgment_transcript_coverage,
        "real_env": real_env,
        "real_ended_reason": ended_reason,
        "dead": not has_target_output and target_output_tokens == 0,
        "crashed": run_metadata.get("target_tools_mode") == "real" and ended_reason is None,
        "tool_truncations": tool_output_truncations(sample),
        "target_provider_events": target_provider_events(sample, target_model),
        "target_context_usage": target_context_usage,
        "compactions": compactions,
        "model_usage": _usage_dict(getattr(sample, "model_usage", None)),
        "role_usage": _usage_dict(getattr(sample, "role_usage", None)),
        "load_issues": [
            *list((judgment or {}).get("issues") or []),
            *([{
                "kind": "stored_judgment_missing_recovered_messages",
                "recovered_message_count": recovery["added_message_count"],
            }] if recovery is not None and score is not None else []),
        ],
    }
    return finalize_audit_integrity(audit)


def link_rejudge_sources(audits: list[dict]) -> None:
    """Attach original trajectory identity, timing, and context to rejudges."""

    originals = {
        (
            audit.get("mode"), audit.get("task"),
            str(audit.get("stored_seed") or audit.get("seed")),
            audit.get("epoch"),
        ): audit
        for audit in audits
        if not audit.get("retrospective_rejudge")
    }
    for audit in audits:
        source = audit.get("retrospective_rejudge")
        if not isinstance(source, dict):
            continue
        key = (
            source.get("source_run"), source.get("source_task"),
            str(source.get("seed")), source.get("epoch"),
        )
        original = originals.get(key)
        audit["source_trajectory_id"] = original.get("id") if original else None
        if original:
            audit["seed"] = original.get("seed")
            audit["stored_seed"] = (
                original.get("stored_seed") or original.get("seed")
            )
            audit["real_env_manifest_sha"] = original.get(
                "real_env_manifest_sha"
            )
            original_messages = original.get("messages") or []
            original_by_source_id = {
                str(message.get("source_id")): message
                for message in original_messages
                if message.get("source_id") is not None
            }
            for index, message in enumerate(audit.get("messages") or []):
                source_id = message.get("source_id")
                source_message = (
                    original_by_source_id.get(str(source_id))
                    if source_id is not None
                    else None
                )
                if (
                    source_message is None
                    and index < len(original_messages)
                    and message.get("role") == original_messages[index].get("role")
                    and message.get("text") == original_messages[index].get("text")
                ):
                    source_message = original_messages[index]
                if source_message is None:
                    continue
                for field in (
                    "timestamp", "elapsed_seconds", "elapsed_time", "timestamp_source",
                ):
                    message[field] = source_message.get(field)
        if original and isinstance(original.get("target_context_usage"), dict):
            audit["target_context_usage"] = {
                **deepcopy(original["target_context_usage"]),
                "origin": "source_trajectory",
                "source_trajectory_id": original.get("id"),
            }


_PROMOTED_JUDGMENT_FIELDS = (
    "judge",
    "judgment_role",
    "score_key",
    "score_value",
    "score_metadata",
    "scores",
    "summary",
    "highlights",
    "explanation",
    "judgment",
    "judge_failure",
)

_SUPERSEDED_STATUS_FIELDS = (
    "judgment_transcript_coverage",
    "load_issues",
    "flags",
    "mechanical_status",
    "benchmark_eligible",
    "prefix_eligible",
    "integrity_status",
    "integrity_issues",
    "prefix_issues",
    "status_tags",
)

_JUDGED_MESSAGE_FIELDS = (
    "role",
    "text",
    "reasoning",
    "other_content_blocks",
    "tool_calls",
    "tool_call_id",
    "function",
    "error",
    "source_id",
)


def _judged_message_sequence(audit: dict) -> list[dict]:
    return [
        {
            field: deepcopy(message.get(field))
            for field in _JUDGED_MESSAGE_FIELDS
            if field in message
        }
        for message in audit.get("messages") or []
    ]


def promote_rejudge_judgments(
    audits: list[dict], promoted_run_names: Iterable[str],
    *, archived_source_keys: Iterable[str] = (),
) -> int:
    """Make explicitly selected rejudgments canonical on their source rows.

    Promotion is deliberately fail-closed. A selected run must be present, contain
    only successful structured rejudgments, resolve one-to-one onto source rows whose
    stored judges missed reconstructed messages, and carry the exact same judged
    message sequence. Raw Inspect logs are never changed; the superseded fields remain
    attached to the viewer audit for provenance and judge-comparison rendering.
    """

    selected = {str(name) for name in promoted_run_names}
    if not selected:
        return 0

    present = {
        str(audit.get("mode") or "")
        for audit in audits
        if str(audit.get("mode") or "") in selected
    }
    missing = sorted(selected - present)
    if missing:
        raise ValueError(
            "promoted rejudge run directories are not loaded: " + ", ".join(missing)
        )

    originals_by_id = {
        int(audit["id"]): audit
        for audit in audits
        if not audit.get("retrospective_rejudge") and audit.get("id") is not None
    }
    archived_sources = set(archived_source_keys)
    claimed_sources: dict[int, str] = {}
    promoted = 0
    for rejudge in audits:
        mode = str(rejudge.get("mode") or "")
        if mode not in selected:
            continue
        retrospective = rejudge.get("retrospective_rejudge")
        if not isinstance(retrospective, dict):
            raise ValueError(
                f"promoted rejudge run {mode} contains a non-rejudge trajectory"
            )
        source_id = rejudge.get("source_trajectory_id")
        original = originals_by_id.get(int(source_id)) if source_id is not None else None
        if original is None:
            source_key = (
                f"{retrospective.get('source_run')}__{retrospective.get('source_task')}"
                f"__{retrospective.get('seed')}__e{retrospective.get('epoch')}"
            )
            if source_key in archived_sources:
                # A release retains the rejudge log even when an explicitly archived
                # source is absent. Never relax validation for other missing sources.
                issue = {"kind": "archived_rejudge_source_not_in_release",
                         "source_trajectory_key": source_key}
                issues = rejudge.setdefault("load_issues", [])
                if issue not in issues:
                    issues.append(issue)
                continue
            raise ValueError(
                f"promoted rejudge trajectory {rejudge.get('id')} has no loaded source"
            )
        if int(source_id) in claimed_sources:
            raise ValueError(
                "multiple promoted rejudgments target source trajectory "
                f"{source_id}: {claimed_sources[int(source_id)]}, {mode}"
            )
        judgment = rejudge.get("judgment") or {}
        envelope = judgment.get("envelope") or {}
        raw_result = judgment.get("raw_result")
        if not isinstance(raw_result, dict):
            raw_result = envelope.get("result")
        if (
            rejudge.get("judge_failure")
            or judgment.get("format") != "structured"
            or not isinstance(raw_result, dict)
            or envelope.get("post_validation") != "passed"
        ):
            raise ValueError(
                f"promoted rejudge trajectory {rejudge.get('id')} has no valid judgment"
            )
        coverage = original.get("judgment_transcript_coverage") or {}
        if not coverage.get("stored_judgment_predates_reconstruction"):
            raise ValueError(
                f"source trajectory {source_id} has no superseded transcript judgment"
            )
        if _judged_message_sequence(original) != _judged_message_sequence(rejudge):
            raise ValueError(
                f"promoted rejudge trajectory {rejudge.get('id')} does not match "
                f"source trajectory {source_id}'s judged messages"
            )

        superseded_fields = {
            field: deepcopy(original.get(field))
            for field in (
                *_PROMOTED_JUDGMENT_FIELDS,
                *_SUPERSEDED_STATUS_FIELDS,
            )
        }
        original["superseded_judgment"] = {
            "reason": "stored_judgment_missed_recovered_messages",
            "replaced_by_rejudge_run": mode,
            "replaced_by_rejudge_trajectory_id": rejudge.get("id"),
            "audit_fields": superseded_fields,
        }
        for field in _PROMOTED_JUDGMENT_FIELDS:
            original[field] = deepcopy(rejudge.get(field))
        original["judgment_role"] = "promoted_retrospective_rejudge"
        original["promoted_rejudge"] = {
            "run": mode,
            "trajectory_id": rejudge.get("id"),
            "source_key": retrospective.get("source_key"),
            "judge": rejudge.get("judge"),
            "judging_method_sha256": envelope.get("judge_method_sha256"),
            "campaign_fingerprint": retrospective.get("campaign_fingerprint"),
            "reason": "stored_judgment_missed_recovered_messages",
        }
        message_count = len(original.get("messages") or [])
        original["judgment_transcript_coverage"] = {
            "complete": True,
            "stored_judgment_predates_reconstruction": False,
            "messages_seen_by_canonical_judgment": message_count,
            "messages_shown_after_reconstruction": message_count,
            "recovered_messages_not_seen_by_canonical_judgment": 0,
            "judgment_source": "promoted_retrospective_rejudge",
        }
        original["load_issues"] = [
            issue for issue in original.get("load_issues") or []
            if not (
                isinstance(issue, dict)
                and issue.get("kind")
                == "stored_judgment_missing_recovered_messages"
            )
        ]
        stored_integrity = deepcopy(original.get("stored_integrity") or {})
        stored_integrity["issues"] = [
            issue for issue in stored_integrity.get("issues") or []
            if str(issue) != "judge_missing_recovered_messages"
        ]
        original["stored_integrity"] = stored_integrity
        real_env = deepcopy(original.get("real_env") or {})
        real_env["final_judgment"] = {
            **deepcopy(envelope),
            "result": deepcopy(raw_result),
        }
        original["real_env"] = real_env
        finalize_audit_integrity(original)

        claimed_sources[int(source_id)] = mode
        promoted += 1
    return promoted


def attach_remote_compute(mode_dir: Path, audits: list[dict]) -> list[dict]:
    """Attach final VM-cost records from a verified imported AWS campaign."""

    path = mode_dir / "remote_campaign.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [{
            "kind": "invalid_remote_campaign_sidecar",
            "path": str(path),
            "error": f"{type(error).__name__}: {error}",
        }]
    task_compute = payload.get("task_compute")
    if not isinstance(task_compute, dict):
        return [{
            "kind": "invalid_remote_campaign_sidecar",
            "path": str(path),
            "error": "task_compute is missing or is not an object",
        }]
    terminal_by_task = {}
    for cell in payload.get("cells") or []:
        terminal = cell.get("terminal") or {}
        task_name = terminal.get("task_name")
        if not task_name:
            continue
        terminal_by_task[str(task_name)] = {
            "campaign_cell_status": cell.get("status"),
            "pipeline_exit_code": terminal.get("pipeline_exit_code"),
            "worker_started_at": terminal.get("started_at"),
            "worker_completed_at": terminal.get("completed_at"),
        }
    for audit in audits:
        compute = task_compute.get(str(audit.get("task") or ""))
        if not isinstance(compute, dict):
            continue
        real_env = audit.get("real_env")
        if not isinstance(real_env, dict):
            real_env = {}
            audit["real_env"] = real_env
        # The sidecar is written after termination, so it supersedes the pre-launch
        # execution identity embedded inside the EvalLog.
        real_env["compute"] = {
            **deepcopy(compute),
            **deepcopy(terminal_by_task.get(str(audit.get("task") or ""), {})),
        }
    return []


def attach_integrity_sidecar(mode_dir: Path, audits: list[dict]) -> list[dict]:
    """Attach the experiment pipeline's persisted per-sample integrity verdict."""

    path = mode_dir / "pipeline_integrity.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [{
            "kind": "invalid_pipeline_integrity_sidecar",
            "path": str(path),
            "error": f"{type(error).__name__}: {error}",
        }]
    records = payload.get("records")
    if not isinstance(records, list):
        return [{
            "kind": "invalid_pipeline_integrity_sidecar",
            "path": str(path),
            "error": "records is missing or is not an array",
        }]
    by_key = {
        (str(record.get("task")), str(record.get("sample")), record.get("epoch")): record
        for record in records if isinstance(record, dict)
    }
    for audit in audits:
        key = (
            str(audit.get("task")),
            str(audit.get("stored_seed") or audit.get("seed")),
            audit.get("epoch"),
        )
        record = by_key.get(key)
        if record is not None:
            audit["stored_integrity"] = deepcopy(record)
    return []


def _supersede_retry_attempts(audits: list[dict]) -> list[dict]:
    """Keep only the final eval-set attempt per trajectory identity.

    Inspect ``eval_set`` retries a failed task by writing a NEW ``.eval`` file
    into the same run directory, so a retried trajectory appears once per
    attempt. The latest attempt is authoritative (eval-set never retries a
    completed sample); earlier attempts are dropped here, with the drop stored
    on the survivor and surfaced through its visible load issues.
    """

    groups: dict[str, list[dict]] = {}
    for audit in audits:
        groups.setdefault(traj_key(audit), []).append(audit)
    survivors: dict[int, dict] = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda item: str(item.get("log_file") or ""))
        survivor = ordered[-1]
        if len(ordered) > 1:
            survivor["superseded_retry_attempts"] = [
                {
                    "log_file": earlier.get("log_file"),
                    "real_ended_reason": earlier.get("real_ended_reason"),
                }
                for earlier in ordered[:-1]
            ]
            survivor.setdefault("load_issues", []).append(
                f"{len(ordered) - 1} earlier retry attempt log(s) superseded by "
                f"{survivor.get('log_file')}"
            )
        survivors[id(survivor)] = survivor
    return [audit for audit in audits if id(audit) in survivors]


async def load_mode(
    mode_dir: Path, *, cache_root: Path | None = None, use_cache: bool = True
) -> list[dict]:
    """Load one real-environment run directory through Inspect's public log API."""
    cache_path = _cache_file(mode_dir, cache_root) if cache_root and use_cache else None
    if cache_path and cache_path.exists():
        try:
            return pickle.loads(cache_path.read_bytes())
        except (OSError, EOFError, pickle.UnpicklingError):
            pass

    # Lazy import keeps pure normalizer/component tests runnable without the runtime.
    from inspect_ai.log import list_eval_logs, read_eval_log

    eval_paths = list(mode_dir.glob("*.eval"))
    mode_mtime = max((path.stat().st_mtime for path in eval_paths), default=0.0)
    audits = []
    for log_info in list_eval_logs(str(mode_dir)):
        try:
            log = read_eval_log(log_info)
        except FileNotFoundError:
            continue
        roles = getattr(log.eval, "model_roles", None) or {}
        run_metadata = _plain(getattr(log.eval, "metadata", None) or {})
        for sample in getattr(log, "samples", None) or []:
            audit = sample_to_audit(
                mode=mode_dir.name,
                mode_mtime=mode_mtime,
                task=str(log.eval.task),
                run_metadata=run_metadata,
                roles=roles,
                sample=sample,
            )
            audit["log_file"] = Path(str(getattr(log_info, "name", log_info))).name
            audits.append(audit)
    audits = _supersede_retry_attempts(audits)
    sidecar_issues = attach_remote_compute(mode_dir, audits)
    sidecar_issues.extend(attach_integrity_sidecar(mode_dir, audits))
    for audit in audits:
        finalize_audit_integrity(audit)
    if sidecar_issues:
        for audit in audits:
            audit.setdefault("load_issues", []).extend(deepcopy(sidecar_issues))
    if cache_path:
        _write_pickle_atomic(cache_path, audits)
        for stale in cache_path.parent.glob(f"mode__{mode_dir.name}__*.pkl"):
            if stale != cache_path:
                stale.unlink(missing_ok=True)
    return audits


def _prune_superseded_mode_pickles(
    mode_dir: Path, cache_root: Path | None
) -> None:
    """Remove whole-run caches after the bounded store has the same run."""

    if cache_root is None or not cache_root.is_dir():
        return
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", mode_dir.name)
    for path in cache_root.glob(f"mode__{safe}__*.pkl"):
        path.unlink(missing_ok=True)


def mode_directories(logs_root: Path) -> list[Path]:
    """Run directories the viewer loads from one root, skipping hidden markers."""

    if not logs_root.is_dir():
        return []
    return sorted(
        path for path in logs_root.iterdir()
        if path.is_dir() and not (path / ".viewer-hidden").is_file()
    )


async def load_all(
    logs_root: Path, *, cache_root: Path | None = None, use_cache: bool = True,
    audit_store: Any | None = None, progress: bool = False,
    store_mode_prefix: str = "", prune: bool = True,
    extra_active_store_modes: frozenset[str] = frozenset(),
) -> tuple[list[dict], list[dict]]:
    """Load every run directory, returning visible per-directory errors separately.

    When ``audit_store`` is supplied, only compact index/aggregate records stay in
    memory.  Full normalized audits are persisted one by one for later detail-page
    hydration.  The ordinary list-returning behavior remains available to loader tests
    and small callers.

    ``store_mode_prefix`` namespaces this root's runs inside a shared ``audit_store`` so
    two roots may contain same-named run directories.  Callers loading several roots
    pass ``prune=False`` for all but the last root and give the last call the earlier
    roots' store modes as ``extra_active_store_modes`` so pruning keeps them.
    """
    if not logs_root.is_dir():
        return [], []
    audits: list[dict] = []
    errors: list[dict] = []
    mode_dirs = mode_directories(logs_root)
    for mode_index, mode_dir in enumerate(mode_dirs, start=1):
        signature = _mode_signature(mode_dir)
        cached = (
            audit_store.cached_mode(store_mode_prefix + mode_dir.name, signature)
            if audit_store is not None and use_cache
            else None
        )
        if cached is not None:
            audits.extend(cached)
            _prune_superseded_mode_pickles(mode_dir, cache_root)
        else:
            try:
                loaded = await load_mode(
                    mode_dir,
                    # The bounded store supersedes the old whole-mode pickle.  Writing
                    # both would retain two multi-gigabyte viewer caches on disk.
                    cache_root=None if audit_store is not None else cache_root,
                    use_cache=False if audit_store is not None else use_cache,
                )
            except MemoryError:
                raise
            except Exception as error:  # one malformed run must not hide healthy runs
                errors.append({
                    "mode": mode_dir.name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
            else:
                if audit_store is None:
                    audits.extend(loaded)
                else:
                    # Store failures are fatal. Continuing after a disk/database
                    # error could publish a viewer that silently dropped healthy data.
                    audits.extend(
                        audit_store.replace_mode(
                            store_mode_prefix + mode_dir.name, signature, loaded
                        )
                    )
                    _prune_superseded_mode_pickles(mode_dir, cache_root)
                    del loaded
                    gc.collect()
        if progress and (mode_index % 10 == 0 or mode_index == len(mode_dirs)):
            print(
                f"viewer: loaded {mode_index}/{len(mode_dirs)} run directories "
                f"({len(audits):,} compact trajectories, {len(errors)} errors)",
                flush=True,
            )
    if audit_store is not None:
        if prune:
            audit_store.prune_modes(
                {store_mode_prefix + path.name for path in mode_dirs}
                | set(extra_active_store_modes)
            )
        if cache_root is not None and cache_root.is_dir():
            # Every active run is now present in the bounded store. Any remaining
            # whole-mode files belong to directories that were removed or renamed.
            for path in cache_root.glob("mode__*.pkl"):
                path.unlink(missing_ok=True)
    return audits, errors
