"""Task-turn semantics shared by the real-environment viewer surfaces.

The saved transcript is a wire-level record. Native scaffolds can add genuine
``user``-role messages of their own (for example Codex's ``environment_context``),
but those are not turns sent by the experiment. The protocol record is therefore the
authoritative source for current runs; transcript counting is only a legacy fallback.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping


def is_experiment_user_message(message: dict) -> bool:
    """Whether a normalized message is a user turn sent by the experiment."""

    return (
        message.get("role") == "user"
        and not message.get("scaffold_injected")
    )


def number_message_turns(messages: Iterable[MutableMapping]) -> None:
    """Add transcript A#/U# indices without treating scaffold preambles as U turns."""

    assistant_turn = 0
    user_turn = 0
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            assistant_turn += 1
        if is_experiment_user_message(message):
            user_turn += 1
        message["assistant_turn"] = (
            assistant_turn if role == "assistant" else None
        )
        message["user_turn"] = (
            user_turn if is_experiment_user_message(message) else None
        )


def _stored_protocol_turn_count(audit: dict) -> int | None:
    """Return the controller-authored task-turn count when the run stored it."""

    protocol = ((audit.get("real_env") or {}).get("protocol") or {})
    follow_up_sent = protocol.get("follow_up_sent")
    if isinstance(follow_up_sent, bool):
        # Every real-environment task starts with one opening user message. The
        # current controller can then send exactly one pinned follow-up.
        return 2 if follow_up_sent else 1
    return None


def experiment_user_turn_count(audit: dict) -> int | None:
    """Count controller-authored user turns for an original or continuation task.

    Current runs store whether the sole optional follow-up was sent, which is robust
    to harness-specific transcript preambles. Legacy runs fall back to their saved
    transcript, excluding scaffold-injected messages and, for continuations, all
    messages on the prefix side of the stored boundary.
    """

    if "_viewer_experiment_user_turn_count" in audit:
        compact_count = audit.get("_viewer_experiment_user_turn_count")
        return (
            compact_count
            if isinstance(compact_count, int)
            and not isinstance(compact_count, bool)
            else None
        )

    stored = _stored_protocol_turn_count(audit)
    if stored is not None:
        return stored

    messages = audit.get("messages") or []
    if not messages:
        return None

    continuation = ((audit.get("real_env") or {}).get("continuation") or {})
    boundary = continuation.get("boundary_index")
    has_boundary = isinstance(boundary, int) and not isinstance(boundary, bool)
    return sum(
        1
        for message in messages
        if is_experiment_user_message(message)
        and (
            not has_boundary
            or (
                isinstance(message.get("number"), int)
                and message["number"] > boundary
            )
        )
    )
