"""Paid current judge call for one stored or live real trajectory."""

from __future__ import annotations

from inspect_ai.model import ChatMessage, Model

from continuation_evidence import (
    continuation_caveat_fields,
    judge_view_messages as continuation_judge_view_messages,
)
from multi_agent_evidence import (
    caveat_fields as activity_log_caveat_fields,
    judge_view_messages as activity_log_judge_view_messages,
    sanitize_artifact_snapshots as sanitize_activity_log_artifact_snapshots,
)
from real_judge_evidence import complete_real_trajectory_evidence

from .evidence import EvidenceCaveat
from .exp_runner import judge_trajectory


async def judge_complete_real_trajectory(
    *,
    family,
    messages: list[ChatMessage],
    real_env: dict,
    model: str | Model | None,
):
    """Apply the current final-judgment prompt/schema/evidence policy.

    A continuation trajectory (``real_env["continuation"]``) is judged on its live
    task only: the prefix conversation is sliced out mechanically by the stored
    boundary index, and the omission is declared as an evidence caveat. This is the
    one shared path, so production and retrospective rejudges scope identically.
    """

    continuation = (real_env or {}).get("continuation")
    if continuation:
        messages = continuation_judge_view_messages(messages, continuation)
    activity_log = (real_env or {}).get("multi_agent")
    if activity_log:
        messages, masking = activity_log_judge_view_messages(messages, activity_log)
        activity_log.setdefault("judge_masking", {}).update(masking)
    artifacts, caveats = complete_real_trajectory_evidence(messages, real_env)
    if activity_log:
        artifacts, masked_paths = sanitize_activity_log_artifact_snapshots(
            artifacts, activity_log
        )
        if masked_paths:
            masking = activity_log.setdefault("judge_masking", {})
            existing = masking.setdefault("artifact_paths_masked", [])
            existing.extend(path for path in masked_paths if path not in existing)
    if continuation:
        caveats = [
            *caveats,
            EvidenceCaveat(**continuation_caveat_fields(continuation)),
        ]
    if activity_log:
        caveats = [
            *caveats,
            EvidenceCaveat(**activity_log_caveat_fields(activity_log)),
        ]
    return await judge_trajectory(
        family=family,
        stage="final",
        messages=messages,
        artifacts=artifacts,
        upstream_caveats=caveats,
        model=model,
    )
