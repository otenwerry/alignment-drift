"""Stored, queryable integrity and warning flags for real-environment runs.

The experiment pipeline and viewer share these rules.  This keeps a run from being
accepted by one layer and silently excluded by another.
"""

from __future__ import annotations

from typing import Any


BLOCKING_TARGET_PROVIDER_EVENT_KINDS = frozenset({
    "content_filter",
    "provider_error",
    "empty_response",
})


JUDGMENT_FAILURE_ISSUES = frozenset({
    "judge_not_judged",
    "missing_environment_judge_score",
    "missing_structured_result",
    "judge_post_validation",
    "unusable_judgment",
    "judge_missing_recovered_messages",
})

ENVIRONMENT_FAILURE_ISSUES = frozenset({"environment_failure"})

TECHNICAL_FAILURE_ISSUES = frozenset({
    "missing_protocol_record",
    "finalize_error",
    "missing_final_artifact_record",
    "gate_error",
    "ended_early",
    "production_prefix_boundary_lost",
    "native_resume_bundle_unavailable",
    "activity_log_not_fully_delivered",
    *JUDGMENT_FAILURE_ISSUES,
})


EVIDENCE_CAVEAT_FLAGS = {
    "messages_excluded_by_family_policy": (
        "messages_excluded_by_family_policy", "judge messages omitted"
    ),
    "native_reasoning_excluded_by_policy": (
        "native_reasoning_excluded_by_policy", "judge reasoning omitted"
    ),
    "tool_calls_excluded_by_family_policy": (
        "tool_calls_excluded_by_family_policy", "judge tool calls omitted"
    ),
    "tool_results_excluded_by_family_policy": (
        "tool_results_excluded_by_family_policy", "judge tool results omitted"
    ),
    "native_reasoning_not_fully_available_upstream": (
        "native_reasoning_not_fully_available_upstream", "reasoning incomplete"
    ),
    "native_system_prompt_unavailable": (
        "native_system_prompt_unavailable", "system prompt unavailable"
    ),
    "changed_non_text_artifacts_not_copied": (
        "changed_non_text_artifacts_not_copied", "non-text artifacts omitted"
    ),
    "target_scaffold_context_compacted": (
        "context_compacted", "context compacted"
    ),
    "target_scaffold_tool_output_pruned": (
        "scaffold_tool_output_pruned", "tool output pruned"
    ),
    "initial_task_snapshot_unavailable_upstream": (
        "initial_task_snapshot_unavailable_upstream", "initial task unavailable"
    ),
    "submission_snapshots_unavailable_upstream": (
        "submission_snapshots_unavailable_upstream",
        "submission snapshots unavailable",
    ),
    "final_artifacts_unavailable_upstream": (
        "missing_final_artifact_record", "final artifacts unavailable"
    ),
    "prior_unrelated_task_omitted": (
        "prior_unrelated_task_omitted", "prior task omitted"
    ),
    "interrupted_native_transcript_reconstructed": (
        "interrupted_native_transcript_reconstructed",
        "interrupted transcript recovered",
    ),
}


def tool_output_truncations(sample: Any) -> list[dict]:
    out = []
    for event in getattr(sample, "events", None) or []:
        if getattr(event, "event", None) != "tool" or not getattr(event, "truncated", None):
            continue
        truncated = getattr(event, "truncated")
        try:
            original, limit = int(truncated[0]), int(truncated[1])
        except (IndexError, TypeError, ValueError):
            original, limit = None, None
        out.append({
            "tool": getattr(event, "function", None),
            "original_bytes": original,
            "limit_bytes": limit,
        })
    return out


def sample_has_target_output(sample: Any) -> bool:
    """Whether stored assistant messages contain visible agent output or a tool call."""

    for message in getattr(sample, "messages", None) or []:
        if getattr(message, "role", None) != "assistant":
            continue
        if getattr(message, "tool_calls", None) or []:
            return True
        content = getattr(message, "content", "")
        if isinstance(content, str):
            if content.strip():
                return True
            continue
        for block in content or []:
            if isinstance(block, str) and block.strip():
                return True
            if getattr(block, "type", None) != "reasoning" \
                    and str(getattr(block, "text", None) or "").strip():
                return True
    return False


def _response_presence(event: Any) -> tuple[bool, bool]:
    """Return (visible/actionable content, readable native reasoning)."""

    output = getattr(event, "output", None)
    choices = getattr(output, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    reasoning = ""
    if isinstance(content, str):
        visible = content
    else:
        visible = "".join(
            block if isinstance(block, str) else (
                (getattr(block, "text", None) or "")
                if getattr(block, "type", None) != "reasoning" else ""
            )
            for block in (content or [])
        )
        reasoning = "".join(
            str(
                (block.get("reasoning") or block.get("summary") or block.get("text") or "")
                if isinstance(block, dict)
                else (
                    getattr(block, "reasoning", None)
                    or getattr(block, "summary", None)
                    or getattr(block, "text", None)
                    or ""
                )
            )
            for block in (content or [])
            if (
                (block.get("type") if isinstance(block, dict) else getattr(block, "type", None))
                == "reasoning"
            )
        )
    return bool(
        visible.strip()
        or str(getattr(message, "refusal", None) or "").strip()
        or (getattr(message, "tool_calls", None) or [])
    ), bool(reasoning.strip())


def _interrupted_opencode_sample(sample: Any) -> bool:
    metadata = _get(sample, "metadata", {}) or {}
    real_env = _get(metadata, "real_env", {}) or {}
    harness = _get(real_env, "harness", {}) or {}
    protocol = _get(real_env, "protocol", {}) or {}
    recovery = _get(real_env, "interrupted_native_transcript", {}) or {}
    return (
        _get(harness, "scaffold") == "opencode"
        and (
            bool(_get(recovery, "reconstructed"))
            or _get(protocol, "ended_reason")
            in {"wall_clock_limit", "cancelled", "interrupted"}
        )
    )


def target_model_events(
    sample: Any, target_model: str | None
) -> tuple[list[Any], bool]:
    """Select agent-call events and report whether model-name fallback was used."""

    events = []
    used_model_fallback = False
    for event in getattr(sample, "events", None) or []:
        if getattr(event, "event", None) != "model":
            continue
        role = getattr(event, "role", None)
        if role is not None:
            if role != "target":
                continue
        else:
            if target_model and getattr(event, "model", None) != target_model:
                continue
            used_model_fallback = True
        events.append(event)
    return events, used_model_fallback


def same_model_request(left: Any, right: Any) -> bool:
    """Whether two recorded model events represent the same provider request."""

    try:
        return all(
            getattr(left, field, None) == getattr(right, field, None)
            for field in ("model", "input", "tools", "config")
        )
    except (TypeError, ValueError):
        return False


def target_provider_events(sample: Any, target_model: str | None) -> list[dict]:
    """Return agent-call anomalies, retaining recovered retries as non-blocking."""

    events, _ = target_model_events(sample, target_model)

    out: list[dict] = []
    for index, event in enumerate(events):
        output = getattr(event, "output", None)
        choices = getattr(output, "choices", None) or []
        choice = choices[0] if choices else None
        usage = getattr(output, "usage", None)
        visible, reasoning = _response_presence(event)
        error = getattr(event, "error", None)
        stop_reason = str(getattr(choice, "stop_reason", None) or "unknown").lower()
        next_event = events[index + 1] if index + 1 < len(events) else None
        next_usage = getattr(getattr(next_event, "output", None), "usage", None)

        if usage is None and error and next_event is not None and next_usage is not None \
                and same_model_request(event, next_event):
            out.append({
                "kind": "failed_attempt_retried",
                "attempt": index + 1,
                "succeeded_on_attempt": index + 2,
                "error": type(error).__name__,
            })
            continue
        if stop_reason == "content_filter":
            out.append({
                "kind": "content_filter",
                "attempt": index + 1,
                "stop_reason": "content_filter",
            })
        elif error:
            out.append({
                "kind": "provider_error",
                "attempt": index + 1,
                "error": type(error).__name__,
            })
        elif not visible and reasoning:
            out.append({
                "kind": "reasoning_only_response",
                "attempt": index + 1,
                "stop_reason": stop_reason,
            })
        elif not visible and next_event is not None:
            out.append({
                "kind": "empty_response_retried",
                "attempt": index + 1,
                "next_attempt": index + 2,
                "stop_reason": stop_reason,
            })
        elif not visible and _interrupted_opencode_sample(sample):
            out.append({
                "kind": "empty_response_interrupted",
                "attempt": index + 1,
                "stop_reason": stop_reason,
            })
        elif not visible:
            out.append({
                "kind": "empty_response",
                "attempt": index + 1,
                "stop_reason": stop_reason,
            })
        elif usage is None:
            out.append({"kind": "missing_usage", "attempt": index + 1})
    return out


def blocking_provider_events(events: list[dict] | None) -> list[dict]:
    return [
        event for event in (events or [])
        if event.get("kind") in BLOCKING_TARGET_PROVIDER_EVENT_KINDS
    ]


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def has_unfinished_action(messages: list[Any] | None) -> bool:
    """Whether the stored trajectory ends with an unresolved assistant tool call."""

    messages = messages or []
    if not messages:
        return False
    last = messages[-1]
    return _get(last, "role") == "assistant" and bool(_get(last, "tool_calls", []) or [])


def _normalized_environment_failed(judgment: dict) -> bool:
    for dimension in judgment.get("dimensions") or []:
        if (
            isinstance(dimension, dict)
            and dimension.get("key") == "environment_failure"
            and dimension.get("status") == "ok"
        ):
            return dimension.get("value") is True
    result = (judgment.get("envelope") or {}).get("result") or {}
    return ((result.get("environment_failure") or {}).get("value") is True)


def _raw_environment_failed(envelope: dict) -> bool:
    result = envelope.get("result") or {}
    return ((result.get("environment_failure") or {}).get("value") is True)


def _status_tags(
    issues: list[str],
    prefix_issues: list[str],
    *,
    ended_reason: str | None,
    has_output: bool,
    unfinished_action: bool,
    source_tags: list[dict] | None = None,
) -> list[dict]:
    """Return the small, human-readable explanation used by the status column."""

    issue_set = set(issues)
    prefix_issue_set = set(prefix_issues)
    all_issues = issue_set | prefix_issue_set
    tags: list[dict] = []

    def add(code: str, label: str, detail: str, *, severity: str) -> None:
        existing = next((tag for tag in tags if tag["code"] == code), None)
        if existing is None:
            tags.append({
                "code": code,
                "label": label,
                "detail": detail,
                "severity": severity,
            })
        elif severity == "error":
            # A reason can have both a prefix-only and an invalidating cause. Red
            # wins only when that reason independently invalidates this trajectory.
            existing["severity"] = "error"

    if "target_no_output" in all_issues:
        add(
            "no_output",
            "no output",
            "The agent produced no visible output or tool call.",
            severity="error",
        )
    elif ended_reason not in (None, "protocol_end") or unfinished_action:
        ended_codes = {
            "wall_clock_limit" if ended_reason == "wall_clock_limit" else "ended_early",
        }
        if unfinished_action:
            ended_codes.add("unfinished_action")
        if ended_reason == "gate_error_end":
            ended_codes.add("gate_error")
        add(
            "ended_partway",
            "ended partway",
            "The scripted run did not reach its normal end.",
            severity="error" if ended_codes & issue_set else "warning",
        )

    if all_issues & JUDGMENT_FAILURE_ISSUES:
        add(
            "judgment_failed",
            "judgment failed",
            "No usable final judgment was stored.",
            severity="error",
        )
    if all_issues & ENVIRONMENT_FAILURE_ISSUES:
        add(
            "environment_failed",
            "environment failed",
            "The judge found that the environment failed.",
            severity="error",
        )
    if "source_prefix_ineligible" in all_issues:
        add(
            "prefix_ineligible",
            "prefix ineligible",
            "This continuation was built from a source that cannot serve as a prefix.",
            severity="error",
        )

    if ended_reason == "wall_clock_limit":
        add(
            "time_limit",
            "time limit",
            "The wall-clock limit ended the run.",
            severity=(
                "error" if "wall_clock_limit" in issue_set else "warning"
            ),
        )
    if unfinished_action:
        add(
            "unfinished_action",
            "unfinished action",
            "The final assistant tool call has no stored result.",
            severity="error" if "unfinished_action" in issue_set else "warning",
        )
    technical_codes = {
        code for code in all_issues
        if code.startswith("target_provider_") or code in TECHNICAL_FAILURE_ISSUES
    }
    if technical_codes:
        add(
            "technical_failure",
            "technical failure",
            "A technical failure blocked completion.",
            severity="error" if technical_codes & issue_set else "warning",
        )

    # A continuation sourced from an ineligible run should expose the source's
    # concrete qualifier without making the continuation itself look timed out.
    if "source_prefix_ineligible" in all_issues:
        for tag in source_tags or []:
            code = str(tag.get("code") or "")
            if code in {"time_limit", "unfinished_action", "technical_failure"}:
                add(
                    code,
                    str(tag.get("label") or code.replace("_", " ")),
                    str(tag.get("detail") or ""),
                    severity=str(tag.get("severity") or "warning"),
                )

    if (issues or prefix_issues) and not tags:
        add(
            "other",
            "other",
            "See the trajectory details for the underlying issue.",
            severity="error" if issues else "warning",
        )
    return tags


PREFIX_ONLY_COMPLETION_ISSUES = frozenset({
    "wall_clock_limit",
    "unfinished_action",
})


def _issue_is_prefix_only(code: str, *, multi_agent: bool) -> bool:
    """Whether an issue preserves the benchmark outcome but blocks reuse.

    Originals and continuations share the same benchmark-eligibility rule.  Their
    display differs: originals call this state ``benchmark_only`` while continuations
    collapse it to ``valid``.  Multi-agent activity-log transfer retains its stricter
    completion contract independently.
    """

    return code == "native_resume_bundle_unavailable" or (
        not multi_agent and code in PREFIX_ONLY_COMPLETION_ISSUES
    )


def mechanical_status_record(
    *,
    issues: list[str],
    prefix_issues: list[str],
    show_benchmark_only: bool,
    ended_reason: str | None,
    has_output: bool,
    unfinished_action: bool,
    source_tags: list[dict] | None = None,
) -> dict:
    """Classify benchmark and prefix eligibility from orthogonal issue lists."""

    issues = list(dict.fromkeys(str(code) for code in issues))
    prefix_issues = list(dict.fromkeys(str(code) for code in prefix_issues))
    if issues:
        status = "invalid"
    elif prefix_issues and show_benchmark_only:
        status = "benchmark_only"
    else:
        status = "valid"
    return {
        "mechanical_status": status,
        "benchmark_eligible": not issues,
        "prefix_eligible": not issues and not prefix_issues,
        "integrity_status": "excluded" if issues else "included",
        "integrity_issues": issues,
        "prefix_issues": prefix_issues,
        "status_tags": _status_tags(
            issues,
            prefix_issues,
            ended_reason=ended_reason,
            has_output=has_output,
            unfinished_action=unfinished_action,
            source_tags=source_tags,
        ),
    }


def _flag(code: str, label: str, *, severity: str, detail: str) -> dict:
    return {
        "code": code,
        "label": label,
        "severity": severity,
        "detail": detail,
    }


def finalize_audit_integrity(audit: dict) -> dict:
    """Attach the complete integrity verdict and visible flags to a viewer audit."""

    real_env = audit.get("real_env") or {}
    protocol = real_env.get("protocol") or {}
    flags: list[dict] = []
    integrity: list[str] = []
    prefix_issues: list[str] = []
    continuation = bool(real_env.get("continuation"))
    multi_agent = real_env.get("multi_agent") or {}
    conditioned = continuation or bool(multi_agent)
    multi_agent_run = bool(multi_agent)

    def add(code: str, label: str, *, severity: str, detail: str,
            excludes: bool = False, prefix_only: bool = False) -> None:
        flags.append(_flag(code, label, severity=severity, detail=detail))
        if excludes:
            integrity.append(code)
        if prefix_only:
            prefix_issues.append(code)

    if not protocol:
        add("missing_protocol_record", "missing protocol", severity="error",
            detail="The real-environment protocol record is missing.", excludes=True)
    if real_env.get("finalize_error"):
        add("finalize_error", "finalize error", severity="error",
            detail=str(real_env["finalize_error"]), excludes=True)
    if audit.get("target_tools_mode") == "real" and not real_env.get("artifacts"):
        add("missing_final_artifact_record", "missing artifacts", severity="error",
            detail="The final artifact snapshot record is missing.", excludes=True)
    if multi_agent:
        exposure = multi_agent.get("exposure") or {}
        if exposure.get("fully_delivered") is not True:
            status = str(exposure.get("status") or "unknown")
            add(
                "activity_log_not_fully_delivered",
                f"activity log: {status.replace('_', ' ')}",
                severity="error",
                detail=(
                    "The complete numbered ACTIVITY_LOG.md was not mechanically "
                    "verified in a successful target-model input "
                    f"(status={status}, covered="
                    f"{exposure.get('covered_line_count', 0)}/"
                    f"{exposure.get('total_line_count', '?')})."
                ),
                excludes=True,
            )

    gate_errors = [
        gate for gate in real_env.get("gates", [])
        if gate.get("verdict") == "error"
    ]
    if gate_errors or protocol.get("ended_reason") == "gate_error_end":
        add("gate_error", "gate error", severity="error",
            detail="The scripted branch was selected after a failed gate call.",
            excludes=True)

    if audit.get("dead"):
        add("target_no_output", "no agent output", severity="error",
            detail="The agent produced no usable output.", excludes=True)
    for event in blocking_provider_events(audit.get("target_provider_events")):
        kind = str(event.get("kind"))
        add(f"target_provider_{kind}", kind.replace("_", " "), severity="error",
            detail=f"Unrecovered agent-provider anomaly on attempt {event.get('attempt')}.",
            excludes=True)

    judgment = audit.get("judgment") or {}
    judge_failure = audit.get("judge_failure")
    if isinstance(judge_failure, dict):
        attempts = judge_failure.get("fresh_attempt_count")
        attempt_detail = f" after {attempts} fresh calls" if attempts else ""
        add(
            "judge_not_judged",
            "not judged",
            severity="error",
            detail=(
                f"The judge produced no usable judgment{attempt_detail}. "
                + str(judge_failure.get("explanation") or "").strip()
            ).strip(),
            excludes=True,
        )
    coverage = audit.get("judgment_transcript_coverage") or {}
    if coverage.get("stored_judgment_predates_reconstruction"):
        missing = coverage.get("recovered_messages_not_seen_by_stored_judgment")
        add(
            "judge_missing_recovered_messages",
            "judgment missed transcript",
            severity="error",
            detail=(
                f"The stored judgment did not receive {missing} recovered transcript "
                "message(s) that are now visible."
            ),
            excludes=True,
        )
    if judgment.get("format") == "structured":
        envelope = judgment.get("envelope") or {}
        if not isinstance(envelope.get("result"), dict):
            add("missing_structured_result", "missing judgment", severity="error",
                detail="The structured judge result is missing.", excludes=True)
        if envelope.get("post_validation") != "passed":
            add("judge_post_validation", "judge validation", severity="error",
                detail="The structured judge result did not record passed validation.",
                excludes=True)
    elif judgment.get("format") not in {"legacy_numeric", None}:
        add("unusable_judgment", "unusable judgment", severity="error",
            detail="The judgment format is not usable.", excludes=True)
    if _normalized_environment_failed(judgment):
        add(
            "environment_failure",
            "environment failed",
            severity="error",
            detail="The structured judge found that the environment failed.",
            excludes=True,
        )

    compute = real_env.get("compute") or {}
    exit_code = compute.get("pipeline_exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        add("pipeline_exit_nonzero", f"exit {exit_code}", severity="warning",
            detail=(
                "The worker reported a non-zero pipeline exit. Historical workers could "
                "report this after a usable trajectory, so this is a warning, not an "
                "automatic exclusion."
            ))
    ended_reason = protocol.get("ended_reason") or audit.get("real_ended_reason")
    if ended_reason == "wall_clock_limit":
        prefix_only = _issue_is_prefix_only(
            "wall_clock_limit", multi_agent=multi_agent_run
        )
        add(
            "wall_clock_limit",
            "clock limit",
            severity="warning" if prefix_only else "error",
            detail="The trajectory ended at its wall-clock limit.",
            excludes=not prefix_only,
            prefix_only=prefix_only,
        )
    elif ended_reason not in (None, "protocol_end", "gate_error_end"):
        # protocol_end is the one normal ending; gate_error_end is already an
        # excluding gate_error flag above. Anything else (interrupted, cancelled,
        # unknown historical values) ended the run before the scripted protocol
        # finished and must stay visible once the metadata "ended" cell is gone.
        add("ended_early", f"ended: {ended_reason}", severity="error",
            detail=f"The run ended before the scripted protocol finished "
                   f"(ended_reason: {ended_reason}).", excludes=True)
    unfinished_action = has_unfinished_action(audit.get("messages"))
    if unfinished_action:
        prefix_only = _issue_is_prefix_only(
            "unfinished_action", multi_agent=multi_agent_run
        )
        add(
            "unfinished_action",
            "unfinished action",
            severity="warning" if prefix_only else "error",
            detail="The final assistant tool call has no stored result.",
            excludes=not prefix_only,
            prefix_only=prefix_only,
        )
    truncations = audit.get("tool_truncations") or []
    if truncations:
        add("tool_output_truncated", f"tool truncation ×{len(truncations)}",
            severity="warning",
            detail="At least one tool result was truncated before the agent saw it.")
    compactions = audit.get("compactions") or []
    if compactions:
        add("context_compacted", f"compacted ×{len(compactions)}", severity="warning",
            detail="Inspect compacted the model context during this trajectory.")
    recovery = real_env.get("interrupted_native_transcript") or {}
    if recovery.get("reconstructed"):
        add(
            "interrupted_native_transcript_reconstructed",
            f"partial transcript recovered ×{recovery.get('added_message_count', '?')}",
            severity="warning",
            detail=str(recovery.get("limitation") or ""),
        )
    continuation_record = real_env.get("continuation") or {}
    native_prefix = (
        continuation_record.get("production_native_resume")
        or continuation_record.get("subscription_native_resume")
        or continuation_record.get("production_fresh_context")
        or continuation_record.get("subscription_fresh_context")
        or {}
    )
    if native_prefix.get("boundary_lost"):
        add(
            "production_prefix_boundary_lost",
            "prefix boundary lost",
            severity="error",
            detail=(
                "The live hand-off boundary could not be recovered exactly after the "
                "native scaffold resumed its session."
            ),
            excludes=True,
        )
    harness_record = real_env.get("harness") or {}
    native_bundle = real_env.get("native_resume_bundle") or {}
    if (
        harness_record.get("mode") in {"production", "subscription"}
        and native_bundle.get("available") is not True
    ):
        add(
            "native_resume_bundle_unavailable",
            "native resume unavailable",
            severity="warning",
            detail=(
                "The native scaffold's session bundle was not saved, so "
                "this trajectory cannot be resumed literally."
            ),
            prefix_only=True,
        )
    monitoring = harness_record.get("native_loss_monitoring") or {}
    if (
        harness_record.get("mode") in {"production", "subscription"}
        and monitoring.get("complete") is False
    ):
        add(
            "production_scaffold_loss_observability",
            "scaffold loss partly unobservable",
            severity="warning",
            detail=(
                "Native scaffold loss monitoring is incomplete. This is a monitoring "
                "limit and does not mean loss occurred in this trajectory."
            ),
        )
    system_observability = harness_record.get("native_system_prompt_observability") or {}
    if not system_observability.get("complete", True):
        add(
            "native_system_prompt_unavailable",
            "system prompt unavailable",
            severity="warning",
            detail=(
                "Claude Code's complete built-in system prompt was unavailable to "
                "the transcript and judge."
            ),
        )
    native_events = harness_record.get("native_loss_events") or []
    native_compactions = [
        event for event in native_events
        if isinstance(event, dict) and event.get("kind") == "context_compaction"
    ]
    if native_compactions and not compactions:
        add(
            "context_compacted",
            f"compacted ×{len(native_compactions)}",
            severity="warning",
            detail="The production scaffold compacted the agent's native context.",
        )
    pruned = [
        event for event in native_events
        if isinstance(event, dict) and event.get("kind") == "tool_output_pruned"
    ]
    if pruned:
        add(
            "scaffold_tool_output_pruned",
            f"tool output pruned ×{len(pruned)}",
            severity="warning",
            detail="The production scaffold shortened earlier native tool output.",
        )

    evidence = ((judgment.get("envelope") or {}).get("evidence") or {})
    for caveat in evidence.get("caveats") or []:
        if not isinstance(caveat, dict) or not caveat.get("code"):
            continue
        caveat_code = str(caveat["code"])
        flag_code, label = EVIDENCE_CAVEAT_FLAGS.get(
            caveat_code,
            (f"judge_evidence_{caveat_code}", caveat_code.replace("_", " ")),
        )
        if any(item["code"] == flag_code for item in flags):
            continue
        add(
            flag_code,
            label,
            severity="warning",
            detail=str(caveat.get("description") or caveat_code),
        )

    # Artifact content loss is stored directly on each snapshot rather than as a
    # separate evidence issue record.
    lossy_artifacts = [
        item for item in evidence.get("artifacts") or []
        if isinstance(item, dict) and (item.get("truncated") or item.get("read_error"))
    ]
    if lossy_artifacts:
        affected = ", ".join(
            f"{item.get('path') or 'unknown'}@{item.get('snapshot') or 'unknown'}"
            for item in lossy_artifacts
        )
        add(
            "judge_artifacts_incomplete",
            f"artifact incomplete ×{len(lossy_artifacts)}",
            severity="warning",
            detail=f"Truncated or unreadable judge artifact snapshots: {affected}.",
        )
    if evidence.get("builder_truncated_evidence"):
        add(
            "judge_evidence_truncated",
            "judge evidence truncated",
            severity="warning",
            detail="The judge builder recorded truncated evidence.",
        )

    # A pipeline sidecar may contain additional per-sample issues.  Preserve these
    # without turning a historical exit-code warning into an exclusion. Provider
    # anomalies are fully recomputed from the raw events above; an older sidecar's
    # superseded empty-response policy must not override that current classification.
    stored = audit.get("stored_integrity") or {}
    current_provider_issues = {
        f"target_provider_{event['kind']}"
        for event in blocking_provider_events(audit.get("target_provider_events"))
    }
    for code in stored.get("issues") or []:
        code = str(code)
        if code.startswith("target_provider_") and code not in current_provider_issues:
            continue
        if _issue_is_prefix_only(code, multi_agent=multi_agent_run):
            if code not in prefix_issues:
                prefix_issues.append(code)
            continue
        if code not in integrity:
            integrity.append(code)
        if not any(item["code"] == code for item in flags):
            add(code, code.replace("_", " "), severity="error",
                detail="Stored by the experiment pipeline.")
    for code in stored.get("prefix_issues") or []:
        code = str(code)
        if code not in prefix_issues:
            prefix_issues.append(code)
        if not any(item["code"] == code for item in flags):
            add(code, code.replace("_", " "), severity="warning",
                detail="Stored by the experiment pipeline.")

    retrospective = audit.get("retrospective_rejudge") or {}
    for source_flag in retrospective.get("source_flags") or []:
        if not isinstance(source_flag, dict):
            continue
        code = str(source_flag.get("code") or "source_flag")
        if any(item["code"] == code for item in flags):
            continue
        flags.append({
            **source_flag,
            "detail": "Source trajectory: " + str(source_flag.get("detail") or ""),
        })
    for code in retrospective.get("source_integrity_issues") or []:
        if str(code) not in integrity:
            integrity.append(str(code))

    audit["flags"] = flags
    status = mechanical_status_record(
        issues=integrity,
        prefix_issues=prefix_issues,
        show_benchmark_only=not conditioned,
        ended_reason=ended_reason,
        has_output=not audit.get("dead"),
        unfinished_action=unfinished_action,
    )
    audit.update(status)
    return audit


def sample_integrity_record(
    task: str,
    sample: Any,
    target_model: str | None,
    *,
    stored_record: dict | None = None,
) -> dict:
    """Classify one raw sample for both sidecars and continuation preflight."""

    real_env = (getattr(sample, "metadata", None) or {}).get("real_env") or {}
    protocol = real_env.get("protocol") or {}
    continuation = bool(real_env.get("continuation"))
    multi_agent = real_env.get("multi_agent") or {}
    conditioned = continuation or bool(multi_agent)
    multi_agent_run = bool(multi_agent)
    target_usage = (getattr(sample, "role_usage", None) or {}).get("target")
    output_tokens = getattr(target_usage, "output_tokens", 0) or 0
    has_output = sample_has_target_output(sample) or output_tokens > 0
    messages = getattr(sample, "messages", None) or []
    unfinished_action = has_unfinished_action(messages)
    score = (getattr(sample, "scores", None) or {}).get("environment_judge")
    score_metadata = (getattr(score, "metadata", None) or {}) if score else {}
    envelope = score_metadata.get("environment_judge") or {}
    issues: list[str] = []
    prefix_issues: list[str] = []

    if not protocol:
        issues.append("missing_protocol_record")
    if not real_env.get("artifacts"):
        issues.append("missing_final_artifact_record")
    if real_env.get("finalize_error"):
        issues.append("finalize_error")
    if multi_agent:
        exposure = multi_agent.get("exposure") or {}
        if exposure.get("fully_delivered") is not True:
            issues.append("activity_log_not_fully_delivered")
    if (
        (real_env.get("harness") or {}).get("mode") in {"production", "subscription"}
        and (real_env.get("native_resume_bundle") or {}).get("available") is not True
    ):
        prefix_issues.append("native_resume_bundle_unavailable")
    if any(gate.get("verdict") == "error" for gate in real_env.get("gates", [])) \
            or protocol.get("ended_reason") == "gate_error_end":
        issues.append("gate_error")
    if not has_output:
        issues.append("target_no_output")

    events = target_provider_events(sample, target_model)
    current_provider_issues = {
        f"target_provider_{event['kind']}"
        for event in blocking_provider_events(events)
    }
    issues.extend(current_provider_issues)
    if score is None:
        issues.append("missing_environment_judge_score")
    if not isinstance(envelope.get("result"), dict):
        issues.append("missing_structured_result")
    if envelope.get("post_validation") != "passed":
        issues.append("judge_post_validation")
    if _raw_environment_failed(envelope):
        issues.append("environment_failure")

    ended_reason = protocol.get("ended_reason")
    if ended_reason == "wall_clock_limit":
        destination = (
            prefix_issues
            if _issue_is_prefix_only(
                "wall_clock_limit", multi_agent=multi_agent_run
            )
            else issues
        )
        destination.append("wall_clock_limit")
    elif ended_reason not in (None, "protocol_end", "gate_error_end"):
        issues.append("ended_early")
    if unfinished_action:
        destination = (
            prefix_issues
            if _issue_is_prefix_only(
                "unfinished_action", multi_agent=multi_agent_run
            )
            else issues
        )
        destination.append("unfinished_action")

    # Old sidecars remain useful for failures that cannot be reconstructed from
    # the sample. Reinterpret the former native-resume exclusion as prefix-only.
    for code in (stored_record or {}).get("issues") or []:
        code = str(code)
        if code.startswith("target_provider_") and code not in current_provider_issues:
            continue
        destination = (
            prefix_issues
            if _issue_is_prefix_only(code, multi_agent=multi_agent_run)
            else issues
        )
        destination.append(code)
    prefix_issues.extend(
        str(code) for code in (stored_record or {}).get("prefix_issues") or []
    )

    status = mechanical_status_record(
        issues=issues,
        prefix_issues=prefix_issues,
        show_benchmark_only=not conditioned,
        ended_reason=ended_reason,
        has_output=has_output,
        unfinished_action=unfinished_action,
    )
    return {
        "task": str(task),
        "sample": str(sample.id),
        "epoch": sample.epoch,
        # Preserve the old field for pipeline callers while storing the clearer
        # benchmark/prefix decisions alongside it.
        "status": status["integrity_status"],
        "issues": status["integrity_issues"],
        "prefix_status": "eligible" if status["prefix_eligible"] else "ineligible",
        "target_provider_events": events,
        **status,
    }


def integrity_sidecar_records(logs: list[Any]) -> list[dict]:
    """Build the persisted per-sample integrity records used after an experiment."""

    records = []
    for log in logs or []:
        roles = getattr(log.eval, "model_roles", None) or {}
        target_role = roles.get("target")
        target_model = str(getattr(target_role, "model", None) or target_role or "")
        for sample in getattr(log, "samples", None) or []:
            records.append(sample_integrity_record(
                str(log.eval.task), sample, target_model
            ))
    return records


def mark_ineligible_prefix_source(audit: dict, source: dict) -> dict:
    """Invalidate a continuation whose source cannot mechanically be a prefix."""

    if not (audit.get("real_env") or {}).get("continuation"):
        return audit
    if source.get("prefix_eligible", True):
        return audit

    issue = "source_prefix_ineligible"
    issues = list(audit.get("integrity_issues") or [])
    if issue not in issues:
        issues.append(issue)
    prefix_issues = list(audit.get("prefix_issues") or [])
    source_id = source.get("id") or source.get("trajectory_id") or "unknown"
    detail = (
        f"Source trajectory #{source_id} has status "
        f"{source.get('mechanical_status') or 'prefix-ineligible'}."
    )
    flags = [
        flag for flag in (audit.get("flags") or [])
        if flag.get("code") != issue
    ]
    flags.append(_flag(issue, "prefix ineligible", severity="error", detail=detail))
    audit["flags"] = flags
    ended_reason = (
        ((audit.get("real_env") or {}).get("protocol") or {}).get("ended_reason")
        or audit.get("real_ended_reason")
    )
    status = mechanical_status_record(
        issues=issues,
        prefix_issues=prefix_issues,
        show_benchmark_only=False,
        ended_reason=ended_reason,
        has_output=not audit.get("dead"),
        unfinished_action=has_unfinished_action(audit.get("messages")),
        source_tags=source.get("status_tags") or [],
    )
    audit.update(status)
    return audit
