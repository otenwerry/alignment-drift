"""Canonical judgment promotion and publication cohort rules; no page rendering."""
from __future__ import annotations

from env_viewer_cache import trajectory_key
from environment_judge import prepare_judge_call
from real_integrity import mark_ineligible_prefix_source
from env_viewer_continuations import prefix_source_trajectory_id
from env_viewer_continuations import continuation_of
from old_runs import old_trajectory_key_reasons
from env_viewer_load import link_rejudge_sources
from env_viewer_load import promote_rejudge_judgments
from env_viewer_store import ViewerAuditStore
from project_paths import OLD_LOGS_ROOT
from project_paths import TRAJECTORY_IDS_FILE
from project_paths import VIEWER_CACHE_ROOT

REGISTRY_FILE = TRAJECTORY_IDS_FILE

CACHE_ROOT = VIEWER_CACHE_ROOT

CURRENT_HARNESS = "subscription"

CLAUDE_CODE_SCOPE = "claude_code"

OLD_SCOPE = "old"

OLD_STORE_PREFIX = "old/"

CURRENT_WINDOW_HARNESSES = (CURRENT_HARNESS, CLAUDE_CODE_SCOPE)

GENERIC_JUDGE_FAMILIES = ("ml_training_data_misuse", "p_hacking")

ARCHIVE_REASON_LABELS = {
    "simple_harness": "Simple-harness run",
    "archived_run_directory": "Superseded run directory",
    "excess_retry_attempt": "Excess retry attempt",
    "glm_context_overrun": "GLM context overrun",
    "stray_preliminary_baseline": "Stray preliminary baseline",
    "superseded_judge_method": "Superseded judge method",
    "held_for_deletion": "Held in logs_old/",
    "not_in_write_up": "Not in the write-up",
    "replaced_invalid_attempt": "Replaced invalid attempt",
}

def _claude_code_scaffold(audit: dict) -> bool:
    """Whether this trajectory's agent ran under the Claude Code scaffold."""

    native = audit.get("native_harness") or audit.get("production_harness") or {}
    stored = (audit.get("real_env") or {}).get("harness") or {}
    scaffold = native.get("scaffold") or stored.get("scaffold") or ""
    return str(scaffold).replace("-", "_") == CLAUDE_CODE_SCOPE

def _is_claude_code(audit: dict) -> bool:
    return (
        audit.get("viewer_scope") == CLAUDE_CODE_SCOPE
        or _claude_code_scaffold(audit)
    )

def _audit_harness(audit: dict) -> str:
    """Viewer window scope: harness, with Claude Code split out of subscription."""

    if _is_claude_code(audit):
        return CLAUDE_CODE_SCOPE
    stored = str(audit.get("harness") or "simple")
    return "subscription" if stored == "production" else stored

def _archive_reason_record(key: str) -> dict:
    return {"key": key, "label": ARCHIVE_REASON_LABELS.get(key, key.replace("_", " "))}

def _archive_reason(
    audit: dict,
    *,
    harness: str,
    archived_run_names: frozenset[str],
    archived_key_reasons: dict[str, str],
    archived_run_reasons: dict[str, str] | None = None,
) -> dict | None:
    """Why this trajectory renders under Old, or None when it is Current.

    The most specific reason wins: an explicitly curated trajectory key, then a run
    directory archived as a whole, then the harness rule.
    """

    key_reason = archived_key_reasons.get(trajectory_key(audit))
    if key_reason is not None:
        return _archive_reason_record(key_reason)
    mode = str(audit.get("mode") or "")
    if mode in archived_run_names:
        return _archive_reason_record(
            (archived_run_reasons or {}).get(mode, "archived_run_directory")
        )
    if harness not in CURRENT_WINDOW_HARNESSES:
        return _archive_reason_record("simple_harness")
    return None

def _is_prefix_generation_audit(audit: dict) -> bool:
    return (audit.get("real_env") or {}).get("family") in {
        "ml_prefix_only",
        "p_hacking_prefix_only",
    }

def _apply_continuation_source_statuses(audits: list[dict]) -> None:
    """Propagate prefix ineligibility through stored continuation chains."""

    by_id = {
        int(audit["id"]): audit
        for audit in audits
        if audit.get("id") is not None and not audit.get("retrospective_rejudge")
    }
    completed: set[int] = set()

    def apply(audit: dict, visiting: set[int]) -> None:
        audit_id = int(audit.get("id") or 0)
        if audit_id in completed or audit_id in visiting:
            return
        visiting = {*visiting, audit_id}
        source_id = audit.get("source_trajectory_id")
        if source_id is None:
            source_id = prefix_source_trajectory_id(audit)
        try:
            source = by_id.get(int(source_id)) if source_id is not None else None
        except (TypeError, ValueError):
            source = None
        if source is not None:
            apply(source, visiting)
            mark_ineligible_prefix_source(audit, source)
        completed.add(audit_id)

    for audit in audits:
        if continuation_of(audit):
            apply(audit, set())

async def _current_judge_methods() -> dict[str, str]:
    methods = {}
    for family in GENERIC_JUDGE_FAMILIES:
        prepared = await prepare_judge_call(
            family=family, stage="final", messages=[], artifacts=[]
        )
        methods[family] = prepared.method_sha256()
    return methods

def _is_current_judgment(audit: dict, methods: dict[str, str]) -> bool:
    judgment = audit.get("judgment") or {}
    if judgment.get("format") == "structured":
        envelope = judgment.get("envelope") or {}
        family = str(judgment.get("family") or envelope.get("family") or "")
    else:
        score_metadata = audit.get("score_metadata") or {}
        envelope = score_metadata.get("environment_judge") or {}
        if not isinstance(envelope, dict):
            return False
        family = str(envelope.get("family") or "")
    stored = envelope.get("judge_method_sha256")
    return bool(stored and stored == methods.get(family))

def _multi_agent_record(audit: dict) -> dict:
    real_env = audit.get("real_env") or {}
    record = real_env.get("multi_agent") if isinstance(real_env, dict) else None
    return record if isinstance(record, dict) else {}

def _link_and_promote_audits(
    audits: list[dict],
    audit_store: ViewerAuditStore,
    promoted_runs: set[str] | frozenset[str],
) -> int:
    """Perform cross-run enrichment without hydrating the complete corpus."""

    backed = [audit_store.is_backed(audit) for audit in audits]
    if not any(backed):
        link_rejudge_sources(audits)
        return promote_rejudge_judgments(
            audits, promoted_runs, archived_source_keys=old_trajectory_key_reasons()
        )
    if not all(backed):
        raise ValueError("viewer audits cannot mix stored and in-memory records")

    originals = {
        (
            audit.get("mode"),
            audit.get("task"),
            str(audit.get("stored_seed") or audit.get("seed")),
            audit.get("epoch"),
        ): audit
        for audit in audits
        if not audit.get("retrospective_rejudge")
    }
    for rejudge_summary in audits:
        source = rejudge_summary.get("retrospective_rejudge")
        if not isinstance(source, dict):
            continue
        key = (
            source.get("source_run"),
            source.get("source_task"),
            str(source.get("seed")),
            source.get("epoch"),
        )
        original_summary = originals.get(key)
        rejudge = audit_store.hydrate(rejudge_summary)
        pair = [rejudge]
        if original_summary is not None:
            pair.insert(0, audit_store.hydrate(original_summary))
        link_rejudge_sources(pair)
        audit_store.save_and_refresh(rejudge_summary, rejudge)

    selected = {str(name) for name in promoted_runs}
    if not selected:
        return 0
    summaries_by_id = {
        int(audit["id"]): audit
        for audit in audits
        if isinstance(audit.get("id"), int)
    }
    materialized: dict[str, tuple[dict, dict]] = {}

    def include(summary: dict) -> None:
        full = audit_store.hydrate(summary)
        store_key = str(full.get("_viewer_audit_store_key") or "")
        materialized[store_key] = (summary, full)

    for summary in audits:
        if str(summary.get("mode") or "") not in selected:
            continue
        include(summary)
        source_id = summary.get("source_trajectory_id")
        try:
            source_summary = summaries_by_id.get(int(source_id))
        except (TypeError, ValueError):
            source_summary = None
        if source_summary is not None:
            include(source_summary)
    promoted = promote_rejudge_judgments(
        [full for _summary, full in materialized.values()], selected,
        archived_source_keys=old_trajectory_key_reasons(),
    )
    for summary, full in materialized.values():
        audit_store.save_and_refresh(summary, full)
    return promoted
