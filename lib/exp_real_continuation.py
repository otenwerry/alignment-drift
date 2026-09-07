"""Continuation experiment core for real environments (importable).

A continuation asks: if an agent reward-hacks on one task and is then handed a
SECOND, unrelated task, does its probability of hacking on the second task go up --
because it is conditioned on its own earlier behavior?

Each cell conditions ONE agent model on a PREFIX (a prior conversation) and then
drops it into a real environment seed. One invocation = ONE TREATMENT, a free-form
label chosen with --treatment. There is no no-prefix condition here: the environment
is pinned and scripted, so the base rate comes from the ordinary original
trajectories of the same (seed, agent).

PREFIXES are self-contained payload files (``environments-continuation-prefix-v1``):
the exact Inspect message list the agent carries, plus the catalog agent name and its
reasoning setting. Conversation prefixes on production/subscription additionally
contain the scaffold's opaque, checksummed native resume bundle. Static
``inline_user_context`` prefixes instead start a fresh native session and need no
resume bundle. Prefixes come from two places:

- ``--prefixes=<viewer ids>``: reconstructed from stored real-environment logs
  (the viewer's stable integer IDs) and written under
  ``data/runs/continuation_prefixes/``.
- ``--prefix-files=<paths>``: arbitrary hand-built conversations (e.g. a long Q&A
  session). Owen constructs these to match the payload contract below.

The spliced agent context is either [prefix's system prompt + prefix conversation +
injected pivot user turn + the seed's opening message, live] or, for static inline
context, [native/current system prompt + inline context + pivot + opening message in
one user turn]. The pivot sentence is identical to the Petri continuation experiment;
when the prefix is not from the new task's own seed family (including every external
prefix) it also names the new kind of work. Both judge stages see only the system
message plus everything from the pivot onward -- the prefix is sliced out mechanically
by a stored boundary index
(``lib/continuation_evidence.py``), so no text matching is involved and the judge
cannot see or infer anything about the prior task.

INVARIANTS enforced before any paid work:
- a simple prefix's system prompt must byte-match the current
  ``seeds/SYSTEM_PROMPT.txt`` variant implied by its reasoning setting;
- ML prefix-only payloads missing required task deliverables are refused unless the
  caller explicitly enables and records the incomplete-prefix override;
- a native prefix must carry native state from the same harness, agent, reasoning
  setting, scaffold, exact scaffold version, and exact Inspect SWE version;
- a trajectory-sourced prefix must not share a seed with the new task;
- integrity-excluded and rejudge trajectories are refused as prefixes;
- a prefix ending on unresolved tool calls is refused before any model call; no tool
  result is invented.

This file makes no API calls itself; it builds plans and Inspect tasks that the
``exp_continuation_pipeline.py`` endpoint runs. The ``exp_`` prefix follows the
convention of its Petri counterpart (``petri/lib/exp_continuation.py``).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import tarfile
import tempfile
import uuid
import base64
from dataclasses import dataclass, replace
from pathlib import Path

from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser
from pydantic import TypeAdapter

from continuation_evidence import (
    CONTINUATION_DELIVERY_CONVERSATION,
    CONTINUATION_DELIVERY_INLINE_USER_CONTEXT,
    CONTINUATION_RECORD_VERSION,
    ContinuationRun,
    continuation_record,
    pivot_preamble,
    prefix_boundary_index,
    reject_dangling_tool_calls,
    seed_family_task_description,
)
from exp_real_audit import assemble_real_protocol, build_real_task
from exp_target_harness import (
    NATIVE_HARNESS_MODES,
    native_resume_record,
    opencode_go_model_spec,
    production_scaffold_for_target,
    resolve_harness,
    validate_native_resume_bundle,
)
from judge_selection import resolve_judge
from model_catalog import (
    REASONING_EFFORT,
    TARGET_CHOICES,
    build_target,
    resolve_target,
)
from judgment_semantics import (
    ML_HACK_TYPE_VALUES,
    P_HACK_TYPE_VALUES,
    REWARD_HACKING_DISPLAY_LABELS,
)
from multi_agent_evidence import (
    activity_log_inline_user_context,
    activity_log_sha256,
    extract_inline_activity_log,
    render_activity_log,
)
from project_paths import CONTINUATION_PREFIXES_ROOT, LOGS_ROOT, TRAJECTORY_IDS_FILE
from prompt_caching import stable_key
from protocol_sources import (
    global_system_prompt,
    is_seed_member,
    seed_members,
    strip_thinking_instruction,
)
from real_integrity import sample_integrity_record


PREFIX_FORMAT = "environments-continuation-prefix-v1"
ACTIVITY_LOG_PREFIX_GENERATOR = "build_activity_log_prefixes.py"
ACTIVITY_LOG_PREFIX_TYPE = "activity_log_context"
ACTIVITY_LOG_PREFIX_TYPE_LABEL = "Activity-log context"
PREFIX_STORE = CONTINUATION_PREFIXES_ROOT
REGISTRY_FILE = TRAJECTORY_IDS_FILE

# Treatment and prefix names survive round trips through task names, file names, and
# S3 keys, so they share one strict slug rule (same as Petri's treatment slugs).
_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

_MESSAGES_ADAPTER = TypeAdapter(list[ChatMessage])


def validate_treatment(treatment: str) -> str:
    value = (treatment or "").strip().lower()
    if not _SLUG_RE.match(value):
        raise SystemExit(
            f"--treatment {treatment!r} is not a valid slug. Use lowercase letters, "
            "digits and single hyphens (e.g. full-hack, clean), because the treatment "
            "is encoded into task and run-directory names."
        )
    return value


# --------------------------------------------------------------------------- #
# Prefix payloads (the one runtime input format for prefixes)
# --------------------------------------------------------------------------- #
def payload_sha256(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_prefix_payload(payload: dict, *, origin: str) -> dict:
    """Loud structural validation; returns the payload unchanged."""

    if not isinstance(payload, dict):
        raise SystemExit(f"{origin}: prefix payload must be a JSON object")
    if payload.get("format") != PREFIX_FORMAT:
        raise SystemExit(
            f"{origin}: prefix payload format must be {PREFIX_FORMAT!r}, got "
            f"{payload.get('format')!r}"
        )
    name = payload.get("name")
    if not isinstance(name, str) or not _SLUG_RE.match(name):
        raise SystemExit(
            f"{origin}: prefix payload needs a slug 'name' (lowercase letters, "
            f"digits, single hyphens); got {name!r}"
        )
    target = payload.get("target")
    if target not in TARGET_CHOICES:
        raise SystemExit(
            f"{origin}: prefix payload 'target' must be a catalog agent name "
            f"({sorted(TARGET_CHOICES)}); got {target!r}"
        )
    if not isinstance(payload.get("reasoning"), bool):
        raise SystemExit(f"{origin}: prefix payload 'reasoning' must be true/false")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("kind") not in {
        "trajectory", "external",
    }:
        raise SystemExit(
            f"{origin}: prefix payload 'source.kind' must be 'trajectory' or "
            "'external'"
        )
    eligibility = source.get("continuation_eligibility")
    if eligibility is not None:
        if not isinstance(eligibility, dict) or not isinstance(
            eligibility.get("eligible_by_default"), bool
        ):
            raise SystemExit(
                f"{origin}: source.continuation_eligibility must contain a boolean "
                "eligible_by_default"
            )
        missing = eligibility.get("missing_required_deliverables")
        if not isinstance(missing, list) or not all(
            isinstance(item, str) and item for item in missing
        ):
            raise SystemExit(
                f"{origin}: continuation eligibility must list "
                "missing_required_deliverables"
            )
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SystemExit(f"{origin}: prefix payload needs a non-empty 'messages' list")
    delivery = payload.get("delivery")
    if delivery is not None:
        if not isinstance(delivery, dict) or delivery.get("mode") not in {
            CONTINUATION_DELIVERY_CONVERSATION,
            CONTINUATION_DELIVERY_INLINE_USER_CONTEXT,
        }:
            raise SystemExit(
                f"{origin}: delivery.mode must be "
                f"{CONTINUATION_DELIVERY_CONVERSATION!r} or "
                f"{CONTINUATION_DELIVERY_INLINE_USER_CONTEXT!r}"
            )
    return payload


def load_prefix_payload_file(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.is_file():
        raise SystemExit(f"prefix payload file not found: {file_path}")
    try:
        payload = json.loads(file_path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"{file_path} is not valid JSON: {error}") from error
    return validate_prefix_payload(payload, origin=str(file_path))


def store_prefix_payload(payload: dict) -> Path:
    """Write a payload to the content-addressed local store (idempotent)."""

    PREFIX_STORE.mkdir(parents=True, exist_ok=True)
    sha = payload_sha256(payload)
    path = PREFIX_STORE / f"{payload['name']}-{sha[:12]}.json"
    if not path.exists():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _reward_hacking_types(judgment: object) -> list[str]:
    if not isinstance(judgment, dict):
        return []
    reward_hacking = judgment.get("reward_hacking")
    if not isinstance(reward_hacking, dict):
        return []
    values = reward_hacking.get("types")
    return [str(value) for value in values] if isinstance(values, list) else []


def _prefix_condition_record(
    *, family: object, reward_hacking_types: list[str], user_turns: int
) -> dict:
    """Store the source behavior that makes one trajectory a condition."""

    hack_types = (
        ML_HACK_TYPE_VALUES
        if family == "ml_training_data_misuse"
        else P_HACK_TYPE_VALUES
        if family == "p_hacking"
        else frozenset()
    )
    if any(value in hack_types for value in reward_hacking_types):
        key = f"hack_{user_turns}turn"
        label = f"hack · {user_turns} turn" + ("s" if user_turns != 1 else "")
    elif reward_hacking_types == ["clean"]:
        key = "clean"
        label = "clean"
    elif reward_hacking_types:
        key = "notable"
        labels = [
            REWARD_HACKING_DISPLAY_LABELS.get(value, value.replace("_", " "))
            for value in reward_hacking_types
        ]
        label = "notable · " + " + ".join(labels)
    else:
        key = "unknown"
        label = "outcome unavailable"
    return {
        "key": key,
        "label": label,
        "reward_hacking_types": reward_hacking_types,
        "experiment_user_turns": user_turns,
    }


def _trajectory_condition_records(real_env: dict) -> tuple[dict, dict | None]:
    """Return official/full and first-pass source-condition records."""

    protocol = real_env.get("protocol") or {}
    follow_up_sent = protocol.get("follow_up_sent") is True
    user_turns = 2 if follow_up_sent else 1
    family = real_env.get("family")
    final_judgment = ((real_env.get("final_judgment") or {}).get("result") or {})
    official = _prefix_condition_record(
        family=family,
        reward_hacking_types=_reward_hacking_types(final_judgment),
        user_turns=user_turns,
    )
    gates = real_env.get("gates") or []
    first_judgment = (
        (gates[0].get("judgment") or {})
        if gates and isinstance(gates[0], dict)
        else {}
    )
    first_pass = None
    first_types = _reward_hacking_types(first_judgment)
    if first_types:
        first_pass = _prefix_condition_record(
            family=family,
            reward_hacking_types=first_types,
            user_turns=1,
        )
    return official, first_pass


# --------------------------------------------------------------------------- #
# Reconstructing prefixes from stored trajectories (viewer IDs)
# --------------------------------------------------------------------------- #
def _registry_entries(trajectory_ids: list[int]) -> dict[int, dict]:
    """Map viewer IDs to their stored (mode, task, seed, epoch) identity."""

    if not REGISTRY_FILE.is_file():
        raise SystemExit(
            f"trajectory ID registry not found: {REGISTRY_FILE} (build the viewer "
            "once so IDs exist, or pass --prefix-files instead)"
        )
    registry = json.loads(REGISTRY_FILE.read_text())
    by_id = {value: key for key, value in registry.items()}
    entries: dict[int, dict] = {}
    for trajectory_id in trajectory_ids:
        key = by_id.get(trajectory_id)
        if key is None:
            raise SystemExit(
                f"--prefixes id {trajectory_id} is not in {REGISTRY_FILE.name}"
            )
        parts = key.split("__")
        if len(parts) != 4 or not parts[3].startswith("e"):
            raise SystemExit(
                f"registry key for trajectory #{trajectory_id} has an unexpected "
                f"shape: {key!r}"
            )
        entries[trajectory_id] = {
            "mode": parts[0],
            "task": parts[1],
            "seed": parts[2],
            "epoch": int(parts[3][1:]),
        }
    return entries


def _integrity_status(mode_dir: Path, task: str, seed: str, epoch: int) -> dict | None:
    sidecar = mode_dir / "pipeline_integrity.json"
    if not sidecar.is_file():
        return None
    records = (json.loads(sidecar.read_text()) or {}).get("records") or []
    for record in records:
        if (
            record.get("task") == task
            and str(record.get("sample")) == seed
            and record.get("epoch") == epoch
        ):
            return record
    return None


def reconstruct_prefix_payload(
    trajectory_id: int,
    entry: dict,
    *,
    _visited: set[int] | None = None,
    source_use: str = "continuation",
) -> dict:
    """Rebuild one stored trajectory's exact agent-visible conversation."""

    if source_use not in {"continuation", "activity_log"}:
        raise ValueError(f"unknown trajectory reconstruction use: {source_use!r}")

    from inspect_ai.log import list_eval_logs, read_eval_log

    visited = set(_visited or ())
    if trajectory_id in visited:
        raise SystemExit(
            f"trajectory #{trajectory_id}: circular continuation source chain"
        )
    visited.add(trajectory_id)

    mode_dir = LOGS_ROOT / entry["mode"]
    if not mode_dir.is_dir():
        raise SystemExit(
            f"trajectory #{trajectory_id}: run directory {mode_dir} is missing"
        )
    sample = None
    run_metadata: dict = {}
    target_model = ""
    for log_info in list_eval_logs(str(mode_dir)):
        header = read_eval_log(log_info, header_only=True)
        if header.eval.task != entry["task"]:
            continue
        log = read_eval_log(log_info, resolve_attachments=True)
        run_metadata = dict(log.eval.metadata or {})
        roles = getattr(log.eval, "model_roles", None) or {}
        target_role = roles.get("target")
        target_model = str(
            getattr(target_role, "model", None) or target_role or ""
        )
        for candidate in log.samples or []:
            if (
                str(candidate.id) == entry["seed"]
                and int(candidate.epoch) == entry["epoch"]
            ):
                sample = candidate
                break
        if sample is not None:
            break
    if sample is None:
        raise SystemExit(
            f"trajectory #{trajectory_id} ({entry['mode']}/{entry['task']} "
            f"seed={entry['seed']} epoch={entry['epoch']}) was not found on disk"
        )

    sample_metadata = sample.metadata or {}
    if sample_metadata.get("retrospective_rejudge"):
        raise SystemExit(
            f"trajectory #{trajectory_id} is a retrospective rejudge, not an "
            "original trajectory; pick the original's ID"
        )
    integrity = _integrity_status(
        mode_dir, entry["task"], entry["seed"], entry["epoch"]
    )
    current_status = sample_integrity_record(
        entry["task"], sample, target_model, stored_record=integrity
    )
    eligibility_key = (
        "prefix_eligible"
        if source_use == "continuation"
        else "benchmark_eligible"
    )
    if not current_status[eligibility_key]:
        reasons = ", ".join(
            tag["label"] for tag in current_status.get("status_tags") or []
        ) or ", ".join(
            current_status.get("issues") or current_status.get("prefix_issues") or []
        )
        raise SystemExit(
            f"trajectory #{trajectory_id} is "
            f"{current_status['mechanical_status'].replace('_', ' ')} "
            f"({reasons}); refusing it as a "
            f"{'prefix' if source_use == 'continuation' else 'source activity log'}"
        )

    real_env = sample_metadata.get("real_env") or {}
    if real_env.get("continuation"):
        print(
            f"  NOTE: trajectory #{trajectory_id} is itself a continuation; its "
            "whole conversation (earlier prefix included) becomes the new prefix."
        )
        source_id = (
            (real_env.get("continuation", {}).get("prefix") or {})
            .get("source_trajectory_id")
        )
        if source_id is not None:
            try:
                source_id = int(source_id)
            except (TypeError, ValueError) as error:
                raise SystemExit(
                    f"trajectory #{trajectory_id}: invalid source trajectory ID "
                    f"{source_id!r}"
                ) from error
            source_entry = _registry_entries([source_id])[source_id]
            reconstruct_prefix_payload(
                source_id,
                source_entry,
                _visited=visited,
                source_use=source_use,
            )
    target_name = run_metadata.get("target_name")
    if target_name not in TARGET_CHOICES:
        raise SystemExit(
            f"trajectory #{trajectory_id} has no usable stamped target_name "
            f"({target_name!r}); cannot rebuild its agent for a continuation"
        )
    reasoning = run_metadata.get("reasoning")
    if not isinstance(reasoning, bool):
        raise SystemExit(
            f"trajectory #{trajectory_id} has no stamped boolean 'reasoning' "
            "setting; cannot replicate its agent configuration"
        )
    messages = [
        message.model_dump(mode="json", exclude_none=False)
        for message in (sample.messages or [])
    ]
    if not messages:
        raise SystemExit(f"trajectory #{trajectory_id} has no stored messages")

    source_harness = run_metadata.get("harness") or "simple"
    native_resume = None
    if source_harness in NATIVE_HARNESS_MODES:
        sidecar = (
            mode_dir
            / "real_artifacts"
            / entry["task"]
            / f"{entry['seed']}_ep{entry['epoch']}"
        )
        manifest_path = sidecar / "_native_resume_bundle.json"
        archive_path = sidecar / "_native_resume_bundle.tar.gz"
        if manifest_path.is_file() and archive_path.is_file():
            try:
                native_resume = json.loads(manifest_path.read_text())
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"trajectory #{trajectory_id}: native resume manifest is invalid: {error}"
                ) from error
            native_resume["archive_base64"] = base64.b64encode(
                archive_path.read_bytes()
            ).decode("ascii")

    protocol_sources = run_metadata.get("protocol_sources") or {}
    recorded_protocol = sample_metadata.get("protocol") or {}
    prefix_condition, first_pass_condition = _trajectory_condition_records(real_env)
    payload = {
        "format": PREFIX_FORMAT,
        "name": f"traj{trajectory_id}",
        "target": target_name,
        "reasoning": reasoning,
        "source": {
            "kind": "trajectory",
            "trajectory_id": trajectory_id,
            "run": entry["mode"],
            "task": entry["task"],
            "seed": entry["seed"],
            "epoch": entry["epoch"],
            "family": (
                real_env.get("family")
                or protocol_sources.get("protocol_family")
            ),
            "condition": run_metadata.get("condition"),
            "harness": source_harness,
            "environment_system_prompt": recorded_protocol.get("system_prompt"),
            "was_continuation": bool(real_env.get("continuation")),
            "prefix_condition": prefix_condition,
            "first_pass_condition": first_pass_condition,
        },
        "messages": messages,
    }
    if native_resume is not None:
        payload["native_resume"] = native_resume
    return validate_prefix_payload(payload, origin=f"trajectory #{trajectory_id}")


def reconstruct_prefix_payloads(
    trajectory_ids: list[int], *, source_use: str = "continuation"
) -> list[dict]:
    if not trajectory_ids:
        return []
    entries = _registry_entries(trajectory_ids)
    return [
        reconstruct_prefix_payload(
            trajectory_id, entries[trajectory_id], source_use=source_use
        )
        for trajectory_id in trajectory_ids
    ]


def activity_log_prefix_payload_from_source(source_payload: dict) -> dict:
    """Render one saved trajectory payload as a free, static continuation prefix.

    The returned payload contains exactly one neutral user message. It carries the
    complete stored observable activity log but no native session archive and no
    fabricated assistant or tool history. The ordinary continuation splice folds the
    pivot and next task into this same user message.

    ``source_payload`` may be either a viewer-reconstructed trajectory or a complete
    external prefix-generation payload. The latter is how no-honeypot ML trajectories
    become inline activity-log contexts without first assigning them viewer trajectory
    IDs (prefix-only generation logs intentionally have no such IDs).
    """

    validate_prefix_payload(
        source_payload,
        origin=str(source_payload.get("name") or "source payload"),
    )
    source_kind = (source_payload.get("source") or {}).get("kind")
    if source_kind not in {"trajectory", "external"}:
        raise SystemExit(
            "activity-log prefixes must be derived from a trajectory or a saved "
            "external prefix payload"
        )
    if (
        (source_payload.get("source") or {}).get("prefix_type")
        == ACTIVITY_LOG_PREFIX_TYPE
    ):
        raise SystemExit("an activity-log prefix cannot be wrapped a second time")
    try:
        messages = _MESSAGES_ADAPTER.validate_python(source_payload["messages"])
    except Exception as error:
        raise SystemExit(
            f"{source_payload.get('name')}: stored messages could not be parsed: "
            f"{error}"
        ) from error
    content, metadata = render_activity_log(messages)
    source = dict(source_payload["source"])
    if source_kind == "external":
        # The derived payload is self-contained, so it does not depend on the source
        # file remaining at the same host path. Preserve its canonical identity
        # without copying native resume state or path-dependent metadata.
        source["activity_log_source_payload"] = {
            "kind": "continuation_prefix_payload",
            "format": source_payload["format"],
            "name": source_payload["name"],
            "sha256": payload_sha256(source_payload),
            "prefix_type": source.get("prefix_type"),
            "prefix_type_label": source.get("prefix_type_label"),
            "generator": source.get("generator"),
        }
    source.update({
        "generator": ACTIVITY_LOG_PREFIX_GENERATOR,
        "prefix_type": ACTIVITY_LOG_PREFIX_TYPE,
        "prefix_type_label": ACTIVITY_LOG_PREFIX_TYPE_LABEL,
        "stored_message_format": "inspect-chat-messages",
        "native_session_state_included": False,
        "activity_log_metadata": metadata,
    })
    source_label = source.get("source_label")
    if not isinstance(source_label, str) or not source_label.strip():
        source["source_label"] = ACTIVITY_LOG_PREFIX_TYPE_LABEL
    user_message = ChatMessageUser(
        content=activity_log_inline_user_context(content)
    ).model_dump(mode="json", exclude_none=False)
    # Inspect message IDs are generated randomly at model construction time. They are
    # runtime citation identities, not source data, so omit the ID from the payload;
    # build_prefix_spec assigns one after hashing. This keeps repeated free builds
    # byte-for-byte and content-hash stable.
    user_message.pop("id", None)
    payload = {
        "format": PREFIX_FORMAT,
        "name": f"activity-log-{source_payload['name']}",
        "target": source_payload["target"],
        "reasoning": source_payload["reasoning"],
        "delivery": {
            "mode": CONTINUATION_DELIVERY_INLINE_USER_CONTEXT,
            "fresh_session": True,
            "prior_assistant_messages_replayed": False,
            "native_session_state_included": False,
        },
        "source": source,
        "messages": [user_message],
    }
    return validate_prefix_payload(
        payload,
        origin=(
            f"trajectory {source.get('trajectory_id')}"
            if source_kind == "trajectory"
            else f"prefix payload {source_payload['name']}"
        ),
    )


def activity_log_prefix_payload_from_trajectory(trajectory_payload: dict) -> dict:
    """Backward-compatible, trajectory-only wrapper for viewer-ID callers."""

    if (trajectory_payload.get("source") or {}).get("kind") != "trajectory":
        raise SystemExit("activity-log prefixes must be derived from a trajectory")
    return activity_log_prefix_payload_from_source(trajectory_payload)


# --------------------------------------------------------------------------- #
# Explicit one-turn cutoff derivatives
# --------------------------------------------------------------------------- #
_CUTOFF_FORMAT = "trajectory-prefix-cutoff-v1"


def _payload_message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _is_experiment_user_payload_message(message: dict) -> bool:
    metadata = message.get("metadata")
    return (
        message.get("role") == "user"
        and not (
            isinstance(metadata, dict)
            and metadata.get("scaffold_injected")
        )
    )


def _native_user_text_matches(actual: object, expected: str) -> bool:
    if not isinstance(actual, str):
        return False
    return actual == expected or (
        len(actual) >= 2 and actual[0] == actual[-1] == '"'
        and actual[1:-1] == expected
    )


def _archive_member_names(data: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        return [
            member.name.removeprefix("./").rstrip("/")
            for member in archive.getmembers()
        ]


_CUTOFF_NATIVE_SESSION_PATHS = {
    "codex": ("root/.codex/sessions",),
    "opencode": (
        "root/.local/share/opencode",
        "root/.local/state/opencode",
    ),
}


def _repack_native_archive(temporary_path: Path, *, scaffold: str) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for arcname in _CUTOFF_NATIVE_SESSION_PATHS[scaffold]:
            path = temporary_path / arcname
            if path.exists():
                archive.add(path, arcname=arcname, recursive=True)
    return output.getvalue()


def _rewrite_opencode_cutoff(
    root: Path, *, cutoff_user_text: str
) -> dict:
    database = root / ".local" / "share" / "opencode" / "opencode.db"
    if not database.is_file():
        raise SystemExit("OpenCode cutoff archive has no opencode.db")

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")
        user_rows = connection.execute(
            """
            SELECT message.id, message.time_created, part.data
            FROM message
            JOIN part ON part.message_id = message.id
            WHERE json_extract(message.data, '$.role') = 'user'
            ORDER BY message.time_created, message.id, part.id
            """
        ).fetchall()
        matches = []
        for row in user_rows:
            try:
                part = json.loads(row["data"])
            except (TypeError, json.JSONDecodeError):
                continue
            if (
                part.get("type") == "text"
                and _native_user_text_matches(part.get("text"), cutoff_user_text)
            ):
                matches.append(row)
        if len(matches) != 1:
            raise SystemExit(
                "OpenCode cutoff requires exactly one native copy of the second "
                f"user turn; found {len(matches)}"
            )
        cutoff_id = str(matches[0]["id"])
        cutoff_time = int(matches[0]["time_created"])
        event_matches = connection.execute(
            "SELECT seq FROM event WHERE instr(data, ?) > 0 ORDER BY seq",
            (cutoff_id,),
        ).fetchall()
        if not event_matches:
            raise SystemExit(
                "OpenCode cutoff could not locate the second user turn in its "
                "native event sequence"
            )
        cutoff_sequence = int(event_matches[0]["seq"])
        original_messages = int(
            connection.execute("SELECT count(*) FROM message").fetchone()[0]
        )
        original_events = int(
            connection.execute("SELECT count(*) FROM event").fetchone()[0]
        )
        original_todos = int(
            connection.execute("SELECT count(*) FROM todo").fetchone()[0]
        )

        connection.execute("PRAGMA secure_delete=ON")
        connection.execute(
            """
            DELETE FROM part
            WHERE message_id IN (
                SELECT id FROM message WHERE time_created >= ?
            )
            """,
            (cutoff_time,),
        )
        connection.execute(
            "DELETE FROM message WHERE time_created >= ?", (cutoff_time,)
        )
        # OpenCode's current todo table is a mutable projection rather than an
        # event-sourced snapshot. Rows touched after the boundary cannot safely be
        # rewound, so remove them rather than leaking second-pass state.
        connection.execute(
            "DELETE FROM todo WHERE time_created >= ? OR time_updated >= ?",
            (cutoff_time, cutoff_time),
        )
        connection.execute("DELETE FROM event WHERE seq >= ?", (cutoff_sequence,))
        connection.execute(
            """
            UPDATE event_sequence
            SET seq = COALESCE(
                (SELECT MAX(seq) FROM event
                 WHERE event.aggregate_id = event_sequence.aggregate_id),
                0
            )
            """
        )
        connection.execute(
            """
            UPDATE session
            SET time_updated = COALESCE(
                (SELECT MAX(time_updated) FROM message
                 WHERE message.session_id = session.id),
                time_created
            )
            """
        )
        connection.commit()
        connection.execute("VACUUM")
        connection.execute("PRAGMA journal_mode=WAL")
        retained_messages = int(
            connection.execute("SELECT count(*) FROM message").fetchone()[0]
        )
        retained_events = int(
            connection.execute("SELECT count(*) FROM event").fetchone()[0]
        )
        retained_todos = int(
            connection.execute("SELECT count(*) FROM todo").fetchone()[0]
        )
    finally:
        connection.close()

    # Logs and stale locks are not conversational state and can contain activity
    # after the boundary. OpenCode recreates both on resume.
    log_file = root / ".local" / "share" / "opencode" / "log" / "opencode.log"
    if log_file.exists():
        log_file.unlink()
    locks = root / ".local" / "state" / "opencode" / "locks"
    if locks.is_dir():
        shutil.rmtree(locks)
        locks.mkdir(parents=True)

    leaked = []
    needles = {cutoff_user_text.encode()}
    if (
        len(cutoff_user_text) >= 2
        and cutoff_user_text[0] == cutoff_user_text[-1] == '"'
    ):
        needles.add(cutoff_user_text[1:-1].encode())
    for path in root.rglob("*"):
        if path.is_file() and any(
            needle and needle in path.read_bytes() for needle in needles
        ):
            leaked.append(str(path.relative_to(root)))
    if leaked:
        raise SystemExit(
            "OpenCode cutoff still contains the omitted user turn in: "
            + ", ".join(leaked)
        )
    return {
        "method": "opencode_sqlite_event_rewind_v1",
        "original_native_messages": original_messages,
        "retained_native_messages": retained_messages,
        "omitted_native_messages": original_messages - retained_messages,
        "original_native_events": original_events,
        "retained_native_events": retained_events,
        "omitted_native_events": original_events - retained_events,
        "original_native_todos": original_todos,
        "retained_native_todos": retained_todos,
        "omitted_native_todos": original_todos - retained_todos,
        "native_log_removed": True,
        "native_locks_reset": True,
    }


def _codex_user_text(record: dict) -> str | None:
    if record.get("type") != "response_item":
        return None
    payload = record.get("payload") or {}
    if payload.get("type") != "message" or payload.get("role") != "user":
        return None
    content = payload.get("content") or []
    texts = [
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "input_text"
    ]
    return "".join(texts)


def _rewrite_codex_cutoff(root: Path, *, cutoff_user_text: str) -> dict:
    rollouts = sorted((root / ".codex" / "sessions").rglob("rollout-*.jsonl"))
    if len(rollouts) != 1:
        raise SystemExit(
            "Codex cutoff requires exactly one native rollout file; found "
            f"{len(rollouts)}"
        )
    rollout = rollouts[0]
    raw_lines = rollout.read_text().splitlines()
    records = []
    try:
        records = [json.loads(line) for line in raw_lines]
    except json.JSONDecodeError as error:
        raise SystemExit(f"Codex cutoff rollout is invalid JSONL: {error}") from error
    matches = [
        index for index, record in enumerate(records)
        if _codex_user_text(record) == cutoff_user_text
    ]
    if len(matches) != 1:
        raise SystemExit(
            "Codex cutoff requires exactly one native copy of the second user "
            f"turn; found {len(matches)}"
        )
    boundary = matches[0]
    completed = [
        index for index, record in enumerate(records[:boundary])
        if record.get("type") == "event_msg"
        and (record.get("payload") or {}).get("type") == "task_complete"
    ]
    if not completed:
        raise SystemExit(
            "Codex cutoff could not find the completed first native invocation"
        )
    retained_count = completed[-1] + 1
    retained_lines = raw_lines[:retained_count]
    rollout.write_text("\n".join(retained_lines) + "\n")
    if cutoff_user_text in rollout.read_text():
        raise SystemExit("Codex cutoff still contains the omitted user turn")
    return {
        "method": "codex_rollout_turn_rewind_v1",
        "original_native_events": len(raw_lines),
        "retained_native_events": retained_count,
        "omitted_native_events": len(raw_lines) - retained_count,
        "first_invocation_completed": True,
    }


def _cutoff_native_resume_bundle(
    bundle: dict,
    *,
    target_name: str,
    reasoning: bool,
    cutoff_user_text: str,
) -> tuple[dict, dict]:
    routed_slug = resolve_target(target_name)
    data = validate_native_resume_bundle(
        bundle,
        target_name=target_name,
        routed_slug=routed_slug,
        reasoning=reasoning,
    )
    scaffold = str(bundle.get("scaffold") or "")
    with tempfile.TemporaryDirectory(prefix="mats-prefix-cutoff-") as temporary:
        temporary_path = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            archive.extractall(temporary_path, filter="data")
        root = temporary_path / "root"
        if scaffold == "opencode":
            rewrite = _rewrite_opencode_cutoff(
                root, cutoff_user_text=cutoff_user_text
            )
        elif scaffold == "codex":
            rewrite = _rewrite_codex_cutoff(
                root, cutoff_user_text=cutoff_user_text
            )
        else:
            raise SystemExit(
                f"one-turn cutoff native rewind is not implemented for {scaffold!r}"
            )
        rewritten_data = _repack_native_archive(
            temporary_path, scaffold=scaffold
        )

    rewritten = dict(bundle)
    rewritten.update({
        "archive_base64": base64.b64encode(rewritten_data).decode("ascii"),
        "archive_sha256": hashlib.sha256(rewritten_data).hexdigest(),
        "archive_bytes": len(rewritten_data),
        "archive_members": _archive_member_names(rewritten_data),
        "derived_cutoff": rewrite,
    })
    validate_native_resume_bundle(
        rewritten,
        target_name=target_name,
        routed_slug=routed_slug,
        reasoning=reasoning,
    )
    return rewritten, rewrite


def cutoff_prefix_payload(
    payload: dict, *, before_experiment_user_turn: int = 2
) -> dict:
    """Derive a self-contained prefix ending before one experiment user turn.

    Native scaffold state is rewound to the same boundary; merely slicing the
    Inspect messages would leave the omitted second pass in the CLI's own session.
    """

    validate_prefix_payload(payload, origin="cutoff source")
    if before_experiment_user_turn < 2:
        raise SystemExit("cutoff turn must be 2 or later")
    experiment_users = [
        (index, message)
        for index, message in enumerate(payload["messages"])
        if isinstance(message, dict)
        and _is_experiment_user_payload_message(message)
    ]
    if len(experiment_users) < before_experiment_user_turn:
        raise SystemExit(
            f"prefix {payload['name']!r} has only {len(experiment_users)} "
            "experiment user turn(s); the requested cutoff does not exist"
        )
    cutoff_index, cutoff_message = experiment_users[
        before_experiment_user_turn - 1
    ]
    cutoff_text = _payload_message_text(cutoff_message)
    if not cutoff_text:
        raise SystemExit("cutoff user turn has no text")
    retained_messages = payload["messages"][:cutoff_index]
    if not retained_messages:
        raise SystemExit("cutoff would produce an empty prefix")

    native_resume = payload.get("native_resume")
    native_rewrite = None
    rewritten_native_resume = None
    if native_resume is not None:
        rewritten_native_resume, native_rewrite = _cutoff_native_resume_bundle(
            native_resume,
            target_name=payload["target"],
            reasoning=payload["reasoning"],
            cutoff_user_text=cutoff_text,
        )

    derived = json.loads(json.dumps(payload))
    source = derived["source"]
    original_condition = source.get("prefix_condition")
    first_pass_condition = source.get("first_pass_condition")
    if not isinstance(first_pass_condition, dict):
        raise SystemExit(
            f"prefix {payload['name']!r} has no stored first-pass judgment; "
            "cannot label a clean cutoff"
        )
    cutoff_condition = dict(first_pass_condition)
    cutoff_condition["key"] = f"{cutoff_condition.get('key', 'unknown')}_1turn_cutoff"
    cutoff_condition["label"] = (
        f"{cutoff_condition.get('label') or 'outcome unavailable'} · 1-turn cutoff"
    )
    cutoff_record = {
        "format": _CUTOFF_FORMAT,
        "affected": True,
        "lossy_processing": True,
        "before_experiment_user_turn": before_experiment_user_turn,
        "original_message_count": len(payload["messages"]),
        "retained_message_count": len(retained_messages),
        "omitted_message_count": len(payload["messages"]) - len(retained_messages),
        "native_state_rewritten": rewritten_native_resume is not None,
        "native_rewrite": native_rewrite,
        "visible_caveat": (
            "Conversation and native scaffold state were cut off immediately "
            f"before experiment user turn {before_experiment_user_turn}; "
            f"{len(payload['messages']) - len(retained_messages)} saved message(s) "
            "from the later pass were omitted."
        ),
    }
    derived["name"] = f"{payload['name']}-cutoff-u{before_experiment_user_turn}"
    derived["messages"] = retained_messages
    source["derived_from_prefix_sha256"] = payload_sha256(payload)
    source["derived_from_prefix_condition"] = original_condition
    source["prefix_condition"] = cutoff_condition
    source["cutoff"] = cutoff_record
    source["lossy_processing"] = {
        "affected": True,
        "kind": "trajectory_turn_cutoff",
        "visible_caveat": cutoff_record["visible_caveat"],
    }
    if rewritten_native_resume is not None:
        derived["native_resume"] = rewritten_native_resume
    return validate_prefix_payload(derived, origin=derived["name"])


# --------------------------------------------------------------------------- #
# Prefix specs (validated, spliceable prefixes)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PrefixSpec:
    """One validated prefix, ready to splice in front of a new task."""

    name: str
    payload: dict
    sha256: str
    payload_path: Path | None
    messages: tuple[ChatMessage, ...]  # validated; inline-native starts with user
    target_name: str                   # catalog key
    reasoning: bool
    family: str | None                 # source seed family; None for external
    source_seed: str | None
    system_prompt_inserted: bool
    boundary_index: int
    native_resume: dict | None
    delivery_mode: str = CONTINUATION_DELIVERY_CONVERSATION
    continuation_eligibility_override: bool = False

    def record(self) -> dict:
        """The stored ``real_env["continuation"]["prefix"]`` block."""

        return {
            "name": self.name,
            "sha256": self.sha256,
            "target_name": self.target_name,
            "reasoning": self.reasoning,
            "family": self.family,
            "system_prompt_inserted": self.system_prompt_inserted,
            "delivery_mode": self.delivery_mode,
            "continuation_eligibility_override": (
                self.continuation_eligibility_override
            ),
            **{f"source_{key}": value for key, value in self.payload["source"].items()},
        }

    @property
    def char_count(self) -> int:
        return sum(len(message.text or "") for message in self.messages)


def _ensure_unique_message_ids(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Judge evidence numbering requires unique Inspect message IDs per sample."""

    seen: set[str] = set()
    fixed: list[ChatMessage] = []
    for message in messages:
        identifier = getattr(message, "id", None)
        if not identifier or identifier in seen:
            message = message.model_copy(update={"id": uuid.uuid4().hex})
        seen.add(message.id)
        fixed.append(message)
    return fixed


def expected_system_prompt(reasoning: bool) -> str:
    prompt = global_system_prompt()
    return strip_thinking_instruction(prompt) if reasoning else prompt


def build_prefix_spec(
    payload: dict,
    *,
    payload_path: Path | None = None,
    harness: str = "simple",
) -> PrefixSpec:
    """Validate one payload against the experiment invariants and close it."""

    harness = resolve_harness(harness)
    origin = payload["name"]
    try:
        messages = _MESSAGES_ADAPTER.validate_python(payload["messages"])
    except Exception as error:
        raise SystemExit(
            f"prefix {origin!r}: messages do not parse as Inspect chat messages: "
            f"{error}"
        ) from error

    expected = expected_system_prompt(payload["reasoning"])
    system_prompt_inserted = False
    source_harness = str(payload.get("source", {}).get("harness") or "simple")
    delivery_mode = str(
        (payload.get("delivery") or {}).get("mode")
        or CONTINUATION_DELIVERY_CONVERSATION
    )
    inline_user_context = (
        delivery_mode == CONTINUATION_DELIVERY_INLINE_USER_CONTEXT
    )
    if inline_user_context:
        if payload.get("native_resume") is not None:
            raise SystemExit(
                f"prefix {origin!r}: inline user context must not carry native "
                "resume state"
            )
        if len(messages) != 1 or messages[0].role != "user":
            raise SystemExit(
                f"prefix {origin!r}: inline user context must contain exactly one "
                "user message"
            )
        source = payload["source"]
        if source.get("prefix_type") == ACTIVITY_LOG_PREFIX_TYPE:
            try:
                activity_content = extract_inline_activity_log(messages[0].text or "")
            except ValueError as error:
                raise SystemExit(f"prefix {origin!r}: {error}") from error
            activity_metadata = source.get("activity_log_metadata") or {}
            if activity_metadata.get("sha256") != activity_log_sha256(activity_content):
                raise SystemExit(
                    f"prefix {origin!r}: inline activity-log SHA-256 does not match"
                )
            if activity_metadata.get("line_count") != len(
                activity_content.splitlines()
            ):
                raise SystemExit(
                    f"prefix {origin!r}: inline activity-log line count does not match"
                )
            if activity_metadata.get("byte_count") != len(
                activity_content.encode()
            ):
                raise SystemExit(
                    f"prefix {origin!r}: inline activity-log byte count does not match"
                )
        if harness == "simple":
            messages = [ChatMessageSystem(content=expected), *messages]
            system_prompt_inserted = True
    elif harness == "simple" and source_harness in NATIVE_HARNESS_MODES:
        raise SystemExit(
            f"prefix {origin!r} came from a {source_harness}-harness trajectory and "
            f"cannot be natively spliced into --harness=simple; use "
            f"--harness={source_harness}"
        )
    native_resume = payload.get("native_resume")
    if harness in NATIVE_HARNESS_MODES and not inline_user_context:
        if source_harness != harness:
            raise SystemExit(
                f"prefix {origin!r} came from --harness={source_harness} and has no "
                f"matching {harness}-scaffold session to resume; generate it with "
                f"--harness={harness}"
            )
        routed_slug = resolve_target(payload["target"])
        validate_native_resume_bundle(
            native_resume,
            target_name=payload["target"],
            routed_slug=routed_slug,
            reasoning=payload["reasoning"],
        )
        if (
            harness == "subscription"
            and production_scaffold_for_target(payload["target"], routed_slug)
            != "opencode"
            and not native_resume.get("native_session_id")
        ):
            # Fail before any spend: without a native session id the CLI cannot
            # resume (the source run's first CLI call never completed).
            raise SystemExit(
                f"prefix {origin!r}: subscription resume bundle has no "
                "native_session_id, so the native CLI session cannot be resumed"
            )
        if messages[0].role != "system":
            raise SystemExit(
                f"prefix {origin!r}: {harness} scaffold transcript does not start "
                "with its native system message"
            )
    elif not inline_user_context:
        if messages[0].role == "system":
            actual = messages[0].text or ""
            if actual != expected:
                raise SystemExit(
                    f"prefix {origin!r}: its recorded system prompt does not byte-match "
                    "the current seeds/SYSTEM_PROMPT.txt "
                    f"({'reasoning-stripped' if payload['reasoning'] else 'verbatim'} "
                    f"variant): prefix sha "
                    f"{hashlib.sha256(actual.encode()).hexdigest()[:12]}, current sha "
                    f"{hashlib.sha256(expected.encode()).hexdigest()[:12]}. A drifted "
                    "global prompt would confound the comparison against the original "
                    "base rate, so this is refused rather than silently replayed."
                )
        else:
            messages = [ChatMessageSystem(content=expected), *messages]
            system_prompt_inserted = True
        extra_system = [
            index for index, message in enumerate(messages[1:], start=2)
            if message.role == "system"
        ]
        if extra_system:
            raise SystemExit(
                f"prefix {origin!r}: system messages beyond the first are not supported "
                f"(found at positions {extra_system})"
            )

    messages = _ensure_unique_message_ids(list(messages))
    try:
        reject_dangling_tool_calls(messages)
    except ValueError as error:
        raise SystemExit(f"prefix {origin!r}: {error}") from error
    source = payload["source"]
    return PrefixSpec(
        name=payload["name"],
        payload=payload,
        sha256=payload_sha256(payload),
        payload_path=payload_path,
        messages=tuple(messages),
        target_name=payload["target"],
        reasoning=payload["reasoning"],
        # Prefix-only control datasets retain their own provenance in ``family``
        # and ``seed``, but continuation planning must compare the task the model
        # actually saw.  Otherwise a checkout control can be spliced straight into
        # another checkout task and is incorrectly labelled cross-family.
        family=source.get("comparison_source_family") or source.get("family"),
        source_seed=source.get("comparison_source_seed") or source.get("seed"),
        system_prompt_inserted=system_prompt_inserted,
        boundary_index=prefix_boundary_index(messages),
        native_resume=(
            native_resume
            if harness in NATIVE_HARNESS_MODES and not inline_user_context
            else None
        ),
        delivery_mode=delivery_mode,
    )


def load_prefix_specs(
    trajectory_ids: list[int],
    payload_files: list[str],
    *,
    harness: str,
    allow_incomplete_prefixes: bool = False,
) -> list[PrefixSpec]:
    """Resolve both prefix sources into validated specs with unique names."""

    specs: list[PrefixSpec] = []
    for payload in reconstruct_prefix_payloads(trajectory_ids):
        path = store_prefix_payload(payload)
        specs.append(build_prefix_spec(payload, payload_path=path, harness=harness))
    for file_path in payload_files:
        payload = load_prefix_payload_file(file_path)
        source = payload.get("source") or {}
        eligibility = source.get("continuation_eligibility")
        ml_prefix_only = (
            source.get("prefix_type") == "ml_prefix_only"
            or source.get("generator") == "exp_ml_prefix.py"
        )
        if isinstance(eligibility, dict):
            eligible_by_default = eligibility.get("eligible_by_default") is True
            missing_deliverables = list(
                eligibility.get("missing_required_deliverables") or []
            )
        elif ml_prefix_only:
            eligible_by_default = source.get("all_deliverables_present") is True
            deliverables = source.get("deliverables") or {}
            missing_deliverables = [
                name
                for name in ("REPORT.md", "models/final/")
                if deliverables.get(name) is not True
            ]
        else:
            eligible_by_default = True
            missing_deliverables = []
        requires_incomplete_override = not eligible_by_default
        if requires_incomplete_override and not allow_incomplete_prefixes:
            detail = ", ".join(missing_deliverables) or "status not recorded"
            raise SystemExit(
                f"prefix {payload['name']!r} is ineligible for continuations by "
                f"default because required ML deliverables are missing ({detail}); "
                "pass --allow-incomplete-prefixes only if task failure is an "
                "intentional prefix condition"
            )
        if requires_incomplete_override:
            detail = ", ".join(missing_deliverables) or "status not recorded"
            print(
                f"  WARNING: prefix {payload['name']!r} is using the explicit "
                "--allow-incomplete-prefixes override "
                f"(missing required deliverables: {detail})"
            )
        source_id = source.get("trajectory_id")
        # AWS workers intentionally receive the payload but not the local registry.
        # Wherever the source logs are available (local runs and the AWS controller),
        # recheck even an old, already-exported trajectory payload under today's rules.
        if source.get("kind") == "trajectory":
            if REGISTRY_FILE.is_file():
                if source_id is None:
                    raise SystemExit(
                        f"{file_path}: trajectory-sourced prefix has no trajectory_id"
                    )
                try:
                    source_id = int(source_id)
                except (TypeError, ValueError) as error:
                    raise SystemExit(
                        f"{file_path}: invalid source trajectory ID {source_id!r}"
                    ) from error
                source_entry = _registry_entries([source_id])[source_id]
                if (
                    (payload.get("delivery") or {}).get("mode")
                    == CONTINUATION_DELIVERY_INLINE_USER_CONTEXT
                    and source.get("prefix_type") == ACTIVITY_LOG_PREFIX_TYPE
                ):
                    current_source = reconstruct_prefix_payload(
                        source_id,
                        source_entry,
                        source_use="activity_log",
                    )
                    current_payload = activity_log_prefix_payload_from_trajectory(
                        current_source
                    )
                    stored_sha = (
                        source.get("activity_log_metadata") or {}
                    ).get("sha256")
                    current_sha = (
                        current_payload["source"].get("activity_log_metadata") or {}
                    ).get("sha256")
                    if stored_sha != current_sha:
                        raise SystemExit(
                            f"{file_path}: exported activity-log prefix differs from "
                            "the current trajectory reconstruction"
                        )
                else:
                    reconstruct_prefix_payload(source_id, source_entry)
            elif not os.environ.get("MATS_REMOTE_RUN_DIR"):
                raise SystemExit(
                    f"{file_path}: cannot verify this trajectory-sourced prefix "
                    f"because {REGISTRY_FILE} is missing"
                )
        spec = build_prefix_spec(
            payload, payload_path=Path(file_path), harness=harness
        )
        if requires_incomplete_override:
            spec = replace(spec, continuation_eligibility_override=True)
        specs.append(spec)
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(
            f"prefix names must be unique within one run; duplicates: {duplicates}"
        )
    if not specs:
        raise SystemExit("no prefixes were supplied (--prefixes and/or --prefix-files)")
    return specs


# --------------------------------------------------------------------------- #
# Cells and tasks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContinuationCell:
    """One (prefix, new-task member) pair; run for N epochs."""

    prefix: PrefixSpec
    member: str
    unit_path: Path
    family: str
    cross_family: bool | None  # None when the prefix has no seed family (external)


def build_continuation_cells(
    prefix_specs: list[PrefixSpec],
    seeds_path: str,
    members: list[str],
) -> list[ContinuationCell]:
    root = Path(seeds_path).resolve()
    if is_seed_member(root):
        units = {root.name: root}
    else:
        available = seed_members(root)
        units = {member: root / member for member in available if member in set(members)}
    missing = sorted(set(members) - set(units))
    if missing:
        raise SystemExit(f"--seeds includes unavailable members: {missing}")

    conflicts = [
        (spec.name, member)
        for spec in prefix_specs
        for member in units
        if spec.source_seed == member
    ]
    if conflicts:
        raise SystemExit(
            "a continuation's new task must come from a different seed than its "
            f"prefix; conflicting (prefix, seed) pairs: {conflicts}. Drop those "
            "seeds from --seeds or use other prefixes."
        )

    cells: list[ContinuationCell] = []
    for spec in prefix_specs:
        for member, unit_path in sorted(units.items()):
            family = unit_path.parent.name
            cross_family = None if spec.family is None else spec.family != family
            # Resolve the pivot descriptor NOW so an unmapped family aborts at plan
            # time rather than mid-generation.
            if cross_family is None or cross_family:
                seed_family_task_description(family)
            cells.append(ContinuationCell(
                prefix=spec,
                member=member,
                unit_path=unit_path,
                family=family,
                cross_family=cross_family,
            ))
    return cells


def continuation_task_name(treatment: str, cell: ContinuationCell, slug: str) -> str:
    model_last = slug.split("/")[-1]
    return f"continuation_{treatment}_{model_last}_{cell.member}_p{cell.prefix.name}"


def build_continuation_tasks(
    cells: list[ContinuationCell],
    *,
    treatment: str,
    run_label: str,
    condition: str,
    gate_model: str,
    judge: str | None,
    artifacts_root: Path | None,
    task_id_suffix: str | None = None,
    execution_metadata: dict | None = None,
    harness: str,
    pressure: str | None = None,
) -> list:
    """One Inspect task per cell, sharing build_real_task with plain audits."""

    harness = resolve_harness(harness)
    judge_model = resolve_judge(judge)
    tasks = []
    for cell in cells:
        prefix = cell.prefix
        unit_sp = (
            expected_system_prompt(prefix.reasoning)
            if harness == "simple"
            else ""
        )
        target_slug = resolve_target(prefix.target_name)
        opencode_go_subscription = (
            harness == "subscription"
            and opencode_go_model_spec(prefix.target_name, target_slug) is not None
        )
        target_build = build_target(
            target_slug,
            reasoning_on=prefix.reasoning,
            effort=REASONING_EFFORT,
            # OpenAI rejects prompt_cache_key values longer than 64 chars.
            prompt_cache_key="environments-cont-" + stable_key(
                (
                    "continuation-target-v1"
                    if harness == "simple"
                    else f"continuation-target-{harness}-v1"
                ),
                target_slug,
                prefix.reasoning,
                prefix.sha256,
                cell.member,
                pressure,
            )[:40],
            construct_model=not opencode_go_subscription,
        )
        descriptor = (
            None
            if cell.cross_family is False
            else seed_family_task_description(cell.family)
        )
        pivot = pivot_preamble(descriptor)
        spec = assemble_real_protocol(
            cell.unit_path, condition, unit_sp, pressure
        )
        record = continuation_record(
            treatment=treatment,
            prefix=prefix.record(),
            prefix_length=len(prefix.messages),
            boundary_index=prefix.boundary_index,
            pivot_preamble_text=pivot,
            opening_user_message=spec.opening_user_message,
            cross_family=cell.cross_family,
            delivery_mode=prefix.delivery_mode,
        )
        record["pressure"] = spec.pressure
        continuation = ContinuationRun(
            prefix_messages=prefix.messages,
            pivot_text=pivot,
            record=record,
            native_resume=prefix.native_resume,
            delivery_mode=prefix.delivery_mode,
        )
        if harness in NATIVE_HARNESS_MODES:
            if (
                prefix.delivery_mode
                == CONTINUATION_DELIVERY_INLINE_USER_CONTEXT
            ):
                record[f"{harness}_fresh_context"] = {
                    "delivery_mode": prefix.delivery_mode,
                    "fresh_session": True,
                    "native_session_resumed": False,
                    "scaffold_system_message_count": 0,
                }
            else:
                native_key = f"{harness}_native_resume"
                record[native_key] = native_resume_record(
                    prefix.native_resume or {}
                )
                record[native_key][
                    "scaffold_system_message_count"
                ] = sum(
                    message.role == "system"
                    for message in prefix.messages[:prefix.boundary_index]
                )
        tasks.append(build_real_task(
            target_name=prefix.target_name,
            target_build=target_build,
            unit_path=cell.unit_path,
            unit_sp=unit_sp,
            reasoning=prefix.reasoning,
            condition=condition,
            judge_model=judge_model,
            gate_model=gate_model,
            run_label=run_label,
            artifacts_root=artifacts_root,
            task_id_suffix=task_id_suffix,
            execution_metadata=execution_metadata,
            continuation=continuation,
            task_name=continuation_task_name(treatment, cell, target_slug),
            extra_metadata={
                "experiment": "continuation",
                "continuation_version": CONTINUATION_RECORD_VERSION,
                "treatment": treatment,
                "prefix_name": prefix.name,
                "prefix_sha256": prefix.sha256,
                "prefix_source": dict(prefix.payload["source"]),
                "prefix_target_name": prefix.target_name,
                "prefix_message_count": len(prefix.messages),
                "prefix_boundary_index": prefix.boundary_index,
                "prefix_system_prompt_inserted": prefix.system_prompt_inserted,
                "prefix_delivery_mode": prefix.delivery_mode,
                "pivot_preamble": pivot,
                "cross_family": cell.cross_family,
            },
            harness=harness,
            pressure=pressure,
        ))
    return tasks


def describe_plan(
    treatment: str, prefix_specs: list[PrefixSpec], cells: list[ContinuationCell]
) -> str:
    """Human-readable plan block for the pipeline's stdout (and --dry-run)."""

    lines = [f"treatment {treatment!r}: {len(prefix_specs)} prefix(es), "
             f"{len(cells)} cell(s)"]
    for spec in prefix_specs:
        source = spec.payload["source"]
        origin = (
            f"trajectory #{source.get('trajectory_id')} "
            f"(seed {source.get('seed')}, {source.get('run')})"
            if source.get("kind") == "trajectory"
            else f"external ({source.get('description') or 'no description'})"
        )
        flags = []
        if spec.system_prompt_inserted:
            flags.append("system prompt inserted")
        flag_note = f"  [{'; '.join(flags)}]" if flags else ""
        lines.append(
            f"  prefix {spec.name}: {origin}, agent {spec.target_name} "
            f"[reasoning:{'on' if spec.reasoning else 'off'}], "
            f"delivery={spec.delivery_mode}, "
            f"{len(spec.messages)} msgs / {spec.char_count:,} chars{flag_note}"
        )
    for cell in cells:
        pivot_kind = (
            "plain pivot" if cell.cross_family is False
            else "pivot names the new work type"
        )
        lines.append(
            f"  cell: {cell.prefix.name} -> {cell.member} "
            f"({cell.family}; {pivot_kind})"
        )
    return "\n".join(lines)
