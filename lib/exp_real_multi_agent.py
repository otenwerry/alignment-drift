"""Fresh-session activity-log experiment core for real environments.

Each cell mounts a line-addressable ``ACTIVITY_LOG.md`` rendered from a different
stored trajectory, then starts the same catalog agent on a new seed in a fresh
sandbox.  Unlike a continuation, no prior messages or native session state are
resumed.  The ordinary real-audit task builder remains the execution path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.model import ChatMessage
from pydantic import TypeAdapter

from exp_real_audit import assemble_real_protocol, build_real_task
from exp_real_continuation import (
    PREFIX_FORMAT,
    expected_system_prompt,
    reconstruct_prefix_payloads,
    validate_treatment,
)
from exp_target_harness import (
    opencode_go_model_spec,
    resolve_harness,
)
from judge_selection import resolve_judge
from model_catalog import REASONING_EFFORT, TARGET_CHOICES, build_target, resolve_target
from multi_agent_evidence import (
    ACTIVITY_LOG_FORMAT,
    ACTIVITY_LOG_PROMPT_TEMPLATE,
    MULTI_AGENT_RECORD_VERSION,
    ActivityLogRun,
    activity_log_sha256,
    effective_opening_user_message,
    render_activity_log,
)
from project_paths import ACTIVITY_LOGS_ROOT
from prompt_caching import stable_key
from protocol_sources import is_seed_member, seed_members


ACTIVITY_LOG_PAYLOAD_FORMAT = "environments-multi-agent-activity-log-v2"
ACTIVITY_LOG_STORE = ACTIVITY_LOGS_ROOT
_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_MESSAGES_ADAPTER = TypeAdapter(list[ChatMessage])


def payload_sha256(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_activity_log_payload(payload: dict, *, origin: str) -> dict:
    if not isinstance(payload, dict):
        raise SystemExit(f"{origin}: activity-log payload must be a JSON object")
    if payload.get("format") != ACTIVITY_LOG_PAYLOAD_FORMAT:
        raise SystemExit(
            f"{origin}: activity-log payload format must be "
            f"{ACTIVITY_LOG_PAYLOAD_FORMAT!r}"
        )
    name = payload.get("name")
    if not isinstance(name, str) or not _SLUG_RE.fullmatch(name):
        raise SystemExit(f"{origin}: activity-log payload needs a slug 'name'")
    if payload.get("target") not in TARGET_CHOICES:
        raise SystemExit(
            f"{origin}: payload target must be a catalog agent name; got "
            f"{payload.get('target')!r}"
        )
    if not isinstance(payload.get("reasoning"), bool):
        raise SystemExit(f"{origin}: payload reasoning must be true/false")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("kind") not in {
        "trajectory", "external"
    }:
        raise SystemExit(
            f"{origin}: source.kind must be 'trajectory' or 'external'"
        )
    activity = payload.get("activity_log")
    if not isinstance(activity, dict) or not isinstance(activity.get("content"), str):
        raise SystemExit(f"{origin}: activity_log.content must be a string")
    metadata = activity.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("format") != ACTIVITY_LOG_FORMAT:
        raise SystemExit(
            f"{origin}: activity_log.metadata.format must be {ACTIVITY_LOG_FORMAT!r}"
        )
    content = activity["content"]
    expected_sha = activity_log_sha256(content)
    if metadata.get("sha256") != expected_sha:
        raise SystemExit(f"{origin}: activity-log content SHA-256 does not match")
    lines = content.splitlines()
    if metadata.get("line_count") != len(lines) or not lines:
        raise SystemExit(f"{origin}: activity-log line count does not match")
    expected_markers = [f"[L{number:06d}]" for number in range(1, len(lines) + 1)]
    if any(
        not line.startswith(marker)
        for line, marker in zip(lines, expected_markers, strict=True)
    ):
        raise SystemExit(f"{origin}: activity-log line markers are not contiguous")
    return payload


def load_activity_log_payload_file(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.is_file():
        raise SystemExit(f"activity-log payload file not found: {file_path}")
    try:
        payload = json.loads(file_path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"{file_path} is not valid JSON: {error}") from error
    return validate_activity_log_payload(payload, origin=str(file_path))


def store_activity_log_payload(payload: dict) -> Path:
    ACTIVITY_LOG_STORE.mkdir(parents=True, exist_ok=True)
    digest = payload_sha256(payload)
    path = ACTIVITY_LOG_STORE / f"{payload['name']}-{digest[:12]}.json"
    if not path.exists():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        try:
            stored = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise SystemExit(f"stored activity-log payload is corrupt: {path}") from error
        if stored != payload:
            raise SystemExit(f"stored activity-log payload drifted: {path}")
    return path


def activity_log_payload_from_trajectory(prefix_payload: dict) -> dict:
    """Convert the validated stored-trajectory representation into a visible log."""

    if prefix_payload.get("format") != PREFIX_FORMAT:
        raise SystemExit("trajectory reconstruction returned an unexpected format")
    try:
        messages = _MESSAGES_ADAPTER.validate_python(prefix_payload["messages"])
    except Exception as error:
        raise SystemExit(
            f"{prefix_payload.get('name')}: stored messages could not be parsed: "
            f"{error}"
        ) from error
    content, metadata = render_activity_log(messages)
    source = dict(prefix_payload["source"])
    source["stored_message_format"] = "inspect-chat-messages"
    source["native_session_state_included"] = False
    return validate_activity_log_payload({
        "format": ACTIVITY_LOG_PAYLOAD_FORMAT,
        "name": f"log-{prefix_payload['name']}",
        "target": prefix_payload["target"],
        "reasoning": prefix_payload["reasoning"],
        "source": source,
        "activity_log": {
            "content": content,
            "metadata": metadata,
        },
    }, origin=f"trajectory {source.get('trajectory_id')}")


@dataclass(frozen=True)
class ActivityLogSpec:
    name: str
    payload: dict
    sha256: str
    payload_path: Path
    markdown_path: Path
    target_name: str
    reasoning: bool
    family: str | None
    source_seed: str | None
    source_harness: str

    def record(self, *, treatment: str, base_prompt: str) -> dict:
        activity = self.payload["activity_log"]
        return {
            "version": MULTI_AGENT_RECORD_VERSION,
            "treatment": treatment,
            "name": self.name,
            "payload_sha256": self.sha256,
            "target_name": self.target_name,
            "reasoning": self.reasoning,
            "source": dict(self.payload["source"]),
            "activity_log": {
                "content": activity["content"],
                "metadata": dict(activity["metadata"]),
            },
            "base_opening_user_message": base_prompt,
            "effective_opening_user_message": effective_opening_user_message(
                base_prompt
            ),
            "prompt_template": ACTIVITY_LOG_PROMPT_TEMPLATE,
            "fresh_session": True,
            "system_prompt_modified": False,
        }


def build_activity_log_spec(
    payload: dict, *, payload_path: Path, harness: str
) -> ActivityLogSpec:
    validate_activity_log_payload(payload, origin=str(payload_path))
    harness = resolve_harness(harness)
    source = payload["source"]
    source_harness = str(source.get("harness") or "simple")
    if source_harness != harness:
        raise SystemExit(
            f"activity log {payload['name']!r} came from --harness="
            f"{source_harness}, but this run requested --harness={harness}"
        )
    if harness == "simple":
        recorded = source.get("environment_system_prompt")
        expected = expected_system_prompt(payload["reasoning"])
        if recorded is not None and recorded != expected:
            raise SystemExit(
                f"activity log {payload['name']!r}: source system prompt no longer "
                "matches the current pinned prompt"
            )
    ACTIVITY_LOG_STORE.mkdir(parents=True, exist_ok=True)
    markdown_path = ACTIVITY_LOG_STORE / (
        f"{payload['name']}-{payload_sha256(payload)[:12]}.md"
    )
    content = payload["activity_log"]["content"]
    if not markdown_path.exists():
        markdown_path.write_text(content)
    elif markdown_path.read_text() != content:
        raise SystemExit(f"activity-log Markdown sidecar drifted: {markdown_path}")
    return ActivityLogSpec(
        name=payload["name"],
        payload=payload,
        sha256=payload_sha256(payload),
        payload_path=payload_path,
        markdown_path=markdown_path,
        target_name=payload["target"],
        reasoning=payload["reasoning"],
        family=source.get("family"),
        source_seed=source.get("seed"),
        source_harness=source_harness,
    )


def load_activity_log_specs(
    trajectory_ids: list[int], payload_files: list[str], *, harness: str
) -> list[ActivityLogSpec]:
    specs: list[ActivityLogSpec] = []
    for prefix_payload in reconstruct_prefix_payloads(
        trajectory_ids, source_use="activity_log"
    ):
        payload = activity_log_payload_from_trajectory(prefix_payload)
        path = store_activity_log_payload(payload)
        specs.append(build_activity_log_spec(payload, payload_path=path, harness=harness))
    for raw_path in payload_files:
        payload = load_activity_log_payload_file(raw_path)
        source = payload.get("source") or {}
        # On a controller/local machine, revalidate trajectory-backed exported files
        # using today's integrity rules. AWS workers intentionally lack the registry.
        if source.get("kind") == "trajectory" and not os.environ.get(
            "MATS_REMOTE_RUN_DIR"
        ):
            source_id = source.get("trajectory_id")
            if source_id is None:
                raise SystemExit(f"{raw_path}: trajectory source has no trajectory_id")
            current = activity_log_payload_from_trajectory(
                reconstruct_prefix_payloads(
                    [int(source_id)], source_use="activity_log"
                )[0]
            )
            if payload_sha256(current) != payload_sha256(payload):
                raise SystemExit(
                    f"{raw_path}: exported activity log differs from current "
                    "trajectory reconstruction"
                )
        path = Path(raw_path)
        specs.append(build_activity_log_spec(payload, payload_path=path, harness=harness))
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(f"activity-log names must be unique: {duplicates}")
    if not specs:
        raise SystemExit("no activity logs were supplied")
    return specs


@dataclass(frozen=True)
class MultiAgentCell:
    activity_log: ActivityLogSpec
    member: str
    unit_path: Path
    family: str
    cross_family: bool | None


def build_multi_agent_cells(
    specs: list[ActivityLogSpec], seeds_path: str, members: list[str]
) -> list[MultiAgentCell]:
    root = Path(seeds_path).resolve()
    if is_seed_member(root):
        units = {root.name: root}
    else:
        units = {
            member: root / member
            for member in seed_members(root)
            if member in set(members)
        }
    missing = sorted(set(members) - set(units))
    if missing:
        raise SystemExit(f"--seeds includes unavailable members: {missing}")
    conflicts = [
        (spec.name, member)
        for spec in specs
        for member in units
        if spec.source_seed == member
    ]
    if conflicts:
        raise SystemExit(
            "the new task must differ from the activity log's source task; "
            f"conflicting pairs: {conflicts}"
        )
    return [
        MultiAgentCell(
            activity_log=spec,
            member=member,
            unit_path=unit_path,
            family=unit_path.parent.name,
            cross_family=(
                None if spec.family is None else spec.family != unit_path.parent.name
            ),
        )
        for spec in specs
        for member, unit_path in sorted(units.items())
    ]


def multi_agent_task_name(treatment: str, cell: MultiAgentCell, slug: str) -> str:
    return (
        f"multi_agent_{treatment}_{slug.split('/')[-1]}_{cell.member}_"
        f"a{cell.activity_log.name}"
    )


def build_multi_agent_tasks(
    cells: list[MultiAgentCell],
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
    harness = resolve_harness(harness)
    judge_model = resolve_judge(judge)
    tasks = []
    for cell in cells:
        activity = cell.activity_log
        unit_sp = expected_system_prompt(activity.reasoning) if harness == "simple" else ""
        target_slug = resolve_target(activity.target_name)
        opencode_go_subscription = (
            harness == "subscription"
            and opencode_go_model_spec(activity.target_name, target_slug) is not None
        )
        target_build = build_target(
            target_slug,
            reasoning_on=activity.reasoning,
            effort=REASONING_EFFORT,
            prompt_cache_key="environments-multi-" + stable_key(
                f"multi-agent-target-{harness}-v1",
                target_slug,
                activity.reasoning,
                activity.sha256,
                cell.member,
                pressure,
            )[:40],
            construct_model=not opencode_go_subscription,
        )
        spec = assemble_real_protocol(cell.unit_path, condition, unit_sp, pressure)
        record = activity.record(
            treatment=treatment, base_prompt=spec.opening_user_message
        )
        record["pressure"] = spec.pressure
        run = ActivityLogRun(content_path=activity.markdown_path, record=record)
        tasks.append(build_real_task(
            target_name=activity.target_name,
            target_build=target_build,
            unit_path=cell.unit_path,
            unit_sp=unit_sp,
            reasoning=activity.reasoning,
            condition=condition,
            judge_model=judge_model,
            gate_model=gate_model,
            run_label=run_label,
            artifacts_root=artifacts_root,
            task_id_suffix=task_id_suffix,
            execution_metadata=execution_metadata,
            activity_log=run,
            task_name=multi_agent_task_name(treatment, cell, target_slug),
            extra_metadata={
                "experiment": "multi_agent",
                "multi_agent_version": MULTI_AGENT_RECORD_VERSION,
                "treatment": treatment,
                "activity_log_name": activity.name,
                "activity_log_sha256": activity.sha256,
                "activity_log_source": dict(activity.payload["source"]),
                "cross_family": cell.cross_family,
            },
            harness=harness,
            pressure=pressure,
        ))
    return tasks


def describe_plan(
    treatment: str, specs: list[ActivityLogSpec], cells: list[MultiAgentCell]
) -> str:
    lines = [
        f"treatment {treatment!r}: {len(specs)} activity log(s), {len(cells)} cell(s)"
    ]
    for spec in specs:
        source = spec.payload["source"]
        origin = (
            f"trajectory #{source.get('trajectory_id')} (seed {source.get('seed')})"
            if source.get("kind") == "trajectory"
            else f"external ({source.get('description') or 'no description'})"
        )
        lines.append(
            f"  {spec.name}: {spec.target_name}, reasoning={spec.reasoning}, "
            f"{origin}, {spec.payload['activity_log']['metadata']['line_count']} lines"
        )
    for cell in cells:
        lines.append(
            f"    {cell.activity_log.name} -> {cell.member} ({cell.family}, "
            f"cross_family={cell.cross_family})"
        )
    return "\n".join(lines)


__all__ = [
    "ACTIVITY_LOG_PAYLOAD_FORMAT",
    "ActivityLogSpec",
    "MultiAgentCell",
    "activity_log_payload_from_trajectory",
    "build_multi_agent_cells",
    "build_multi_agent_tasks",
    "describe_plan",
    "load_activity_log_specs",
    "payload_sha256",
    "store_activity_log_payload",
    "validate_activity_log_payload",
    "validate_treatment",
]
