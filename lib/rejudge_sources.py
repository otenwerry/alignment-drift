"""Free discovery and exact-input fingerprinting for retrospective rejudging."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from inspect_ai.log import list_eval_logs, read_eval_log, resolve_sample_attachments
from inspect_ai.model import ChatMessage

from environment_judge import JUDGE_SCHEMA_VERSION, prepare_judge_call
from interrupted_native_transcript import recover_interrupted_opencode_messages
from old_runs import old_source_run_names
from project_paths import ENVIRONMENTS_ROOT
from real_integrity import (
    finalize_audit_integrity,
    sample_has_target_output,
    target_provider_events,
    tool_output_truncations,
)
from real_judge_evidence import complete_real_trajectory_evidence


FAMILIES = ("ml_training_data_misuse", "p_hacking")
REJUDGE_METHOD_VERSION = "retrospective-current-judge-v2"


@dataclass(frozen=True)
class SourceTrajectory:
    messages: tuple[ChatMessage, ...]
    real_env: dict
    source: dict

    @property
    def family(self) -> str:
        return str(self.source["family"])


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _plain(dump(mode="json"))
    return str(value)


def _role_model(roles: dict, role: str) -> str | None:
    value = (roles or {}).get(role)
    if value is None:
        return None
    return str(getattr(value, "model", None) or value)


def _source_family(real_env: dict, run_metadata: dict) -> str:
    candidates = [
        real_env.get("family"),
        run_metadata.get("dimension_scope"),
        str(run_metadata.get("seed_dir") or "").split("/", 1)[0],
    ]
    family = next((str(item) for item in candidates if item in FAMILIES), None)
    if family is None:
        raise ValueError(
            "cannot identify trajectory family from real_env.family, "
            "dimension_scope, or seed_dir"
        )
    return family


def source_key(source: dict) -> str:
    identity = {
        key: source.get(key)
        for key in ("source_run", "source_log", "source_task", "seed", "epoch")
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def discover_run_dirs(logs_root: Path, selection: str) -> list[Path]:
    available = sorted(
        path for path in logs_root.iterdir()
        if path.is_dir() and path.name.startswith("real-v")
    ) if logs_root.is_dir() else []
    if selection == "all":
        if not available:
            raise ValueError(f"no original real-v* runs found under {logs_root}")
        return available
    requested = (
        sorted(old_source_run_names())
        if selection == "old"
        else [item.strip() for item in selection.split(",") if item.strip()]
    )
    if not requested:
        raise ValueError(
            "--source-runs must be `all`, `old`, or a comma-separated run list"
        )
    by_name = {path.name: path for path in available}
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(
            f"unknown original run directories {unknown}; available: {sorted(by_name)}"
        )
    return [by_name[name] for name in requested]


def load_sources(
    run_dirs: Sequence[Path], *, family: str = "all"
) -> list[SourceTrajectory]:
    if family not in ("all", *FAMILIES):
        raise ValueError(f"unknown family {family!r}; choices: all, {', '.join(FAMILIES)}")
    sources: list[SourceTrajectory] = []
    errors: list[str] = []
    for run_dir in run_dirs:
        for log_info in list_eval_logs(str(run_dir)):
            try:
                log = read_eval_log(log_info)
                run_metadata = _plain(getattr(log.eval, "metadata", None) or {})
                roles = getattr(log.eval, "model_roles", None) or {}
                log_name = Path(str(getattr(log, "location", "") or log_info)).name
                for raw_sample in getattr(log, "samples", None) or []:
                    sample = resolve_sample_attachments(raw_sample, "full")
                    sample_metadata = _plain(getattr(sample, "metadata", None) or {})
                    real_env = sample_metadata.get("real_env")
                    if not isinstance(real_env, dict):
                        raise ValueError("sample has no real_env metadata")
                    messages = tuple(getattr(sample, "messages", None) or ())
                    if not messages:
                        raise ValueError("sample has no stored messages")
                    source_family = _source_family(real_env, run_metadata)
                    if family != "all" and source_family != family:
                        continue
                    target_model = _role_model(roles, "target")
                    recovery = None
                    if ((real_env.get("harness") or {}).get("scaffold")) == "opencode":
                        recovered, recovery = recover_interrupted_opencode_messages(
                            messages,
                            list(getattr(sample, "events", None) or []),
                            target_model=target_model,
                            applied_before_judging=True,
                            attachments=getattr(sample, "attachments", None) or {},
                        )
                        if recovery is not None:
                            recovery["recovered_during_retrospective_loading"] = True
                            messages = tuple(recovered)
                            real_env["interrupted_native_transcript"] = recovery
                            policy = real_env.setdefault("judge_evidence_policy", {})
                            reasons = policy.setdefault("lossy_reasons", [])
                            if "interrupted_native_transcript_reconstructed" not in reasons:
                                reasons.append(
                                    "interrupted_native_transcript_reconstructed"
                                )
                            policy["lossy"] = True
                    target_usage = (
                        getattr(sample, "role_usage", None) or {}
                    ).get("target")
                    source_audit = finalize_audit_integrity({
                        "real_env": real_env,
                        "real_ended_reason": (
                            (real_env.get("protocol") or {}).get("ended_reason")
                        ),
                        "target_tools_mode": run_metadata.get("target_tools_mode"),
                        "judgment": None,
                        "target_provider_events": target_provider_events(
                            sample, target_model
                        ),
                        "tool_truncations": tool_output_truncations(sample),
                        "compactions": [
                            _plain(event)
                            for event in getattr(sample, "events", None) or []
                            if getattr(event, "event", None) == "compaction"
                        ],
                        "dead": (
                            not sample_has_target_output(sample)
                            and (getattr(target_usage, "output_tokens", 0) or 0) == 0
                        ),
                    })
                    record = {
                        "source_run": run_dir.name,
                        "source_log": log_name,
                        "source_task": str(log.eval.task),
                        "seed": str(getattr(sample, "id", "")),
                        "epoch": getattr(sample, "epoch", None),
                        "family": source_family,
                        "target": target_model,
                        "original_judge": _role_model(roles, "judge"),
                        "condition": run_metadata.get("condition"),
                        "pressure": (
                            run_metadata.get("pressure")
                            or real_env.get("pressure")
                        ),
                        "harness": (
                            run_metadata.get("harness")
                            or (real_env.get("harness") or {}).get("mode")
                            or "simple"
                        ),
                        "real_env_manifest_sha": run_metadata.get(
                            "real_env_manifest_sha"
                        ),
                        "source_score_keys": sorted(
                            str(key) for key in (getattr(sample, "scores", None) or {})
                        ),
                        "source_integrity_status": source_audit["integrity_status"],
                        "source_integrity_issues": source_audit["integrity_issues"],
                        "source_flags": source_audit["flags"],
                        "source_target_provider_events": source_audit[
                            "target_provider_events"
                        ],
                        "interrupted_native_transcript": recovery,
                    }
                    record["source_key"] = source_key(record)
                    sources.append(SourceTrajectory(
                        messages=messages,
                        real_env=real_env,
                        source=record,
                    ))
            except Exception as error:
                errors.append(f"{run_dir.name}/{log_info}: {type(error).__name__}: {error}")
    if errors:
        raise ValueError("source validation failed before judging:\n" + "\n".join(errors))
    if not sources:
        raise ValueError("the selected original runs contain no matching trajectories")
    keys = [item.source["source_key"] for item in sources]
    if len(keys) != len(set(keys)):
        raise ValueError("selected source trajectories do not have unique identities")
    return sorted(sources, key=lambda item: item.source["source_key"])


def judging_method_sha256() -> str:
    paths = sorted((ENVIRONMENTS_ROOT / "lib" / "environment_judge").glob("*.py"))
    paths.extend([
        ENVIRONMENTS_ROOT / "lib" / "interrupted_native_transcript.py",
        ENVIRONMENTS_ROOT / "lib" / "real_judge_evidence.py",
        *sorted((ENVIRONMENTS_ROOT / "judge_instructions").rglob("*.md")),
    ])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ENVIRONMENTS_ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


async def prepare_sources(
    sources: Sequence[SourceTrajectory], *, judge_model: str
) -> tuple[list[SourceTrajectory], dict]:
    prepared_sources: list[SourceTrajectory] = []
    prompt_rows = []
    for item in sources:
        artifacts, caveats = complete_real_trajectory_evidence(
            item.messages, item.real_env
        )
        prepared = await prepare_judge_call(
            family=item.family,
            stage="final",
            messages=item.messages,
            artifacts=artifacts,
            upstream_caveats=caveats,
        )
        prompt_sha = prepared.metadata()["prompt_sha256"]
        source = {
            **item.source,
            "current_judge_input_sha256": prompt_sha,
            "current_judge_method_sha256": prepared.method_sha256(),
            "upstream_caveat_codes": [entry.code for entry in prepared.evidence.caveats],
        }
        prepared_sources.append(replace(item, source=source))
        prompt_rows.append({
            "source_key": source["source_key"],
            "prompt_sha256": prompt_sha,
        })

    method_sha = judging_method_sha256()
    method_identity = {
        "rejudge_method_version": REJUDGE_METHOD_VERSION,
        "judge_schema_version": JUDGE_SCHEMA_VERSION,
        "judge_model": judge_model,
        "judging_method_sha256": method_sha,
    }
    method_fingerprint = hashlib.sha256(
        json.dumps(method_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    identity = {
        **method_identity,
        "method_fingerprint": method_fingerprint,
        "inputs": sorted(prompt_rows, key=lambda row: row["source_key"]),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    campaign = {**identity, "campaign_fingerprint": fingerprint}
    return prepared_sources, campaign
