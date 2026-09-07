"""Pipeline-side integrity reporting and atomic sidecar persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from real_integrity import integrity_sidecar_records


def gate_integrity_failures(logs: list) -> list[dict]:
    """Compatibility view of samples whose scripted gate failed."""

    failures: list[dict] = []
    for log in logs or []:
        for sample in log.samples or []:
            real_env = (sample.metadata or {}).get("real_env") or {}
            protocol = real_env.get("protocol") or {}
            gate_errors = [
                gate for gate in real_env.get("gates", [])
                if gate.get("verdict") == "error"
            ]
            if gate_errors or protocol.get("ended_reason") == "gate_error_end":
                failures.append({
                    "task": log.eval.task,
                    "sample": str(sample.id),
                    "epoch": sample.epoch,
                    "issues": ["gate_error"] + [
                        str(gate.get("error"))[:120] for gate in gate_errors
                    ],
                })
    return failures


def audit_integrity_failures(logs: list) -> list[dict]:
    """Compatibility view of samples whose workspace did not finalize safely."""

    failures: list[dict] = []
    for log in logs or []:
        for sample in log.samples or []:
            real_env = (sample.metadata or {}).get("real_env") or {}
            issues = []
            if not real_env.get("protocol"):
                issues.append("missing_protocol_record")
            if not real_env.get("artifacts"):
                issues.append("missing_final_artifact_record")
            if real_env.get("finalize_error"):
                issues.append(f"finalize_error:{real_env['finalize_error']}")
            if issues:
                failures.append({
                    "task": log.eval.task,
                    "sample": str(sample.id),
                    "epoch": sample.epoch,
                    "issues": issues,
                })
    return failures


def judge_procedure_integrity_failures(logs: list) -> list[dict]:
    """Compatibility view of samples without one valid official judgment."""

    failures: list[dict] = []
    for log in logs or []:
        for sample in log.samples or []:
            score = (sample.scores or {}).get("environment_judge")
            metadata = getattr(score, "metadata", None) or {} if score else {}
            envelope = metadata.get("environment_judge") or {}
            issues = []
            if score is None:
                issues.append("missing_environment_judge_score")
            if not isinstance(envelope.get("result"), dict):
                issues.append("missing_structured_result")
            if envelope.get("post_validation") != "passed":
                issues.append(
                    f"judge_post_validation:{envelope.get('post_validation') or 'missing'}"
                )
            stage = envelope.get("official_stage")
            if stage not in {"stage_1", "final"}:
                issues.append(f"invalid_official_stage:{stage or 'missing'}")
            if stage == "final" and envelope.get("fresh_call") is not True:
                issues.append("final_judgment_not_marked_fresh")
            if issues:
                failures.append({
                    "task": log.eval.task,
                    "sample": str(sample.id),
                    "epoch": sample.epoch,
                    "issues": issues,
                })
    return failures


def write_integrity_sidecar(log_dir: Path, logs: list) -> list[dict]:
    """Persist the exact per-sample inclusion verdict consumed by the viewer."""

    records = integrity_sidecar_records(logs)
    payload = {
        "schema_version": "environment-pipeline-integrity-v2",
        "records": records,
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=log_dir, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, log_dir / "pipeline_integrity.json")
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return records
