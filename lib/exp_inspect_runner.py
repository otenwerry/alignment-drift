"""Thin environments-owned wrapper around Inspect's public ``eval_set``.

This file can launch paid model calls, hence the ``exp_`` prefix.  Importing it does not
run anything.  Every invocation writes a runtime/accounting manifest before spend and
updates it after Inspect returns or raises.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cost_tracking import (
    cost_tracking_provenance,
    estimate_usage_cost,
    install_cost_tracking,
)
from prompt_caching import install_inspect_warmup, write_report as write_cache_report


RUNTIME_MANIFEST_FILENAME = "runtime_accounting.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_manifest(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _usage_dict(usage: Any) -> dict:
    def read(*names: str):
        for name in names:
            value = getattr(usage, name, None)
            if isinstance(value, (int, float)):
                return value
        return 0

    total_cost = getattr(usage, "total_cost", None)
    return {
        "input": read("input_tokens"),
        "output": read("output_tokens"),
        "cache_read": read("input_tokens_cache_read"),
        "cache_write": read("input_tokens_cache_write"),
        "total_cost": total_cost if isinstance(total_cost, (int, float)) else None,
    }


def summarize_log_usage(logs: list[Any]) -> dict:
    """Aggregate Inspect usage while retaining model-level cost provenance."""

    by_model: dict[str, dict] = {}
    for log in logs:
        stats = getattr(log, "stats", None)
        for slug, usage in (getattr(stats, "model_usage", None) or {}).items():
            row = by_model.setdefault(
                slug,
                {
                    "input": 0,
                    "output": 0,
                    "cache_read": 0,
                    "cache_write": 0,
                    "total_cost": 0.0,
                    "all_calls_costed": True,
                },
            )
            current = _usage_dict(usage)
            for field in ("input", "output", "cache_read", "cache_write"):
                row[field] += current[field]
            if current["total_cost"] is None:
                row["all_calls_costed"] = False
            else:
                row["total_cost"] += current["total_cost"]

    total = 0.0
    exact = True
    unknown: list[str] = []
    subscription_models: list[str] = []
    for slug, row in by_model.items():
        # Do not pass a synthetic 0 total through as billed when one or more calls lacked it.
        usage = dict(row)
        if not row["all_calls_costed"]:
            usage["total_cost"] = None
        priced = estimate_usage_cost(slug, usage)
        row["priced"] = priced
        if priced["cost_usd"] is None:
            if priced.get("source") == "subscription_not_metered":
                subscription_models.append(slug)
            else:
                unknown.append(slug)
            exact = False
        else:
            total += priced["cost_usd"]
            exact = exact and bool(priced["exact"])
    return {
        "total_cost_usd": total,
        "exact": exact and bool(by_model),
        "unknown_models": sorted(unknown),
        "subscription_models_not_metered": sorted(subscription_models),
        "total_excludes_subscription_usage": bool(subscription_models),
        "by_model": by_model,
    }


def read_saved_logs(
    log_dir: str | Path, summary_logs: list[Any] | None = None
) -> list[Any]:
    """Read the complete logs that Inspect saved after a run."""

    from inspect_ai.log import list_eval_logs, read_eval_log

    locations = [getattr(log, "location", "") for log in summary_logs or []]
    if locations and all(locations):
        return [read_eval_log(location) for location in locations]
    return [read_eval_log(log_info) for log_info in list_eval_logs(str(log_dir))]


def run_eval(
    tasks: list[Any],
    epochs: int,
    concurrency: int,
    log_dir: str | Path,
    max_sandboxes: int | None = None,
    time_limit: int | None = None,
):
    """Run an Inspect eval set with environment accounting installed first.

    The returned logs are reloaded from disk so they include the complete saved samples.
    Individual sample failures remain in the logs and do not make this wrapper discard
    completed samples.
    """

    if not tasks:
        raise SystemExit("refusing to call Inspect with no tasks")
    if epochs < 1:
        raise SystemExit(f"epochs must be >= 1, got {epochs}")
    if concurrency < 1:
        raise SystemExit(f"concurrency must be >= 1, got {concurrency}")

    from inspect_ai import eval_set

    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / RUNTIME_MANIFEST_FILENAME
    installed = install_cost_tracking()
    prompt_cache_warmup_installed = install_inspect_warmup()
    manifest = {
        "schema_version": "environments-runtime-accounting-v1",
        "status": "running",
        "started_at": _now(),
        "finished_at": None,
        "task_count": len(tasks),
        "epochs": epochs,
        "concurrency": concurrency,
        "max_sandboxes": max_sandboxes,
        "time_limit_seconds": time_limit,
        "response_cache_enabled": False,
        "provider_prompt_cache": {
            "enabled": True,
            "warmup_barrier_installed": prompt_cache_warmup_installed,
            "report": "prompt_cache_report.json",
        },
        "cost_tracking": cost_tracking_provenance(),
        "cost_install": installed.to_dict(),
        "usage": None,
        "error": None,
    }
    _write_manifest(manifest_path, manifest)

    kwargs: dict[str, Any] = {
        "epochs": epochs,
        "max_tasks": len(tasks),
        "max_samples": concurrency,
        "max_connections": concurrency,
        "log_dir": str(output_dir),
    }
    if max_sandboxes is not None:
        kwargs["max_sandboxes"] = max_sandboxes
    if time_limit is not None:
        kwargs["time_limit"] = time_limit

    try:
        success, summary_logs = eval_set(tasks, **kwargs)
        logs = read_saved_logs(output_dir, summary_logs)
        if not logs:
            raise RuntimeError(
                f"Inspect returned without saving any eval logs in {output_dir}"
            )
        cache_report = write_cache_report(output_dir, logs)
        manifest.update(
            status="complete" if success else "completed_with_failures",
            finished_at=_now(),
            usage=summarize_log_usage(logs),
            provider_prompt_cache={
                **manifest["provider_prompt_cache"],
                "verified_models": cache_report["verified_models"],
                "unverified_models": cache_report["unverified_models"],
            },
        )
        _write_manifest(manifest_path, manifest)
        return success, logs
    except BaseException as exc:
        manifest.update(
            status="failed",
            finished_at=_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _write_manifest(manifest_path, manifest)
        raise


_MODEL_SNAPSHOT_SUFFIX_RE = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2})$")


def _model_family_name(slug: str) -> str:
    """Comparable model name when an included-usage service drops snapshot dates."""

    return _MODEL_SNAPSHOT_SUFFIX_RE.sub("", slug.rsplit("/", 1)[-1])


def dead_targets(logs: list[Any], target_models: list[str]) -> list[str]:
    """Models that produced zero output tokens across all completed logs.

    Inspect records the model name actually served. Included-usage routes can serve an
    undated family name even when the requested catalog entry has a dated snapshot.
    Exact names win; an undated match is accepted only when it identifies one target.
    """

    output_by_target = {slug.rsplit("/", 1)[-1]: 0 for slug in target_models}
    family_targets: dict[str, list[str]] = {}
    for target in output_by_target:
        family_targets.setdefault(_model_family_name(target), []).append(target)
    for log in logs:
        stats = getattr(log, "stats", None)
        for slug, usage in (getattr(stats, "model_usage", None) or {}).items():
            name = slug.rsplit("/", 1)[-1]
            target = name if name in output_by_target else None
            if target is None:
                matches = family_targets.get(_model_family_name(name), [])
                if len(matches) == 1:
                    target = matches[0]
            if target is not None:
                output_by_target[target] += getattr(usage, "output_tokens", 0) or 0
    return sorted(name for name, output in output_by_target.items() if output == 0)
