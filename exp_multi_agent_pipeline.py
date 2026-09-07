"""End-to-end activity-log transfer pipeline for real environments.

Each cell starts a fresh agent on a new task. The only treatment is a mounted
``ACTIVITY_LOG.md`` rendered from another agent instance's stored trajectory and the
fixed opening user-role wrapper naming that file. No conversation or native session
is resumed, and the environment system prompt is unchanged.

Usage (from ``mats/environments``)::

  uv run exp_multi_agent_pipeline.py --treatment=prior-hack \
      --activity-logs=12,17 --seed-dir=p_hacking --seeds=all --epochs=5 \
      --harness=simple --pressure=low
  uv run exp_multi_agent_pipeline.py ... --dry-run

``--activity-log-files`` accepts exported
``environments-multi-agent-activity-log-v2`` JSON payloads. Every trajectory remains
stored, but the integrity layer excludes a run unless the complete numbered file is
mechanically verified in target-model context. Judge evidence mechanically replaces
the previous agent's log contents while retaining the current agent's actions.

``--retry-failed=<id> --retry-pipeline-failures`` explicitly includes completed cells
whose stored pipeline exit was nonzero or missing, as well as the ordinary unlaunched
and infrastructure-failure cells.

This script can make paid model calls and launch EC2 instances; ``--dry-run`` is free.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import traceback
from datetime import datetime

_ENVIRONMENTS = pathlib.Path(__file__).resolve().parent
for _path in (str(_ENVIRONMENTS / "lib"),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import exp_real_audit_pipeline as base_pipeline
from exp_aws_trajectory import (
    DEFAULT_INSTANCE_TYPE,
    DEFAULT_REGION,
    DEFAULT_VM_CONCURRENCY,
    AwsTrajectoryError,
    campaign_ok,
    resume_campaign,
    retry_failed,
    run_campaign,
    sha256_file,
)
from exp_inspect_runner import dead_targets, run_eval
from exp_real_audit import (
    DEFAULT_SANDBOX_CONCURRENCY,
    ported_members,
    resolve_gate_model,
    resolve_time_limit,
)
from exp_real_multi_agent import (
    build_multi_agent_cells,
    build_multi_agent_tasks,
    describe_plan,
    load_activity_log_specs,
    validate_treatment,
)
from exp_target_harness import resolve_harness
from judge_selection import resolve_judge
from model_catalog import resolve_target
from pipeline_cli import positive_int_arg, validate_cli_args, value_arg
from project_paths import DATA_ROOT as DATA, VIEWER_ROOT as OUT
from protocol_sources import (
    load_protocol_sources,
    reject_retired_fixed_system_prompt_flag,
    resolve_condition,
    resolve_pressure,
    resolve_seeds,
)


DEFAULT_CONCURRENCY = 50

_VALUE_FLAGS = {
    "--treatment", "--activity-logs", "--activity-log-files", "--seed-dir",
    "--seeds", "--epochs", "--harness", "--condition", "--pressure",
    "--judge", "--gate-model", "--concurrency", "--sandbox-concurrency",
    "--time-limit", "--compute", "--vm-concurrency", "--aws-region",
    "--aws-instance-type", "--aws-bucket", "--aws-secret-env",
    "--resume-campaign", "--retry-failed",
}
_SWITCH_FLAGS = {"--skip-viewer", "--dry-run", "--retry-pipeline-failures"}
_REJECTED_FLAGS = {
    "--reasoning": (
        "--reasoning is read from each activity-log source so the same agent "
        "configuration is replicated"
    ),
    "--targets": "--targets is read from each activity-log source",
    "--prefixes": "use --activity-logs=<viewer ids> for this fresh-session pipeline",
    "--prefix-files": "use --activity-log-files=<paths>",
    "--aws-setup": "--aws-setup lives in exp_real_audit_pipeline.py",
    "--aws-smoke-test": "--aws-smoke-test lives in exp_real_audit_pipeline.py",
}


def _arg(flag: str, default: str | None = None) -> str | None:
    return value_arg(sys.argv, flag, default)


def _posint(flag: str, default: int | None) -> int | None:
    return positive_int_arg(sys.argv, flag, default)


def _parse_ids(flag: str) -> list[int]:
    raw = _arg(flag)
    if raw is None:
        return []
    try:
        values = [int(item) for item in raw.split(",") if item.strip()]
    except ValueError as error:
        raise SystemExit(f"{flag} must be comma-separated integers") from error
    if not values:
        raise SystemExit(f"{flag} had no usable ids")
    return values


def _parse_paths(flag: str) -> list[str]:
    raw = _arg(flag)
    if raw is None:
        return []
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit(f"{flag} had no usable paths")
    return values


def _parse_args() -> dict:
    reject_retired_fixed_system_prompt_flag()
    validate_cli_args(
        sys.argv,
        value_flags=_VALUE_FLAGS,
        switch_flags=_SWITCH_FLAGS,
        rejected_flags=_REJECTED_FLAGS,
    )
    treatment_arg = _arg("--treatment")
    if treatment_arg is None:
        raise SystemExit("--treatment is required (for example prior-hack)")
    activity_ids = _parse_ids("--activity-logs")
    activity_files = _parse_paths("--activity-log-files")
    if not activity_ids and not activity_files:
        raise SystemExit(
            "supply --activity-logs=<viewer ids> and/or --activity-log-files=<paths>"
        )
    seed_dir = _arg("--seed-dir")
    if seed_dir is None:
        raise SystemExit("--seed-dir is required for the NEW task")
    seeds_path, available = resolve_seeds(seed_dir)
    ported = ported_members(seeds_path)
    seeds_arg = _arg("--seeds")
    if seeds_arg is None:
        raise SystemExit("--seeds is required; use --seeds=all or a comma list")
    if seeds_arg.strip() == "all":
        seeds = list(ported)
        if not seeds:
            raise SystemExit(f"no member of {seed_dir} has a real_env workspace")
    else:
        seeds = [item.strip() for item in seeds_arg.split(",") if item.strip()]
        unknown = sorted(set(seeds) - set(available))
        if unknown:
            raise SystemExit(f"unknown --seeds {unknown}; available: {available}")
    epochs = _posint("--epochs", None)
    if epochs is None:
        raise SystemExit("--epochs is required; use a positive integer")
    harness = resolve_harness(_arg("--harness"))
    condition = resolve_condition(_arg("--condition"))
    seed_path = pathlib.Path(seeds_path)
    family = (
        seed_path.parent.name
        if (seed_path / "manifest.json").is_file()
        else seed_path.name
    )
    if family in {"ml_prefix_only", "p_hacking_prefix_only"}:
        raise SystemExit(f"{family} is not a judged destination family")
    pressure = resolve_pressure(_arg("--pressure"), family)
    compute = _arg("--compute") or "aws"
    if compute not in {"aws", "local"}:
        raise SystemExit("--compute must be aws or local")
    judge_arg = _arg("--judge")
    return {
        "treatment": validate_treatment(treatment_arg),
        "activity_ids": activity_ids,
        "activity_files": activity_files,
        "seed_dir": seed_dir,
        "seeds_path": seeds_path,
        "seeds": seeds,
        "family": family,
        "epochs": epochs,
        "harness": harness,
        "condition": condition,
        "pressure": pressure,
        "judge": judge_arg,
        "judge_resolved": resolve_judge(judge_arg),
        "gate_model": resolve_gate_model(_arg("--gate-model"), judge_arg),
        "concurrency": _posint("--concurrency", DEFAULT_CONCURRENCY),
        "sandbox_concurrency": _posint(
            "--sandbox-concurrency", DEFAULT_SANDBOX_CONCURRENCY
        ),
        "time_limit": resolve_time_limit(_arg("--time-limit"), family),
        "skip_viewer": "--skip-viewer" in sys.argv,
        "compute": compute,
        "vm_concurrency": _posint("--vm-concurrency", DEFAULT_VM_CONCURRENCY),
        "aws_region": _arg("--aws-region", DEFAULT_REGION),
        "aws_instance_type": _arg("--aws-instance-type", DEFAULT_INSTANCE_TYPE),
        "aws_bucket": _arg("--aws-bucket"),
        "aws_secret_env": [
            name.strip()
            for name in (_arg("--aws-secret-env", "") or "").split(",")
            if name.strip()
        ],
        "dry_run": "--dry-run" in sys.argv,
        "reasoning": None,
    }


def _load_plan(cfg: dict):
    print("[plan] loading activity logs, validating invariants, building cells ...")
    specs = load_activity_log_specs(
        cfg["activity_ids"], cfg["activity_files"], harness=cfg["harness"]
    )
    cells = build_multi_agent_cells(specs, cfg["seeds_path"], cfg["seeds"])
    for unit_path in {cell.unit_path for cell in cells}:
        load_protocol_sources(unit_path, pressure=cfg["pressure"])
    print(describe_plan(cfg["treatment"], specs, cells))
    expected = len(cells) * cfg["epochs"]
    print(
        f"  {len(cells)} cell(s) x epochs={cfg['epochs']} = {expected} "
        "fresh-session treatment trajectory(ies)"
    )
    return specs, cells, expected


def run_local_stage(cfg: dict, cells: list) -> tuple[object, pathlib.Path, int, bool]:
    expected_n = len(cells) * cfg["epochs"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    names = {cell.activity_log.name for cell in cells}
    activity_label = next(iter(names)) if len(names) == 1 else f"{len(names)}logs"
    remote_run_name = os.environ.get("MATS_REMOTE_RUN_DIR")
    log_dir = DATA / "logs" / (
        remote_run_name
        or f"multi-agent-{cfg['treatment']}-{activity_label}-"
        f"{cfg['epochs']}ep-{timestamp}"
    )
    task_suffix = os.environ.get("MATS_REMOTE_TASK_SUFFIX")
    execution_metadata = None
    if os.environ.get("MATS_REMOTE_COMPUTE_JSON"):
        try:
            execution_metadata = json.loads(os.environ["MATS_REMOTE_COMPUTE_JSON"])
        except json.JSONDecodeError as error:
            raise SystemExit(f"invalid MATS_REMOTE_COMPUTE_JSON: {error}") from error

    print("=" * 72)
    print(f"STAGE 1/2  MULTI-AGENT ACTIVITY LOG + JUDGE  ->  {log_dir.name}")
    print("=" * 72)
    print(
        f"  treatment={cfg['treatment']}  seeds={cfg['seeds']}  "
        f"condition={cfg['condition']}  pressure={cfg['pressure']}"
    )
    print(
        f"  harness={cfg['harness']}  judge={cfg['judge_resolved']}  "
        f"gate={cfg['gate_model']}"
    )
    print(
        f"  expected={expected_n}  epochs={cfg['epochs']}  "
        f"concurrency={cfg['concurrency']}  "
        f"sandbox_concurrency={cfg['sandbox_concurrency']}  "
        f"time_limit={cfg['time_limit']}s"
    )
    print(
        "  fresh sandbox/session; unchanged system prompt; ACTIVITY_LOG.md mounted; "
        "prior-agent contents omitted from judge evidence\n"
    )
    target_models = sorted({
        resolve_target(cell.activity_log.target_name) for cell in cells
    })
    try:
        tasks = build_multi_agent_tasks(
            cells,
            treatment=cfg["treatment"],
            run_label=log_dir.name,
            condition=cfg["condition"],
            gate_model=cfg["gate_model"],
            judge=cfg["judge"],
            artifacts_root=log_dir / "real_artifacts",
            task_id_suffix=task_suffix,
            execution_metadata=execution_metadata,
            harness=cfg["harness"],
            pressure=cfg["pressure"],
        )
        success, logs = run_eval(
            tasks,
            cfg["epochs"],
            cfg["concurrency"],
            log_dir,
            max_sandboxes=cfg["sandbox_concurrency"],
            time_limit=cfg["time_limit"],
        )
    except SystemExit:
        raise
    except Exception as error:
        print("\n!! MULTI-AGENT STAGE CRASHED (viewer will use existing logs):")
        print(f"   {type(error).__name__}: {error}")
        traceback.print_exc()
        return None, log_dir, expected_n, False

    actual_n = sum(len(log.samples or []) for log in logs)
    for log in logs:
        print(
            f"  {log.eval.task}: status={log.status}, "
            f"samples={len(log.samples or [])}"
        )
    dead = dead_targets(logs, target_models)
    records = base_pipeline.write_integrity_sidecar(log_dir, logs)
    failures = [record for record in records if record["status"] == "excluded"]
    if actual_n != expected_n:
        print(f"WARNING: expected {expected_n} trajectories but found {actual_n}")
    if dead:
        print(f"WARNING: agent(s) with no output: {dead}")
    if failures:
        print(f"WARNING: {len(failures)} trajectory(ies) failed integrity")
        for failure in failures:
            print(
                f"  {failure['task']}/{failure['sample']} epoch "
                f"{failure['epoch']}: {', '.join(failure['issues'])}"
            )
    integrity_ok = (
        bool(success)
        and actual_n == expected_n
        and not dead
        and not failures
    )
    return logs, log_dir, expected_n, integrity_ok


def _aws_payload_descriptors(specs: list) -> list[dict]:
    return [
        {
            "name": spec.name,
            "sha256": spec.sha256,
            "file_sha256": sha256_file(spec.payload_path),
            "local_path": str(spec.payload_path),
            "target": spec.target_name,
            "target_model": resolve_target(spec.target_name),
            "reasoning": spec.reasoning,
        }
        for spec in specs
    ]


def main() -> None:
    try:
        resume_id = _arg("--resume-campaign")
        retry_id = _arg("--retry-failed")
        if "--retry-pipeline-failures" in sys.argv and not retry_id:
            raise SystemExit(
                "--retry-pipeline-failures requires --retry-failed=<campaign>"
            )
        if resume_id and retry_id:
            raise SystemExit(
                "--resume-campaign and --retry-failed are mutually exclusive"
            )
        if resume_id or retry_id:
            control = base_pipeline.parse_aws_control_args()
            if resume_id:
                if control["dry_run"]:
                    raise SystemExit("--dry-run is not meaningful with --resume-campaign")
                state = resume_campaign(control, _ENVIRONMENTS, DATA, resume_id)
            else:
                control["harness"] = resolve_harness(control.get("harness"))
                state = retry_failed(
                    control, _ENVIRONMENTS, DATA, retry_id, dry_run=control["dry_run"]
                )
                if control["dry_run"]:
                    return
            post_ok = asyncio.run(base_pipeline.run_env_post_stages({
                "skip_viewer": control["skip_viewer"]
            }))
            if not (campaign_ok(state) and post_ok):
                raise SystemExit(1)
            return

        cfg = _parse_args()
        specs, cells, _expected = _load_plan(cfg)
        if cfg["compute"] == "aws":
            cfg["multi_agent"] = {
                "treatment": cfg["treatment"],
                "payloads": _aws_payload_descriptors(specs),
            }
            targets = {spec.target_name: resolve_target(spec.target_name) for spec in specs}
            cfg["targets"] = list(targets)
            cfg["target_models"] = list(targets.values())
            state = run_campaign(cfg, _ENVIRONMENTS, DATA, dry_run=cfg["dry_run"])
            if cfg["dry_run"]:
                return
            audit_ok = campaign_ok(state)
        else:
            if cfg["dry_run"]:
                print("\n[dry-run] validated; no generation, judge, or compute cost.")
                return
            base_pipeline.require_docker()
            _logs, _log_dir, _expected_n, audit_ok = run_local_stage(cfg, cells)
        post_ok = asyncio.run(base_pipeline.run_env_post_stages(cfg))
    except AwsTrajectoryError as error:
        raise SystemExit(f"AWS trajectory error: {error}") from error

    print("\n" + "=" * 72)
    if audit_ok and post_ok:
        print(f"PIPELINE DONE.  open {OUT / 'index.html'}")
    else:
        print(
            "PIPELINE FINISHED WITH INTEGRITY FAILURES. Inspect "
            f"{OUT / 'index.html'}."
        )
    print("=" * 72)
    if not (audit_ok and post_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
