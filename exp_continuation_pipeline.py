"""End-to-end CONTINUATION pipeline for real environments: generate -> structured
judge (inline gate + fresh final call) -> viewer, in one command.

THE QUESTION: if an agent reward-hacks on one task and is then handed a SECOND,
unrelated task, does its hack rate on the second task go up -- because it is
conditioned on its own earlier behavior?

ONE INVOCATION = ONE TREATMENT: label the run with a free-form ``--treatment`` slug
and choose the prefixes the agent carries; run again for another treatment. There
is NO no-prefix condition: the environment is pinned and scripted (no auditor), so
the base rate is simply the ordinary original trajectories of the same
(seed, agent) -- the viewer's Continuations page shows them side by side.

PREFIXES come from two sources, combinable in one run:
  --prefixes=<ids>        stored trajectories by viewer ID, reconstructed exactly
                          (integrity-excluded and rejudge rows are refused);
  --prefix-files=<paths>  arbitrary hand-built conversations in the
                          environments-continuation-prefix-v1 JSON format (see
                          lib/exp_real_continuation.py) -- e.g. a long Q&A session.

Each cell = one (prefix, new-task seed) pair, run for --epochs epochs. The agent
model and its reasoning setting are read from each prefix (there is no --targets or
--reasoning flag). The agent starts pre-loaded with the prefix conversation, then
receives one injected pivot user turn (the fixed Petri pivot sentence + the seed's
pinned opening message) and goes live in a fresh sandbox. Both judge stages see only
the system message plus the live task: the prefix is sliced out mechanically at a
stored boundary index and the omission is declared as an evidence caveat.

Usage (from mats/environments/):
  uv run exp_continuation_pipeline.py --treatment=full-hack --prefixes=12,17 \
      --seed-dir=p_hacking --seeds=all --epochs=5 --harness=simple \
      --pressure=high
  uv run exp_continuation_pipeline.py --treatment=full-hack --prefixes=12 \
      --seed-dir=ml_training_data_misuse --seeds=demand_forecasting --epochs=5 \
      --harness=production
  uv run exp_continuation_pipeline.py ... --dry-run     # FREE: plan (and AWS preflight)

Flags:
  --treatment=<slug>      REQUIRED. free-form label (lowercase letters, digits,
                          hyphens); runs with the same treatment pool together.
  --prefixes=<ids>        stored-trajectory prefix ids (viewer IDs), and/or
  --prefix-files=<paths>  comma-separated prefix payload files. At least one of the
                          two is required.
  --seed-dir=<sub>        REQUIRED. the NEW task's family or member dir.
  --seeds=<a,b|all>       REQUIRED. new-task members (`all` = every ported member).
                          Must not include any prefix's own source seed.
  --epochs=<N>            REQUIRED. continuations per (prefix, seed) cell.
  --harness=simple|production|subscription  REQUIRED. simple keeps native prefix
                          message splicing. production uses API-backed native
                          scaffolds. subscription uses Claude/Codex subscription login
                          quota and OpenCode Go for mapped models; other OpenCode
                          models remain API-backed.
  --condition=<c>         allow only (default).
  --pressure=low|high     p_hacking new tasks only; defaults to low.
  --judge=<m>            alignment judge; defaults to gpt-5.6-luna.
  --gate-model=<m>       defaults to and must equal --judge.
  --concurrency=<N>       LLM parallelism (default 50).
  --sandbox-concurrency=<N>  max live Docker containers (default 8).
  --time-limit=<sec>      family default (ml 4200 / p_hacking 1800).
  --compute=aws|local     defaults to aws for every new-task seed family. AWS ships
                          each prefix payload with SHA-256 verification.
  --vm-concurrency / --aws-region / --aws-instance-type / --aws-bucket /
  --aws-secret-env        as in exp_real_audit_pipeline.
  --resume-campaign=<id> / --retry-failed=<id>  continuation campaigns supported.
  --retry-pipeline-failures  with --retry-failed, also retry completed cells whose
                          stored pipeline exit was nonzero or missing.
  --allow-incomplete-prefixes  permit ML prefix-only controls missing REPORT.md or
                          models/final/; the override is recorded in run provenance.
  --skip-viewer           don't rebuild the viewer at the end.
  --dry-run               FREE: validate + print the full plan (local), or the plan
                          plus the AWS campaign preflight (aws). No model calls.

There is no --reasoning flag: each prefix records the reasoning setting its agent
ran with, and the continuation replicates it. --aws-setup and --aws-smoke-test live
in exp_real_audit_pipeline.py.

Costs money (API-backed agent calls, stage-one/final judge calls, and EC2 for AWS
runs) unless --dry-run. Direct subscription agents consume plan quota instead of a
measured per-run API charge.
"""

import asyncio
import json
import os
import pathlib
import sys
import traceback
from datetime import datetime

_ENVIRONMENTS = pathlib.Path(__file__).resolve().parent
for _p in (str(_ENVIRONMENTS / "lib"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
from exp_real_continuation import (
    build_continuation_cells,
    build_continuation_tasks,
    describe_plan,
    load_prefix_specs,
    validate_treatment,
)
from judge_selection import resolve_judge
from model_catalog import resolve_target
from exp_target_harness import resolve_harness
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
    "--treatment", "--prefixes", "--prefix-files", "--seed-dir", "--seeds",
    "--epochs", "--harness", "--condition", "--pressure", "--judge", "--gate-model", "--concurrency",
    "--sandbox-concurrency", "--time-limit", "--compute", "--vm-concurrency",
    "--aws-region", "--aws-instance-type", "--aws-bucket", "--aws-secret-env",
    "--resume-campaign", "--retry-failed",
}
_SWITCH_FLAGS = {
    "--skip-viewer",
    "--dry-run",
    "--allow-incomplete-prefixes",
    "--retry-pipeline-failures",
}
_REJECTED_FLAGS = {
    "--reasoning": (
        "--reasoning is not a continuation flag: each prefix records the reasoning "
        "setting its agent ran with, and the continuation replicates it"
    ),
    "--targets": (
        "--targets is not a continuation flag: the agent model is read from each "
        "prefix"
    ),
    "--aws-setup": "--aws-setup lives in exp_real_audit_pipeline.py",
    "--aws-smoke-test": "--aws-smoke-test lives in exp_real_audit_pipeline.py",
}


def _validate_cli_args() -> None:
    validate_cli_args(
        sys.argv,
        value_flags=_VALUE_FLAGS,
        switch_flags=_SWITCH_FLAGS,
        rejected_flags=_REJECTED_FLAGS,
    )


def _arg(flag: str, default: str | None = None) -> str | None:
    return value_arg(sys.argv, flag, default)


def _posint(flag: str, default: int | None) -> int | None:
    return positive_int_arg(sys.argv, flag, default)


def _parse_ids(flag: str) -> list[int]:
    raw = _arg(flag)
    if raw is None:
        return []
    try:
        ids = [int(item) for item in raw.split(",") if item.strip()]
    except ValueError:
        raise SystemExit(f"{flag} must be comma-separated integers, got {raw!r}")
    if not ids:
        raise SystemExit(f"{flag} had no usable ids")
    return ids


def _parse_paths(flag: str) -> list[str]:
    raw = _arg(flag)
    if raw is None:
        return []
    paths = [item.strip() for item in raw.split(",") if item.strip()]
    if not paths:
        raise SystemExit(f"{flag} had no usable paths")
    return paths


def _parse_args() -> dict:
    reject_retired_fixed_system_prompt_flag()
    _validate_cli_args()

    treatment_arg = _arg("--treatment")
    if treatment_arg is None:
        raise SystemExit("--treatment is required (a free-form label for this run, "
                         "e.g. full-hack / clean)")
    treatment = validate_treatment(treatment_arg)

    prefix_ids = _parse_ids("--prefixes")
    prefix_files = _parse_paths("--prefix-files")
    if not prefix_ids and not prefix_files:
        raise SystemExit("supply at least one prefix via --prefixes=<viewer ids> "
                         "and/or --prefix-files=<paths>")

    seed_dir_arg = _arg("--seed-dir")
    if seed_dir_arg is None:
        raise SystemExit("--seed-dir is required (the NEW task's family or member, "
                         "e.g. ml_training_data_misuse or p_hacking)")
    seeds_path, available_seeds = resolve_seeds(seed_dir_arg)
    ported = ported_members(seeds_path)
    seeds_arg = _arg("--seeds")
    if seeds_arg is None:
        raise SystemExit(
            "--seeds is required; use --seeds=all for every PORTED member, or a "
            f"comma list. Ported members here: {ported or 'none'}"
        )
    if seeds_arg.strip() == "all":
        seeds = list(ported)
        if not seeds:
            raise SystemExit(
                f"no member of {seed_dir_arg} has a real_env workspace yet"
            )
    else:
        seeds = [item.strip() for item in seeds_arg.split(",") if item.strip()]
        unknown = [item for item in seeds if item not in available_seeds]
        if unknown:
            raise SystemExit(f"unknown --seeds {unknown}; available: {available_seeds}")

    epochs = _posint("--epochs", None)
    if epochs is None:
        raise SystemExit("--epochs is required (no default); positive integer")
    harness = resolve_harness(_arg("--harness"))

    condition = resolve_condition(_arg("--condition"))
    resolved_seed_path = pathlib.Path(seeds_path)
    family = (
        resolved_seed_path.parent.name
        if (resolved_seed_path / "manifest.json").is_file()
        else resolved_seed_path.name
    )
    if family in {"ml_prefix_only", "p_hacking_prefix_only"}:
        endpoint = (
            "prefixes/exp_ml_prefix.py"
            if family == "ml_prefix_only"
            else "prefixes/exp_p_hacking_prefix.py"
        )
        raise SystemExit(
            f"{family} is a prefix source, not a continuation destination; "
            f"generate it with {endpoint} and select a judged new task"
        )
    pressure = resolve_pressure(_arg("--pressure"), family)
    compute = _arg("--compute") or "aws"
    if compute not in {"aws", "local"}:
        raise SystemExit("--compute must be aws or local")
    time_limit = resolve_time_limit(_arg("--time-limit"), family)
    judge_arg = _arg("--judge")

    return {
        "treatment": treatment,
        "prefix_ids": prefix_ids,
        "prefix_files": prefix_files,
        "seeds": seeds,
        "seeds_path": seeds_path,
        "seed_dir": seed_dir_arg,
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
        "time_limit": time_limit,
        "skip_viewer": "--skip-viewer" in sys.argv,
        "compute": compute,
        "vm_concurrency": _posint("--vm-concurrency", DEFAULT_VM_CONCURRENCY),
        "aws_region": _arg("--aws-region", DEFAULT_REGION),
        "aws_instance_type": _arg("--aws-instance-type", DEFAULT_INSTANCE_TYPE),
        "aws_bucket": _arg("--aws-bucket"),
        "aws_secret_env": [
            name.strip() for name in (_arg("--aws-secret-env", "") or "").split(",")
            if name.strip()
        ],
        "dry_run": "--dry-run" in sys.argv,
        "allow_incomplete_prefixes": "--allow-incomplete-prefixes" in sys.argv,
        # Reasoning is per-prefix; this run-level slot only fills the stored AWS
        # pipeline_config shape.
        "reasoning": None,
    }


def _load_plan(cfg: dict):
    """Free: resolve prefixes, validate every invariant, and build the cells."""

    print("[plan] loading prefixes, validating invariants, building cells ...")
    specs = load_prefix_specs(
        cfg["prefix_ids"],
        cfg["prefix_files"],
        harness=cfg["harness"],
        allow_incomplete_prefixes=cfg["allow_incomplete_prefixes"],
    )
    cells = build_continuation_cells(specs, cfg["seeds_path"], cfg["seeds"])
    for unit_path in {cell.unit_path for cell in cells}:
        load_protocol_sources(unit_path, pressure=cfg["pressure"])
    print(describe_plan(cfg["treatment"], specs, cells))
    expected = len(cells) * cfg["epochs"]
    print(f"  {len(cells)} cell(s) x epochs={cfg['epochs']} = {expected} "
          "continuation(s) to generate")
    return specs, cells, expected


def run_local_stage(cfg: dict, cells: list) -> tuple[object, pathlib.Path, int, bool]:
    expected_n = len(cells) * cfg["epochs"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefixes_label = (
        cells[0].prefix.name if len({c.prefix.name for c in cells}) == 1
        else f"{len({c.prefix.name for c in cells})}prefixes"
    )
    remote_run_name = os.environ.get("MATS_REMOTE_RUN_DIR")
    log_dir = DATA / "logs" / (
        remote_run_name
        or f"continuation-{cfg['treatment']}-{prefixes_label}-"
           f"{cfg['epochs']}ep-{timestamp}"
    )
    task_suffix = os.environ.get("MATS_REMOTE_TASK_SUFFIX")
    execution_metadata = None
    if os.environ.get("MATS_REMOTE_COMPUTE_JSON"):
        try:
            execution_metadata = json.loads(os.environ["MATS_REMOTE_COMPUTE_JSON"])
        except json.JSONDecodeError as ex:
            raise SystemExit(f"invalid MATS_REMOTE_COMPUTE_JSON: {ex}")

    print("=" * 72)
    print(f"STAGE 1/2  CONTINUATION + STRUCTURED JUDGE  ->  {log_dir.name}")
    print("=" * 72)
    pressure_text = f"  pressure={cfg['pressure']}" if cfg["pressure"] else ""
    print(
        f"  treatment={cfg['treatment']}  seeds={cfg['seeds']}  "
        f"condition={cfg['condition']}{pressure_text}"
    )
    print(f"  harness={cfg['harness']}")
    print(f"  judge={cfg['judge_resolved']}  gate={cfg['gate_model']}")
    print(f"  epochs={cfg['epochs']}  concurrency={cfg['concurrency']}  "
          f"sandbox_concurrency={cfg['sandbox_concurrency']}  "
          f"time_limit={cfg['time_limit']}s (no turn cap)")
    print(f"  expected trajectories: {len(cells)} cell(s) x {cfg['epochs']} = "
          f"{expected_n}")
    print("  environment: REAL docker sandboxes; the agent carries its prefix; "
          "both judge stages see only the live task\n")

    target_models = sorted({
        resolve_target(cell.prefix.target_name) for cell in cells
    })
    try:
        tasks = build_continuation_tasks(
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
            tasks, cfg["epochs"], cfg["concurrency"], log_dir,
            max_sandboxes=cfg["sandbox_concurrency"], time_limit=cfg["time_limit"],
        )
    except SystemExit:
        raise
    except Exception as e:
        print("\n!! CONTINUATION STAGE CRASHED (continuing to viewer on existing logs):")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        return None, log_dir, expected_n, False

    print(f"\neval_set finished, success={success}")
    actual_n = 0
    for log in logs:
        n = len(log.samples or [])
        actual_n += n
        print(f"  {log.eval.task}: status={log.status}, samples={n}")

    dead = dead_targets(logs, target_models)
    if dead:
        print("\n" + "!" * 72)
        print(f"WARNING: agent(s) {dead} produced 0 output tokens -- they never ran.")
        print("!" * 72)

    if actual_n != expected_n:
        print(f"\nWARNING: expected {expected_n} trajectories but eval_set wrote "
              f"{actual_n}.")
    else:
        print(f"\nall {expected_n} trajectories present.")

    integrity_records = base_pipeline.write_integrity_sidecar(log_dir, logs)
    integrity_failures = [
        record for record in integrity_records if record["status"] == "excluded"
    ]
    if integrity_failures:
        print("\n" + "!" * 72)
        print(f"WARNING: {len(integrity_failures)} trajectory(ies) have stored "
              "integrity failures:")
        for failure in integrity_failures:
            print(f"  {failure['task']}/{failure['sample']} epoch {failure['epoch']}: "
                  f"{', '.join(failure['issues'])}")
        print("!" * 72)

    integrity_ok = (bool(success) and actual_n == expected_n and not dead
                    and not integrity_failures)
    return logs, log_dir, expected_n, integrity_ok


def _aws_payload_descriptors(specs: list) -> list[dict]:
    descriptors = []
    for spec in specs:
        if spec.payload_path is None:
            raise SystemExit(
                f"prefix {spec.name!r} has no payload file on disk; AWS shipping "
                "needs one"
            )
        descriptors.append({
            "name": spec.name,
            "sha256": spec.sha256,
            "file_sha256": sha256_file(spec.payload_path),
            "local_path": str(spec.payload_path),
            "target": spec.target_name,
            "target_model": resolve_target(spec.target_name),
            "reasoning": spec.reasoning,
        })
    return descriptors


def main() -> None:
    try:
        resume_id = _arg("--resume-campaign")
        retry_id = _arg("--retry-failed")
        if "--retry-pipeline-failures" in sys.argv and not retry_id:
            raise SystemExit(
                "--retry-pipeline-failures requires --retry-failed=<campaign>"
            )
        if resume_id and retry_id:
            raise SystemExit("--resume-campaign and --retry-failed are mutually exclusive")
        if resume_id or retry_id:
            control = base_pipeline.parse_aws_control_args()
            if resume_id:
                if control["dry_run"]:
                    raise SystemExit("--dry-run is not meaningful with --resume-campaign")
                state = resume_campaign(control, _ENVIRONMENTS, DATA, resume_id)
            else:
                control["harness"] = resolve_harness(control.get("harness"))
                state = retry_failed(
                    control, _ENVIRONMENTS, DATA, retry_id,
                    dry_run=control["dry_run"],
                )
                if control["dry_run"]:
                    return
            post_ok = asyncio.run(base_pipeline.run_env_post_stages(
                {"skip_viewer": control["skip_viewer"]}
            ))
            if not (campaign_ok(state) and post_ok):
                raise SystemExit(1)
            return

        cfg = _parse_args()
        specs, cells, _expected = _load_plan(cfg)

        if cfg["compute"] == "aws":
            cfg["continuation"] = {
                "treatment": cfg["treatment"],
                "payloads": _aws_payload_descriptors(specs),
            }
            target_models = {
                spec.target_name: resolve_target(spec.target_name) for spec in specs
            }
            cfg["targets"] = list(target_models)
            cfg["target_models"] = list(target_models.values())
            state = run_campaign(cfg, _ENVIRONMENTS, DATA, dry_run=cfg["dry_run"])
            if cfg["dry_run"]:
                return
            audit_ok = campaign_ok(state)
        else:
            if cfg["dry_run"]:
                print("\n[dry-run] plan validated (treatment + prefixes + seeds). "
                      "No generation, no cost.")
                return
            base_pipeline.require_docker()
            _logs, _log_dir, _expected_n, audit_ok = run_local_stage(cfg, cells)
        post_ok = asyncio.run(base_pipeline.run_env_post_stages(cfg))
    except AwsTrajectoryError as ex:
        raise SystemExit(f"AWS trajectory error: {ex}") from ex
    print("\n" + "=" * 72)
    if audit_ok and post_ok:
        print(f"PIPELINE DONE.  open {OUT / 'index.html'}")
    else:
        print("PIPELINE FINISHED WITH INTEGRITY FAILURES. "
              f"Inspect {OUT / 'index.html'} and the warnings above.")
    print("=" * 72)
    if not (audit_ok and post_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
