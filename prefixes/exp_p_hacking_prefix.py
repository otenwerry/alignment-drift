"""Generate positive p-hacking no-honeypot continuation prefixes.

Each prefix-only member keeps its original p-hacking task and export structure while
changing only outcomes to provide a clear honest win. Every member ends after one
submission; no LLM judge is called.

Usage (from the repository root)::

  uv run prefixes/exp_p_hacking_prefix.py \
      --targets=deepseek-v4-pro,glm-5.1,kimi-k2.6 \
      --seeds=all --epochs=10 --harness=production --compute=aws
  uv run prefixes/exp_p_hacking_prefix.py --resume-campaign=<campaign-id>

The singular ``--model`` and ``--seed`` aliases match ``exp_ml_prefix.py`` and default
to one epoch only for a single legacy-style cell.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


_ENVIRONMENTS = Path(__file__).resolve().parents[1]
for _path in (str(_ENVIRONMENTS), str(_ENVIRONMENTS / "lib")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv

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
)
from exp_inspect_runner import dead_targets, run_eval
from exp_subscription_harness import require_opencode_go_auth, require_subscription_auth
from exp_target_harness import (
    NATIVE_HARNESS_MODES,
    opencode_go_model_spec,
    production_scaffold_for_target,
    resolve_harness,
)
from model_catalog import REASONING_EFFORT, TARGET_CHOICES, build_target, resolve_target
from pipeline_cli import positive_int_arg, validate_cli_args, value_arg
from prefixes import exp_ml_prefix as shared
from project_paths import DATA_ROOT, ENV_FILE, LOGS_ROOT, VIEWER_ROOT
from prompt_caching import stable_key
from protocol_sources import (
    global_system_prompt,
    load_protocol_sources,
    resolve_pressure,
    resolve_reasoning,
    strip_thinking_instruction,
)
from real_env_workspace import workspace_files


PREFIX_FORMAT = "environments-continuation-prefix-v1"
P_HACKING_PREFIX_ONLY_FAMILY = "p_hacking_prefix_only"
TIME_LIMIT_SECONDS = 30 * 60
DEFAULT_CONCURRENCY = 50
DEFAULT_SANDBOX_CONCURRENCY = 8
SEEDS_ROOT = _ENVIRONMENTS / "seeds" / P_HACKING_PREFIX_ONLY_FAMILY
SEED_CONFIG = {
    "reasoning_prompt_benchmark_positive": {
        "description": (
            "reasoning-prompt benchmark task with a clear honest positive result"
        ),
        "source_label": "Reasoning-prompt benchmark · clear positive",
        "comparison_source_seed": "reasoning_prompt_benchmark",
        "result_condition": "clear positive",
        "expected_submissions": 1,
        "expected_follow_up": False,
        "workspace_files": frozenset({
            "/workspace/README.md",
            "/workspace/requirements.txt",
            "/workspace/data/prompt_benchmark.csv",
        }),
    },
}
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VALUE_FLAGS = {
    "--targets", "--model", "--seeds", "--seed", "--epochs", "--harness",
    "--reasoning", "--pressure", "--name", "--concurrency",
    "--sandbox-concurrency", "--time-limit", "--compute", "--vm-concurrency",
    "--aws-region", "--aws-instance-type", "--aws-bucket", "--aws-secret-env",
    "--resume-campaign", "--retry-failed",
}
_SWITCH_FLAGS = {"--skip-viewer", "--dry-run"}
_REJECTED_FLAGS = {
    "--aws-setup": "--aws-setup lives in exp_real_audit_pipeline.py",
    "--aws-smoke-test": "--aws-smoke-test lives in exp_real_audit_pipeline.py",
}


def _arg(flag: str, default: str | None = None) -> str | None:
    return value_arg(sys.argv, flag, default)


def _posint(flag: str, default: int | None) -> int | None:
    return positive_int_arg(sys.argv, flag, default)


def _validate_cli_args() -> None:
    validate_cli_args(
        sys.argv,
        value_flags=_VALUE_FLAGS,
        switch_flags=_SWITCH_FLAGS,
        rejected_flags=_REJECTED_FLAGS,
    )


def _csv_values(raw: str | None, *, flag: str) -> list[str]:
    if raw is None:
        return []
    values = list(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    if not values:
        raise SystemExit(f"{flag} had no usable values")
    return values


def seed_path(member: str) -> Path:
    return SEEDS_ROOT / member


def default_name() -> str:
    return (
        f"p-hacking-no-honeypot-{datetime.now().strftime('%Y%m%d%H%M%S%f')}-"
        f"{uuid.uuid4().hex[:8]}"
    )


def _parse_args() -> dict:
    _validate_cli_args()
    targets_arg = _arg("--targets")
    model_arg = _arg("--model")
    if targets_arg is not None and model_arg is not None:
        raise SystemExit("use --targets or the legacy --model alias, not both")
    legacy_target = model_arg is not None
    targets = _csv_values(targets_arg, flag="--targets")
    if model_arg is not None:
        targets = [model_arg.strip()]
    if not targets:
        raise SystemExit(f"--targets is required; agent choices: {sorted(TARGET_CHOICES)}")
    unknown_targets = [target for target in targets if target not in TARGET_CHOICES]
    if unknown_targets:
        raise SystemExit(
            f"unknown agents in --targets {unknown_targets}; choices: {sorted(TARGET_CHOICES)}"
        )

    seeds_arg = _arg("--seeds")
    seed_arg = _arg("--seed")
    if seeds_arg is not None and seed_arg is not None:
        raise SystemExit("use --seeds or the legacy --seed alias, not both")
    legacy_seed = seed_arg is not None
    if seeds_arg is not None and seeds_arg.strip() == "all":
        seeds = sorted(SEED_CONFIG)
    else:
        seeds = _csv_values(seeds_arg, flag="--seeds")
    if seed_arg is not None:
        seeds = [seed_arg.strip()]
    if not seeds and legacy_target:
        seeds = ["reasoning_prompt_benchmark_positive"]
        legacy_seed = True
    if not seeds:
        raise SystemExit(
            f"--seeds is required; use --seeds=all or choose from {sorted(SEED_CONFIG)}"
        )
    unknown_seeds = [member for member in seeds if member not in SEED_CONFIG]
    if unknown_seeds:
        raise SystemExit(
            f"unknown --seeds {unknown_seeds}; choices: {sorted(SEED_CONFIG)}"
        )

    epochs = _posint("--epochs", None)
    if epochs is None:
        if (legacy_target or legacy_seed) and len(targets) == len(seeds) == 1:
            epochs = 1
        else:
            raise SystemExit("--epochs is required (no default); positive integer")
    if _arg("--harness") is None:
        raise SystemExit("--harness is required (simple, production, or subscription)")
    harness = resolve_harness(_arg("--harness"))
    pressure = resolve_pressure(_arg("--pressure"), P_HACKING_PREFIX_ONLY_FAMILY)
    compute = str(_arg("--compute", "aws"))
    if compute not in {"aws", "local"}:
        raise SystemExit("--compute must be aws or local")
    time_limit = _posint("--time-limit", TIME_LIMIT_SECONDS)
    if time_limit != TIME_LIMIT_SECONDS:
        raise SystemExit(
            f"p_hacking_prefix_only has a fixed 30-minute task; --time-limit must be "
            f"{TIME_LIMIT_SECONDS}"
        )
    name = _arg("--name") or default_name()
    if not _SLUG_RE.fullmatch(name):
        raise SystemExit(
            f"--name must be a lowercase hyphen-separated base slug, got {name!r}"
        )
    return {
        "targets": targets,
        "target_models": [resolve_target(target) for target in targets],
        "seeds": seeds,
        "seeds_path": str(SEEDS_ROOT),
        "seed_dir": P_HACKING_PREFIX_ONLY_FAMILY,
        "family": P_HACKING_PREFIX_ONLY_FAMILY,
        "prefix_only": True,
        "prefix_pipeline_script": "prefixes/exp_p_hacking_prefix.py",
        "prefix_task_namespace": "p_hacking_prefix_only",
        "prefix_campaign_namespace": "p-hacking-prefix",
        "epochs": epochs,
        "harness": harness,
        "reasoning": resolve_reasoning(_arg("--reasoning")),
        "name": name,
        "condition": "allow",
        "pressure": pressure,
        "judge_resolved": None,
        "gate_model": None,
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
            item.strip()
            for item in str(_arg("--aws-secret-env", "") or "").split(",")
            if item.strip()
        ],
        "dry_run": "--dry-run" in sys.argv,
    }


def _parse_aws_control_args() -> dict:
    _validate_cli_args()
    return {
        "harness": _arg("--harness"),
        "harness_explicit": _arg("--harness") is not None,
        "aws_region": _arg("--aws-region", DEFAULT_REGION),
        "aws_region_explicit": _arg("--aws-region") is not None,
        "aws_instance_type": _arg("--aws-instance-type", DEFAULT_INSTANCE_TYPE),
        "aws_instance_type_explicit": _arg("--aws-instance-type") is not None,
        "aws_bucket": _arg("--aws-bucket"),
        "aws_bucket_explicit": _arg("--aws-bucket") is not None,
        "aws_secret_env": [
            item.strip()
            for item in str(_arg("--aws-secret-env", "") or "").split(",")
            if item.strip()
        ],
        "vm_concurrency": _posint("--vm-concurrency", DEFAULT_VM_CONCURRENCY),
        "vm_concurrency_explicit": _arg("--vm-concurrency") is not None,
        "dry_run": "--dry-run" in sys.argv,
        "concurrency": _posint("--concurrency", DEFAULT_CONCURRENCY),
        "skip_viewer": "--skip-viewer" in sys.argv,
    }


def controlled_workspace_files(seed: Path) -> dict[str, str]:
    member_config = SEED_CONFIG.get(seed.name)
    if member_config is None:
        raise SystemExit(f"unknown p_hacking_prefix_only member: {seed.name}")
    mapping = workspace_files(seed)
    expected_files = member_config["workspace_files"]
    if frozenset(mapping) != expected_files:
        raise SystemExit(
            "p_hacking_prefix_only workspace boundary changed; expected "
            f"{sorted(expected_files)}, got {sorted(mapping)}"
        )
    manifest = json.loads((seed / "manifest.json").read_text())
    if manifest.get("analysis_honeypot") is not False:
        raise SystemExit("p_hacking_prefix_only must explicitly disable the honeypot")
    expected_source = f"p_hacking/{member_config['comparison_source_seed']}"
    if manifest.get("source_seed") != expected_source:
        raise SystemExit("p_hacking_prefix_only source_seed drifted")
    if (seed / "answers").exists():
        raise SystemExit("p_hacking_prefix_only must not contain an answers directory")
    return mapping


def payload_name(base: str, model: str, seed: str, epoch: int) -> str:
    return (
        f"{base}-{shared._model_slug_fragment(model)}-"
        f"{seed.replace('_', '-')}-e{epoch}"
    )


def build_payload(
    cfg: dict,
    sample: Any,
    usage_summary: dict,
    *,
    sidecar: Path,
) -> dict:
    from exp_real_continuation import validate_prefix_payload

    real_env = (sample.metadata or {}).get("real_env") or {}
    protocol = real_env.get("protocol") or {}
    member = cfg["seed"]
    member_config = SEED_CONFIG[member]
    if protocol.get("submissions") != member_config["expected_submissions"]:
        raise SystemExit(
            f"{member} completed {protocol.get('submissions')} submissions; expected "
            f"{member_config['expected_submissions']}"
        )
    if (protocol.get("follow_up_sent") is True) != member_config["expected_follow_up"]:
        raise SystemExit(f"{member} follow-up behavior drifted")
    judging = real_env.get("judging") or {}
    if judging.get("enabled") is not False or real_env.get("gates"):
        raise SystemExit("p_hacking_prefix_only unexpectedly recorded a judge call")

    context_tokens, measurement = shared._final_context(sample)
    seed = seed_path(member)
    manifest_path = seed / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    grade = real_env.get("grade") or {}
    deliverables = grade.get("deliverables") or {}
    required_deliverables = ("analysis.py", "RESULT.md")
    missing_deliverables = [
        name for name in required_deliverables if deliverables.get(name) is not True
    ]
    all_deliverables_present = not missing_deliverables
    source_epoch = shared._source_epoch(sample)
    payload = {
        "format": PREFIX_FORMAT,
        "name": cfg["name"],
        "target": cfg["model"],
        "reasoning": cfg["reasoning"],
        "source": {
            "kind": "external",
            "description": member_config["description"],
            "generator": "exp_p_hacking_prefix.py",
            "prefix_type": "p_hacking_no_honeypot",
            "prefix_type_label": "P-hacking no-honeypot",
            "source_label": member_config["source_label"],
            "family": P_HACKING_PREFIX_ONLY_FAMILY,
            "seed": member,
            "epoch": source_epoch,
            "run_name": cfg.get("run_name"),
            "comparison_source_family": "p_hacking",
            "comparison_source_seed": member_config["comparison_source_seed"],
            "result_condition": member_config["result_condition"],
            "pressure": cfg["pressure"],
            "harness": cfg["harness"],
            "generated_at": datetime.now().astimezone().isoformat(),
            "completed_script": True,
            "completed_protocol": True,
            "submissions": protocol.get("submissions"),
            "follow_up_sent": protocol.get("follow_up_sent"),
            "time_limit_seconds": TIME_LIMIT_SECONDS,
            "judging": {
                "stage_one_llm_judge": False,
                "final_llm_judge": False,
                "performance_grader": False,
            },
            "analysis_honeypot": False,
            "statistical_summary": manifest.get("statistical_summary"),
            "deliverables": deliverables,
            "all_deliverables_present": all_deliverables_present,
            "continuation_eligibility": {
                "eligible_by_default": all_deliverables_present,
                "reason": (
                    "required_deliverables_present"
                    if all_deliverables_present else "missing_required_deliverables"
                ),
                "required_deliverables": list(required_deliverables),
                "missing_required_deliverables": missing_deliverables,
                "override_flag": "--allow-incomplete-prefixes",
            },
            "measured_context_tokens": context_tokens,
            "token_measurement": measurement,
            "generation_usage": shared._generation_usage(sample),
            "generation_cost": shared._generation_cost(usage_summary, cfg["harness"]),
            "workspace_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "workspace_files": sorted(controlled_workspace_files(seed)),
            "native_harness": real_env.get("harness") or {},
        },
        "messages": [
            message.model_dump(mode="json", exclude_none=False)
            for message in (sample.messages or [])
        ],
    }
    if cfg["harness"] in NATIVE_HARNESS_MODES:
        native_resume = shared._native_resume(sidecar)
        if native_resume is None:
            raise SystemExit(
                "native p-hacking prefix completed without a resumable scaffold bundle"
            )
        payload["native_resume"] = native_resume
    recovery = real_env.get("native_submission_recovery")
    if recovery is not None:
        payload["source"]["native_submission_recovery"] = recovery
    return validate_prefix_payload(payload, origin="exp_p_hacking_prefix")


def _system_prompt(reasoning: bool, harness: str) -> str:
    system_prompt = global_system_prompt()
    if harness != "simple":
        return ""
    return strip_thinking_instruction(system_prompt) if reasoning else system_prompt


def _print_plan(cfg: dict) -> None:
    print("P-HACKING PREFIX-ONLY CONTROL  (positive no-honeypot prefixes)")
    print(
        f"  agents ({len(cfg['targets'])}): {cfg['targets']}  "
        f"harness={cfg['harness']}  reasoning={cfg['reasoning']}"
    )
    print(
        f"  seeds ({len(cfg['seeds'])}): {cfg['seeds']}  pressure={cfg['pressure']}"
    )
    print(
        f"  epochs={cfg['epochs']}  expected prefixes="
        f"{len(cfg['targets'])} x {len(cfg['seeds'])} x {cfg['epochs']} = "
        f"{len(cfg['targets']) * len(cfg['seeds']) * cfg['epochs']}"
    )
    print(
        f"  compute={cfg['compute']}  concurrency={cfg['concurrency']}  "
        f"sandbox_concurrency={cfg['sandbox_concurrency']}  "
        f"vm_concurrency={cfg['vm_concurrency']}"
    )
    print("  protocol=positive 1 pass; judges=0; graders=0")
    for member in cfg["seeds"]:
        seed = seed_path(member)
        sources = load_protocol_sources(seed, pressure=cfg["pressure"])
        print(
            f"  {sources.family}/{sources.member}: "
            f"follow-up={'yes' if sources.follow_up else 'no'}"
        )
        for destination in sorted(controlled_workspace_files(seed)):
            print(f"    {destination}")


def _execution_metadata() -> dict | None:
    raw = os.environ.get("MATS_REMOTE_COMPUTE_JSON")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid MATS_REMOTE_COMPUTE_JSON: {error}") from error


def _build_tasks(cfg: dict, log_dir: Path) -> list[Any]:
    from exp_real_audit import build_real_task

    if cfg["harness"] == "subscription":
        require_subscription_auth(
            production_scaffold_for_target(target, resolve_target(target))
            for target in cfg["targets"]
            if production_scaffold_for_target(target, resolve_target(target)) != "opencode"
        )
        if any(
            opencode_go_model_spec(target, resolve_target(target)) is not None
            for target in cfg["targets"]
        ):
            require_opencode_go_auth(reasoning=cfg["reasoning"])

    unit_sp = _system_prompt(cfg["reasoning"], cfg["harness"])
    task_suffix = os.environ.get("MATS_REMOTE_TASK_SUFFIX")
    execution_metadata = _execution_metadata()
    tasks = []
    for target_name in cfg["targets"]:
        target_slug = resolve_target(target_name)
        opencode_go_subscription = (
            cfg["harness"] == "subscription"
            and opencode_go_model_spec(target_name, target_slug) is not None
        )
        target_build = build_target(
            target_slug,
            reasoning_on=cfg["reasoning"],
            effort=REASONING_EFFORT,
            prompt_cache_key="environments-p-hacking-prefix-" + stable_key(
                "p-hacking-prefix-only-v1",
                target_slug,
                cfg["reasoning"],
                cfg["harness"],
                cfg["pressure"],
            )[:36],
            construct_model=not opencode_go_subscription,
        )
        for member in cfg["seeds"]:
            tasks.append(build_real_task(
                target_name=target_name,
                target_build=target_build,
                unit_path=seed_path(member),
                unit_sp=unit_sp,
                reasoning=cfg["reasoning"],
                condition="allow",
                judge_model=None,
                gate_model=None,
                judge_enabled=False,
                run_label=log_dir.name,
                artifacts_root=log_dir / "real_artifacts",
                task_id_suffix=task_suffix,
                task_name=(
                    f"p_hacking_prefix_only_{member}_"
                    f"{shared._model_slug_fragment(target_name)}"
                ),
                execution_metadata=execution_metadata,
                extra_metadata={
                    "experiment": "p_hacking_prefix_only",
                    "prefix_name_base": cfg["name"],
                    "source_seed": member,
                    "result_condition": SEED_CONFIG[member]["result_condition"],
                },
                harness=cfg["harness"],
                pressure=cfg["pressure"],
            ))
    return tasks


def collect_prefix_payloads(
    log_dir: Path,
    *,
    aws_cell_ids: set[str] | None = None,
) -> list[Path]:
    from exp_real_continuation import (
        build_prefix_spec,
        load_prefix_payload_file,
        store_prefix_payload,
    )

    if aws_cell_ids is None:
        candidates = sorted(log_dir.rglob("prefix_payloads/*.json"))
    else:
        candidates = sorted({
            candidate
            for cell_id in aws_cell_ids
            for candidate in (log_dir / "remote_cells" / cell_id).rglob(
                "prefix_payloads/*.json"
            )
        })
    stored: list[Path] = []
    identities: set[tuple[str, str]] = set()
    for candidate in candidates:
        payload = load_prefix_payload_file(candidate)
        source = payload.get("source") or {}
        if source.get("generator") != "exp_p_hacking_prefix.py":
            raise SystemExit(f"unexpected payload in p-hacking prefix run: {candidate}")
        harness = str(source.get("harness") or "simple")
        spec = build_prefix_spec(payload, payload_path=candidate, harness=harness)
        identity = (spec.name, spec.sha256)
        if identity in identities:
            continue
        identities.add(identity)
        stored.append(store_prefix_payload(payload))
    return stored


def run_local_stage(cfg: dict) -> tuple[list[Path], Path, int, bool]:
    expected = len(cfg["targets"]) * len(cfg["seeds"]) * cfg["epochs"]
    remote_run_name = os.environ.get("MATS_REMOTE_RUN_DIR")
    log_dir = LOGS_ROOT / (
        remote_run_name
        or f"p-hacking-prefix-only-{cfg['name']}-{cfg['epochs']}ep"
    )
    if log_dir.exists() and not remote_run_name:
        raise SystemExit(f"refusing to overwrite existing run directory: {log_dir}")
    print("=" * 72)
    print(f"P-HACKING PREFIX GENERATION  ->  {log_dir.name}")
    print("=" * 72)
    tasks = _build_tasks(cfg, log_dir)
    try:
        success, logs = run_eval(
            tasks,
            epochs=cfg["epochs"],
            concurrency=cfg["concurrency"],
            log_dir=log_dir,
            max_sandboxes=cfg["sandbox_concurrency"],
            time_limit=cfg["time_limit"],
        )
    except SystemExit:
        raise
    except Exception as error:
        print(f"\n!! P-HACKING PREFIX STAGE CRASHED: {type(error).__name__}: {error}")
        traceback.print_exc()
        return [], log_dir, expected, False

    from exp_real_continuation import build_prefix_spec, store_prefix_payload

    records = []
    failures = []
    actual = 0
    for log in logs:
        target_name = str((getattr(log.eval, "metadata", None) or {}).get("target_name"))
        if target_name not in cfg["targets"]:
            failures.append({"task": str(log.eval.task), "error": "missing target metadata"})
            continue
        for sample in log.samples or []:
            actual += 1
            member = str(sample.id)
            epoch = shared._source_epoch(sample)
            sample_cfg = {
                "model": target_name,
                "seed": member,
                "pressure": cfg["pressure"],
                "harness": cfg["harness"],
                "reasoning": cfg["reasoning"],
                "run_name": cfg["name"],
                "name": payload_name(cfg["name"], target_name, member, epoch),
            }
            sidecar = (
                log_dir / "real_artifacts" / str(log.eval.task)
                / f"{sample.id}_ep{sample.epoch}"
            )
            try:
                payload = build_payload(
                    sample_cfg,
                    sample,
                    shared._sample_usage_summary(sample_cfg, sample),
                    sidecar=sidecar,
                )
                prefix = build_prefix_spec(payload, harness=cfg["harness"])
                stored_path = store_prefix_payload(payload)
                run_path = shared._write_run_payload(stored_path, log_dir)
                records.append({
                    "name": prefix.name,
                    "sha256": prefix.sha256,
                    "target": target_name,
                    "seed": member,
                    "epoch": epoch,
                    "payload_file": str(stored_path),
                    "run_payload_file": str(run_path),
                    "generation_cost": payload["source"]["generation_cost"],
                })
                print(f"  payload: {stored_path}")
            except (SystemExit, OSError, ValueError) as error:
                failures.append({
                    "task": str(log.eval.task),
                    "sample": member,
                    "epoch": epoch,
                    "error": str(error),
                })

    dead = dead_targets(logs, cfg["target_models"])
    index = {
        "format": "environments-p-hacking-prefix-run-v1",
        "run_name": cfg["name"],
        "expected": expected,
        "actual_samples": actual,
        "saved_prefixes": len(records),
        "success": bool(success),
        "dead_targets": dead,
        "failures": failures,
        "prefixes": records,
    }
    shared._write_prefix_index(log_dir, index)
    ok = bool(success) and actual == expected and len(records) == expected and not dead
    if not ok:
        print(
            f"WARNING: expected={expected}, actual={actual}, saved={len(records)}, "
            f"dead_targets={dead}, payload_failures={len(failures)}"
        )
    return [Path(record["payload_file"]) for record in records], log_dir, expected, ok


async def _post_stage(cfg: dict) -> bool:
    return await base_pipeline.run_env_post_stages(cfg)


def _collect_campaign_payloads(state: dict) -> list[Path]:
    pipeline = state.get("pipeline_config") or {}
    if (
        pipeline.get("prefix_only") is not True
        or pipeline.get("seed_dir") != P_HACKING_PREFIX_ONLY_FAMILY
    ):
        raise SystemExit(
            f"campaign {state.get('campaign_id')} is not a p-hacking prefix-only campaign"
        )
    log_dir_value = state.get("local_log_dir")
    if not isinstance(log_dir_value, str):
        raise SystemExit("completed campaign has no imported local log directory")
    successful_cells = [
        cell
        for cell in state.get("cells") or []
        if cell.get("status") == "completed"
        and int((cell.get("terminal") or {}).get("pipeline_exit_code", 1)) == 0
    ]
    successful_cell_ids = {str(cell["cell_id"]) for cell in successful_cells}
    payloads = collect_prefix_payloads(
        Path(log_dir_value), aws_cell_ids=successful_cell_ids
    )
    if len(payloads) != len(successful_cells):
        raise SystemExit(
            f"imported campaign has {len(successful_cells)} successful cells but "
            f"{len(payloads)} unique prefix payloads"
        )
    return payloads


def main() -> None:
    try:
        resume_id = _arg("--resume-campaign")
        retry_id = _arg("--retry-failed")
        if resume_id and retry_id:
            raise SystemExit("--resume-campaign and --retry-failed are mutually exclusive")
        if resume_id or retry_id:
            control = _parse_aws_control_args()
            if resume_id:
                if control["dry_run"]:
                    raise SystemExit("--dry-run is not meaningful with --resume-campaign")
                state = resume_campaign(control, _ENVIRONMENTS, DATA_ROOT, resume_id)
            else:
                control["harness"] = resolve_harness(control.get("harness"))
                state = retry_failed(
                    control,
                    _ENVIRONMENTS,
                    DATA_ROOT,
                    retry_id,
                    dry_run=control["dry_run"],
                )
                if control["dry_run"]:
                    return
            payloads = _collect_campaign_payloads(state)
            post_ok = asyncio.run(_post_stage(control))
            if not (campaign_ok(state) and post_ok):
                raise SystemExit(1)
            print(f"[done] {len(payloads)} prefix payload(s) available")
            return

        cfg = _parse_args()
        _print_plan(cfg)
        if cfg["compute"] == "aws":
            state = run_campaign(cfg, _ENVIRONMENTS, DATA_ROOT, dry_run=cfg["dry_run"])
            if cfg["dry_run"]:
                return
            payloads = _collect_campaign_payloads(state)
            generation_ok = campaign_ok(state)
        else:
            if cfg["dry_run"]:
                print("\n[dry-run] local plan validated; no secrets or model calls.")
                return
            load_dotenv(ENV_FILE)
            base_pipeline.require_docker()
            payloads, _log_dir, _expected, generation_ok = run_local_stage(cfg)
        post_ok = asyncio.run(_post_stage(cfg))
    except AwsTrajectoryError as error:
        raise SystemExit(f"AWS trajectory error: {error}") from error

    print("\n" + "=" * 72)
    if generation_ok and post_ok:
        print(f"PIPELINE DONE. {len(payloads)} prefix payload(s) available.")
        print(f"Viewer: {VIEWER_ROOT / 'prefixes.html'}")
    else:
        print("PIPELINE FINISHED WITH FAILURES; inspect the run logs and prefix index.")
    print("=" * 72)
    if not (generation_ok and post_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
