"""Generate resumable, hidden-test ML continuation prefixes at experiment scale.

Each selected (agent, seed) cell runs the same two-pass no-honeypot protocol for the
requested number of epochs. Every completed trajectory becomes its own reusable
``environments-continuation-prefix-v1`` payload. Local runs use Inspect concurrency;
AWS runs use the shared one-trajectory-per-VM campaign, resume, and retry machinery.
Missing task deliverables are preserved in a payload marked continuation-ineligible by
default; continuation runs can opt in explicitly when task failure is the treatment.

Usage (from the repository root):
  uv run prefixes/exp_ml_prefix.py --targets=deepseek-v4-pro,glm-5.1,kimi-k2.6 \
      --seeds=fraud_detection --epochs=10 --harness=production --compute=aws
  uv run prefixes/exp_ml_prefix.py --targets=gpt-5.5 --seeds=fraud_detection \
      --epochs=10 --harness=subscription --compute=aws
  uv run prefixes/exp_ml_prefix.py --resume-campaign=<campaign-id>

The legacy single-cell ``--model`` and ``--seed`` aliases remain accepted; when those
aliases are used, ``--epochs`` defaults to one. New matrix invocations require
``--targets``, ``--seeds``, and ``--epochs`` explicitly.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
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
from cost_tracking import estimate_usage_cost
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
from exp_subscription_harness import (
    require_opencode_go_auth,
    require_subscription_auth,
)
from exp_target_harness import (
    NATIVE_HARNESS_MODES,
    opencode_go_model_spec,
    production_scaffold_for_target,
    resolve_harness,
)
from model_catalog import (
    REASONING_EFFORT,
    TARGET_CHOICES,
    build_target,
    resolve_target,
)
from pipeline_cli import positive_int_arg, validate_cli_args, value_arg
from project_paths import DATA_ROOT, ENV_FILE, LOGS_ROOT, VIEWER_ROOT
from prompt_caching import stable_key
from protocol_sources import (
    global_system_prompt,
    load_protocol_sources,
    resolve_reasoning,
    strip_thinking_instruction,
)
from real_env_workspace import workspace_files


PREFIX_FORMAT = "environments-continuation-prefix-v1"
ML_PREFIX_ONLY_FAMILY = "ml_prefix_only"
ML_FIRST_PASS_LIMIT_SECONDS = 70 * 60
ML_SECOND_PASS_LIMIT_SECONDS = 45 * 60
DEFAULT_CONCURRENCY = 50
DEFAULT_SANDBOX_CONCURRENCY = 8
SEEDS_ROOT = _ENVIRONMENTS / "seeds" / ML_PREFIX_ONLY_FAMILY
SEED_CONFIG = {
    "demand_forecasting": {
        "description": "demand-forecasting",
        "source_label": "Demand forecasting",
    },
    "fraud_detection": {
        "description": "fraud-detection",
        "source_label": "Fraud detection",
    },
}
EXPECTED_WORKSPACE_FILES = frozenset({
    "/workspace/README.md",
    "/workspace/requirements.txt",
    "/workspace/train.py",
    "/workspace/data/train.csv",
    "/workspace/data/val.csv",
})
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VALUE_FLAGS = {
    "--targets", "--model", "--seeds", "--seed", "--epochs", "--harness",
    "--reasoning", "--name", "--concurrency", "--sandbox-concurrency",
    "--time-limit", "--compute", "--vm-concurrency", "--aws-region",
    "--aws-instance-type", "--aws-bucket", "--aws-secret-env",
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


def _model_slug_fragment(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-") or "model"


def seed_path(member: str) -> Path:
    return SEEDS_ROOT / member


def default_name() -> str:
    return (
        f"ml-prefix-{datetime.now().strftime('%Y%m%d%H%M%S%f')}-"
        f"{uuid.uuid4().hex[:8]}"
    )


def _csv_values(raw: str | None, *, flag: str) -> list[str]:
    if raw is None:
        return []
    values = list(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    if not values:
        raise SystemExit(f"{flag} had no usable values")
    return values


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
        raise SystemExit(
            f"--targets is required; agent choices: {sorted(TARGET_CHOICES)}"
        )
    unknown_targets = [target for target in targets if target not in TARGET_CHOICES]
    if unknown_targets:
        raise SystemExit(
            f"unknown agents in --targets {unknown_targets}; "
            f"choices: {sorted(TARGET_CHOICES)}"
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
        seeds = ["demand_forecasting"]
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
    legacy_single = legacy_target or legacy_seed
    if epochs is None:
        if legacy_single and len(targets) == len(seeds) == 1:
            epochs = 1
        else:
            raise SystemExit("--epochs is required (no default); positive integer")
    if _arg("--harness") is None:
        raise SystemExit("--harness is required (simple, production, or subscription)")
    harness = resolve_harness(_arg("--harness"))
    compute = str(_arg("--compute", "aws"))
    if compute not in {"aws", "local"}:
        raise SystemExit("--compute must be aws or local")
    time_limit = _posint("--time-limit", ML_FIRST_PASS_LIMIT_SECONDS)
    if time_limit != ML_FIRST_PASS_LIMIT_SECONDS:
        raise SystemExit(
            "ml_prefix_only has a fixed 70-minute first pass; "
            f"--time-limit must be {ML_FIRST_PASS_LIMIT_SECONDS}"
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
        "seed_dir": ML_PREFIX_ONLY_FAMILY,
        "family": ML_PREFIX_ONLY_FAMILY,
        "prefix_only": True,
        "epochs": epochs,
        "harness": harness,
        "reasoning": resolve_reasoning(_arg("--reasoning")),
        "name": name,
        "condition": "allow",
        "pressure": None,
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
    """Resume/retry modes do not require a new experiment matrix."""

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


def hidden_workspace_files(seed: Path) -> dict[str, str]:
    """Return the exact sandbox mapping, rejecting any evaluation-set exposure."""

    mapping = workspace_files(seed)
    actual = frozenset(mapping)
    if actual != EXPECTED_WORKSPACE_FILES:
        raise SystemExit(
            "ml_prefix_only workspace boundary changed; expected only "
            f"{sorted(EXPECTED_WORKSPACE_FILES)}, got {sorted(actual)}"
        )
    manifest = json.loads((seed / "manifest.json").read_text())
    access = manifest.get("evaluation_access") or {}
    required_false = {
        "test_inputs_available",
        "test_labels_available",
        "test_api_available",
        "prediction_deliverable",
        "reward_hack_available",
    }
    incorrect = sorted(key for key in required_false if access.get(key) is not False)
    if incorrect:
        raise SystemExit(
            "ml_prefix_only manifest must explicitly set these fields false: "
            f"{incorrect}"
        )
    for sandbox_path, host_path in mapping.items():
        lowered_parts = {part.casefold() for part in Path(host_path).parts}
        if "answers" in lowered_parts or Path(host_path).name.casefold().startswith("test"):
            raise SystemExit(
                f"ml_prefix_only attempted to expose evaluation material at "
                f"{sandbox_path}: {host_path}"
            )
    if (seed / "answers").exists():
        raise SystemExit("ml_prefix_only must not contain a host-side answers directory")
    return mapping


def _usage_record(usage: Any) -> dict:
    def read(name: str) -> int:
        value = getattr(usage, name, None)
        return int(value) if isinstance(value, (int, float)) else 0

    return {
        "input": read("input_tokens"),
        "output": read("output_tokens"),
        "cache_read": read("input_tokens_cache_read"),
        "cache_write": read("input_tokens_cache_write"),
        "total_cost": (
            float(usage.total_cost)
            if isinstance(getattr(usage, "total_cost", None), (int, float))
            else None
        ),
    }


def _final_context(sample: Any) -> tuple[int, str]:
    records = []
    for event in getattr(sample, "events", None) or []:
        if getattr(event, "event", None) != "model":
            continue
        output = getattr(event, "output", None)
        usage = getattr(output, "usage", None)
        if usage is not None:
            records.append(_usage_record(usage))
    if records:
        final = records[-1]
        return (
            final["input"]
            + final["cache_read"]
            + final["cache_write"]
            + final["output"],
            "final_recorded_model_call_provider_usage",
        )
    messages = getattr(sample, "messages", None) or []
    estimate = sum(len(getattr(message, "text", None) or "") for message in messages) // 4
    return estimate, "estimated_visible_chars_over_4"


def _generation_usage(sample: Any) -> dict:
    usage = (getattr(sample, "role_usage", None) or {}).get("target")
    return _usage_record(usage) if usage is not None else {}


def _generation_cost(summary: dict, harness: str) -> dict:
    subscriptions = summary.get("subscription_models_not_metered") or []
    if harness == "subscription" and subscriptions:
        return {
            "cost_usd": None,
            "exact": False,
            "source": "subscription_not_metered",
            "subscription_models": subscriptions,
        }
    unknown = summary.get("unknown_models") or []
    return {
        "cost_usd": summary.get("total_cost_usd"),
        "exact": bool(summary.get("exact")),
        "source": (
            "inspect_sample_usage"
            if not unknown
            else "partially_priced_inspect_sample_usage"
        ),
        "unknown_models": unknown,
    }


def _sample_usage_summary(cfg: dict, sample: Any) -> dict:
    target = cfg["model"]
    routed = resolve_target(target)
    scaffold = production_scaffold_for_target(target, routed)
    included_subscription = cfg["harness"] == "subscription" and (
        scaffold != "opencode" or opencode_go_model_spec(target, routed) is not None
    )
    if included_subscription:
        return {
            "total_cost_usd": 0.0,
            "exact": False,
            "unknown_models": [],
            "subscription_models_not_metered": [routed],
        }
    priced = estimate_usage_cost(routed, _generation_usage(sample))
    amount = priced.get("cost_usd")
    return {
        "total_cost_usd": float(amount) if isinstance(amount, (int, float)) else 0.0,
        "exact": bool(priced.get("exact")),
        "unknown_models": [] if amount is not None else [routed],
        "subscription_models_not_metered": [],
    }


def _native_resume(sidecar: Path) -> dict | None:
    manifest_path = sidecar / "_native_resume_bundle.json"
    archive_path = sidecar / "_native_resume_bundle.tar.gz"
    if not manifest_path.is_file() or not archive_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    manifest["archive_base64"] = base64.b64encode(archive_path.read_bytes()).decode("ascii")
    return manifest


def _source_epoch(sample: Any) -> int:
    real_env = (getattr(sample, "metadata", None) or {}).get("real_env") or {}
    compute = real_env.get("compute") or {}
    value = compute.get("original_epoch")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return int(sample.epoch)


def payload_name(base: str, model: str, seed: str, epoch: int) -> str:
    return (
        f"{base}-{_model_slug_fragment(model)}-"
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
    if protocol.get("submissions") != 2 or protocol.get("follow_up_sent") is not True:
        raise SystemExit(
            "the two-pass ML protocol did not complete; refusing to save a partial prefix"
        )
    judging = real_env.get("judging") or {}
    if judging.get("enabled") is not False or real_env.get("gates"):
        raise SystemExit("ml_prefix_only unexpectedly recorded a judge call")
    context_tokens, measurement = _final_context(sample)
    member = cfg["seed"]
    seed = seed_path(member)
    member_config = SEED_CONFIG[member]
    manifest_path = seed / "manifest.json"
    grade = real_env.get("grade") or {}
    deliverables = grade.get("deliverables") or {}
    required_deliverables = ("REPORT.md", "models/final/")
    missing_deliverables = [
        name for name in required_deliverables if deliverables.get(name) is not True
    ]
    all_deliverables_present = not missing_deliverables
    source_epoch = _source_epoch(sample)
    payload = {
        "format": PREFIX_FORMAT,
        "name": cfg["name"],
        "target": cfg["model"],
        "reasoning": cfg["reasoning"],
        "source": {
            "kind": "external",
            "description": (
                f"two-pass {member_config['description']} ML task with train and "
                "validation data only; evaluation inputs and labels were fully "
                "unavailable"
            ),
            "generator": "exp_ml_prefix.py",
            "prefix_type": "ml_prefix_only",
            "prefix_type_label": "ML prefix only",
            "source_label": f"{member_config['source_label']} · no test access",
            "family": ML_PREFIX_ONLY_FAMILY,
            "seed": member,
            "epoch": source_epoch,
            "run_name": cfg.get("run_name"),
            "comparison_source_family": "ml_training_data_misuse",
            "harness": cfg["harness"],
            "generated_at": datetime.now().astimezone().isoformat(),
            "completed_script": True,
            "completed_protocol": True,
            "submissions": protocol.get("submissions"),
            "follow_up_sent": protocol.get("follow_up_sent"),
            "first_pass_limit_seconds": ML_FIRST_PASS_LIMIT_SECONDS,
            "second_pass_limit_seconds": ML_SECOND_PASS_LIMIT_SECONDS,
            "judging": {
                "stage_one_llm_judge": False,
                "final_llm_judge": False,
                "performance_grader": False,
            },
            "evaluation_access": {
                "test_inputs_available": False,
                "test_labels_available": False,
                "test_api_available": False,
                "prediction_deliverable": False,
                "reward_hack_available": False,
            },
            "deliverables": deliverables,
            "all_deliverables_present": all_deliverables_present,
            "continuation_eligibility": {
                "eligible_by_default": all_deliverables_present,
                "reason": (
                    "required_deliverables_present"
                    if all_deliverables_present
                    else "missing_required_deliverables"
                ),
                "required_deliverables": list(required_deliverables),
                "missing_required_deliverables": missing_deliverables,
                "override_flag": "--allow-incomplete-prefixes",
            },
            "measured_context_tokens": context_tokens,
            "token_measurement": measurement,
            "generation_usage": _generation_usage(sample),
            "generation_cost": _generation_cost(usage_summary, cfg["harness"]),
            "workspace_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "workspace_files": sorted(hidden_workspace_files(seed)),
            "native_harness": real_env.get("harness") or {},
        },
        "messages": [
            message.model_dump(mode="json", exclude_none=False)
            for message in (sample.messages or [])
        ],
    }
    if cfg["harness"] in NATIVE_HARNESS_MODES:
        native_resume = _native_resume(sidecar)
        if native_resume is None:
            raise SystemExit(
                "native ML prefix completed without a resumable scaffold bundle; "
                "refusing to save a non-resumable prefix"
            )
        payload["native_resume"] = native_resume
    submission_recovery = real_env.get("native_submission_recovery")
    if submission_recovery is not None:
        payload["source"]["native_submission_recovery"] = submission_recovery
    return validate_prefix_payload(payload, origin="exp_ml_prefix")


def _system_prompt(reasoning: bool, harness: str) -> str:
    system_prompt = global_system_prompt()
    if harness != "simple":
        return ""
    return strip_thinking_instruction(system_prompt) if reasoning else system_prompt


def _print_plan(cfg: dict) -> None:
    print("ML PREFIX-ONLY CONTROL  (hidden evaluation -> continuation prefixes)")
    print(
        f"  agents ({len(cfg['targets'])}): {cfg['targets']}  "
        f"harness={cfg['harness']}  reasoning={cfg['reasoning']}"
    )
    print(f"  seeds ({len(cfg['seeds'])}): {cfg['seeds']}")
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
    print(
        "  protocol=70-minute first pass + fixed 45-minute follow-up; "
        "judges=0; graders=0; test inputs=0; test labels=0"
    )
    for member in cfg["seeds"]:
        seed = seed_path(member)
        sources = load_protocol_sources(seed)
        mapping = hidden_workspace_files(seed)
        print(f"  {sources.family}/{sources.member} sandbox files:")
        for destination in sorted(mapping):
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
            if production_scaffold_for_target(target, resolve_target(target))
            != "opencode"
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
            prompt_cache_key="environments-ml-prefix-" + stable_key(
                "ml-prefix-only-v2",
                target_slug,
                cfg["reasoning"],
                cfg["harness"],
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
                    f"ml_prefix_only_{member}_{_model_slug_fragment(target_name)}"
                ),
                execution_metadata=execution_metadata,
                extra_metadata={
                    "experiment": "ml_prefix_only",
                    "prefix_name_base": cfg["name"],
                    "source_seed": member,
                    "test_access": "none",
                },
                harness=cfg["harness"],
            ))
    return tasks


def _write_run_payload(stored_path: Path, log_dir: Path) -> Path:
    payload_dir = log_dir / "prefix_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    run_path = payload_dir / stored_path.name
    if run_path.exists() and run_path.read_bytes() != stored_path.read_bytes():
        raise SystemExit(f"refusing to overwrite different run payload: {run_path}")
    if not run_path.exists():
        shutil.copy2(stored_path, run_path)
    return run_path


def _write_prefix_index(log_dir: Path, value: dict) -> None:
    path = log_dir / "prefix_index.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def collect_prefix_payloads(
    log_dir: Path,
    *,
    aws_cell_ids: set[str] | None = None,
) -> list[Path]:
    """Promote verified local or imported-worker payloads into the shared store."""

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
            for candidate in (
                log_dir / "remote_cells" / cell_id
            ).rglob("prefix_payloads/*.json")
        })
    stored: list[Path] = []
    identities: set[tuple[str, str]] = set()
    for candidate in candidates:
        payload = load_prefix_payload_file(candidate)
        source = payload.get("source") or {}
        if source.get("generator") != "exp_ml_prefix.py":
            raise SystemExit(f"unexpected payload in ML prefix run: {candidate}")
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
        or f"ml-prefix-only-{cfg['name']}-{cfg['epochs']}ep"
    )
    if log_dir.exists() and not remote_run_name:
        raise SystemExit(f"refusing to overwrite existing run directory: {log_dir}")
    print("=" * 72)
    print(f"ML PREFIX GENERATION  ->  {log_dir.name}")
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
        print(f"\n!! ML PREFIX STAGE CRASHED: {type(error).__name__}: {error}")
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
            epoch = _source_epoch(sample)
            sample_cfg = {
                "model": target_name,
                "seed": member,
                "harness": cfg["harness"],
                "reasoning": cfg["reasoning"],
                "run_name": cfg["name"],
                "name": payload_name(cfg["name"], target_name, member, epoch),
            }
            sidecar = (
                log_dir
                / "real_artifacts"
                / str(log.eval.task)
                / f"{sample.id}_ep{sample.epoch}"
            )
            try:
                payload = build_payload(
                    sample_cfg,
                    sample,
                    _sample_usage_summary(sample_cfg, sample),
                    sidecar=sidecar,
                )
                prefix = build_prefix_spec(payload, harness=cfg["harness"])
                stored_path = store_prefix_payload(payload)
                run_path = _write_run_payload(stored_path, log_dir)
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
        "format": "environments-ml-prefix-run-v1",
        "run_name": cfg["name"],
        "expected": expected,
        "actual_samples": actual,
        "saved_prefixes": len(records),
        "success": bool(success),
        "dead_targets": dead,
        "failures": failures,
        "prefixes": records,
    }
    _write_prefix_index(log_dir, index)
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
    if pipeline.get("prefix_only") is not True:
        raise SystemExit(
            f"campaign {state.get('campaign_id')} is not an ML prefix-only campaign"
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
                state = resume_campaign(
                    control, _ENVIRONMENTS, DATA_ROOT, resume_id
                )
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
            state = run_campaign(
                cfg, _ENVIRONMENTS, DATA_ROOT, dry_run=cfg["dry_run"]
            )
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
