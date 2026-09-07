"""Retrospectively apply the current environment judge to stored trajectories.

This endpoint never regenerates agent behavior. It loads complete stored Inspect
messages and artifacts, then calls the same final-judgment function used by new real
runs. The output is a new Inspect eval log, so cost, exact judge input, validation, and
evidence-loss records use the normal environment pipeline.

Usage (costs judge API money unless ``--dry-run`` is present):
  uv run exp_judge_tests.py --dry-run
  uv run exp_rejudge.py --source-runs=old --dry-run
  uv run exp_rejudge.py --source-runs=old
  uv run exp_rejudge.py --source-runs=real-v1-... --family=p_hacking

Flags:
  --source-runs=<old|all|judge-tests|a,b>  REQUIRED. ``old`` selects the archived
                           pre-canonical runs; ``judge-tests`` uses the fixed viewer
                           cohort; other values select original ``real-v*`` run dirs.
  --family=<name|all>      optional; default all.
  --judge=<model>          optional; defaults to gpt-5.6-luna.
  --concurrency=<N>        optional; default 10 judge calls at once.
  --only-interrupted-native
                           select only sources whose interrupted native transcript
                           was reconstructed from stored target events.
  --dry-run                build and fingerprint every exact input, but make no calls.
  --force                  create a new attempt directory even if this exact campaign
                           already has successful saved judgments.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from inspect_ai import Task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.solver import Generate, Solver, TaskState, solver  # noqa: E402

from environment_judge.exp_rejudge import retrospective_rejudge  # noqa: E402
from exp_inspect_runner import run_eval  # noqa: E402
from judge_selection import judge_shortname, resolve_judge  # noqa: E402
from old_runs import old_run_names, old_trajectory_keys  # noqa: E402
from project_paths import ENV_FILE, JUDGE_TEST_SOURCES_FILE, LOGS_ROOT  # noqa: E402
from rejudge_sources import (  # noqa: E402
    FAMILIES,
    REJUDGE_METHOD_VERSION,
    SourceTrajectory,
    discover_run_dirs,
    load_sources,
    prepare_sources,
)


_VALUE_FLAGS = {"--source-runs", "--family", "--judge", "--concurrency"}
_SWITCH_FLAGS = {
    "--dry-run",
    "--force",
    "--only-interrupted-native",
    "--skip-viewer",
}


def _validate_cli() -> None:
    valid = sorted(_VALUE_FLAGS | _SWITCH_FLAGS)
    for argument in sys.argv[1:]:
        flag, separator, _value = argument.partition("=")
        if flag in _VALUE_FLAGS and separator:
            continue
        if flag in _SWITCH_FLAGS and not separator:
            continue
        raise SystemExit(f"unknown or malformed argument {argument!r}; valid: {valid}")


def _arg(name: str, default: str | None = None) -> str | None:
    prefix = f"--{name}="
    return next((item[len(prefix):] for item in sys.argv if item.startswith(prefix)), default)


def _positive_int(name: str, default: int) -> int:
    raw = _arg(name, str(default))
    try:
        value = int(raw or "")
    except ValueError as error:
        raise SystemExit(f"--{name} must be an integer, got {raw!r}") from error
    if value < 1:
        raise SystemExit(f"--{name} must be >= 1, got {value}")
    return value


@solver
def preserve_stored_trajectory() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        return state

    return solve


def _completed_source_inputs(log_dirs: list[Path]) -> dict[str, list[dict[str, str]]]:
    from inspect_ai.log import list_eval_logs, read_eval_log

    completed: dict[str, list[dict[str, str]]] = {}
    for log_dir in log_dirs:
        if not log_dir.is_dir():
            continue
        for log_info in list_eval_logs(str(log_dir)):
            try:
                log = read_eval_log(log_info)
            except (FileNotFoundError, ValueError):
                continue
            run_judge_model = str((log.eval.metadata or {}).get("judge_model") or "")
            for sample in getattr(log, "samples", None) or []:
                source = (getattr(sample, "metadata", None) or {}).get(
                    "retrospective_rejudge"
                ) or {}
                score = (getattr(sample, "scores", None) or {}).get("environment_judge")
                envelope = (
                    (getattr(score, "metadata", None) or {}).get("environment_judge")
                    if score is not None else {}
                ) or {}
                if (
                    source.get("source_key")
                    and isinstance(getattr(score, "value", None), dict)
                    and envelope.get("post_validation") == "passed"
                ):
                    completed.setdefault(str(source["source_key"]), []).append({
                        "input_sha256": str(
                            source.get("current_judge_input_sha256") or ""
                        ),
                        "judge_method_sha256": str(
                            envelope.get("judge_method_sha256") or ""
                        ),
                        "judge_model": run_judge_model,
                    })
    return completed


def _source_is_complete(
    item: SourceTrajectory,
    completed: dict[str, list[dict[str, str]]],
    *,
    judge_model: str,
) -> bool:
    return any(
        record["input_sha256"] == item.source["current_judge_input_sha256"]
        and record["judge_method_sha256"]
        == item.source["current_judge_method_sha256"]
        and record["judge_model"] == judge_model
        for record in completed.get(item.source["source_key"], [])
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower()


def _interrupted_native_sources(
    sources: list[SourceTrajectory],
) -> list[SourceTrajectory]:
    return [
        item for item in sources
        if isinstance(
            item.source.get("interrupted_native_transcript"), dict
        )
    ]


def build_rejudge_task(
    sources: list[SourceTrajectory], *, judge_model: str, campaign: dict
) -> Task:
    samples = []
    for item in sources:
        source_record = {
            **item.source,
            "rejudge_method_version": REJUDGE_METHOD_VERSION,
            "campaign_fingerprint": campaign["campaign_fingerprint"],
            "method_fingerprint": campaign["method_fingerprint"],
            "judging_method_sha256": campaign["judging_method_sha256"],
        }
        samples.append(Sample(
            id=source_record["source_key"],
            input=list(item.messages),
            metadata={
                "real_env": item.real_env,
                "retrospective_rejudge": source_record,
            },
        ))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Task(
        dataset=MemoryDataset(samples, name="stored_real_trajectories"),
        solver=preserve_stored_trajectory(),
        scorer=retrospective_rejudge(),
        model=judge_model,
        model_roles={"judge": judge_model},
        name=f"retrospective_rejudge_{timestamp}",
        metadata={
            "experiment_kind": "retrospective_rejudge",
            "judgment_role": "retrospective_rejudge",
            **campaign,
        },
    )


def _write_campaign(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "rejudge_campaign.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


async def _plan() -> tuple[list[SourceTrajectory], dict, str, Path]:
    source_selection = _arg("source-runs")
    if not source_selection:
        raise SystemExit("--source-runs is required; use --source-runs=all explicitly")
    family = _arg("family", "all") or "all"
    if family not in ("all", *FAMILIES):
        raise SystemExit(
            f"unknown family {family!r}; choices: all, {', '.join(FAMILIES)}"
        )
    judge_model = resolve_judge(_arg("judge"))
    try:
        judge_test_keys: set[str] | None = None
        if source_selection == "judge-tests":
            manifest_path = JUDGE_TEST_SOURCES_FILE
            if not manifest_path.is_file():
                raise ValueError(
                    "judge-test source manifest is missing; select explicit saved log sources instead"
                )
            manifest = json.loads(manifest_path.read_text())
            keys = manifest.get("source_keys")
            if not isinstance(keys, list) or not keys:
                raise ValueError(f"judge-test source manifest is empty: {manifest_path}")
            judge_test_keys = {str(key) for key in keys}
            run_dirs = [
                path for path in discover_run_dirs(LOGS_ROOT, "all")
                if any(key.startswith(path.name + "__") for key in judge_test_keys)
            ]
            if not run_dirs:
                raise ValueError("judge-test source runs were not found in stored logs")
        else:
            run_dirs = discover_run_dirs(LOGS_ROOT, source_selection)
        sources = load_sources(
            run_dirs, family="all" if judge_test_keys is not None else family
        )
        if source_selection == "old":
            archived_runs = old_run_names()
            archived_keys = old_trajectory_keys()
            sources = [
                item for item in sources
                if (
                    item.source["source_run"] in archived_runs
                    or (
                        f"{item.source['source_run']}__{item.source['source_task']}__"
                        f"{item.source['seed']}__e{item.source['epoch']}"
                    ) in archived_keys
                )
            ]
            if not sources:
                raise ValueError("the Old archive contains no matching trajectories")
        if judge_test_keys is not None:
            sources = [
                item for item in sources
                if (
                    f"{item.source['source_run']}__{item.source['source_task']}__"
                    f"{item.source['seed']}__e{item.source['epoch']}"
                ) in judge_test_keys
            ]
            if len(sources) != len(judge_test_keys):
                found = {
                    f"{item.source['source_run']}__{item.source['source_task']}__"
                    f"{item.source['seed']}__e{item.source['epoch']}"
                    for item in sources
                }
                missing = sorted(judge_test_keys - found)
                raise ValueError(
                    "judge-test manifest sources were not found in stored logs: "
                    + ", ".join(missing[:3])
                )
            if family != "all":
                sources = [item for item in sources if item.family == family]
                if not sources:
                    raise ValueError(
                        f"judge-test cohort has no trajectories in family {family!r}"
                    )
        if "--only-interrupted-native" in sys.argv:
            sources = _interrupted_native_sources(sources)
            if not sources:
                raise ValueError(
                    "the selected source runs contain no reconstructed interrupted "
                    "native trajectories"
                )
        sources, campaign = await prepare_sources(sources, judge_model=judge_model)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    judge_name = judge_shortname(judge_model) or judge_model.split("/")[-1]
    run_name = (
        f"rejudge-current-{_safe_name(judge_name)}-"
        f"{campaign['method_fingerprint'][:12]}"
    )
    # One fresh directory per invocation batch: Inspect's eval_set refuses to add a
    # new task to a directory that already holds another batch's logs, and a shared
    # directory would also overwrite the earlier batch's campaign/accounting sidecars.
    # Cross-batch resume is unaffected — completed sources are found by scanning every
    # rejudge-current-* directory, not by directory identity.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if "--force" in sys.argv:
        log_dir = LOGS_ROOT / f"{run_name}-rerun-{timestamp}"
    else:
        log_dir = LOGS_ROOT / f"{run_name}-{timestamp}"
    return sources, campaign, judge_model, log_dir


async def async_main() -> None:
    _validate_cli()
    concurrency = _positive_int("concurrency", 10)
    sources, campaign, judge_model, log_dir = await _plan()
    prior_log_dirs = sorted(LOGS_ROOT.glob("rejudge-current-*"))
    completed = (
        {} if "--force" in sys.argv else _completed_source_inputs(prior_log_dirs)
    )
    pending = [
        item for item in sources
        if not _source_is_complete(item, completed, judge_model=judge_model)
    ]
    completed_count = len(sources) - len(pending)
    families = Counter(item.family for item in sources)
    caveats = Counter(
        code
        for item in sources
        for code in item.source.get("upstream_caveat_codes") or []
    )
    source_integrity = Counter(
        str(item.source.get("source_integrity_status") or "unknown")
        for item in sources
    )
    print(
        f"Rejudge plan: {len(sources)} stored trajectories; "
        f"{completed_count} already complete; {len(pending)} pending."
    )
    print("Families: " + ", ".join(f"{key}={value}" for key, value in sorted(families.items())))
    print(
        "Source integrity: "
        + ", ".join(
            f"{key}={value}" for key, value in sorted(source_integrity.items())
        )
    )
    print(f"Current judge: {judge_model}")
    print(f"Judging-method SHA-256: {campaign['judging_method_sha256']}")
    print(f"Campaign fingerprint: {campaign['campaign_fingerprint']}")
    print(f"Output: {log_dir}")
    if caveats:
        print("Stored evidence caveats: " + ", ".join(
            f"{key}={value}" for key, value in sorted(caveats.items())
        ))
    if "--dry-run" in sys.argv:
        print("Dry run complete: no model calls were made and no files were written.")
        return

    if pending:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE)
        campaign_record = {
            **campaign,
            "status": "running",
            "source_trajectory_count": len(sources),
            "already_complete_count": completed_count,
            "pending_count": len(pending),
            "source_runs": sorted({item.source["source_run"] for item in sources}),
        }
        _write_campaign(log_dir, campaign_record)
        task = build_rejudge_task(pending, judge_model=judge_model, campaign=campaign)
        success, _logs = run_eval(
            [task], epochs=1, concurrency=concurrency, log_dir=log_dir
        )
        campaign_record.update(
            status="complete" if success else "completed_with_failures",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        _write_campaign(log_dir, campaign_record)
    else:
        success = True
        print("Every exact source input already has a successful saved judgment.")

    if not success:
        raise SystemExit(1)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
