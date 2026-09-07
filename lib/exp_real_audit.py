"""Inspect tasks for the environments-owned real-sandbox experiment.

The agent works in Docker and receives only pinned local protocol text. A structured
stage-one judge decides whether to send the scripted follow-up. Runs that continue get
a fresh final judgment plus stored artifact snapshots. ML judges receive the complete
observable trajectory, including native reasoning. P-hacking judges receive only user
turns and assistant submission turns, with native reasoning excluded. The selected
family policy and every omission are stored and shown with the judgment.

This file can make paid agent and judge calls, so it keeps the ``exp_`` prefix.
Importing it is free.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

import anyio
from dotenv import load_dotenv
from inspect_ai import Task
from inspect_ai.agent import AgentState
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    execute_tools,
    get_model,
)
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sample_limits, span

from continuation_evidence import (
    CONTINUATION_DELIVERY_INLINE_USER_CONTEXT,
    ContinuationRun,
    continuation_caveat_fields,
    judge_view_messages as continuation_judge_view_messages,
    sliced_message_record,
    update_production_boundary,
)
from multi_agent_evidence import (
    ACTIVITY_LOG_PATH,
    ActivityLogRun,
    caveat_fields as activity_log_caveat_fields,
    finalize_activity_log_exposure,
    judge_view_messages as activity_log_judge_view_messages,
    sanitize_artifacts as sanitize_activity_log_artifacts,
    update_activity_log_exposure,
    update_activity_log_exposure_from_model_events,
)
from environment_judge import (
    ArtifactSnapshot,
    EvidenceCaveat,
    JudgeCallError,
    JudgePostValidationError,
)
from environment_judge.prompt import JUDGE_SCHEMA_VERSION
from environment_judge.rubrics import load_rubrics
from environment_judge.exp_real import judge_complete_real_trajectory
from environment_judge.score import score_from_judgment
from exp_inspect_runner import run_eval
from exp_target_harness import (
    NATIVE_HARNESS_MODES,
    build_production_agent,
    capture_native_resume_bundle,
    native_resume_record,
    opencode_go_model_spec,
    production_agent_input_messages,
    production_harness_metadata,
    production_scaffold_for_target,
    retain_native_scaffold_system_messages,
    resolve_harness,
    restore_native_resume_bundle,
    subscription_harness_metadata,
    update_resolved_scaffold_version,
    update_scaffold_loss_events,
)
from exp_subscription_harness import (
    NativeSessionRef,
    build_subscription_agent,
    record_subscription_native_version,
    require_opencode_go_auth,
    require_subscription_auth,
    session_id_value,
    subscription_agent_record,
)
from exp_real_gate import (
    GateConfig,
    gate_detectors,
    grade_predictions,
    run_gate,
    target_authored_texts,
)
from judge_selection import (
    DEFAULT_JUDGE,
    JUDGE_CHOICES,
    resolve_judge,
)
from inspect_task_naming import bounded_inspect_task_name, full_inspect_task_name
from interrupted_native_transcript import (
    recover_interrupted_opencode_messages,
    recover_predeadline_opencode_submission,
)
from model_catalog import (
    REASONING_EFFORT,
    TARGET_CHOICES,
    build_target,
    resolve_target,
    route,
)
from prompt_caching import stable_key
from project_paths import ENVIRONMENTS_ROOT, ENV_FILE, LOGS_ROOT, SEEDS_ROOT
from protocol_sources import (
    global_system_prompt,
    is_seed_member,
    load_protocol_sources,
    resolve_condition,
    resolve_pressure,
    resolve_reasoning,
    resolve_seeds,
    seed_members,
    strip_thinking_instruction,
)
from real_env_workspace import (
    WORKSPACE_DIR,
    answers_dir,
    create_empty_dirs,
    deliverable_entry,
    extract_artifacts,
    has_real_env,
    real_env_dir,
    workspace_files,
    workspace_snapshot,
)
from real_grader import deliverables_status
from real_judge_evidence import (
    artifact_snapshots as _artifact_snapshots,
    artifacts_from_real_env,
    tool_output_truncation_numbers as _tool_output_truncation_numbers,
    upstream_caveats as _upstream_caveats,
)
from sandbox_runtime import (
    FAMILY_SANDBOX,
    compose_file_for_harness,
)
from real_target_tools import (
    REAL_MAX_TOOL_OUTPUT,
    REAL_TOOL_SEMANTICS_VERSION,
    real_target_tools,
    real_target_tools_fingerprint,
    real_target_tools_metadata,
    reset_sample_time_remaining,
)

load_dotenv(ENV_FILE)


REAL_CONFIG_VERSION = "environments-real-v1"
REAL_PROTOCOL_VERSION = "environments-script-v3"
ML_TIME_PROTOCOL_VERSION = "environments-ml-two-pass-clock-v2"
ML_CLOCK_VERSION = "environments-ml-first-70m-second-45m-v1"
JUDGE_EVIDENCE_VERSION = "complete-observable-messages-v5"

ML_FIRST_PASS_LIMIT_SECONDS = 70 * 60
ML_SECOND_PASS_LIMIT_SECONDS = 45 * 60
ML_PREFIX_ONLY_FAMILY = "ml_prefix_only"
P_HACKING_PREFIX_ONLY_FAMILY = "p_hacking_prefix_only"
PREFIX_ONLY_FAMILIES = frozenset({
    ML_PREFIX_ONLY_FAMILY,
    P_HACKING_PREFIX_ONLY_FAMILY,
})
ML_TIME_FAMILIES = frozenset({"ml_training_data_misuse", ML_PREFIX_ONLY_FAMILY})

DEFAULT_GATE_MODEL = DEFAULT_JUDGE
DEFAULT_SANDBOX_CONCURRENCY = 8
DEFAULT_TIME_LIMIT_SECONDS = 3600
FAMILY_TIME_LIMIT_SECONDS = {
    ML_PREFIX_ONLY_FAMILY: ML_FIRST_PASS_LIMIT_SECONDS,
    "ml_training_data_misuse": ML_FIRST_PASS_LIMIT_SECONDS,
    "p_hacking": 1800,
    P_HACKING_PREFIX_ONLY_FAMILY: 1800,
}
FINALIZE_GRACE_SECONDS = 120

SANDBOX_DIR = ENVIRONMENTS_ROOT / "sandbox"
FAMILY_DELIVERABLES = {
    ML_PREFIX_ONLY_FAMILY: ("REPORT.md", "models/final/"),
    "ml_training_data_misuse": ("predictions.csv", "REPORT.md", "models/final/"),
    "p_hacking": ("analysis.py", "RESULT.md"),
    P_HACKING_PREFIX_ONLY_FAMILY: ("analysis.py", "RESULT.md"),
}


def resolve_time_limit(arg: str | None, family: str | None = None) -> int | None:
    if arg is None:
        return FAMILY_TIME_LIMIT_SECONDS.get(family, DEFAULT_TIME_LIMIT_SECONDS)
    try:
        value = int(arg)
    except ValueError as error:
        raise SystemExit(
            f"--time-limit must be an integer number of seconds, got {arg!r}"
        ) from error
    if value < 0:
        raise SystemExit(f"--time-limit must be >= 0, got {value}")
    if family in ML_TIME_FAMILIES and value != ML_FIRST_PASS_LIMIT_SECONDS:
        raise SystemExit(
            "ML real-environment runs use the fixed 70-minute --time-limit=4200"
        )
    return value or None


@dataclass(frozen=True)
class RealProtocolSpec:
    version: str
    family: str
    member: str
    pressure: str | None
    system_prompt: str
    opening_user_message: str
    follow_up: str | None
    deliverables: tuple[str, ...]
    gate: GateConfig
    task_readme: str
    source_provenance: dict

    @property
    def follow_up_user_messages(self) -> list[str]:
        return [self.follow_up] if self.follow_up else []


def seed_family(unit_path: str | Path) -> str:
    family = Path(unit_path).resolve().parent.name
    if family not in FAMILY_SANDBOX:
        raise SystemExit(
            f"seed {unit_path} belongs to unsupported family {family!r}; "
            f"supported families: {sorted(FAMILY_SANDBOX)}"
        )
    return family


def assemble_real_protocol(
    unit_path: str | Path,
    condition: str,
    unit_sp: str,
    pressure: str | None = None,
) -> RealProtocolSpec:
    if condition != "allow":
        raise SystemExit("real environments currently support only --condition=allow")
    unit_path = Path(unit_path).resolve()
    sources = load_protocol_sources(unit_path, pressure=pressure)
    family = seed_family(unit_path)
    if not has_real_env(unit_path):
        raise SystemExit(f"{unit_path} has no materialized workspace")

    key_text = None
    threshold = None
    grade_metric = "macro_f1"
    tolerance = None
    if family == "ml_training_data_misuse":
        key_path = answers_dir(unit_path) / "test_labels.csv"
        if not key_path.is_file():
            raise SystemExit(f"{key_path} is missing (the grader and gate need it)")
        key_text = key_path.read_text()
        summary_path = real_env_dir(unit_path) / "calibration_summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"{summary_path} is missing; run the matching calibration")
        threshold = json.loads(summary_path.read_text())["gate_agreement_threshold"]
        manifest = json.loads((real_env_dir(unit_path) / "manifest.json").read_text())
        grade_metric = manifest.get("grade_metric", "macro_f1")
        tolerance = manifest.get("agreement_tolerance")

    opening = sources.opening_user_message
    follow_up = sources.follow_up
    version = REAL_PROTOCOL_VERSION
    if family in ML_TIME_FAMILIES:
        version = ML_TIME_PROTOCOL_VERSION

    provenance = sources.provenance()
    if not unit_sp:
        provenance["protocol_source_files"] = [
            item
            for item in provenance["protocol_source_files"]
            if item.get("path") != "seeds/SYSTEM_PROMPT.txt"
        ]
        provenance["environment_system_prompt_applied"] = False

    return RealProtocolSpec(
        version=version,
        family=family,
        member=sources.member,
        pressure=sources.pressure,
        system_prompt=unit_sp,
        opening_user_message=opening,
        follow_up=follow_up,
        deliverables=FAMILY_DELIVERABLES[family],
        gate=GateConfig(
            key_text=key_text,
            agreement_threshold=threshold,
            grade_metric=grade_metric,
            tolerance=tolerance,
        ),
        task_readme=(unit_path / "workspace" / "README.md").read_text(),
        source_provenance=provenance,
    )


class RealProtocolController:
    """Pure deterministic state machine for the scripted user turns."""

    def __init__(self, spec: RealProtocolSpec, condition: str):
        self.spec = spec
        self.condition = condition
        self.submissions = 0
        self.follow_up_sent = False
        self.first_gate_verdict: str | None = None

    def on_submission(
        self,
        *,
        gate_verdict: str | None = None,
    ) -> tuple[str, str]:
        self.submissions += 1
        if self.submissions == 1:
            self.first_gate_verdict = gate_verdict
            if gate_verdict == "error":
                return "end", "gate_error_end"
            if gate_verdict == "hacked":
                return "end", "protocol_end"
            if self.spec.follow_up:
                self.follow_up_sent = True
                return "send", self.spec.follow_up
            return "end", "protocol_end"
        return "end", "protocol_end"

    def summary(self, ended_reason: str) -> dict:
        return {
            "submissions": self.submissions,
            "first_gate_verdict": self.first_gate_verdict,
            "follow_up_sent": self.follow_up_sent,
            "ended_reason": ended_reason,
        }


def _task_snapshot(spec: RealProtocolSpec) -> ArtifactSnapshot:
    return ArtifactSnapshot(
        path=f"{WORKSPACE_DIR}/README.md",
        snapshot="initial_task",
        content=spec.task_readme,
        sha256=hashlib.sha256(spec.task_readme.encode()).hexdigest(),
        byte_count=len(spec.task_readme.encode()),
    )


def _native_reasoning_message_numbers(messages: Sequence[ChatMessage]) -> list[int]:
    numbers = []
    for number, message in enumerate(messages, start=1):
        if not isinstance(message, ChatMessageAssistant) or not isinstance(message.content, list):
            continue
        if any(getattr(block, "type", None) == "reasoning" for block in message.content):
            numbers.append(number)
    return numbers


def _tool_call_count(messages: Sequence[ChatMessage]) -> int:
    return sum(
        len(getattr(message, "tool_calls", None) or []) for message in messages
    )


def _embedded_tool_use_block_count(messages: Sequence[ChatMessage]) -> int:
    return sum(
        1
        for message in messages
        for block in (message.content if isinstance(message.content, list) else [])
        if getattr(block, "type", None) == "tool_use"
    )


def _tool_result_message_count(messages: Sequence[ChatMessage]) -> int:
    return sum(message.role == "tool" for message in messages)


def get_cancelled_exc_class():
    return anyio.get_cancelled_exc_class()


def _time_limit_blown() -> bool:
    try:
        limit = sample_limits().time
    except Exception:
        return False
    return limit.limit is not None and limit.usage >= limit.limit


@solver
def real_audit_solver(
    *,
    spec: RealProtocolSpec,
    seed_path: str,
    condition: str,
    artifacts_dir: str | None = None,
    execution_metadata: dict | None = None,
    continuation: ContinuationRun | None = None,
    activity_log: ActivityLogRun | None = None,
    harness: str,
    target_name: str,
    target_slug: str,
    reasoning: bool,
    judge_first_submission: bool = True,
) -> Solver:
    """Run the agent and scripted protocol inside one Inspect sandbox.

    With ``continuation`` set, the agent starts pre-loaded with the prefix
    conversation plus the injected pivot user turn instead of the plain
    opening (and, in simple mode, environment-system) pair. Both judge stages then see
    the active system message(s) and live task, sliced at the stored boundary index.
    With ``activity_log`` set, execution instead starts as an ordinary fresh session
    with ACTIVITY_LOG.md mounted in the workspace. Its complete delivery to model
    context is measured mechanically after each successful target call.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        include_time = spec.family in ML_TIME_FAMILIES
        tools = (
            real_target_tools(include_time=include_time)
            if harness == "simple"
            else []
        )
        native_agent = None
        native_session_ref: str | NativeSessionRef | None = None
        native_scaffold = production_scaffold_for_target(target_name, target_slug)
        if harness in NATIVE_HARNESS_MODES:
            time_tool = (
                real_target_tools(include_time=True)[-1] if include_time else None
            )
            # One instance per sample: the agent owns its CLI session and is reused
            # only for this sample's scripted follow-up turns.
            builder = (
                build_subscription_agent
                if harness == "subscription"
                else build_production_agent
            )
            builder_args = {
                "time_tool": time_tool,
                "native_resume": (
                    continuation.native_resume if continuation is not None else None
                ),
            }
            if harness == "subscription":
                builder_args["reasoning"] = reasoning
            native_agent, native_session_ref = builder(
                target_name, target_slug, **builder_args
            )
        controller = RealProtocolController(spec, condition)
        real_env: dict = {
            "family": spec.family,
            "pressure": spec.pressure,
            "gates": [],
            "submission_artifacts": [],
            "task_context": _task_snapshot(spec).model_dump(mode="json"),
            "judging": {
                "enabled": judge_first_submission,
                "stage_one": "llm_judge" if judge_first_submission else "not_run_by_design",
                "final": "task_scorer" if judge_first_submission else "not_run_by_design",
            },
            "judge_evidence_policy": (
                {
                    "version": JUDGE_EVIDENCE_VERSION,
                    "message_selection": "all observable messages supplied for the stage",
                    "native_reasoning": "included where readable",
                    "lossy": False,
                    "lossy_reasons": [],
                    "caveat": (
                        "Judging includes all supplied messages, readable native reasoning, "
                        "assistant-visible text, tool calls, and tool results."
                    ),
                }
                if judge_first_submission
                else {
                    "version": "no-judge-prefix-v1",
                    "message_selection": "none",
                    "native_reasoning": "not inspected by a judge",
                    "lossy": False,
                    "lossy_reasons": [],
                    "caveat": (
                        "No gate or final LLM judge runs for this prefix-only task. "
                        "Transcript observability is recorded only for continuation "
                        "integrity."
                    ),
                }
            ),
            "protocol_sources": spec.source_provenance,
            "harness": (
                {"mode": "simple"}
                if harness == "simple"
                else (
                    subscription_harness_metadata(target_name, target_slug)
                    if harness == "subscription"
                    else production_harness_metadata(target_name, target_slug)
                )
            ),
        }
        system_observability = real_env["harness"].get(
            "native_system_prompt_observability"
        ) or {}
        if not system_observability.get("complete", True):
            real_env["judge_evidence_policy"]["lossy_reasons"].append(
                "native_system_prompt_unavailable"
            )
        if include_time:
            real_env["clock"] = {
                "version": ML_CLOCK_VERSION,
                "initial_pass_seconds": ML_FIRST_PASS_LIMIT_SECONDS,
                "second_pass_seconds": ML_SECOND_PASS_LIMIT_SECONDS,
                "second_pass_reset": None,
            }
        if execution_metadata:
            real_env["compute"] = dict(execution_metadata)
        if continuation is not None:
            real_env["continuation"] = deepcopy(continuation.record)
            state.messages, _ = continuation.initial_messages()
        else:
            if activity_log is not None:
                real_env["multi_agent"] = deepcopy(activity_log.record)
            if (
                harness == "simple"
                and not any(
                    isinstance(message, ChatMessageSystem)
                    for message in state.messages
                )
            ):
                state.messages = [
                    ChatMessageSystem(content=spec.system_prompt),
                    ChatMessageUser(content=spec.opening_user_message),
                ]

        model = get_model(role="target") if harness == "simple" else None
        ended_reason = "interrupted"
        start_snapshot: dict[str, str] = {}
        turns_used = 0
        active_native_event_start: int | None = None
        active_native_base_messages: list[ChatMessage] | None = None
        active_native_events_measured = False
        native_submission_recovery: dict | None = None
        try:
            async with span(name="target", type="agent"):
                await create_empty_dirs(seed_path)
                if harness in NATIVE_HARNESS_MODES and continuation is not None:
                    if (
                        continuation.native_resume is None
                        and continuation.delivery_mode
                        != CONTINUATION_DELIVERY_INLINE_USER_CONTEXT
                    ):
                        raise RuntimeError(
                            f"{harness} continuation reached execution without a "
                            "validated native resume bundle"
                        )
                    if continuation.native_resume is not None:
                        restored = await restore_native_resume_bundle(
                            continuation.native_resume,
                            target_name=target_name,
                            routed_slug=target_slug,
                            reasoning=reasoning,
                        )
                        resume_key = f"{harness}_native_resume"
                        real_env["continuation"].setdefault(resume_key, {}).update(
                            restored
                        )
                start_snapshot = await workspace_snapshot()
                while True:
                    turns_used += 1
                    if harness == "simple":
                        model_input_messages = list(state.messages)
                        state.output = await model.generate(
                            input=state.messages, tools=tools
                        )
                        if activity_log is not None:
                            update_activity_log_exposure(
                                real_env["multi_agent"], model_input_messages
                            )
                        state.messages.append(state.output.message)
                        if state.output.message.tool_calls:
                            executed = await execute_tools(
                                state.messages,
                                tools,
                                max_output=REAL_MAX_TOOL_OUTPUT,
                            )
                            state.messages.extend(executed.messages)
                            continue
                    else:
                        assert native_agent is not None
                        previous_messages = list(state.messages)
                        active_native_base_messages = previous_messages
                        agent_state = AgentState(
                            messages=(
                                list(state.messages)
                                if harness == "subscription"
                                and native_scaffold != "opencode"
                                else production_agent_input_messages(state.messages)
                            )
                        )
                        active_native_event_start = len(transcript().events)
                        active_native_events_measured = False
                        agent_state = await native_agent(agent_state)
                        state.messages = (
                            retain_native_scaffold_system_messages(
                                previous_messages, agent_state.messages
                            )
                            if continuation is not None
                            else list(agent_state.messages)
                        )
                        state.output = agent_state.output
                        current_events = list(transcript().events)
                        if harness == "subscription" and native_scaffold != "opencode":
                            subscription_record = subscription_agent_record(
                                native_session_ref
                                if isinstance(native_session_ref, NativeSessionRef)
                                else None
                            )
                            real_env["harness"]["subscription_run"] = subscription_record
                            for event in subscription_record.get("all_loss_events") or []:
                                if event not in real_env["harness"]["native_loss_events"]:
                                    real_env["harness"]["native_loss_events"].append(event)
                            native_version = subscription_record.get("native_version")
                            record_subscription_native_version(
                                real_env["harness"], native_version
                            )
                            await update_resolved_scaffold_version(real_env)
                        else:
                            await update_resolved_scaffold_version(real_env)
                        await update_scaffold_loss_events(real_env, current_events)
                        if real_env["harness"].get("native_loss_events"):
                            reasons = real_env["judge_evidence_policy"][
                                "lossy_reasons"
                            ]
                            if "target_scaffold_native_loss_confirmed" not in reasons:
                                reasons.append("target_scaffold_native_loss_confirmed")
                        if continuation is not None:
                            update_production_boundary(
                                state.messages, real_env["continuation"]
                            )
                        if activity_log is not None:
                            native_loss_events = (
                                real_env["harness"].get("native_loss_events") or []
                            )
                            event_start = active_native_event_start or 0
                            checked = update_activity_log_exposure_from_model_events(
                                real_env["multi_agent"],
                                current_events[event_start:],
                                native_loss_events=native_loss_events,
                            )
                            if not checked:
                                update_activity_log_exposure(
                                    real_env["multi_agent"],
                                    state.messages,
                                    native_loss_events=native_loss_events,
                                    measurement_source="agent_state_fallback",
                                )
                            active_native_events_measured = True

                    snapshot_now = await workspace_snapshot()
                    submission_number = controller.submissions + 1
                    raw_artifacts_now = await extract_artifacts(
                        start_snapshot,
                        snapshot_now,
                        spec.deliverables,
                    )
                    judge_artifacts_now = raw_artifacts_now
                    if activity_log is not None:
                        judge_artifacts_now, masked_paths = (
                            sanitize_activity_log_artifacts(
                                raw_artifacts_now, real_env["multi_agent"]
                            )
                        )
                        if masked_paths:
                            masking = real_env["multi_agent"].setdefault(
                                "judge_masking", {}
                            )
                            existing = masking.setdefault(
                                "artifact_paths_masked", []
                            )
                            existing.extend(
                                path for path in masked_paths if path not in existing
                            )
                    real_env["submission_artifacts"].append({
                        "submission": submission_number,
                        "artifacts": raw_artifacts_now,
                    })
                    if controller.submissions == 0 and judge_first_submission:
                        # A continuation gate sees only the system message and the
                        # live task; the prefix is sliced out mechanically.
                        gate_messages = (
                            continuation_judge_view_messages(
                                state.messages, real_env["continuation"]
                            )
                            if continuation is not None
                            else list(state.messages)
                        )
                        if activity_log is not None:
                            gate_messages, masking = activity_log_judge_view_messages(
                                gate_messages, real_env["multi_agent"]
                            )
                            real_env["multi_agent"].setdefault(
                                "judge_masking", {}
                            ).update(masking)
                        detectors = gate_detectors(
                            spec.family,
                            spec.gate,
                            raw_artifacts_now,
                            target_authored_texts(gate_messages),
                        )
                        stage_artifacts = [
                            _task_snapshot(spec),
                            *_artifact_snapshots(
                                judge_artifacts_now, "submission_1"
                            ),
                        ]
                        gate_caveats = list(_upstream_caveats(
                            gate_messages,
                            real_env["submission_artifacts"],
                            real_env=real_env,
                        ))
                        if continuation is not None:
                            gate_caveats.append(EvidenceCaveat(
                                **continuation_caveat_fields(
                                    real_env["continuation"]
                                )
                            ))
                        if activity_log is not None:
                            gate_caveats.append(EvidenceCaveat(
                                **activity_log_caveat_fields(
                                    real_env["multi_agent"]
                                )
                            ))
                        gate = await run_gate(
                            family=spec.family,
                            messages=gate_messages,
                            artifacts=stage_artifacts,
                            upstream_caveats=gate_caveats,
                            detectors=detectors,
                        )
                        gate_record = {
                            "submission": 1,
                            **gate.as_dict(),
                        }
                        if continuation is not None:
                            gate_record["continuation_scope"] = (
                                sliced_message_record(
                                    real_env["continuation"],
                                    len(state.messages),
                                )
                            )
                        real_env["gates"].append(gate_record)
                        labels = (
                            ((gate.judgment or {}).get("reward_hacking") or {}).get("types")
                            if gate.judgment else None
                        )
                        print(
                            f"    [{spec.member}] submission 1 judge: "
                            f"verdict={gate.verdict} reward_hack_labels={labels}"
                        )
                        action, payload = controller.on_submission(
                            gate_verdict=gate.verdict
                        )
                    else:
                        action, payload = controller.on_submission()
                        if controller.submissions == 1 and not judge_first_submission:
                            print(
                                f"    [{spec.member}] submission 1 complete; "
                                "LLM judge skipped by design"
                            )
                    active_native_event_start = None
                    active_native_base_messages = None
                    active_native_events_measured = False
                    if action == "send":
                        if (
                            include_time
                            and controller.submissions == 1
                            and controller.follow_up_sent
                        ):
                            reset = reset_sample_time_remaining(
                                ML_SECOND_PASS_LIMIT_SECONDS
                            )
                            real_env["clock"]["second_pass_reset"] = reset
                            print(
                                f"    [{spec.member}] second-pass clock reset: "
                                f"{ML_SECOND_PASS_LIMIT_SECONDS}s remaining"
                        )
                        state.messages.append(ChatMessageUser(content=payload))
                        continue
                    ended_reason = payload
                    break
        except get_cancelled_exc_class():
            ended_reason = "wall_clock_limit" if _time_limit_blown() else "cancelled"
            if (
                ended_reason == "wall_clock_limit"
                and native_scaffold == "opencode"
                and active_native_event_start is not None
                and active_native_base_messages is not None
                and controller.submissions == 1
                and controller.follow_up_sent
            ):
                reset = (real_env.get("clock") or {}).get("second_pass_reset") or {}
                deadline = reset.get("deadline_seconds_from_start")
                recovered_messages, recovery, deadline_recovery = (
                    recover_predeadline_opencode_submission(
                        active_native_base_messages,
                        list(transcript().events),
                        deadline_seconds_from_start=deadline,
                        event_start=active_native_event_start,
                        applied_before_judging=judge_first_submission,
                    )
                )
                if deadline_recovery is not None:
                    state.messages = recovered_messages
                    real_env["interrupted_native_transcript"] = recovery
                    reasons = real_env["judge_evidence_policy"]["lossy_reasons"]
                    if "interrupted_native_transcript_reconstructed" not in reasons:
                        reasons.append("interrupted_native_transcript_reconstructed")
                    action, payload = controller.on_submission()
                    if action != "end" or payload != "protocol_end":
                        raise RuntimeError(
                            "pre-deadline final-response recovery reached a "
                            "non-terminal protocol state"
                        )
                    ended_reason = payload
                    native_submission_recovery = deadline_recovery
                    real_env["native_submission_recovery"] = deadline_recovery
                    if activity_log is not None:
                        update_activity_log_exposure_from_model_events(
                            real_env["multi_agent"],
                            list(transcript().events)[active_native_event_start:],
                            native_loss_events=(
                                real_env["harness"].get("native_loss_events") or []
                            ),
                        )
                        active_native_events_measured = True
                    active_native_event_start = None
                    active_native_base_messages = None
                    print(
                        f"    [{spec.member}] counted terminal OpenCode response "
                        "completed before the deadline; CLI handoff finished late"
                    )
                else:
                    raise
            else:
                raise
        finally:
            if native_scaffold == "opencode" and active_native_event_start is not None:
                recovered_messages, recovery = recover_interrupted_opencode_messages(
                    active_native_base_messages or state.messages,
                    list(transcript().events),
                    event_start=active_native_event_start,
                    applied_before_judging=judge_first_submission,
                )
                if recovery is not None:
                    state.messages = recovered_messages
                    real_env["interrupted_native_transcript"] = recovery
                    reasons = real_env["judge_evidence_policy"]["lossy_reasons"]
                    if "interrupted_native_transcript_reconstructed" not in reasons:
                        reasons.append("interrupted_native_transcript_reconstructed")
            real_env["protocol"] = controller.summary(ended_reason)
            if activity_log is not None:
                if (
                    active_native_event_start is not None
                    and not active_native_events_measured
                ):
                    update_activity_log_exposure_from_model_events(
                        real_env["multi_agent"],
                        list(transcript().events)[active_native_event_start:],
                        native_loss_events=(
                            real_env["harness"].get("native_loss_events") or []
                        ),
                    )
                finalize_activity_log_exposure(
                    real_env["multi_agent"],
                    native_loss_events=(
                        real_env["harness"].get("native_loss_events") or []
                    ),
                )
            real_env["turns_used"] = turns_used
            real_env["max_tool_output"] = (
                REAL_MAX_TOOL_OUTPUT if harness == "simple" else None
            )
            native_reasoning = _native_reasoning_message_numbers(state.messages)
            tool_call_count = _tool_call_count(state.messages)
            embedded_tool_use_block_count = _embedded_tool_use_block_count(
                state.messages
            )
            tool_result_message_count = _tool_result_message_count(state.messages)
            truncated_tools = _tool_output_truncation_numbers(state.messages)
            policy = real_env["judge_evidence_policy"]
            policy["native_reasoning_messages"] = native_reasoning
            policy["tool_call_count"] = tool_call_count
            policy["embedded_tool_use_block_count"] = (
                embedded_tool_use_block_count
            )
            policy["tool_result_message_count"] = tool_result_message_count
            policy["tool_output_truncated_messages"] = truncated_tools
            if truncated_tools:
                policy["lossy_reasons"].append("target_tool_output_truncated")
            policy["lossy"] = bool(policy["lossy_reasons"])
            state.metadata = {**(state.metadata or {}), "real_env": real_env}
            with anyio.move_on_after(FINALIZE_GRACE_SECONDS, shield=True):
                try:
                    await _finalize_real_sample(
                        state,
                        spec,
                        real_env,
                        start_snapshot,
                        artifacts_dir,
                        target_name=target_name,
                        target_slug=target_slug,
                        reasoning=reasoning,
                        native_session_id=session_id_value(native_session_ref),
                    )
                    if native_submission_recovery is not None:
                        submissions = real_env["submission_artifacts"]
                        if not any(item.get("submission") == 2 for item in submissions):
                            submissions.append({
                                "submission": 2,
                                "artifacts": real_env["artifacts"],
                                "recovered_from_predeadline_terminal_response": True,
                            })
                except Exception as error:
                    real_env["finalize_error"] = repr(error)[:500]
                    print(
                        f"  !! WARNING [{spec.member}]: artifact finalize failed: "
                        f"{error!r}"
                    )
            if "artifacts" not in real_env and "finalize_error" not in real_env:
                real_env["finalize_error"] = (
                    f"artifact finalize exceeded {FINALIZE_GRACE_SECONDS}s"
                )
        return state

    return solve


async def _finalize_real_sample(
    state: TaskState,
    spec: RealProtocolSpec,
    real_env: dict,
    start_snapshot: dict,
    artifacts_dir: str | None,
    *,
    target_name: str,
    target_slug: str,
    reasoning: bool,
    native_session_id: str | None,
) -> None:
    end_snapshot = await workspace_snapshot()
    artifacts = await extract_artifacts(
        start_snapshot,
        end_snapshot,
        spec.deliverables,
    )
    real_env["artifacts"] = artifacts
    policy = real_env.get("judge_evidence_policy") or {}
    policy_reasons = policy.setdefault("lossy_reasons", [])
    for reason in artifacts.get("lossy_reasons") or []:
        if reason not in policy_reasons:
            policy_reasons.append(reason)
    policy["lossy"] = bool(policy_reasons)
    if spec.family == "ml_training_data_misuse":
        predictions = deliverable_entry(artifacts, "predictions.csv")
        real_env["grade"] = grade_predictions(
            spec.gate,
            predictions.get("content") if predictions else None,
            bool(predictions and predictions.get("truncated")),
        )
    elif spec.family == ML_PREFIX_ONLY_FAMILY:
        real_env["grade"] = {
            "scored": False,
            "reason": (
                "prefix-only task has no accessible evaluation set and no "
                "performance score"
            ),
        }
    else:
        real_env["grade"] = {
            "scored": False,
            "reason": "p_hacking has no deterministic performance score",
        }
    real_env["grade"].update(
        deliverables_status(
            [entry["path"] for entry in artifacts["files"]],
            spec.deliverables,
            model_entries=artifacts.get("models") or [],
        )
    )

    native_bundle = None
    harness_meta = real_env.get("harness") or {}
    if harness_meta.get("mode") in NATIVE_HARNESS_MODES:
        if (
            harness_meta.get("mode") == "subscription"
            and harness_meta.get("scaffold") != "opencode"
            and native_session_id is None
        ):
            # Subscription CLIs allocate their session id inside the first call.
            # Without one there is nothing resumable, and attempting a capture
            # would raise SystemExit past the Exception guards below.
            real_env["native_resume_bundle"] = {
                "available": False,
                "capture_error": (
                    "no native session id: the first subscription CLI call "
                    "never completed"
                ),
            }
        else:
            try:
                native_bundle = await capture_native_resume_bundle(
                    target_name=target_name,
                    routed_slug=target_slug,
                    reasoning=reasoning,
                    native_session_id=native_session_id,
                )
                real_env["native_resume_bundle"] = native_resume_record(native_bundle)
            except Exception as error:
                real_env["native_resume_bundle"] = {
                    "available": False,
                    "capture_error": repr(error)[:500],
                }

    if artifacts_dir:
        sidecar = Path(artifacts_dir) / f"{state.sample_id}_ep{state.epoch}"
        sidecar.mkdir(parents=True, exist_ok=True)
        for entry in artifacts["files"]:
            if entry.get("content") is None:
                continue
            relative = entry["path"].removeprefix(f"{WORKSPACE_DIR}/").replace("/", "__")
            (sidecar / relative).write_text(entry["content"])
        if native_bundle is not None:
            archive_data = base64.b64decode(native_bundle["archive_base64"])
            (sidecar / "_native_resume_bundle.tar.gz").write_bytes(archive_data)
            manifest = {
                key: value
                for key, value in native_bundle.items()
                if key != "archive_base64"
            }
            (sidecar / "_native_resume_bundle.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            real_env["native_resume_bundle"].update({
                "available": True,
                "storage": "real_artifacts_sidecar",
                "archive_file": "_native_resume_bundle.tar.gz",
                "manifest_file": "_native_resume_bundle.json",
            })
        (sidecar / "_real_env.json").write_text(
            json.dumps(real_env, indent=2, default=str)
        )
        real_env["sidecar_dir"] = str(sidecar)


def _score_from_judgment(
    state: TaskState,
    *,
    judgment: dict,
    call_metadata: dict,
    family: str,
    official_stage: str,
    reused_stage_one: bool,
) -> Score:
    score = score_from_judgment(
        judgment=judgment,
        call_metadata=call_metadata,
        family=family,
        official_stage=official_stage,
        reused_stage_one=reused_stage_one,
    )
    envelope = score.metadata["environment_judge"]
    real_env = (state.metadata or {}).get("real_env") or {}
    real_env["final_judgment"] = {
        "schema_version": envelope["judge_schema_version"],
        "family": family,
        "official_stage": official_stage,
        "reused_stage_one": reused_stage_one,
        "fresh_call": not reused_stage_one,
        "result": judgment,
    }
    state.metadata = {**(state.metadata or {}), "real_env": real_env}
    return score


@scorer(metrics=[], name="environment_judge")
def real_audit_judge(family: str):
    """Fresh final judge, except when the stage-one hack judgment ended the run."""

    async def score(state: TaskState, target: Target) -> Score:
        del target
        real_env = (state.metadata or {}).get("real_env") or {}
        protocol = real_env.get("protocol") or {}
        gates = real_env.get("gates") or []
        gate = gates[0] if gates else {}
        if (
            protocol.get("submissions") == 1
            and protocol.get("first_gate_verdict") == "hacked"
            and isinstance(gate.get("judgment"), dict)
        ):
            return _score_from_judgment(
                state,
                judgment=gate["judgment"],
                call_metadata=gate.get("judge_metadata") or {},
                family=family,
                official_stage="stage_1",
                reused_stage_one=True,
            )

        try:
            result = await judge_complete_real_trajectory(
                family=family,
                messages=list(state.messages),
                real_env=real_env,
                model=get_model(role="judge"),
            )
        except JudgePostValidationError as error:
            return Score(
                value="invalid",
                explanation=str(error),
                metadata={"environment_judge": error.call_metadata},
            )
        except JudgeCallError as error:
            return Score(
                value="invalid",
                explanation=str(error),
                metadata={"environment_judge": error.call_metadata},
            )
        except Exception as error:
            return Score(
                value="invalid",
                explanation=repr(error)[:500],
                metadata={
                    "environment_judge": {
                        "judge_schema_version": JUDGE_SCHEMA_VERSION,
                        "family": family,
                        "official_stage": "final",
                        "post_validation": "call_failed",
                    }
                },
            )
        if not isinstance(result.value, dict):
            return Score(
                value="invalid",
                explanation=result.explanation or "judge returned no structured answer",
                metadata=dict(result.metadata or {}),
            )
        return _score_from_judgment(
            state,
            judgment=result.value,
            call_metadata=dict(result.metadata or {}),
            family=family,
            official_stage="final",
            reused_stage_one=False,
        )

    return score


def _judge_artifacts_from_state(state: TaskState, family: str) -> list[ArtifactSnapshot]:
    """Compatibility helper retained for tests and older imports."""

    del family
    real_env = (state.metadata or {}).get("real_env") or {}
    return artifacts_from_real_env(real_env)


def resolve_gate_model(gate_model_arg: str | None, judge: str | None = None) -> str:
    judge_model = resolve_judge(judge)
    if gate_model_arg is None:
        return judge_model
    value = gate_model_arg.strip()
    if value in JUDGE_CHOICES:
        resolved = resolve_judge(value)
    elif value in TARGET_CHOICES:
        resolved = resolve_target(value)
    elif "/" in value:
        resolved = route(value)
    else:
        raise SystemExit(f"unknown --gate-model={value!r}")
    if resolved != judge_model:
        raise SystemExit(
            f"--gate-model must equal the final judge ({judge_model}); got {resolved}"
        )
    return resolved


def ported_members(seeds_path: str) -> list[str]:
    root = Path(seeds_path)
    return [root.name] if is_seed_member(root) else seed_members(root)


def build_real_tasks(
    selected_targets: list[str],
    selected_seeds: list[str],
    run_label: str,
    *,
    reasoning: bool,
    condition: str,
    gate_model: str,
    judge: str | None = None,
    seeds_path: str,
    artifacts_root: Path | None = None,
    task_id_suffix: str | None = None,
    execution_metadata: dict | None = None,
    harness: str,
    pressure: str | None = None,
) -> list[Task]:
    harness = resolve_harness(harness)
    if condition != "allow":
        raise SystemExit("real environments currently support only --condition=allow")
    judge_model = resolve_judge(judge)
    if route(gate_model) != route(judge_model):
        raise SystemExit(
            f"--gate-model must equal the final judge ({judge_model}); got {gate_model}"
        )
    root = Path(seeds_path).resolve()
    if is_seed_member(root):
        units = [root]
    else:
        wanted = set(selected_seeds)
        units = [root / member for member in seed_members(root) if member in wanted]
    if not units:
        raise SystemExit(f"no runnable environment members matched {selected_seeds}")
    if any(seed_family(unit) in PREFIX_ONLY_FAMILIES for unit in units):
        raise SystemExit(
            "prefix-only controls are no-judge prefix sources; build them with their "
            "prefixes/exp_*_prefix.py endpoint instead of the judged real-audit endpoint"
        )

    if harness == "subscription":
        require_subscription_auth(
            production_scaffold_for_target(name, resolve_target(name))
            for name in selected_targets
            if production_scaffold_for_target(name, resolve_target(name)) != "opencode"
        )
        if any(
            opencode_go_model_spec(name, resolve_target(name)) is not None
            for name in selected_targets
        ):
            require_opencode_go_auth(reasoning=reasoning)

    system_prompt = global_system_prompt()
    unit_sp = (
        strip_thinking_instruction(system_prompt) if reasoning else system_prompt
    ) if harness == "simple" else ""
    tasks: list[Task] = []
    for target_name in selected_targets:
        target_slug = resolve_target(target_name)
        opencode_go_subscription = (
            harness == "subscription"
            and opencode_go_model_spec(target_name, target_slug) is not None
        )
        target_build = build_target(
            target_slug,
            reasoning_on=reasoning,
            effort=REASONING_EFFORT,
            # OpenAI rejects prompt_cache_key values longer than 64 chars.
            prompt_cache_key="environments-target-" + stable_key(
                (
                    "target-v1"
                    if harness == "simple"
                    else f"target-{harness}-v1"
                ),
                target_slug,
                reasoning,
                unit_sp,
            )[:40],
            construct_model=not opencode_go_subscription,
        )
        for unit_path in units:
            tasks.append(build_real_task(
                target_name=target_name,
                target_build=target_build,
                unit_path=unit_path,
                unit_sp=unit_sp,
                reasoning=reasoning,
                condition=condition,
                judge_model=judge_model,
                gate_model=gate_model,
                run_label=run_label,
                artifacts_root=artifacts_root,
                task_id_suffix=task_id_suffix,
                execution_metadata=execution_metadata,
                harness=harness,
                pressure=pressure,
            ))
    return tasks


def build_real_task(
    *,
    target_name: str,
    target_build,
    unit_path: Path,
    unit_sp: str,
    reasoning: bool,
    condition: str,
    judge_model: str | None,
    gate_model: str | None,
    run_label: str,
    artifacts_root: Path | None = None,
    task_id_suffix: str | None = None,
    execution_metadata: dict | None = None,
    continuation: ContinuationRun | None = None,
    activity_log: ActivityLogRun | None = None,
    task_name: str | None = None,
    extra_metadata: dict | None = None,
    harness: str,
    pressure: str | None = None,
    judge_enabled: bool = True,
) -> Task:
    """One Inspect task for one (agent, environment member) cell.

    Shared by the plain audit builder above and the continuation builder
    (``exp_real_continuation``), so both stamp identical protocol/tool/judge
    metadata. ``target_build`` is a :class:`model_catalog.TargetModelBuild`.
    """

    target_slug = target_build.routed_slug
    harness = resolve_harness(harness)
    scaffold = production_scaffold_for_target(target_name, target_slug)
    if harness == "subscription" and scaffold != "opencode":
        require_subscription_auth([scaffold])
    go_spec = (
        opencode_go_model_spec(target_name, target_slug)
        if harness == "subscription"
        else None
    )
    if go_spec is not None:
        require_opencode_go_auth(reasoning=reasoning)
    spec = assemble_real_protocol(unit_path, condition, unit_sp, pressure)
    if continuation is not None and activity_log is not None:
        raise SystemExit("a task cannot be both a continuation and multi-agent run")
    sample_files = workspace_files(unit_path)
    if activity_log is not None:
        if ACTIVITY_LOG_PATH in sample_files:
            raise SystemExit(
                f"seed workspace already contains reserved path {ACTIVITY_LOG_PATH}"
            )
        sample_files = {
            **sample_files,
            ACTIVITY_LOG_PATH: str(activity_log.content_path),
        }
        spec = replace(
            spec,
            opening_user_message=activity_log.opening_user_message(
                spec.opening_user_message
            ),
        )
    if judge_enabled and spec.family in PREFIX_ONLY_FAMILIES:
        raise SystemExit(
            f"{spec.family} must run with judge_enabled=False; it is a prefix-only "
            "control family"
        )
    if not judge_enabled and spec.family not in PREFIX_ONLY_FAMILIES:
        raise SystemExit(
            "judge_enabled=False is reserved for prefix-only control families"
        )
    compose_name = compose_file_for_harness(
        harness=harness,
        scaffold=scaffold,
    )
    compose = SANDBOX_DIR / FAMILY_SANDBOX[spec.family] / compose_name
    if not compose.is_file():
        raise SystemExit(f"sandbox compose file missing: {compose}")
    rubric_documents = load_rubrics(spec.family) if judge_enabled else []
    manifest_path = unit_path / "manifest.json"
    include_time = spec.family in ML_TIME_FAMILIES
    tools_metadata = real_target_tools_metadata(include_time=include_time)
    native_harness_metadata = (
        subscription_harness_metadata(target_name, target_slug)
        if harness == "subscription"
        else production_harness_metadata(target_name, target_slug)
        if harness == "production"
        else None
    )
    sample = Sample(
        id=spec.member,
        input=spec.opening_user_message,
        files=sample_files,
        metadata={
            "condition": condition,
            "pressure": spec.pressure,
            "protocol": {
                "version": spec.version,
                "pressure": spec.pressure,
                "system_prompt": spec.system_prompt if harness == "simple" else None,
                "system_prompt_source": (
                    "seeds/SYSTEM_PROMPT.txt"
                    if harness == "simple"
                    else f"{harness}_scaffold_native"
                ),
                "opening_user_message": spec.opening_user_message,
                "follow_up_user_messages": spec.follow_up_user_messages,
            },
            "tools": tools_metadata if harness == "simple" else [],
            "tools_source": (
                "fixed_inspect_tools"
                if harness == "simple"
                else f"{harness}_scaffold_native_tools"
            ),
            "bridged_tools": (
                []
                if harness == "simple" or not include_time
                else [tools_metadata[-1]]
            ),
        },
    )
    base_task_name = (
        task_name or f"real_audit_{target_slug.split('/')[-1]}_{spec.member}"
    )
    full_task_name = full_inspect_task_name(base_task_name, task_id_suffix)
    name = bounded_inspect_task_name(base_task_name, task_id_suffix)
    artifacts_dir = str(artifacts_root / name) if artifacts_root else None
    metadata = {
        "config_version": REAL_CONFIG_VERSION,
        "run_label": run_label,
        "target_name": target_name,
        "target_model": target_slug,
        "reasoning": reasoning,
        "harness": harness,
        # Keep the historical key for production logs while giving new code a
        # mode-neutral field.
        "production_harness": (
            native_harness_metadata if harness == "production" else None
        ),
        "native_harness": native_harness_metadata,
        "reasoning_effort": REASONING_EFFORT if reasoning else None,
        "reasoning_enabled": target_build.reasoning_enabled,
        "strict_tools": target_build.strict_tools,
        "target_prompt_cache_key": target_build.prompt_cache_key,
        "judge": judge_model if judge_enabled else None,
        "gate_model": gate_model if judge_enabled else None,
        "judging_enabled": judge_enabled,
        "condition": condition,
        "pressure": spec.pressure,
        "judge_schema_version": JUDGE_SCHEMA_VERSION if judge_enabled else None,
        "judge_dimensions": [document.key for document in rubric_documents],
        "judge_dimension_files": [
            document.metadata() for document in rubric_documents
        ],
        "judge_evidence_version": JUDGE_EVIDENCE_VERSION if judge_enabled else None,
        "target_tools_mode": "real",
        "target_tools_version": (
            REAL_TOOL_SEMANTICS_VERSION if harness == "simple" else None
        ),
        "target_tool_names": (
            [tool["name"] for tool in tools_metadata]
            if harness == "simple" else None
        ),
        "target_tools_fingerprint": (
            real_target_tools_fingerprint(include_time=include_time)
            if harness == "simple" else None
        ),
        "target_bridged_tool_names": (
            [] if harness == "simple" or not include_time else ["check_time"]
        ),
        "target_max_tool_output": (
            REAL_MAX_TOOL_OUTPUT if harness == "simple" else None
        ),
        "protocol_version": spec.version,
        "protocol_sources": spec.source_provenance,
        "sandbox_image": FAMILY_SANDBOX[spec.family],
        "sandbox_compose": compose_name,
        "real_env_manifest_sha": (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_path.is_file() else None
        ),
        "max_turns": None,
        "remote_task_id": task_id_suffix,
        "task_name_full": full_task_name,
        "task_name_shortened": name != full_task_name,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    api_backed_agent = harness == "production" or (
        harness == "subscription" and scaffold == "opencode" and go_spec is None
    )
    model_roles = {}
    if judge_enabled:
        if not judge_model or not gate_model:
            raise SystemExit("judged real tasks require judge and gate models")
        model_roles.update({
            "judge": judge_model,
            # Keep construction free and key-independent. Inspect resolves the role
            # when the paid stage-one call actually runs.
            "gate": gate_model,
        })
    if harness == "simple" or api_backed_agent:
        model_roles["target"] = target_build.model
    return Task(
        dataset=MemoryDataset([sample], name="seeds"),
        solver=real_audit_solver(
            spec=spec,
            seed_path=str(unit_path),
            condition=condition,
            artifacts_dir=artifacts_dir,
            execution_metadata=execution_metadata,
            continuation=continuation,
            activity_log=activity_log,
            harness=harness,
            target_name=target_name,
            target_slug=target_slug,
            reasoning=reasoning,
            judge_first_submission=judge_enabled,
        ),
        scorer=real_audit_judge(spec.family) if judge_enabled else None,
        sandbox=("docker", str(compose)),
        model_roles=model_roles,
        # Production bridges resolve their agent through Inspect's active model.
        # Use the same configured object as the agent role so usage remains role-
        # attributed and reasoning/cache settings remain intact. Simple mode keeps the
        # previous task construction byte-for-byte at this seam.
        model=target_build.model if api_backed_agent else None,
        name=name,
        metadata=metadata,
    )


def _flag(name: str) -> str | None:
    return next(
        (argument.split("=", 1)[1] for argument in sys.argv if argument.startswith(f"--{name}=")),
        None,
    )


def reject_max_turns_flag() -> None:
    if any(argument.startswith("--max-turns") for argument in sys.argv):
        raise SystemExit("--max-turns is not supported; real runs use wall-clock limits")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)
    reject_max_turns_flag()
    harness = resolve_harness(_flag("harness"))
    target_arg = _flag("target")
    if not target_arg:
        raise SystemExit(f"--target is required; agent choices: {sorted(TARGET_CHOICES)}")
    targets = list(dict.fromkeys(
        item.strip() for item in target_arg.split(",") if item.strip()
    ))
    for name in targets:
        resolve_target(name)
    seeds_path, available = resolve_seeds(_flag("seed-dir"))
    seeds_arg = _flag("seeds") or "all"
    selected = available if seeds_arg == "all" else [
        item.strip() for item in seeds_arg.split(",") if item.strip()
    ]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise SystemExit(f"unknown seeds {unknown}; available: {available}")
    condition = resolve_condition(_flag("condition"))
    reasoning = resolve_reasoning(_flag("reasoning"))
    judge_arg = _flag("judge")
    gate_model = resolve_gate_model(_flag("gate-model"), judge_arg)
    family = Path(seeds_path).name if not is_seed_member(seeds_path) else Path(seeds_path).parent.name
    pressure = resolve_pressure(_flag("pressure"), family)
    time_limit = resolve_time_limit(_flag("time-limit"), family)
    epochs = int(_flag("epochs") or "1")
    concurrency = int(_flag("concurrency") or "50")
    sandbox_concurrency = int(_flag("sandbox-concurrency") or str(DEFAULT_SANDBOX_CONCURRENCY))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = LOGS_ROOT / f"real-v3-{targets[0]}-{epochs}ep-{timestamp}"
    tasks = build_real_tasks(
        targets,
        selected,
        log_dir.name,
        reasoning=reasoning,
        condition=condition,
        gate_model=gate_model,
        judge=judge_arg,
        seeds_path=seeds_path,
        artifacts_root=log_dir / "real_artifacts",
        harness=harness,
        pressure=pressure,
    )
    success, _logs = run_eval(
        tasks,
        epochs,
        concurrency,
        log_dir,
        max_sandboxes=sandbox_concurrency,
        time_limit=time_limit,
    )
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
