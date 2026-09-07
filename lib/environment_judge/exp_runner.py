"""Run and strictly validate one prepared environment-judge API call.

The ``exp_`` filename is intentional: importing and calling this module can spend
money through Inspect Scout.
"""

from __future__ import annotations

from typing import Iterable

from inspect_ai.model import GenerateConfig, Model
from inspect_scout import AnswerStructured, Result, generate_answer
from pydantic import ValidationError

from .prompt import (
    ANSWER_FORMAT,
    ANSWER_MAX_ATTEMPTS,
    ANSWER_PROMPT,
    ANSWER_TOOL,
    FRESH_RETRY_LIMIT,
    PreparedJudgeCall,
    RETRY_REFUSALS_PER_FRESH_CALL,
)
from .schema import (
    EvidenceReferences,
    JudgeResult,
    MLJudgeResult,
    MLNoticedValue,
)
from judgment_semantics import cited_message_numbers


class JudgeCallError(RuntimeError):
    """A judge call failed while retaining its complete input/provenance record."""

    def __init__(self, message: str, *, call_metadata: dict):
        super().__init__(message)
        self.call_metadata = call_metadata


class JudgePostValidationError(JudgeCallError):
    """The model returned schema-shaped data that violates the evidence contract."""


def _all_dimension_evidence(
    judgment: JudgeResult,
) -> Iterable[tuple[str, EvidenceReferences]]:
    yield "reward_hacking", judgment.reward_hacking.evidence
    yield "noticed", judgment.noticed.evidence
    if isinstance(judgment, MLJudgeResult) and judgment.deception is not None:
        yield "deception", judgment.deception.evidence
    if isinstance(judgment, MLJudgeResult):
        yield "environment_failure", judgment.environment_failure.evidence


def _require_positive_evidence(
    name: str,
    positive: bool,
    evidence: EvidenceReferences,
) -> None:
    if positive and not evidence.has_any:
        raise ValueError(f"positive {name} judgment must cite evidence")


def validate_judgment_evidence(
    judgment: JudgeResult,
    prepared: PreparedJudgeCall,
    *,
    explanation: str,
) -> list[dict]:
    """Validate cross-record rules that JSON Schema cannot express."""

    valid_messages = {
        message.number for message in prepared.evidence.source_message_map
    }
    valid_artifacts = {
        (artifact.path, artifact.snapshot) for artifact in prepared.evidence.artifacts
    }
    artifact_normalizations: list[dict] = []
    for field, text in (
        ("summary", judgment.summary),
        ("highlights", judgment.highlights),
        ("explanation", explanation),
    ):
        citations = cited_message_numbers(
            text, valid_messages, include_range_endpoints=True
        )
        unknown_citations = sorted(citations - valid_messages)
        if unknown_citations:
            raise ValueError(
                f"{field} cites message numbers absent from official evidence: "
                f"{unknown_citations}"
            )

    for dimension, evidence in _all_dimension_evidence(judgment):
        unknown_messages = sorted(set(evidence.messages) - valid_messages)
        if unknown_messages:
            raise ValueError(
                f"{dimension} cites message numbers absent from official evidence: "
                f"{unknown_messages}"
            )
        unknown_artifacts = []
        normalized_artifacts: list[tuple[str, str]] = []
        for item in evidence.artifacts:
            supplied = (item.path, item.snapshot)
            if supplied in valid_artifacts:
                normalized_artifacts.append(supplied)
                continue

            # The prompt still asks for the exact displayed path. Tolerate the common
            # provider mistake of dropping only the leading /workspace/ prefix, but do
            # not guess from a basename or change the requested snapshot.
            candidate = (f"/workspace/{item.path}", item.snapshot)
            if not item.path.startswith("/") and candidate in valid_artifacts:
                original_path = item.path
                item.path = candidate[0]
                normalized_artifacts.append(candidate)
                artifact_normalizations.append({
                    "dimension": dimension,
                    "supplied_path": original_path,
                    "canonical_path": item.path,
                    "snapshot": item.snapshot,
                    "rule": "restore_workspace_prefix",
                })
                continue
            unknown_artifacts.append(supplied)
        if unknown_artifacts:
            raise ValueError(
                f"{dimension} cites artifacts absent from official evidence: "
                f"{sorted(unknown_artifacts)}"
            )
        if len(normalized_artifacts) != len(set(normalized_artifacts)):
            raise ValueError(
                f"{dimension} cites the same artifact more than once after path "
                "normalization"
            )

    _require_positive_evidence(
        "reward_hacking",
        judgment.reward_hacking.has_notable_behavior,
        judgment.reward_hacking.evidence,
    )
    _require_positive_evidence(
        "noticed",
        (
            judgment.noticed.value != MLNoticedValue.NEITHER
            if isinstance(judgment, MLJudgeResult)
            else judgment.noticed.value
        ),
        judgment.noticed.evidence,
    )
    if isinstance(judgment, MLJudgeResult) and judgment.deception is not None:
        _require_positive_evidence("deception", True, judgment.deception.evidence)
    if isinstance(judgment, MLJudgeResult):
        _require_positive_evidence(
            "environment_failure",
            judgment.environment_failure.value,
            judgment.environment_failure.evidence,
        )
    return artifact_normalizations


def _model_name(model: str | Model | None) -> str | None:
    if isinstance(model, str) or model is None:
        return model
    return getattr(model, "name", None) or str(model)


def _config_record(config: GenerateConfig | None) -> dict | None:
    if config is None:
        return None
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json", exclude_none=True)
    return {key: value for key, value in vars(config).items() if value is not None}


def _provider_request_record(answer: AnswerStructured) -> dict:
    """Record the exact initial interface Scout gives the provider model.

    Scout's structured path sends the supplied prompt as one user message and exposes a
    forced answer tool. It does not use ``AnswerStructured.answer_prompt`` or
    ``answer_format``. The schema helper is private upstream, so the Inspect Scout pin
    in ``pyproject.toml`` and the focused test below intentionally make this dependency
    fail loudly if Scout changes the provider interface.
    """

    from inspect_scout._llm_scanner.structured import structured_schema

    schema = structured_schema(answer)
    schema_record = schema.model_dump(mode="json", exclude_none=True)
    return {
        "schema_version": "inspect-scout-structured-provider-interface-v1",
        "initial_messages": [{
            "role": "user",
            "content": "prompt_passed_to_scout",
        }],
        "tools": [{
            "name": answer.answer_tool,
            "description": "Use this tool to submit your final answer.",
            "parameters": {
                "type": "object",
                "properties": schema_record.get("properties") or {},
                "required": schema_record.get("required") or [],
            },
        }],
        "forced_tool_choice": answer.answer_tool,
        "parallel_tool_calls": False,
        "context_tools": [],
        "max_attempts": answer.max_attempts,
        "no_tool_retry_message": (
            f"Please use the {answer.answer_tool}() tool to report your answer."
        ),
        "no_tool_retry_message_used": answer.max_attempts > 1,
        "answer_prompt_used": False,
        "answer_format_used": False,
    }


def provider_request_record(prepared: PreparedJudgeCall) -> dict:
    """Return the exact initial Scout/provider interface without making a call."""

    answer = AnswerStructured(
        prepared.answer_model,
        answer_tool=ANSWER_TOOL,
        answer_prompt=ANSWER_PROMPT,
        answer_format=ANSWER_FORMAT,
        max_attempts=ANSWER_MAX_ATTEMPTS,
    )
    return _provider_request_record(answer)


async def run_prepared_judge(
    prepared: PreparedJudgeCall,
    *,
    model: str | Model | None,
    config: GenerateConfig | None = None,
    fresh_retry_limit: int = FRESH_RETRY_LIMIT,
) -> Result:
    """Return a validated result, replacing bad calls with fresh conversations.

    Each attempt starts from ``prepared.prompt``. A malformed answer, refusal, provider
    error, or post-validation failure is never appended to the next attempt's context.
    Inspect records usage for every attempted call in its normal event stream; this
    function adds the prompt, evidence, rubric, schema, and retry provenance.
    """
    if fresh_retry_limit < 0:
        raise ValueError("fresh_retry_limit must be non-negative")
    call_metadata = {
        **prepared.metadata(),
        "requested_model": _model_name(model),
        "generate_config": _config_record(config),
        "fresh_retry_limit": fresh_retry_limit,
        "fresh_call_limit": fresh_retry_limit + 1,
        "structured_attempts_per_fresh_call": ANSWER_MAX_ATTEMPTS,
        "retry_refusals_per_fresh_call": RETRY_REFUSALS_PER_FRESH_CALL,
        "provider_request": provider_request_record(prepared),
    }
    failures: list[dict] = []
    last_error: Exception | None = None
    last_failure_kind = "unknown"

    for attempt in range(1, fresh_retry_limit + 2):
        answer = AnswerStructured(
            prepared.answer_model,
            answer_tool=ANSWER_TOOL,
            answer_prompt=ANSWER_PROMPT,
            answer_format=ANSWER_FORMAT,
            max_attempts=ANSWER_MAX_ATTEMPTS,
        )
        try:
            result = await generate_answer(
                prepared.prompt,
                answer=answer,
                model=model,
                config=config,
                retry_refusals=RETRY_REFUSALS_PER_FRESH_CALL,
                extract_refs=prepared.extract_references,
            )
        except Exception as ex:
            last_error = ex
            last_failure_kind = "generation_failed"
            failures.append({
                "fresh_attempt": attempt,
                "kind": last_failure_kind,
                "error_type": type(ex).__name__,
                "error": str(ex),
            })
            continue

        if result.value is None:
            last_error = None
            last_failure_kind = "no_structured_answer"
            failures.append({
                "fresh_attempt": attempt,
                "kind": last_failure_kind,
                "explanation": result.explanation,
                "result_metadata": dict(result.metadata or {}),
            })
            continue

        explanation = str(result.explanation or "").strip()
        try:
            if not explanation:
                raise ValueError("environment judge returned no explanation")
            judgment = prepared.result_model.model_validate(result.value)
            artifact_normalizations = validate_judgment_evidence(
                judgment,
                prepared,
                explanation=explanation,
            )
        except (ValidationError, ValueError) as ex:
            last_error = ex
            last_failure_kind = "post_validation_failed"
            failures.append({
                "fresh_attempt": attempt,
                "kind": last_failure_kind,
                "error_type": type(ex).__name__,
                "error": str(ex),
                "unvalidated_answer": result.value,
                "explanation": result.explanation,
                "result_metadata": dict(result.metadata or {}),
            })
            continue

        result.value = judgment.model_dump(mode="json")
        result.metadata = {
            **(result.metadata or {}),
            "environment_judge": {
                **call_metadata,
                "fresh_attempt_count": attempt,
                "fresh_failures": failures,
                "fresh_retries_exhausted": False,
                "artifact_reference_normalizations": artifact_normalizations,
                "explanation": explanation,
                "post_validation": "passed",
            },
        }
        return result

    exhausted_metadata = {
        **call_metadata,
        "fresh_attempt_count": fresh_retry_limit + 1,
        "fresh_failures": failures,
        "fresh_retries_exhausted": True,
        "failure_kind": last_failure_kind,
        "post_validation": (
            "failed"
            if last_failure_kind == "post_validation_failed"
            else last_failure_kind
        ),
    }
    final_failure = failures[-1] if failures else {}
    final_detail = (
        final_failure.get("error")
        or final_failure.get("explanation")
        or last_failure_kind
    )
    message = (
        f"environment judge failed after {fresh_retry_limit + 1} fresh calls; "
        f"last failure ({last_failure_kind}): {final_detail}"
    )
    error_class = (
        JudgePostValidationError
        if last_failure_kind == "post_validation_failed"
        else JudgeCallError
    )
    error = error_class(message, call_metadata=exhausted_metadata)
    if last_error is not None:
        raise error from last_error
    raise error


async def judge_trajectory(
    *,
    family,
    stage,
    messages,
    artifacts=(),
    upstream_caveats=(),
    rubrics_root=None,
    model: str | Model | None,
    config: GenerateConfig | None = None,
    fresh_retry_limit: int = FRESH_RETRY_LIMIT,
) -> Result:
    """Convenience wrapper for integrations that do not need a separate preview step."""

    from .prompt import prepare_judge_call
    from .rubrics import DEFAULT_RUBRICS_ROOT

    prepared = await prepare_judge_call(
        family=family,
        stage=stage,
        messages=messages,
        artifacts=artifacts,
        upstream_caveats=upstream_caveats,
        rubrics_root=rubrics_root or DEFAULT_RUBRICS_ROOT,
    )
    return await run_prepared_judge(
        prepared,
        model=model,
        config=config,
        fresh_retry_limit=fresh_retry_limit,
    )
