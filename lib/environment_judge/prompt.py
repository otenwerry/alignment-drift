"""Assemble a complete, inspectable judge call without making a model request."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

from inspect_ai.model import ChatMessage

from .evidence import (
    ArtifactSnapshot,
    EvidenceBundle,
    EvidenceCaveat,
    render_evidence,
)
from .instructions import (
    DEFAULT_INSTRUCTIONS_ROOT,
    InstructionDocument,
    load_instructions,
)
from .rubrics import DEFAULT_RUBRICS_ROOT, RubricDocument, load_rubrics
from .schema import (
    JudgeAnswerType,
    JudgeFamily,
    JudgeResultType,
    JudgeStage,
    answer_model_for_family,
    result_model_for_family,
)


JUDGE_SCHEMA_VERSION = "environment-judge-v15"
ANSWER_TOOL = "submit_judgment"
ANSWER_PROMPT = "Use the {{ answer_tool }}() tool to return the judgment requested below:"
ANSWER_FORMAT = "Return the final judgment only by calling {{ answer_tool }}()."
# One structured-output attempt per provider conversation. If it is malformed, the
# runner discards that conversation and starts again from the original prompt.
ANSWER_MAX_ATTEMPTS = 1
FRESH_RETRY_LIMIT = 3
RETRY_REFUSALS_PER_FRESH_CALL = 0


def _build_prompt(
    *,
    instructions: Sequence[InstructionDocument],
    rubrics: Sequence[RubricDocument],
    evidence: EvidenceBundle,
) -> str:
    instruction_text = {document.key: document.content for document in instructions}
    rubric_sections = []
    for rubric in rubrics:
        rubric_sections.append(
            f"## Dimension: {rubric.key}\n\n{rubric.content.rstrip()}"
        )
    return "\n\n".join([
        "# Overall judge instructions\n\n" + instruction_text["overall"],
        "# Dimension rubrics\n\n" + "\n\n".join(rubric_sections),
        "# Numbered observable trajectory\n\n" + (
            evidence.rendered_messages or "(no messages were supplied for this stage)"
        ),
        "# Artifact snapshots\n\n" + evidence.rendered_artifacts,
    ])


@dataclass(frozen=True)
class PreparedJudgeCall:
    family: JudgeFamily
    stage: JudgeStage
    prompt: str
    result_model: JudgeResultType
    answer_model: JudgeAnswerType
    output_schema: dict
    instructions: tuple[InstructionDocument, ...]
    rubrics: tuple[RubricDocument, ...]
    evidence: EvidenceBundle
    extract_references: object

    def method_sha256(self) -> str:
        """Identity of the judge contract, excluding trajectory evidence and stage.

        Stage-one and final calls use the same instructions, rubrics, schema, and
        response interface.  Their evidence and rendered prompts can differ, but they
        are the same judging method.  Storing this digest lets the viewer separate
        current judgments from historical ones without guessing from run names.
        """

        payload = {
            "judge_schema_version": JUDGE_SCHEMA_VERSION,
            "family": self.family,
            "instructions": [item.metadata() for item in self.instructions],
            "rubrics": [item.metadata() for item in self.rubrics],
            "output_schema": self.output_schema,
            "scout_answer": {
                "answer_tool": ANSWER_TOOL,
                "answer_prompt": ANSWER_PROMPT,
                "answer_format": ANSWER_FORMAT,
                "max_attempts": ANSWER_MAX_ATTEMPTS,
                "fresh_retry_limit": FRESH_RETRY_LIMIT,
                "retry_refusals_per_fresh_call": RETRY_REFUSALS_PER_FRESH_CALL,
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()

    def metadata(self) -> dict:
        """Complete input/provenance record suitable for an Inspect score."""

        return {
            "judge_schema_version": JUDGE_SCHEMA_VERSION,
            "judge_method_sha256": self.method_sha256(),
            "family": self.family,
            "stage": self.stage,
            "fresh_call": True,
            "prompt_passed_to_scout": self.prompt,
            "prompt_sha256": sha256(self.prompt.encode("utf-8")).hexdigest(),
            "instructions": [
                instruction.metadata() for instruction in self.instructions
            ],
            "rubrics": [rubric.metadata() for rubric in self.rubrics],
            "evidence": self.evidence.model_dump(mode="json"),
            "output_schema": self.output_schema,
            "scout_answer": {
                "answer_tool": ANSWER_TOOL,
                "answer_prompt": ANSWER_PROMPT,
                "answer_format": ANSWER_FORMAT,
                "max_attempts": ANSWER_MAX_ATTEMPTS,
                "fresh_retry_limit": FRESH_RETRY_LIMIT,
                "retry_refusals_per_fresh_call": RETRY_REFUSALS_PER_FRESH_CALL,
            },
        }

    def instruction_text(self, key: str) -> str:
        """Return one loaded instruction by its stable metadata key."""

        for instruction in self.instructions:
            if instruction.key == key:
                return instruction.content
        raise KeyError(f"prepared judge call has no instruction {key!r}")


async def prepare_judge_call(
    *,
    family: JudgeFamily,
    stage: JudgeStage,
    messages: Sequence[ChatMessage],
    artifacts: Sequence[ArtifactSnapshot] = (),
    upstream_caveats: Sequence[EvidenceCaveat] = (),
    instructions_root: Path = DEFAULT_INSTRUCTIONS_ROOT,
    rubrics_root: Path = DEFAULT_RUBRICS_ROOT,
) -> PreparedJudgeCall:
    """Build a free prompt preview for one independent stage-1 or final call."""

    if stage not in ("stage_1", "final"):
        raise ValueError(f"unsupported judge stage: {stage!r}")
    result_model = result_model_for_family(family)
    answer_model = answer_model_for_family(family)
    instructions = load_instructions(family, root=instructions_root)
    rubrics = load_rubrics(family, root=rubrics_root)
    rendered = await render_evidence(
        messages,
        artifacts,
        family=family,
        upstream_caveats=upstream_caveats,
    )
    prompt = _build_prompt(
        instructions=instructions,
        rubrics=rubrics,
        evidence=rendered.bundle,
    )
    return PreparedJudgeCall(
        family=family,
        stage=stage,
        prompt=prompt,
        result_model=result_model,
        answer_model=answer_model,
        output_schema=answer_model.model_json_schema(),
        instructions=instructions,
        rubrics=rubrics,
        evidence=rendered.bundle,
        extract_references=rendered.extract_references,
    )
