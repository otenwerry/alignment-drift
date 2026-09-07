"""Loss-visible rendering of the evidence supplied to the environment judge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Sequence

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ContentReasoning,
    ContentToolUse,
)
from inspect_scout import MessagesPreprocessor, message_numbering
from pydantic import Field, model_validator

from .schema import (
    ArtifactReference,
    JudgeFamily,
    MessageNumber,
    NonEmptyText,
    StrictModel,
)


class ArtifactSnapshot(StrictModel):
    """One complete artifact state at a named boundary.

    This layer never caps content.  ``truncated`` and ``read_error`` describe loss that
    already happened upstream, and are rendered in the prompt as well as stored in
    queryable metadata.
    """

    path: NonEmptyText
    snapshot: NonEmptyText
    content: str | None
    sha256: str | None = None
    byte_count: Annotated[int, Field(ge=0)] | None = None
    truncated: bool = False
    read_error: str | None = None

    @model_validator(mode="after")
    def unavailable_content_is_explained(self) -> "ArtifactSnapshot":
        if self.content is None and not self.read_error:
            raise ValueError("an artifact without content must include read_error")
        return self

    @property
    def reference(self) -> ArtifactReference:
        return ArtifactReference(path=self.path, snapshot=self.snapshot)


class EvidenceCaveat(StrictModel):
    """A stored and prompt-visible qualification on the evidence."""

    code: NonEmptyText
    description: NonEmptyText
    source: Literal["judge_builder", "upstream"]
    messages: list[MessageNumber]
    artifacts: list[ArtifactReference]


class MessageSource(StrictModel):
    """Map a prompt-local number back to the original Inspect trajectory."""

    number: Annotated[int, Field(ge=1)]
    label: NonEmptyText
    source_index: Annotated[int, Field(ge=0)]
    source_id: NonEmptyText
    role: NonEmptyText


class EvidenceBundle(StrictModel):
    rendered_messages: str
    rendered_artifacts: str
    source_message_map: list[MessageSource]
    artifacts: list[ArtifactSnapshot]
    caveats: list[EvidenceCaveat]
    native_reasoning_policy: Literal["included", "excluded"]
    system_messages_policy: Literal["included", "excluded"]
    assistant_visible_text_policy: Literal["included", "submissions_only"]
    tool_calls_policy: Literal["included", "excluded"]
    tool_results_policy: Literal["included", "excluded"]
    native_reasoning_message_count: Annotated[int, Field(ge=0)]
    native_reasoning_block_count: Annotated[int, Field(ge=0)]
    native_reasoning_plaintext_block_count: Annotated[int, Field(ge=0)]
    native_reasoning_summary_block_count: Annotated[int, Field(ge=0)]
    native_reasoning_unavailable_block_count: Annotated[int, Field(ge=0)]
    source_tool_call_count: Annotated[int, Field(ge=0)]
    source_embedded_tool_use_block_count: Annotated[int, Field(ge=0)]
    source_tool_result_message_count: Annotated[int, Field(ge=0)]
    source_message_count: Annotated[int, Field(ge=0)]
    selected_message_count: Annotated[int, Field(ge=0)]
    omitted_message_count: Annotated[int, Field(ge=0)]
    message_selection: Literal[
        "complete_observable_trajectory_for_stage",
        "user_turns_and_assistant_submissions_for_stage",
        "system_user_and_assistant_content_without_tool_usage_for_stage",
    ]
    builder_truncated_evidence: Literal[False] = False

    @model_validator(mode="after")
    def unique_sources(self) -> "EvidenceBundle":
        numbers = [item.number for item in self.source_message_map]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("source message numbers must be contiguous from 1")
        if len(self.source_message_map) != self.selected_message_count:
            raise ValueError("selected message count must match the source message map")
        if (
            self.source_message_count - self.selected_message_count
            != self.omitted_message_count
        ):
            raise ValueError("source, selected, and omitted message counts are inconsistent")
        if (
            self.native_reasoning_plaintext_block_count
            + self.native_reasoning_summary_block_count
            + self.native_reasoning_unavailable_block_count
            != self.native_reasoning_block_count
        ):
            raise ValueError("native reasoning availability counts are inconsistent")
        artifact_keys = [(item.path, item.snapshot) for item in self.artifacts]
        if len(artifact_keys) != len(set(artifact_keys)):
            raise ValueError("artifact path/snapshot pairs must be unique")
        return self


def _native_reasoning_blocks(message: ChatMessage) -> list[ContentReasoning]:
    if not isinstance(message, ChatMessageAssistant) or not isinstance(message.content, list):
        return []
    return [item for item in message.content if isinstance(item, ContentReasoning)]


def _embedded_tool_use_block_count(message: ChatMessage) -> int:
    if not isinstance(message.content, list):
        return 0
    return sum(isinstance(item, ContentToolUse) for item in message.content)


def _select_messages(
    messages: Sequence[ChatMessage], family: JudgeFamily
) -> tuple[list[tuple[int, ChatMessage]], dict[str, str]]:
    """Apply the explicit family evidence policy without changing source order."""

    indexed = list(enumerate(messages))
    if family in {"ml_training_data_misuse", "p_hacking"}:
        return indexed, {
            "native_reasoning_policy": "included",
            "system_messages_policy": "included",
            "assistant_visible_text_policy": "included",
            "tool_calls_policy": "included",
            "tool_results_policy": "included",
            "message_selection": "complete_observable_trajectory_for_stage",
        }
    raise ValueError(f"unsupported judge family: {family!r}")


def _remap_upstream_caveat(
    caveat: EvidenceCaveat,
    source_number_to_prompt_number: dict[int, int],
) -> EvidenceCaveat:
    """Map original one-based message positions onto the selected prompt messages."""

    return caveat.model_copy(update={
        "messages": [
            source_number_to_prompt_number[number]
            for number in caveat.messages
            if number in source_number_to_prompt_number
        ],
    })


def _render_artifacts(artifacts: Sequence[ArtifactSnapshot]) -> str:
    if not artifacts:
        return "(no artifact snapshots were supplied for this stage)"

    sections: list[str] = []
    for index, artifact in enumerate(artifacts, start=1):
        header = [
            f"[A{index}] path={artifact.path!r} snapshot={artifact.snapshot!r}",
            f"sha256={artifact.sha256 or 'not_recorded'} ",
            f"bytes={artifact.byte_count if artifact.byte_count is not None else 'not_recorded'}",
        ]
        statuses: list[str] = []
        if artifact.truncated:
            statuses.append("TRUNCATED BEFORE JUDGING")
        if artifact.content is None:
            statuses.append(f"UNAVAILABLE: {artifact.read_error}")
        status = f"status={'; '.join(statuses) if statuses else 'complete'}"
        content = artifact.content if artifact.content is not None else "(content unavailable)"
        sections.append("\n".join([
            *header,
            status,
            "----- BEGIN ARTIFACT CONTENT -----",
            content,
            "----- END ARTIFACT CONTENT -----",
        ]))
    return "\n\n".join(sections)


@dataclass(frozen=True)
class RenderedEvidence:
    """Evidence plus Scout's reference extractor for this one fresh prompt."""

    bundle: EvidenceBundle
    extract_references: object


async def render_evidence(
    messages: Sequence[ChatMessage],
    artifacts: Sequence[ArtifactSnapshot] = (),
    *,
    family: JudgeFamily,
    upstream_caveats: Sequence[EvidenceCaveat] = (),
) -> RenderedEvidence:
    """Render the selected family's exact, stored message-evidence policy."""

    messages = list(messages)
    artifacts = list(artifacts)
    message_ids = [message.id for message in messages]
    if len(message_ids) != len(set(message_ids)):
        raise ValueError("Inspect message IDs must be unique for evidence numbering")

    selected, policy = _select_messages(messages, family)
    selected_messages = [message for _, message in selected]
    render, extract_refs, label_for_id = message_numbering(
        MessagesPreprocessor(
            exclude_system=False,
            exclude_reasoning=policy["native_reasoning_policy"] == "excluded",
            exclude_tool_usage=policy["tool_calls_policy"] == "excluded",
        ),
        label_for_id=True,
    )
    rendered_messages = await render(selected_messages)
    source_map: list[MessageSource] = []
    reasoning_numbers: list[int] = []
    incomplete_reasoning_numbers: list[int] = []
    tool_call_numbers: list[int] = []
    reasoning_block_count = 0
    reasoning_plaintext_block_count = 0
    reasoning_summary_block_count = 0
    reasoning_unavailable_block_count = 0
    source_number_to_prompt_number: dict[int, int] = {}
    for source_index, message in selected:
        label = label_for_id(message.id)
        if label is None or not label.startswith("M") or not label[1:].isdigit():
            raise RuntimeError(f"Scout did not number Inspect message {message.id!r}")
        number = int(label[1:])
        source_number_to_prompt_number[source_index + 1] = number
        source_map.append(MessageSource(
            number=number,
            label=label,
            source_index=source_index,
            source_id=message.id,
            role=message.role,
        ))
        message_reasoning = _native_reasoning_blocks(message)
        message_reasoning_blocks = len(message_reasoning)
        if message_reasoning_blocks:
            reasoning_numbers.append(number)
            reasoning_block_count += message_reasoning_blocks
            message_incomplete = False
            for block in message_reasoning:
                if block.redacted and block.summary:
                    reasoning_summary_block_count += 1
                    message_incomplete = True
                elif not block.redacted and block.reasoning:
                    reasoning_plaintext_block_count += 1
                else:
                    reasoning_unavailable_block_count += 1
                    message_incomplete = True
            if message_incomplete:
                incomplete_reasoning_numbers.append(number)
        if (
            isinstance(message, ChatMessageAssistant)
            and (
                bool(message.tool_calls)
                or _embedded_tool_use_block_count(message) > 0
            )
        ):
            tool_call_numbers.append(number)

    source_tool_call_count = sum(
        len(message.tool_calls or [])
        for message in messages
        if isinstance(message, ChatMessageAssistant)
    )
    source_embedded_tool_use_block_count = sum(
        _embedded_tool_use_block_count(message) for message in messages
    )
    source_tool_result_message_count = sum(
        isinstance(message, ChatMessageTool) for message in messages
    )

    caveats = [
        _remap_upstream_caveat(caveat, source_number_to_prompt_number)
        for caveat in upstream_caveats
    ]
    omitted_count = len(messages) - len(selected_messages)
    if policy["tool_calls_policy"] == "excluded" and (
        source_tool_call_count or source_embedded_tool_use_block_count
    ):
        caveats.append(EvidenceCaveat(
            code="tool_calls_excluded_by_family_policy",
            description=(
                "Assistant reasoning and visible text are retained, but the judge does "
                f"not receive {source_tool_call_count} ordinary tool call(s) or "
                f"{source_embedded_tool_use_block_count} embedded tool-use block(s)."
            ),
            source="judge_builder",
            messages=tool_call_numbers,
            artifacts=[],
        ))
    if policy["tool_results_policy"] == "excluded" and (
        source_tool_result_message_count or source_embedded_tool_use_block_count
    ):
        caveats.append(EvidenceCaveat(
            code="tool_results_excluded_by_family_policy",
            description=(
                f"The judge does not receive {source_tool_result_message_count} "
                "tool-result message(s) or the results inside "
                f"{source_embedded_tool_use_block_count} embedded tool-use block(s)."
            ),
            source="judge_builder",
            messages=[],
            artifacts=[],
        ))
    if reasoning_summary_block_count or reasoning_unavailable_block_count:
        caveats.append(EvidenceCaveat(
            code="native_reasoning_not_fully_available_upstream",
            description=(
                "The provider did not expose the complete text of every native "
                f"reasoning block: {reasoning_summary_block_count} block(s) are "
                "represented only by provider summaries and "
                f"{reasoning_unavailable_block_count} block(s) have no readable "
                "reasoning. The hidden original reasoning is not reconstructed."
            ),
            source="upstream",
            messages=incomplete_reasoning_numbers,
            artifacts=[],
        ))

    return RenderedEvidence(
        bundle=EvidenceBundle(
            rendered_messages=rendered_messages,
            rendered_artifacts=_render_artifacts(artifacts),
            source_message_map=source_map,
            artifacts=artifacts,
            caveats=caveats,
            **policy,
            native_reasoning_message_count=len(reasoning_numbers),
            native_reasoning_block_count=reasoning_block_count,
            native_reasoning_plaintext_block_count=reasoning_plaintext_block_count,
            native_reasoning_summary_block_count=reasoning_summary_block_count,
            native_reasoning_unavailable_block_count=(
                reasoning_unavailable_block_count
            ),
            source_tool_call_count=source_tool_call_count,
            source_embedded_tool_use_block_count=(
                source_embedded_tool_use_block_count
            ),
            source_tool_result_message_count=source_tool_result_message_count,
            source_message_count=len(messages),
            selected_message_count=len(selected_messages),
            omitted_message_count=omitted_count,
        ),
        extract_references=extract_refs,
    )
