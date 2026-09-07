"""Load the small, fixed rubric set owned by the environments experiment."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .instructions import DEFAULT_INSTRUCTIONS_ROOT
from .schema import JudgeFamily


DEFAULT_RUBRICS_ROOT = DEFAULT_INSTRUCTIONS_ROOT / "judge_dimensions"

RUBRICS_BY_FAMILY = {
    "ml_training_data_misuse": (
        ("reward_hacking", "ml_training_data_misuse/reward_hacking.md"),
        ("noticed", "ml_training_data_misuse/noticed.md"),
        ("deception", "ml_training_data_misuse/deception.md"),
        ("environment_failure", "ml_training_data_misuse/environment_failure.md"),
    ),
    "p_hacking": (
        ("reward_hacking", "p_hacking/reward_hacking.md"),
        ("noticed", "p_hacking/noticed.md"),
    ),
}


@dataclass(frozen=True)
class RubricDocument:
    key: str
    relative_path: str
    content: str
    sha256: str

    def metadata(self) -> dict:
        return {
            "key": self.key,
            "path": self.relative_path,
            "content": self.content,
            "sha256": self.sha256,
        }


def load_rubrics(
    family: JudgeFamily,
    *,
    root: Path = DEFAULT_RUBRICS_ROOT,
) -> tuple[RubricDocument, ...]:
    """Load the selected family's complete ordered rubric set.

    File order is fixed here because prompt order and schema agreement are part of the
    experiment contract, not a behavior inferred from Markdown metadata.
    """

    try:
        specs = RUBRICS_BY_FAMILY[family]
    except KeyError as ex:
        raise ValueError(f"unsupported judge family: {family!r}") from ex

    documents: list[RubricDocument] = []
    for key, relative_path in specs:
        path = root / relative_path
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError as ex:
            raise FileNotFoundError(
                f"required {family} judge rubric does not exist: {path}"
            ) from ex
        if not content.strip():
            raise ValueError(f"judge rubric is empty: {path}")
        documents.append(RubricDocument(
            key=key,
            relative_path=relative_path,
            content=content,
            sha256=sha256(content.encode("utf-8")).hexdigest(),
        ))
    return tuple(documents)
