"""Load every human-written environment-judge instruction from one Markdown tree."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .schema import JudgeFamily


DEFAULT_INSTRUCTIONS_ROOT = (
    Path(__file__).resolve().parents[2] / "judge_instructions"
)

FAMILY_INSTRUCTIONS = {
    "ml_training_data_misuse": (
        ("overall", "overall_instructions/ml_training_data_misuse.md"),
    ),
    "p_hacking": (
        ("overall", "overall_instructions/p_hacking.md"),
    ),
}


@dataclass(frozen=True)
class InstructionDocument:
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


def _load_document(
    key: str, relative_path: str, *, root: Path
) -> InstructionDocument:
    path = root / relative_path
    try:
        raw_content = path.read_text(encoding="utf-8")
    except FileNotFoundError as ex:
        raise FileNotFoundError(
            f"required judge instruction does not exist: {path}"
        ) from ex
    if not raw_content.strip():
        raise ValueError(f"judge instruction is empty: {path}")
    content = raw_content.rstrip("\n")
    return InstructionDocument(
        key=key,
        relative_path=relative_path,
        content=content,
        sha256=sha256(content.encode("utf-8")).hexdigest(),
    )


def load_instructions(
    family: JudgeFamily,
    *,
    root: Path = DEFAULT_INSTRUCTIONS_ROOT,
) -> tuple[InstructionDocument, ...]:
    """Load the selected family's overall instructions."""

    try:
        specs = FAMILY_INSTRUCTIONS[family]
    except KeyError as ex:
        raise ValueError(f"unsupported judge family: {family!r}") from ex
    documents = tuple(
        _load_document(key, relative_path, root=root)
        for key, relative_path in specs
    )
    keys = {document.key for document in documents}
    required = {"overall"}
    if keys != required:
        raise ValueError(
            f"judge instructions must contain exactly {sorted(required)}; got {sorted(keys)}"
        )
    return documents
