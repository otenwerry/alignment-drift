"""Environment-owned agent protocol and seed discovery.

The real runner needs only the agent-facing protocol: one global system prompt, a
family opening-message template, optional scripted follow-ups, and member-specific
opening text.

All paths resolve under :mod:`environments/seeds`.  The loader also returns file hashes
so every run can stamp the exact source material it used.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from project_paths import ENVIRONMENTS_ROOT, SEEDS_ROOT, SYSTEM_PROMPT_PATH


SHARED_DIR_NAME = "_shared"
SCENARIO_FILENAME = "scenario.md"
USER_PROMPT_FILENAME = "USER_PROMPT.txt"
FOLLOW_UP_FILENAME = "FOLLOW_UP.txt"

USER_PROMPT_PLACEHOLDER = "{{SEED_SPECIFIC_PROMPT}}"
PRESSURE_PLACEHOLDER = "{{PRESSURE}}"
SCENARIO_USER_PROMPT_CORE_MARK = "[USER_PROMPT_CORE]"
SCENARIO_LOW_PRESSURE_MARK = "[LOW_PRESSURE]"
SCENARIO_HIGH_PRESSURE_MARK = "[HIGH_PRESSURE]"
P_HACKING_FAMILY = "p_hacking"
P_HACKING_PREFIX_ONLY_FAMILY = "p_hacking_prefix_only"
P_HACKING_PROTOCOL_FAMILIES = frozenset({
    P_HACKING_FAMILY,
    P_HACKING_PREFIX_ONLY_FAMILY,
})
PRESSURE_LEVELS = ("low", "high")

SUPPORTED_CONDITION = "allow"

THINKING_INSTRUCTION = (
    "Reason step by step inside `<thinking></thinking>` tags before taking actions. "
)


@dataclass(frozen=True)
class PressurePromptFields:
    """The core and interchangeable pressure fragments in p-hacking scenario.md."""

    core: str
    low: str
    high: str
    source_path: Path

    def render(self, pressure: str) -> str:
        """Insert one selected fragment, joining its boundaries with one space."""

        selected = {"low": self.low, "high": self.high}[pressure]
        before, after = self.core.split(PRESSURE_PLACEHOLDER, 1)
        parts = [before.strip(), selected.strip(), after.strip()]
        return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class ProtocolSources:
    """Resolved, agent-facing protocol for one real-environment member."""

    family: str
    member: str
    member_dir: Path
    pressure: str | None
    system_prompt: str
    opening_user_message: str
    follow_up: str | None
    source_files: tuple[Path, ...]

    @property
    def follow_up_user_messages(self) -> tuple[str, ...]:
        return (self.follow_up,) if self.follow_up is not None else ()

    def provenance(self) -> dict:
        """Stored, queryable identity for every protocol source file."""

        return {
            "protocol_source": "environments",
            "protocol_family": self.family,
            "protocol_member": self.member,
            "pressure": self.pressure,
            "protocol_source_files": [
                {
                    "path": _display_path(path),
                    "sha256": _sha256(path),
                }
                for path in self.source_files
            ],
        }


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ENVIRONMENTS_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_seeds(path: Path, *, seeds_root: Path = SEEDS_ROOT) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(Path(seeds_root).resolve())
    except ValueError as exc:
        raise SystemExit(f"seed path {path} is outside {seeds_root}") from exc
    return resolved


def is_seed_member(path: str | Path, *, seeds_root: Path = SEEDS_ROOT) -> bool:
    """Whether ``path`` is one complete environments-owned member."""

    member = Path(path)
    if not member.is_dir() or member.name == SHARED_DIR_NAME:
        return False
    if not (
        (member / SCENARIO_FILENAME).is_file()
        and (member / "manifest.json").is_file()
        and (member / "workspace").is_dir()
    ):
        return False
    try:
        _inside_seeds(member, seeds_root=seeds_root)
    except SystemExit:
        return False
    return True


def seed_members(collection: str | Path, *, seeds_root: Path = SEEDS_ROOT) -> list[str]:
    """Sorted active member names immediately below one family directory.

    Archived members remain directly addressable for exact historical reproduction,
    but family-wide ``--seeds=all`` must not spend money on them.
    """

    root = _inside_seeds(Path(collection), seeds_root=seeds_root)
    if not root.is_dir():
        return []
    return sorted(
        child.name
        for child in root.iterdir()
        if is_seed_member(child, seeds_root=seeds_root)
        and not json.loads((child / "manifest.json").read_text()).get("archived", False)
    )


def resolve_seeds(
    seed_dir: str | None,
    *,
    seeds_root: Path = SEEDS_ROOT,
) -> tuple[str, list[str]]:
    """Resolve ``--seed-dir`` within the environments-owned seed tree.

    Real runs require a family or member name.  A family returns all immediate runnable
    members; a member returns itself as the sole choice.
    """

    if seed_dir is None or not seed_dir.strip():
        families = sorted(
            path.name
            for path in Path(seeds_root).iterdir()
            if path.is_dir() and seed_members(path, seeds_root=seeds_root)
        )
        raise SystemExit(
            f"--seed-dir is required; available environment families: {families}"
        )

    candidate = _inside_seeds(Path(seeds_root) / seed_dir, seeds_root=seeds_root)
    if is_seed_member(candidate, seeds_root=seeds_root):
        return str(candidate), [candidate.name]
    members = seed_members(candidate, seeds_root=seeds_root)
    if members:
        return str(candidate), members

    families = sorted(
        path.name
        for path in Path(seeds_root).iterdir()
        if path.is_dir() and seed_members(path, seeds_root=seeds_root)
    )
    raise SystemExit(
        f"--seed-dir={seed_dir!r} is not a runnable environments family or member; "
        f"available families: {families}"
    )


def resolve_condition(condition: str | None) -> str:
    """Resolve the real protocol's currently supported condition."""

    value = SUPPORTED_CONDITION if condition is None else condition.strip()
    if value != SUPPORTED_CONDITION:
        raise SystemExit(
            f"real environments currently support only --condition={SUPPORTED_CONDITION}; "
            f"got {value!r}"
        )
    return value


def resolve_pressure(pressure: str | None, family: str) -> str | None:
    """Default p-hacking to low pressure; reject pressure elsewhere."""

    value = pressure.strip().casefold() if pressure is not None else None
    if family in P_HACKING_PROTOCOL_FAMILIES:
        if value is None:
            return "low"
        if value not in PRESSURE_LEVELS:
            raise SystemExit(
                f"--pressure must be low or high for p_hacking; got {pressure!r}"
            )
        return value
    if value is not None:
        raise SystemExit(
            f"--pressure applies only to p_hacking protocols; {family} does not use it"
        )
    return None


def resolve_reasoning(value: str | None) -> bool:
    """Resolve the agent ``--reasoning=yes|no`` flag; absent means yes."""

    if value is None:
        return True
    normalized = value.strip().casefold()
    if normalized in {"yes", "on", "true", "1"}:
        return True
    if normalized in {"no", "off", "false", "0"}:
        return False
    raise SystemExit(f"--reasoning must be yes|no, got {value!r}")


def reject_retired_fixed_system_prompt_flag(
    args: Sequence[str] | None = None,
) -> None:
    """Fail if the retired ``--fixed_sp`` switch appears."""

    actual = sys.argv if args is None else args
    if any(argument.startswith("--fixed_sp") for argument in actual):
        raise SystemExit(
            "--fixed_sp was removed: environments always reads "
            "seeds/SYSTEM_PROMPT.txt. Drop the flag."
        )


def global_system_prompt(*, path: Path = SYSTEM_PROMPT_PATH) -> str:
    """Read the environment-owned global agent prompt, failing on missing/empty."""

    if not path.is_file():
        raise SystemExit(f"global agent system prompt is missing: {path}")
    text = path.read_text().strip()
    if not text:
        raise SystemExit(f"global agent system prompt is empty: {path}")
    return text


def strip_thinking_instruction(system_prompt: str) -> str:
    """Remove the prompted scratchpad line when native model reasoning is enabled."""

    if THINKING_INSTRUCTION not in system_prompt:
        print(
            "  !! WARNING: reasoning is ON but the '<thinking> tags' instruction was "
            "not found in environments/seeds/SYSTEM_PROMPT.txt; using it unchanged"
        )
        return system_prompt
    return system_prompt.replace(THINKING_INSTRUCTION, "")


def _read_required(path: Path, description: str) -> str:
    if not path.is_file():
        raise SystemExit(f"{description} is missing: {path}")
    text = path.read_text().strip()
    if not text:
        raise SystemExit(f"{description} is empty: {path}")
    return text


def _read_optional(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text().strip()
    if not text:
        raise SystemExit(f"protocol source exists but is empty: {path}")
    return text


def parse_pressure_prompt(path: str | Path) -> PressurePromptFields:
    """Parse one strict p-hacking core/low/high scenario prompt."""

    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"member opening prompt is missing: {source}")
    text = source.read_text()
    marks = (
        SCENARIO_USER_PROMPT_CORE_MARK,
        SCENARIO_LOW_PRESSURE_MARK,
        SCENARIO_HIGH_PRESSURE_MARK,
    )
    counts = {mark: text.count(mark) for mark in marks}
    if any(count != 1 for count in counts.values()):
        raise SystemExit(
            f"{source} must contain each pressure marker exactly once; got {counts}"
        )
    positions = [text.index(mark) for mark in marks]
    if positions != sorted(positions):
        raise SystemExit(
            f"{source} pressure markers must appear in this order: {marks}"
        )
    if text[:positions[0]].strip():
        raise SystemExit(
            f"{source} cannot contain text before {SCENARIO_USER_PROMPT_CORE_MARK}"
        )

    core_tail = text.split(SCENARIO_USER_PROMPT_CORE_MARK, 1)[1]
    core, low_tail = core_tail.split(SCENARIO_LOW_PRESSURE_MARK, 1)
    low, high = low_tail.split(SCENARIO_HIGH_PRESSURE_MARK, 1)
    core = core.strip()
    if not core:
        raise SystemExit(f"{source} has an empty user-prompt core")
    placeholder_count = core.count(PRESSURE_PLACEHOLDER)
    if placeholder_count != 1:
        raise SystemExit(
            f"{source} user-prompt core must contain {PRESSURE_PLACEHOLDER} "
            f"exactly once; got {placeholder_count}"
        )
    return PressurePromptFields(
        core=core,
        low=low.strip(),
        high=high.strip(),
        source_path=source,
    )


def load_protocol_sources(
    member_path: str | Path,
    *,
    pressure: str | None = None,
    seeds_root: Path = SEEDS_ROOT,
) -> ProtocolSources:
    """Load and compose the complete agent-facing protocol for one member."""

    member = _inside_seeds(Path(member_path), seeds_root=seeds_root)
    if not is_seed_member(member, seeds_root=seeds_root):
        raise SystemExit(
            f"{member} is not a runnable environment member (needs scenario.md, "
            "manifest.json, and workspace/)"
        )
    family = member.parent.name
    resolved_pressure = resolve_pressure(pressure, family)
    shared = member.parent / SHARED_DIR_NAME
    scenario_path = member / SCENARIO_FILENAME
    prompt_path = shared / USER_PROMPT_FILENAME
    follow_up_path = shared / FOLLOW_UP_FILENAME
    readme_path = member / "workspace" / "README.md"

    if not prompt_path.is_file():
        raise SystemExit(f"family opening-prompt template is missing: {prompt_path}")
    if not readme_path.is_file():
        raise SystemExit(f"workspace task specification is missing: {readme_path}")

    member_opening = (
        parse_pressure_prompt(scenario_path).render(resolved_pressure)
        if resolved_pressure is not None
        else _read_required(scenario_path, "member opening prompt")
    )
    template = prompt_path.read_text().strip()
    count = template.count(USER_PROMPT_PLACEHOLDER)
    if count != 1:
        raise SystemExit(
            f"{prompt_path} must contain {USER_PROMPT_PLACEHOLDER} exactly once; got {count}"
        )
    opening = template.replace(USER_PROMPT_PLACEHOLDER, member_opening).strip()
    follow_up = _read_optional(follow_up_path)

    # Prefix-only controls can pin a member to one submission while retaining the
    # family's byte-identical shared follow-up for members that use it.
    manifest_path = member / "manifest.json"
    manifest_controls_follow_up = False
    if family == P_HACKING_PREFIX_ONLY_FAMILY:
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as error:
            raise SystemExit(f"invalid prefix-only manifest {manifest_path}: {error}") from error
        configured_follow_up = manifest.get("follow_up")
        if not isinstance(configured_follow_up, bool):
            raise SystemExit(
                f"{manifest_path} must contain a boolean follow_up field"
            )
        manifest_controls_follow_up = True
        if not configured_follow_up:
            follow_up = None

    system_prompt_path = Path(seeds_root) / SYSTEM_PROMPT_PATH.name
    source_files = [system_prompt_path, prompt_path, scenario_path, readme_path]
    if follow_up is not None:
        source_files.append(follow_up_path)
    if manifest_controls_follow_up:
        source_files.append(manifest_path)

    return ProtocolSources(
        family=family,
        member=member.name,
        member_dir=member,
        pressure=resolved_pressure,
        system_prompt=global_system_prompt(path=system_prompt_path),
        opening_user_message=opening,
        follow_up=follow_up,
        source_files=tuple(source_files),
    )


def protocol_source_plan(protocols: Iterable[ProtocolSources]) -> list[dict]:
    """Free pre-run summary suitable for printing or a dry-run manifest."""

    return [protocol.provenance() for protocol in protocols]
