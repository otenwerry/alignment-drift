"""Shared real-environment sandbox routing.

The local task builder and the AWS campaign controller must resolve a seed family to
the same Docker sandbox.  Keeping that mapping here means a future family is registered
once for both compute backends.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


FAMILY_SANDBOX = {
    "ml_prefix_only": "ml",
    "ml_training_data_misuse": "ml",
    "p_hacking": "p_hacking",
    "p_hacking_prefix_only": "p_hacking",
}

COMPOSE_FILE = "compose.yaml"
SUBSCRIPTION_COMPOSE_FILE = "compose.subscription.yaml"
SUPPORTED_COMPOSE_FILES = frozenset({COMPOSE_FILE, SUBSCRIPTION_COMPOSE_FILE})


def seed_family_from_path(seeds_path: str | Path) -> str:
    """Resolve a registered family from either its collection or one member path."""

    path = Path(seeds_path)
    for candidate in (path.name, path.parent.name):
        if candidate in FAMILY_SANDBOX:
            return candidate
    raise ValueError(
        f"seed path {seeds_path!s} has no registered real-environment sandbox; "
        f"registered families: {sorted(FAMILY_SANDBOX)}"
    )


def compose_file_for_harness(*, harness: str, scaffold: str) -> str:
    """Return the compose variant used by the real task builder."""

    if harness == "subscription" and scaffold != "opencode":
        return SUBSCRIPTION_COMPOSE_FILE
    return COMPOSE_FILE


def sandbox_compose_relative(family: str, compose_file: str) -> str:
    """Repo-relative compose path stored in AWS jobs and campaign records."""

    sandbox = FAMILY_SANDBOX.get(family)
    if sandbox is None:
        raise ValueError(
            f"unsupported seed family {family!r}; registered families: "
            f"{sorted(FAMILY_SANDBOX)}"
        )
    if compose_file not in SUPPORTED_COMPOSE_FILES:
        raise ValueError(f"unsupported sandbox compose file {compose_file!r}")
    return PurePosixPath(
        "environments", "sandbox", sandbox, compose_file
    ).as_posix()


def discover_sandbox_compose_files(environments_root: Path) -> list[str]:
    """Return every checked-in real-sandbox compose file, in stable order."""

    sandbox_root = environments_root.resolve() / "sandbox"
    paths = [
        (PurePosixPath("environments") / path.relative_to(environments_root.resolve()).as_posix()).as_posix()
        for path in sandbox_root.glob("*/compose*.yaml")
        if path.is_file()
    ]
    if not paths:
        raise ValueError(f"no sandbox compose files found under {sandbox_root}")
    return sorted(paths)
