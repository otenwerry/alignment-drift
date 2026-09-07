"""Cache, build-lock, and stable-ID infrastructure for the static viewer."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator


MODULE_ROOT = Path(__file__).resolve().parent
CACHE_DEPENDENCY_FILES = (
    MODULE_ROOT / "env_viewer_cache.py",
    MODULE_ROOT / "env_viewer_load.py",
    MODULE_ROOT / "env_viewer_store.py",
    MODULE_ROOT / "env_viewer_turns.py",
    MODULE_ROOT / "interrupted_native_transcript.py",
    MODULE_ROOT / "judgment_semantics.py",
    MODULE_ROOT / "real_integrity.py",
)


def module_signature(paths: tuple[Path, ...] = CACHE_DEPENDENCY_FILES) -> str:
    """Fingerprint every local module whose behavior is stored in cached audits."""

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def mode_signature(mode_dir: Path) -> str:
    rows = []
    paths = list(mode_dir.glob("*.eval"))
    for name in ("remote_campaign.json", "pipeline_integrity.json"):
        sidecar = mode_dir / name
        if sidecar.is_file():
            paths.append(sidecar)
    for path in sorted(paths):
        stat = path.stat()
        rows.append((path.name, stat.st_mtime_ns, stat.st_size))
    payload = json.dumps([module_signature(), rows], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def cache_file(mode_dir: Path, cache_root: Path) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", mode_dir.name)
    return cache_root / f"mode__{safe}__{mode_signature(mode_dir)}.pkl"


def write_pickle_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def trajectory_key(audit: dict) -> str:
    """Stable identity used by the existing environments data files."""

    retrospective = audit.get("retrospective_rejudge")
    if isinstance(retrospective, dict) and retrospective.get("source_key"):
        return f"{audit['mode']}__rejudge__{retrospective['source_key']}"
    stored_seed = audit.get("stored_seed") or audit["seed"]
    return (
        f"{audit['mode']}__{audit['task']}__{stored_seed}__e{audit['epoch']}"
    )


def assign_stable_ids(audits: list[dict], registry_file: Path) -> None:
    """Assign append-only IDs, refusing identities that would overwrite a page."""

    keys = [trajectory_key(audit) for audit in audits]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:3])
        suffix = f" (+{len(duplicates) - 3} more)" if len(duplicates) > 3 else ""
        raise ValueError(
            "duplicate trajectory identities would overwrite viewer pages: "
            f"{preview}{suffix}"
        )

    try:
        registry = json.loads(registry_file.read_text()) if registry_file.exists() else {}
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read trajectory ID registry {registry_file}: {error}") from error
    if not isinstance(registry, dict) or not all(
        isinstance(key, str) and isinstance(value, int)
        for key, value in registry.items()
    ):
        raise ValueError(f"invalid trajectory ID registry: {registry_file}")

    from project_paths import READ_ONLY_DATA

    next_id = max(registry.values(), default=0) + 1
    new = [audit for audit in audits if trajectory_key(audit) not in registry]
    if new and READ_ONLY_DATA:
        raise ValueError(f"Published registry is missing {len(new)} trajectory IDs; dataset is incomplete")
    for audit in sorted(
        new,
        key=lambda item: (
            item.get("mtime", 0), item.get("task", ""),
            item.get("seed", ""), item.get("epoch", 0),
        ),
    ):
        registry[trajectory_key(audit)] = next_id
        next_id += 1
    if new:
        registry_file.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(registry, indent=2, sort_keys=True) + "\n"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=registry_file.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, registry_file)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    for audit in audits:
        audit["id"] = registry[trajectory_key(audit)]


@contextmanager
def viewer_build_lock(data_root: Path) -> Iterator[None]:
    """Serialize viewer builds and release the lock automatically after crashes."""

    path = data_root / ".viewer_build.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
