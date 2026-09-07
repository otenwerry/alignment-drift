"""Real-environment workspace plumbing: host-side trees -> sandbox /workspace, plus
in-sandbox snapshots and end-of-run artifact extraction.

Each ported seed keeps its materialized tree at environments/seeds/<family>/<member>/:
  workspace/    exactly what lands in the sandbox at /workspace (committed bytes)
  answers/      host-side canon (e.g. the test-label key) for the grader/gate; NEVER
                shipped into the sandbox beyond what workspace/ itself contains
  manifest.json sha256 provenance written by the envgen generator

``real_env_dir()`` accepts either that member directory or another path whose last two
components are ``<family>/<member>``.

Artifact extraction is capped per file (ARTIFACT_CAP_BYTES) with explicit truncated
flags stored alongside. The capped copy is the ONLY copy anywhere — sample metadata,
the gate/judge prompts, the grade input, and the sidecar dir all hold the same
head -c bytes (the container is gone once the sample returns). What survives a cut is
the flag: `truncated` + true `bytes` on the file entry, `input_truncated` on the grade,
and a loud badge/banner in the viewer (surface-lossy rule: nothing is silently
dropped)."""

from __future__ import annotations

import hashlib
import json
import posixpath
from pathlib import Path, PurePosixPath

from inspect_ai.util import sandbox

ENVIRONMENTS_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS_SEEDS = ENVIRONMENTS_ROOT / "seeds"

WORKSPACE_DIR = "/workspace"
HOST_ONLY_WORKSPACE_PARTS = frozenset({
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
})
MODELS_SUBDIR = "models"
# Per-file cap for extracted artifact copies. No full copy exists past this point (the
# sidecar dir mirrors the same capped bytes), so a cut must travel as flags: "truncated"
# + true "bytes" on the entry, input_truncated on the grade, badge/banner in the viewer.
ARTIFACT_CAP_BYTES = 200_000
# Extensions extracted as text artifacts when created/modified during the run.
ARTIFACT_SUFFIXES = {".py", ".md", ".csv", ".txt", ".json"}


def deliverable_path(name: str, workspace_dir: str = WORKSPACE_DIR) -> str:
    """Canonical sandbox path of a protocol deliverable.

    The task README tells the agent exactly where each deliverable lives (the
    workspace root: "write the test-set predictions to `predictions.csv`"), and
    extract_artifacts' always_include pulls from this same path. Every consumer — the
    gate's key_agreement detector, the gate/judge deliverable rendering, the final
    grade, and the presence check — must resolve a deliverable through here, so they
    all read the SAME file; a same-named draft in a subdirectory is not a deliverable.
    """
    return posixpath.join(workspace_dir, name)


def deliverable_entry(artifacts: dict, name: str) -> dict | None:
    """The extracted-artifact entry for a deliverable, or None if it was not in the
    final workspace (extract_artifacts always includes deliverables that exist)."""
    target = deliverable_path(name)
    return next(
        (f for f in artifacts.get("files", []) if f.get("path") == target), None
    )


def real_env_dir(seed_path: str | Path) -> Path:
    """Return the environments-owned member directory for ``seed_path``."""
    seed = Path(seed_path).resolve()
    return ENVIRONMENTS_SEEDS / seed.parent.name / seed.name


def has_real_env(seed_path: str | Path) -> bool:
    return (real_env_dir(seed_path) / "workspace").is_dir()


def answers_dir(seed_path: str | Path) -> Path:
    return real_env_dir(seed_path) / "answers"


def workspace_files(seed_path: str | Path) -> dict[str, str]:
    """Sample.files mapping: absolute sandbox path -> absolute host path.

    .gitkeep placeholders are skipped (they exist only so git keeps empty dirs);
    their parent dirs are reported by ensure_dirs() and created by the solver.

    A manifest may also declare ``workspace_imports``. These are hash-pinned files
    elsewhere in the committed seed tree that land at named workspace-relative
    destinations. Imports let a control seed reuse the exact train/validation bytes
    of another seed without copying its test material. Sources must themselves live
    below another seed's ``workspace/`` directory; host-only ``answers/`` trees can
    never be imported.
    """
    member = real_env_dir(seed_path)
    ws = member / "workspace"
    if not ws.is_dir():
        raise SystemExit(
            f"{seed_path} has no real_env/workspace/ tree; generate it with the "
            "matching envgen/gen_*.py script before running a real audit"
        )
    mapping: dict[str, str] = {}
    for f in sorted(ws.rglob("*")):
        relative_parts = f.relative_to(ws).parts
        if any(part in HOST_ONLY_WORKSPACE_PARTS for part in relative_parts):
            continue
        if f.is_file() and f.name != ".gitkeep" and f.suffix != ".pyc":
            rel = f.relative_to(ws)
            mapping[posixpath.join(WORKSPACE_DIR, rel.as_posix())] = str(f)
    manifest_path = member / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as error:
            raise SystemExit(f"invalid seed manifest {manifest_path}: {error}") from error
        imports = manifest.get("workspace_imports") or {}
        if not isinstance(imports, dict):
            raise SystemExit(f"{manifest_path}: workspace_imports must be an object")
        seeds_root = ENVIRONMENTS_SEEDS.resolve()
        for destination, record in sorted(imports.items()):
            destination_path = PurePosixPath(str(destination))
            if (
                destination_path.is_absolute()
                or not destination_path.parts
                or ".." in destination_path.parts
            ):
                raise SystemExit(
                    f"{manifest_path}: invalid workspace import destination "
                    f"{destination!r}"
                )
            sandbox_path = posixpath.join(WORKSPACE_DIR, destination_path.as_posix())
            if sandbox_path in mapping:
                raise SystemExit(
                    f"{manifest_path}: workspace import overwrites {sandbox_path}"
                )
            if not isinstance(record, dict):
                raise SystemExit(
                    f"{manifest_path}: workspace import {destination!r} must be an object"
                )
            source_value = record.get("source")
            expected_sha = record.get("sha256")
            if not isinstance(source_value, str) or not isinstance(expected_sha, str):
                raise SystemExit(
                    f"{manifest_path}: workspace import {destination!r} needs source "
                    "and sha256 strings"
                )
            source = (seeds_root / source_value).resolve()
            try:
                source_relative = source.relative_to(seeds_root)
            except ValueError as error:
                raise SystemExit(
                    f"{manifest_path}: workspace import source escapes the seed tree: "
                    f"{source_value!r}"
                ) from error
            if (
                len(source_relative.parts) < 4
                or source_relative.parts[2] != "workspace"
                or "answers" in source_relative.parts
            ):
                raise SystemExit(
                    f"{manifest_path}: workspace import source must be under a seed's "
                    f"workspace tree, not {source_value!r}"
                )
            if not source.is_file():
                raise SystemExit(
                    f"{manifest_path}: workspace import source is missing: {source}"
                )
            actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                raise SystemExit(
                    f"{manifest_path}: workspace import {destination!r} hash mismatch; "
                    f"expected {expected_sha}, got {actual_sha}"
                )
            mapping[sandbox_path] = str(source)
    if not mapping:
        raise SystemExit(f"{ws} is empty")
    return mapping


def ensure_dirs(seed_path: str | Path) -> list[str]:
    """Sandbox dirs that must exist but ship empty (parents of .gitkeep files)."""
    ws = real_env_dir(seed_path) / "workspace"
    return sorted(
        posixpath.join(WORKSPACE_DIR, f.parent.relative_to(ws).as_posix())
        for f in ws.rglob(".gitkeep")
    )


# --------------------------------------------------------------------------- #
# in-sandbox operations (call only inside a running sample)
# --------------------------------------------------------------------------- #

async def create_empty_dirs(seed_path: str | Path) -> None:
    dirs = ensure_dirs(seed_path)
    if dirs:
        await sandbox().exec(["mkdir", "-p", *dirs])


def parse_sha_listing(stdout: str) -> dict[str, str]:
    """`sha256sum` output -> {sandbox path: sha256}."""
    snapshot: dict[str, str] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, path = line.partition("  ")
        if len(digest) == 64 and path:
            snapshot[path] = digest
    return snapshot


async def workspace_snapshot() -> dict[str, str]:
    """{path: sha256} for every file under /workspace, excluding models/ (weights are
    large and their identity is captured by models_listing instead)."""
    result = await sandbox().exec(
        ["bash", "-lc",
         f"find {WORKSPACE_DIR} -type f -not -path '{WORKSPACE_DIR}/{MODELS_SUBDIR}/*' "
         "-print0 | xargs -0 -r sha256sum"],
        timeout=120,
    )
    if not result.success:
        raise RuntimeError(f"workspace snapshot failed: {result.stderr[:500]}")
    return parse_sha_listing(result.stdout)


async def models_listing() -> list[dict]:
    """Name + size of everything under /workspace/models/ (no content extraction)."""
    result = await sandbox().exec(
        ["bash", "-lc",
         f"find {WORKSPACE_DIR}/{MODELS_SUBDIR} -type f -printf '%s %p\\n' 2>/dev/null || true"],
        timeout=60,
    )
    listing = []
    for line in result.stdout.splitlines():
        size, _, path = line.strip().partition(" ")
        if size.isdigit() and path:
            listing.append({"path": path, "bytes": int(size)})
    return listing


async def read_artifact(path: str, cap: int = ARTIFACT_CAP_BYTES) -> dict:
    """One artifact file -> {path, content, truncated, bytes} with the cap surfaced."""
    result = await sandbox().exec(
        ["bash", "-lc", f"wc -c < {_shquote(path)} && head -c {cap} {_shquote(path)}"],
        timeout=60,
    )
    if not result.success:
        return {"path": path, "content": None, "truncated": False,
                "error": (result.stderr or "unreadable")[:200]}
    first_line, _, rest = result.stdout.partition("\n")
    try:
        size = int(first_line.strip())
    except ValueError:
        size = len(rest)
    return {
        "path": path,
        "content": rest,
        "bytes": size,
        "truncated": size > cap,
    }


async def extract_artifacts(
    start_snapshot: dict[str, str],
    end_snapshot: dict[str, str],
    always_include: tuple[str, ...] = (),
) -> dict:
    """Everything the run created or modified (by snapshot diff), as capped copies.

    always_include: workspace-relative deliverable names (e.g. "RESULT.md") pulled even
    if unchanged, so the grader/judge always see the final deliverables."""
    changed = sorted(
        path for path, digest in end_snapshot.items()
        if start_snapshot.get(path) != digest
    )
    wanted: list[str] = []
    for path in changed:
        if Path(path).suffix.lower() in ARTIFACT_SUFFIXES:
            wanted.append(path)
    for name in always_include:
        path = posixpath.join(WORKSPACE_DIR, name)
        if path in end_snapshot and path not in wanted:
            wanted.append(path)
    skipped = [p for p in changed if p not in wanted]
    files = [await read_artifact(p) for p in wanted]
    for entry in files:
        entry["sha256"] = end_snapshot.get(entry.get("path"))
    truncated = [entry.get("path") for entry in files if entry.get("truncated")]
    lossy_reasons = []
    if skipped:
        lossy_reasons.append("changed_non_text_files_not_copied")
    if truncated:
        lossy_reasons.append("artifact_content_truncated")
    return {
        "files": files,
        "changed_paths": changed,
        "skipped_non_text": skipped,   # surfaced, not silently dropped
        "models": await models_listing(),
        "lossy": bool(lossy_reasons),
        "lossy_reasons": lossy_reasons,
        "truncated_paths": truncated,
        "selection": "changed text files plus named deliverables",
    }


def _shquote(path: str) -> str:
    return "'" + path.replace("'", "'\\''") + "'"
