"""Deterministic, credential-filtered source bundles for AWS workers."""

from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


_EXCLUDED_DIRS = frozenset({
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", "mats-local", "archive", "outputs",
})
_SECRET_FILE_NAMES = frozenset({
    "credentials", "credentials.json", "application_default_credentials.json",
    "id_rsa", "id_ed25519", ".netrc", ".npmrc", ".pypirc",
})
_SECRET_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx"})


class SourceBundleError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_root(environments_root: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(environments_root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip()).resolve()


def _bundle_path_allowed(relative: Path) -> bool:
    parts = relative.parts
    if not parts or relative.is_absolute() or ".." in parts:
        return False
    if parts[0] == "data":
        return False
    if any(part in _EXCLUDED_DIRS for part in parts):
        return False
    name = relative.name
    if name == ".env" or name.startswith(".env."):
        return False
    if name in _SECRET_FILE_NAMES or relative.suffix.lower() in _SECRET_SUFFIXES:
        return False
    return True


def source_file_list(repo: Path, environments_root: Path) -> list[Path]:
    """Current tracked and non-ignored source bytes under ``environments_root``."""

    try:
        source_relative = environments_root.resolve().relative_to(repo.resolve())
    except ValueError as error:
        raise SourceBundleError(
            f"environments root {environments_root} is outside git root {repo}"
        ) from error
    result = subprocess.run(
        [
            "git", "-C", str(repo), "ls-files", "-z", "--cached", "--others",
            "--exclude-standard", "--", source_relative.as_posix(),
        ],
        capture_output=True,
        check=True,
    )
    files = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        path = repo / relative
        if _bundle_path_allowed(relative.relative_to(source_relative)) and (path.is_file() or path.is_symlink()):
            files.append(relative)
    return sorted(set(files), key=lambda path: path.as_posix())


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    if info.isfile():
        info.mode = 0o755 if info.mode & 0o111 else 0o644
    return info


def build_source_bundle(environments_root: Path, output_dir: Path) -> dict:
    """Archive only the standalone environments runtime from the current worktree."""

    repo = repo_root(environments_root)
    files = source_file_list(repo, environments_root)
    if not files:
        raise SourceBundleError(f"no source files found under {environments_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "source.tar.gz.tmp"
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                root_info = tarfile.TarInfo("mats")
                root_info.type = tarfile.DIRTYPE
                root_info.mode = 0o755
                archive.addfile(_tar_filter(root_info))
                for relative in files:
                    archive.add(
                        repo / relative,
                        arcname=(
                            PurePosixPath("mats/environments") / relative.relative_to(
                                environments_root.resolve().relative_to(repo.resolve())
                            ).as_posix()
                        ).as_posix(),
                        recursive=False,
                        filter=_tar_filter,
                    )
    sha = sha256_file(temporary)
    final = output_dir / f"source-{sha}.tar.gz"
    os.replace(temporary, final)
    return {
        "path": str(final),
        "sha256": sha,
        "bytes": final.stat().st_size,
        "files": len(files),
        "repo_root": str(repo),
    }


def runtime_hash(environments_root: Path, runtime_version: str) -> str:
    repo = repo_root(environments_root)
    candidates = runtime_dependency_files(repo, environments_root)
    digest = hashlib.sha256(runtime_version.encode())
    for relative in candidates:
        path = repo / relative
        if not path.is_file():
            raise SourceBundleError(f"runtime dependency file is missing: {path}")
        canonical = path.relative_to(environments_root.resolve()).as_posix()
        digest.update(canonical.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def runtime_dependency_files(repo: Path, environments_root: Path) -> list[str]:
    """Files whose bytes must match the reusable AMI's installed runtime.

    Every sandbox is included automatically. A future registered seed family therefore
    does not need a second AWS-only dependency list when its Docker files are added.
    """

    try:
        environments_relative = environments_root.resolve().relative_to(repo.resolve())
    except ValueError as error:
        raise SourceBundleError(
            f"environments root {environments_root} is outside git root {repo}"
        ) from error
    required = [
        environments_relative / "pyproject.toml",
        environments_relative / "uv.lock",
    ]
    sandbox_root = environments_root.resolve() / "sandbox"
    if not sandbox_root.is_dir():
        raise SourceBundleError(f"runtime sandbox directory is missing: {sandbox_root}")
    sandbox_files = [
        path.relative_to(repo.resolve())
        for path in sandbox_root.rglob("*")
        if path.is_file()
        and _bundle_path_allowed(path.relative_to(repo.resolve()))
    ]
    return sorted(
        {path.as_posix() for path in [*required, *sandbox_files]}
    )
