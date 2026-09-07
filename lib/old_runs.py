"""Shared manifest for runs and trajectories shown and rejudged as Old."""

from __future__ import annotations

import json
from pathlib import Path

from project_paths import ENVIRONMENTS_ROOT


OLD_RUNS_FILE = ENVIRONMENTS_ROOT / "viewer_old_runs.json"
OLD_RUNS_FORMAT = "environments-viewer-old-runs-v1"


def _old_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("format") != OLD_RUNS_FORMAT:
        raise ValueError(f"unsupported old-run manifest format: {path}")
    return payload


def _string_set(payload: dict, field: str, path: Path) -> frozenset[str]:
    values = payload.get(field, [])
    if (
        not isinstance(values, list)
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(f"invalid {field} in {path}")
    return frozenset(values)


def old_run_names(path: Path = OLD_RUNS_FILE) -> frozenset[str]:
    payload = _old_manifest(path)
    names = payload.get("run_directories")
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError(f"invalid run_directories in {path}")
    return frozenset(names)


def old_trajectory_keys(path: Path = OLD_RUNS_FILE) -> frozenset[str]:
    return _string_set(_old_manifest(path), "trajectory_keys", path)


def old_trajectory_key_reasons(path: Path = OLD_RUNS_FILE) -> dict[str, str]:
    """Map every archived trajectory key to the curation reason that archived it.

    Curation notes that list their keys explicitly (GLM context overruns, the stray
    baseline) claim those keys.  Every remaining key belongs to the first note without
    an explicit key list, which is the excess-retry policy recorded by selector counts.
    """

    payload = _old_manifest(path)
    keys = old_trajectory_keys(path)
    reasons: dict[str, str] = {}
    fallback: str | None = None
    for note in payload.get("curation_notes", []) or []:
        if not isinstance(note, dict) or not note.get("reason"):
            continue
        reason = str(note["reason"])
        listed = [
            str(key) for key in (note.get("trajectory_keys", []) or [])
        ] + [
            str(item.get("trajectory_key"))
            for item in (note.get("trajectories", []) or [])
            if isinstance(item, dict) and item.get("trajectory_key")
        ]
        if listed:
            for key in listed:
                if key in keys:
                    reasons[key] = reason
        elif fallback is None:
            fallback = reason
    for key in keys:
        reasons.setdefault(key, fallback or "listed_in_viewer_old_runs")
    return reasons


def old_run_directory_reasons(path: Path = OLD_RUNS_FILE) -> dict[str, str]:
    """Curation reason for whole archived run directories that a note names."""

    payload = _old_manifest(path)
    names = old_run_names(path)
    reasons: dict[str, str] = {}
    for note in payload.get("curation_notes", []) or []:
        if not isinstance(note, dict) or not note.get("reason"):
            continue
        for name in note.get("run_directories", []) or []:
            if str(name) in names:
                reasons[str(name)] = str(note["reason"])
    return reasons


def old_archive_policies(path: Path = OLD_RUNS_FILE) -> dict[str, str]:
    """Curation policy text per reason, for the viewer's Old explanation."""

    payload = _old_manifest(path)
    return {
        str(note["reason"]): str(note.get("policy") or "")
        for note in (payload.get("curation_notes", []) or [])
        if isinstance(note, dict) and note.get("reason")
    }


def old_prefix_files(path: Path = OLD_RUNS_FILE) -> frozenset[str]:
    """Purpose-built prefix payload basenames shown in the Old viewer window."""

    return _string_set(_old_manifest(path), "prefix_files", path)


def promoted_rejudge_run_names(path: Path = OLD_RUNS_FILE) -> frozenset[str]:
    """Rejudge runs explicitly approved to replace their source judgments."""

    names = _string_set(
        _old_manifest(path), "promoted_rejudge_run_directories", path
    )
    from project_paths import LOGS_ROOT, READ_ONLY_DATA

    # Fresh experiment corpora do not contain the historical study's rejudges.
    # Published analysis requires the complete set and keeps strict validation.
    if path == OLD_RUNS_FILE and not READ_ONLY_DATA:
        return frozenset(name for name in names if (LOGS_ROOT / name).is_dir())
    return names


def old_source_run_names(path: Path = OLD_RUNS_FILE) -> frozenset[str]:
    """Original real-v* directories needed by retrospective rejudging."""

    keys = old_trajectory_keys(path)
    malformed = sorted(key for key in keys if "__" not in key)
    if malformed:
        raise ValueError(f"invalid trajectory_keys in {path}: {malformed[:3]}")
    source_names = old_run_names(path) | frozenset(
        key.split("__", 1)[0] for key in keys
    )
    return frozenset(name for name in source_names if name.startswith("real-v"))
