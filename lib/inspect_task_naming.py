"""Filesystem-safe Inspect task names with deterministic identity preservation."""

from __future__ import annotations

import hashlib


# Inspect derives SQLite sample-buffer filenames from the task name. Leave enough
# room under the 255-byte filename-component limit for its timestamp, task id,
# process id, and SQLite's temporary ``-journal``/``-wal`` suffixes.
MAX_INSPECT_TASK_NAME_CHARS = 160


def full_inspect_task_name(base_name: str, task_id_suffix: str | None = None) -> str:
    """Return the complete human-readable name before filesystem compaction."""

    return f"{base_name}_{task_id_suffix}" if task_id_suffix else base_name


def bounded_inspect_task_name(
    base_name: str,
    task_id_suffix: str | None = None,
    *,
    limit: int = MAX_INSPECT_TASK_NAME_CHARS,
) -> str:
    """Bound a task name while retaining its remote suffix and full-name hash."""

    full_name = full_inspect_task_name(base_name, task_id_suffix)
    if len(full_name) <= limit:
        return full_name

    digest = hashlib.sha256(full_name.encode()).hexdigest()[:12]
    tail = f"_{digest}"
    if task_id_suffix:
        tail += f"_{task_id_suffix}"
    available = limit - len(tail)
    if available < 1:
        raise ValueError(
            f"task suffix is too long for the {limit}-character Inspect name limit"
        )
    head = base_name[:available].rstrip("_-") or "task"
    return f"{head}{tail}"
