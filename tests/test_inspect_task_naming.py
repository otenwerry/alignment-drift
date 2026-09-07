"""Regression tests for Inspect's filesystem-facing task identifiers."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from inspect_task_naming import (  # noqa: E402
    MAX_INSPECT_TASK_NAME_CHARS,
    bounded_inspect_task_name,
    full_inspect_task_name,
)


def test_long_continuation_name_leaves_room_for_sqlite_journal(tmp_path) -> None:
    treatment = "purpose-built-prefix-native-fixed"
    prefix_name = "nq1234-2000-deepseek-v4-pro-20260812001909"
    task_suffix = (
        "purpose-built-prefix-native-fixed-nq1234-2000-de-b948ed03"
    )
    base = (
        f"continuation_{treatment}_deepseek-v4-pro-20260423_"
        f"checkout_redesign_p{prefix_name}"
    )

    full_name = full_inspect_task_name(base, task_suffix)
    bounded = bounded_inspect_task_name(base, task_suffix)

    assert len(full_name) == 191
    assert len(bounded) <= MAX_INSPECT_TASK_NAME_CHARS
    assert bounded.endswith(task_suffix)
    assert bounded == bounded_inspect_task_name(base, task_suffix)

    # Inspect prefixes this with a timestamp and appends a task id/process id. The
    # old 191-character task name produced the 253-character filename from the failed
    # campaign, which SQLite could not extend with ``-journal``.
    filename = f"2026-08-12T21-09-55-00-00_{bounded}_{'Z' * 22}.eval.1699.db"
    assert len(filename + "-journal") <= 255
    connection = sqlite3.connect(tmp_path / filename)
    try:
        connection.execute("CREATE TABLE sample (id INTEGER)")
        connection.commit()
    finally:
        connection.close()


def test_short_task_name_is_unchanged() -> None:
    assert bounded_inspect_task_name("continuation_clean_model_seed_pprefix") == (
        "continuation_clean_model_seed_pprefix"
    )
