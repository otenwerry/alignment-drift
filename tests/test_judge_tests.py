"""The Judge Tests endpoint must target only the saved viewer cohort."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ENVIRONMENTS = Path(__file__).resolve().parents[1]


def test_judge_tests_defaults_to_fixed_cohort_without_running_it() -> None:
    main = Mock()
    with patch.object(
        sys,
        "argv",
        ["exp_judge_tests.py", "--dry-run"],
    ), patch.dict(
        sys.modules, {"exp_rejudge": SimpleNamespace(main=main)}
    ):
        runpy.run_path(str(ENVIRONMENTS / "exp_judge_tests.py"), run_name="__main__")
        arguments = list(sys.argv)

    assert "--source-runs=judge-tests" in arguments
    main.assert_called_once_with()


def test_explicit_source_selection_is_preserved() -> None:
    main = Mock()
    with patch.object(
        sys,
        "argv",
        [
            "exp_judge_tests.py",
            "--source-runs=one",
            "--judge=gpt-5.6-luna",
            "--dry-run",
        ],
    ), patch.dict(sys.modules, {"exp_rejudge": SimpleNamespace(main=main)}):
        runpy.run_path(str(ENVIRONMENTS / "exp_judge_tests.py"), run_name="__main__")
        arguments = list(sys.argv)

    assert arguments.count("--source-runs=one") == 1
    assert "--source-runs=judge-tests" not in arguments
