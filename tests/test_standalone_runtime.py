"""Free checks for the environments-owned runtime boundaries."""

from __future__ import annotations

import ast
import sys
import importlib
from pathlib import Path


ENVIRONMENTS = Path(__file__).resolve().parents[1]
LIB = ENVIRONMENTS / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from cost_tracking import estimate_usage_cost  # noqa: E402
from judge_selection import resolve_judge  # noqa: E402
from model_catalog import resolve_target  # noqa: E402
from protocol_sources import (  # noqa: E402
    load_protocol_sources,
    parse_pressure_prompt,
    resolve_seeds,
)
from environment_judge.rubrics import load_rubrics  # noqa: E402


EXPECTED_MEMBERS = {
    "ml_prefix_only": {
        "demand_forecasting",
        "fraud_detection",
    },
    "ml_training_data_misuse": {
        "demand_forecasting",
        "fraud_detection",
        "rating_prediction",
    },
    "p_hacking": {
        "checkout_redesign",
        "reasoning_prompt_benchmark",
        "retrieval_practice",
    },
}


def test_production_python_has_no_petri_runtime_imports() -> None:
    offenders = []
    for path in ENVIRONMENTS.rglob("*.py"):
        relative = path.relative_to(ENVIRONMENTS)
        # Dot directories hold third-party caches (.venv, .uv-cache), including
        # deliberately non-UTF-8 fixture files.
        if any(part.startswith(".") for part in relative.parts[:-1]) or (
            "tests" in relative.parts
        ):
            continue
        tree = ast.parse(path.read_text(), filename=str(relative))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if module == "petri" or module.startswith(("petri.", "mats.petri")):
                    offenders.append(f"{relative}:{node.lineno}:{module}")
    assert offenders == []


def test_all_protocols_are_local_and_complete() -> None:
    for family, expected in EXPECTED_MEMBERS.items():
        root, members = resolve_seeds(family)
        assert set(members) == expected
        for member in members:
            member_path = Path(root) / member
            protocol = load_protocol_sources(
                member_path,
                pressure="high" if family == "p_hacking" else None,
            )
            assert protocol.family == family
            assert protocol.member == member
            assert protocol.system_prompt
            assert protocol.opening_user_message
            rendered = (
                parse_pressure_prompt(member_path / "scenario.md").render("high")
                if family == "p_hacking"
                else (member_path / "scenario.md").read_text().strip()
            )
            assert protocol.opening_user_message.startswith(rendered + "\n\n")
            assert "{{PRESSURE}}" not in protocol.opening_user_message
            assert "{{SEED_SPECIFIC_PROMPT}}" not in protocol.opening_user_message
            assert protocol.source_files
            assert all(path.is_relative_to(ENVIRONMENTS) for path in protocol.source_files)
            provenance = protocol.provenance()
            assert provenance["protocol_source"] == "environments"
            assert all(
                len(source["sha256"]) == 64
                for source in provenance["protocol_source_files"]
            )


def test_active_rubrics_route_by_family_without_petri_types() -> None:
    ml = load_rubrics("ml_training_data_misuse")
    p_hacking = load_rubrics("p_hacking")
    assert [rubric.key for rubric in ml] == [
        "reward_hacking",
        "noticed",
        "deception",
        "environment_failure",
    ]
    assert [rubric.key for rubric in p_hacking] == [
        "reward_hacking",
        "noticed",
    ]
    assert all(len(rubric.sha256) == 64 for rubric in (*ml, *p_hacking))


def test_model_and_judge_selection_use_environment_catalogs() -> None:
    assert resolve_target("gpt-5.5") == "openai/gpt-5.5-2026-04-23"
    assert resolve_judge("gpt-5.6-luna") == "openai/gpt-5.6-luna"


def test_openrouter_fallback_cost_is_labeled_estimated() -> None:
    priced = estimate_usage_cost(
        "openrouter/qwen/qwen3-32b",
        {"input": 1_000_000, "output": 100_000},
    )
    assert priced == {
        "cost_usd": 0.11,
        "exact": False,
        "source": "openrouter-derived",
    }


def test_zero_billed_cost_remains_exact() -> None:
    priced = estimate_usage_cost(
        "openrouter/qwen/qwen3-32b",
        {"input": 1_000, "output": 100, "total_cost": 0.0},
    )
    assert priced == {
        "cost_usd": 0.0,
        "exact": True,
        "source": "billed_or_direct_list",
    }


def test_inspect_runner_is_lazy_on_import() -> None:
    runner = importlib.import_module("exp" + "_inspect_runner")
    assert callable(runner.run_eval)
