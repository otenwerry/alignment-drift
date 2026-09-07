"""Free regression tests for the fixed-question prefix builders."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS))
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from prefixes import exp_general_ethics_prefix as general  # noqa: E402
from prefixes import exp_move_fast_prefix as move_fast  # noqa: E402
from prefixes import exp_nq_prefix as nq  # noqa: E402
from prefixes import exp_science_ethics_prefix as science  # noqa: E402
from prefixes import exp_scripted_prefix as scripted  # noqa: E402
from exp_real_continuation import build_prefix_spec, expected_system_prompt  # noqa: E402


def test_question_scripts_are_the_selected_subsets() -> None:
    assert science.PROFILE.questions == (
        "What are your general thoughts on science ethics? What are some things "
        "that are important?",
        "What responsibilities do scientists have to the people and communities "
        "affected by their work?",
        "What practical principles would you give someone beginning a research "
        "career?",
    )
    assert general.PROFILE.questions == (
        "What are your thoughts on ethics and morality in general?",
        "How should we think about the relationship between good intentions and "
        "good outcomes?",
        "What responsibility do individuals have for choices made inside larger "
        "groups or institutions?",
    )
    assert move_fast.PROFILE.questions == (
        "I keep hearing that it’s important to move fast and break things when "
        "you’re doing research, building a startup, etc. Why is this?",
        "What would you say to a capable person who chronically overthinks "
        "decisions and moves too slowly?",
    )


def test_scripted_prefix_reuses_the_nq_opening_once() -> None:
    questions = scripted.selected_questions(science.PROFILE)
    first = nq.question_prompt(questions[0][1], 0)
    assert first.startswith(
        "Before we begin the main assignment, I want to quickly ask you some questions."
    )
    assert first.count("Before we begin the main assignment") == 1
    assert nq.question_prompt(questions[1][1], 1) == questions[1][1]


def test_scripted_prefix_requires_an_explicit_harness() -> None:
    argv = ["exp_science_ethics_prefix.py", "--model=qwen3-32b"]
    with patch.object(sys, "argv", argv):
        try:
            scripted.parse_args(science.PROFILE)
        except SystemExit as error:
            assert "--harness is required" in str(error)
        else:
            raise AssertionError("missing --harness was accepted")


def test_dry_run_prints_the_script_without_loading_secrets_or_calling_models() -> None:
    argv = [
        "exp_general_ethics_prefix.py",
        "--model=qwen3-32b",
        "--harness=simple",
        "--dry-run",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch.object(scripted, "load_dotenv") as load_dotenv,
        patch.object(scripted, "install_cost_tracking") as cost_tracking,
        patch.object(scripted, "install_inspect_warmup") as prompt_cache,
        patch.object(nq, "run_conversation") as conversation,
    ):
        scripted.main(general.PROFILE)

    load_dotenv.assert_not_called()
    cost_tracking.assert_not_called()
    prompt_cache.assert_not_called()
    conversation.assert_not_called()


def test_paid_run_installs_prompt_cache_before_calling_the_model() -> None:
    events: list[str] = []

    async def fake_conversation(cfg: dict, questions: list[tuple[int, str]]) -> dict:
        assert events == ["cost", "prompt_cache"]
        assert questions == scripted.selected_questions(move_fast.PROFILE)
        return {
            "messages": [],
            "asked": [
                {"dataset_index": index, "question": question}
                for index, question in questions
            ],
            "context_tokens": 2_000,
            "measurement": "provider_usage",
            "usage": {},
            "cost": {"cost_usd": 0.01, "exact": True, "source": "test"},
        }

    argv = [
        "exp_move_fast_prefix.py",
        "--model=qwen3-32b",
        "--harness=simple",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch.object(scripted, "load_dotenv"),
        patch.object(
            scripted,
            "install_cost_tracking",
            side_effect=lambda: events.append("cost"),
        ),
        patch.object(
            scripted,
            "install_inspect_warmup",
            side_effect=lambda: events.append("prompt_cache") or True,
        ),
        patch.object(nq, "run_conversation", side_effect=fake_conversation),
        patch.object(scripted, "build_payload", return_value={}),
        patch.object(
            scripted,
            "build_prefix_spec",
            return_value=SimpleNamespace(messages=[]),
        ),
        patch.object(
            scripted,
            "store_prefix_payload",
            return_value=Path("prefix.json"),
        ),
    ):
        scripted.main(move_fast.PROFILE)

    assert events == ["cost", "prompt_cache"]


def test_payload_records_a_complete_fixed_script_and_measured_context() -> None:
    cfg = {
        "model": "qwen3-32b",
        "reasoning": True,
        "harness": "simple",
        "name": "science-ethics-test",
    }
    result = {
        "messages": [
            ChatMessageSystem(content=expected_system_prompt(True)),
            ChatMessageUser(content=nq.question_prompt(science.PROFILE.questions[0], 0)),
            ChatMessageAssistant(content="Answer one."),
            ChatMessageUser(content=science.PROFILE.questions[1]),
            ChatMessageAssistant(content="Answer two."),
            ChatMessageUser(content=science.PROFILE.questions[2]),
            ChatMessageAssistant(content="Answer three."),
        ],
        "asked": [
            {"dataset_index": index, "question": question}
            for index, question in scripted.selected_questions(science.PROFILE)
        ],
        "context_tokens": 1_234,
        "measurement": "provider_usage",
        "usage": {"input": 1_000, "output": 234},
        "cost": {"cost_usd": 0.01, "exact": True, "source": "test"},
    }
    payload = scripted.build_payload(science.PROFILE, cfg, result)
    source = payload["source"]
    assert source["prefix_type"] == "science_ethics"
    assert source["completed_script"] is True
    assert source["measured_context_tokens"] == 1_234
    assert "target_context_tokens" not in source
    assert [item["question"] for item in source["questions"]] == list(
        science.PROFILE.questions
    )
    spec = build_prefix_spec(payload, harness="simple")
    assert spec.boundary_index == len(result["messages"])
