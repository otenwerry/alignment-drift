"""Free regression tests for the Natural Questions prefix builder."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS))
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from prefixes import exp_nq_prefix as nq  # noqa: E402


def test_first_question_explains_the_pre_assignment_interlude_once() -> None:
    first = nq.question_prompt("Who wrote Hamlet?", 0)
    assert first.startswith("Before we begin the main assignment")
    assert first.endswith("First question: Who wrote Hamlet?")
    assert nq.question_prompt("Where is Lima?", 1) == "Where is Lima?"


def test_context_size_includes_provider_cache_tokens() -> None:
    assert nq._context_tokens_from_usage({
        "input": 100,
        "cache_read": 20_000,
        "cache_write": 5_000,
        "output": 400,
    }) == 25_500


def test_nq_prefix_requires_explicit_harness() -> None:
    argv = [
        "exp_nq_prefix.py",
        "--model=qwen3-32b",
        "--tokens=1000",
    ]
    with patch.object(sys, "argv", argv):
        try:
            nq._parse_args()
        except SystemExit as error:
            assert "--harness is required" in str(error)
        else:
            raise AssertionError("missing --harness was accepted")


def test_nq_prefix_accepts_subscription_harness() -> None:
    argv = [
        "exp_nq_prefix.py",
        "--model=qwen3-32b",
        "--tokens=1000",
        "--harness=subscription",
    ]
    with patch.object(sys, "argv", argv):
        assert nq._parse_args()["harness"] == "subscription"


def test_dry_run_uses_disposable_cache_and_never_loads_secrets() -> None:
    seen: dict[str, object] = {}

    def fake_load_questions(cache_dir: Path = nq.HF_CACHE) -> list[str]:
        seen["cache_dir"] = cache_dir
        seen["existed_during_call"] = cache_dir.is_dir()
        return ["question one", "question two"]

    argv = [
        "exp_nq_prefix.py",
        "--model=qwen3-32b",
        "--tokens=1000",
        "--harness=simple",
        "--max-questions=2",
        "--dry-run",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch.object(nq, "load_questions", side_effect=fake_load_questions),
        patch.object(nq, "load_dotenv") as load_dotenv,
    ):
        nq.main()

    cache_dir = seen["cache_dir"]
    assert isinstance(cache_dir, Path)
    assert seen["existed_during_call"] is True
    assert not cache_dir.exists()
    load_dotenv.assert_not_called()


def test_paid_run_installs_prompt_cache_before_conversation() -> None:
    events: list[str] = []

    async def fake_conversation(cfg: dict, selected: list[tuple[int, str]]) -> dict:
        assert events == ["cost", "prompt_cache"]
        assert selected == [(0, "question one")]
        return {
            "messages": [],
            "asked": [{"dataset_index": 0, "question": "question one"}],
            "context_tokens": 1_000,
            "measurement": "provider_usage",
            "reached_target_tokens": True,
            "usage": {},
            "cost": {"cost_usd": 0.01, "exact": True, "source": "test"},
        }

    argv = [
        "exp_nq_prefix.py",
        "--model=qwen3-32b",
        "--tokens=1000",
        "--harness=simple",
        "--max-questions=1",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch.object(nq, "load_dotenv"),
        patch.object(nq, "load_questions", return_value=["question one"]),
        patch.object(
            nq,
            "install_cost_tracking",
            side_effect=lambda: events.append("cost"),
        ),
        patch.object(
            nq,
            "install_inspect_warmup",
            side_effect=lambda: events.append("prompt_cache") or True,
        ),
        patch.object(nq, "run_conversation", side_effect=fake_conversation),
        patch.object(nq, "build_payload", return_value={}),
        patch.object(
            nq,
            "build_prefix_spec",
            return_value=SimpleNamespace(messages=[]),
        ),
        patch.object(nq, "store_prefix_payload", return_value=Path("prefix.json")),
    ):
        nq.main()

    assert events == ["cost", "prompt_cache"]


def test_paid_run_refuses_spend_when_prompt_cache_install_fails() -> None:
    argv = [
        "exp_nq_prefix.py",
        "--model=qwen3-32b",
        "--tokens=1000",
        "--harness=simple",
        "--max-questions=1",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch.object(nq, "load_dotenv"),
        patch.object(nq, "load_questions", return_value=["question one"]),
        patch.object(nq, "install_cost_tracking"),
        patch.object(nq, "install_inspect_warmup", return_value=False),
        patch.object(nq, "run_conversation") as conversation,
    ):
        try:
            nq.main()
        except SystemExit as error:
            assert "refusing" in str(error)
        else:
            raise AssertionError("paid run continued without prompt-cache routing")

    conversation.assert_not_called()
