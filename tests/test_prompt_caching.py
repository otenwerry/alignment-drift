"""Free checks for provider prompt-prefix caching."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig


LIB = Path(__file__).resolve().parents[1] / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

import prompt_caching  # noqa: E402


def test_warmup_runs_one_call_before_fanning_out_fresh_calls() -> None:
    barrier = prompt_caching.PromptCacheWarmupBarrier()
    started: list[int] = []
    first_finished = False

    async def call() -> int:
        nonlocal first_finished
        number = len(started) + 1
        started.append(number)
        if number == 1:
            await asyncio.sleep(0.02)
            first_finished = True
        else:
            assert first_finished
        return number

    async def run() -> list[int]:
        return await asyncio.gather(
            *(barrier.run("same-prefix", call) for _ in range(4))
        )

    assert sorted(asyncio.run(run())) == [1, 2, 3, 4]
    assert barrier.stats() == {
        "warmed_prefixes": 1,
        "requests_held_for_warmup": 3,
        "failed_warmups": 0,
    }


def test_request_grouping_uses_stable_prefix_not_final_question_or_message_ids() -> None:
    model = SimpleNamespace(model_name="judge", base_url="https://example.invalid")
    first = [
        ChatMessageSystem(content="long stable rubric " * 300, id="system-a"),
        ChatMessageUser(content="trajectory one", id="question-a"),
    ]
    second = [
        ChatMessageSystem(content="long stable rubric " * 300, id="system-b"),
        ChatMessageUser(content="trajectory two", id="question-b"),
    ]
    kwargs_one = {"input": first, "tools": [], "config": GenerateConfig()}
    kwargs_two = {"input": second, "tools": [], "config": GenerateConfig()}

    assert prompt_caching._request_key(model, (), kwargs_one) == prompt_caching._request_key(
        model, (), kwargs_two
    )


def test_cache_report_never_calls_zero_usage_verified() -> None:
    used = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        input_tokens_cache_read=80,
        input_tokens_cache_write=0,
        total_cost=0.01,
    )
    unused = SimpleNamespace(
        input_tokens=50,
        output_tokens=10,
        input_tokens_cache_read=0,
        input_tokens_cache_write=0,
        total_cost=0.02,
    )
    log = SimpleNamespace(
        stats=SimpleNamespace(model_usage={"openai/used": used, "openai/unused": unused})
    )

    report = prompt_caching.build_report([log])

    assert report["verified_models"] == ["openai/used"]
    assert report["models"]["openai/unused"]["status"] == "no_cache_activity_observed"
    assert report["inspect_response_cache_enabled"] is False
