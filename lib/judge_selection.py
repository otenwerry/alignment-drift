"""One environment-owned place for selecting the structured judge model."""

from __future__ import annotations

import os

from model_catalog import route


JUDGE_CHOICES: dict[str, str] = {
    "gpt-5.6-luna": "openai/gpt-5.6-luna",
    "deepseek-v4-pro": "openrouter/deepseek/deepseek-v4-pro-20260423",
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "opus-4.8": "anthropic/claude-opus-4-8",
    "sonnet-4.6": "anthropic/claude-sonnet-4-6",
}

DEFAULT_JUDGE_NAME = "gpt-5.6-luna"
JUDGE_ENV_VAR = "ENVIRONMENTS_JUDGE"


def resolve_judge(judge: str | None = None) -> str:
    """Resolve an explicit flag, then ``ENVIRONMENTS_JUDGE``, then the default."""

    value = judge or os.environ.get(JUDGE_ENV_VAR) or DEFAULT_JUDGE_NAME
    value = value.strip()
    if not value:
        raise SystemExit(
            f"empty judge; pass a shortname {sorted(JUDGE_CHOICES)} or provider/model"
        )
    if value in JUDGE_CHOICES:
        return route(JUDGE_CHOICES[value])
    if "/" in value:
        return route(value)
    raise SystemExit(
        f"unknown judge {value!r}; choices: {sorted(JUDGE_CHOICES)}, or provider/model"
    )


def judge_shortname(slug: str) -> str | None:
    """Reverse lookup for display when the routed slug is in the catalog."""

    routed = route(slug)
    return next(
        (name for name, value in JUDGE_CHOICES.items() if route(value) == routed),
        None,
    )


DEFAULT_JUDGE = resolve_judge()
