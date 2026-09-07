"""Environment-owned model catalog, provider routing, and agent construction.

The catalog contains experiment configuration, not Petri behavior.  Inspect is imported
only when an agent model is actually constructed, so free planning and validation can
use the rest of this module without loading the evaluation runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


DIRECT_VENDORS = frozenset({"anthropic", "openai"})

# OpenRouter and first-party OpenAI do not always use the same snapshot spelling.
OPENROUTER_TO_DIRECT = {
    "openrouter/openai/gpt-5.5-20260423": "openai/gpt-5.5-2026-04-23",
    "openrouter/openai/gpt-5.4-mini": "openai/gpt-5.4-mini-2026-03-17",
}

TARGET_CHOICES: dict[str, str] = {
    "qwen3-32b": "openrouter/qwen/qwen3-32b",
    "qwen3.7-max": "openrouter/qwen/qwen3.7-max",
    "opus-4.6": "anthropic/claude-opus-4-6",
    "sonnet-4.6": "anthropic/claude-sonnet-4-6",
    "qwen2.5-72b": "openrouter/qwen/qwen-2.5-72b-instruct",
    "llama-3.3-70b": "openrouter/meta-llama/llama-3.3-70b-instruct",
    "deepseek-v3": "openrouter/deepseek/deepseek-chat",
    "gemma-3-27b": "openrouter/google/gemma-3-27b-it",
    "mistral-small": "openrouter/mistralai/mistral-small-3.1-24b-instruct",
    "deepseek-r1": "openrouter/deepseek/deepseek-r1",
    "glm-5.2": "openrouter/z-ai/glm-5.2",
    "glm-5.1": "openrouter/z-ai/glm-5.1",
    "gpt-5.4-mini": "openrouter/openai/gpt-5.4-mini",
    "grok-4.20": "openrouter/x-ai/grok-4.20",
    "kimi-k2.6": "openrouter/moonshotai/kimi-k2.6-20260420",
    "deepseek-v4-pro": "openrouter/deepseek/deepseek-v4-pro-20260423",
    "gpt-5.5": "openrouter/openai/gpt-5.5-20260423",
    "mimo-v2.5-pro": "openrouter/xiaomi/mimo-v2.5-pro",
    "minimax-m2.7": "openrouter/minimax/minimax-m2.7",
    "gpt-5.6-sol": "openrouter/openai/gpt-5.6-sol",
    "gpt-5.6-luna": "openrouter/openai/gpt-5.6-luna",
}

REASONING_EFFORT = "medium"


@dataclass(frozen=True)
class TargetModelBuild:
    """The model object and exact settings that should be stamped into a run."""

    model: Any
    routed_slug: str
    reasoning_on: bool
    reasoning_effort: str | None
    reasoning_enabled: bool | None
    reasoning_history: str | None
    strict_tools: bool | None
    prompt_cache_key: str | None

    def metadata(self) -> dict:
        return {
            "target_model": self.routed_slug,
            "target_reasoning": self.reasoning_on,
            "target_reasoning_effort": self.reasoning_effort,
            "target_reasoning_enabled": self.reasoning_enabled,
            "target_reasoning_history": self.reasoning_history,
            "target_strict_tools": self.strict_tools,
            "target_prompt_cache_key": self.prompt_cache_key,
        }


def route(slug: str) -> str:
    """Route first-party vendors directly and all other catalog slugs as written."""

    if not slug or "/" not in slug:
        return slug
    if slug in OPENROUTER_TO_DIRECT:
        return OPENROUTER_TO_DIRECT[slug]
    if slug.startswith("openrouter/"):
        remainder = slug.removeprefix("openrouter/")
        if remainder.split("/", 1)[0] in DIRECT_VENDORS:
            return remainder
    return slug


def resolve_target(name: str) -> str:
    """Resolve a catalog shortname to its routed provider/model slug."""

    value = name.strip()
    if value not in TARGET_CHOICES:
        raise SystemExit(f"unknown agent {value!r}; choices: {sorted(TARGET_CHOICES)}")
    return route(TARGET_CHOICES[value])


def resolve_target_names(names: Iterable[str]) -> list[str]:
    """Resolve names in order, rejecting emptiness and removing duplicates."""

    selected = list(dict.fromkeys(name.strip() for name in names if name.strip()))
    if not selected:
        raise SystemExit(f"no agent names were supplied; choices: {sorted(TARGET_CHOICES)}")
    return [resolve_target(name) for name in selected]


def same_model(left: str, right: str) -> bool:
    """Provider-route-aware equality for roles that must use the same model."""

    return route(left) == route(right)


def build_target(
    slug: str,
    *,
    reasoning_on: bool,
    effort: str = REASONING_EFFORT,
    prompt_cache_key: str | None = None,
    construct_model: bool = True,
) -> TargetModelBuild:
    """Construct an Inspect agent model with provider-appropriate reasoning settings.

    OpenRouter exposes a boolean reasoning switch and needs non-strict tool schemas.
    First-party Anthropic/OpenAI use Inspect's public ``GenerateConfig`` fields.  Native
    reasoning is not fed back into context for those first-party providers so providers
    with summarized/redacted reasoning remain comparable.
    """

    from inspect_ai.model import GenerateConfig, get_model

    routed = route(slug)
    model_kwargs: dict[str, Any] = {}
    config_kwargs: dict[str, Any] = {}
    reasoning_enabled: bool | None = None
    strict_tools: bool | None = None
    reasoning_history: str | None = None
    effective_effort: str | None = effort if reasoning_on else None

    if routed.startswith("openrouter/"):
        reasoning_enabled = bool(reasoning_on)
        strict_tools = False
        model_kwargs.update(
            reasoning_enabled=reasoning_enabled,
            strict_tools=strict_tools,
        )
    elif routed.startswith("anthropic/"):
        if reasoning_on:
            reasoning_history = "none"
            config_kwargs.update(
                reasoning_effort=effort,
                reasoning_history=reasoning_history,
            )
        else:
            config_kwargs["reasoning_effort"] = "none"
    elif routed.startswith("openai/"):
        if prompt_cache_key is not None:
            model_kwargs["prompt_cache_key"] = prompt_cache_key
        if reasoning_on:
            reasoning_history = "none"
            config_kwargs.update(
                reasoning_effort=effort,
                reasoning_summary="auto",
                reasoning_history=reasoning_history,
            )
        else:
            config_kwargs["reasoning_effort"] = "none"

    if config_kwargs:
        model_kwargs["config"] = GenerateConfig(**config_kwargs)
    model = (
        get_model(routed, **model_kwargs)
        if construct_model and model_kwargs
        else routed
    )
    return TargetModelBuild(
        model=model,
        routed_slug=routed,
        reasoning_on=reasoning_on,
        reasoning_effort=effective_effort,
        reasoning_enabled=reasoning_enabled,
        reasoning_history=reasoning_history,
        strict_tools=strict_tools,
        prompt_cache_key=prompt_cache_key if routed.startswith("openai/") else None,
    )


def build_target_model(
    routed_slug: str,
    *,
    reasoning_on: bool,
    effort: str,
    prompt_cache_key: str | None = None,
):
    """Compatibility tuple for the existing task builder during migration."""

    built = build_target(
        routed_slug,
        reasoning_on=reasoning_on,
        effort=effort,
        prompt_cache_key=prompt_cache_key,
    )
    return built.model, built.reasoning_enabled, built.strict_tools
