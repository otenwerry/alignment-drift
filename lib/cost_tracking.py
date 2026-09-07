"""Environment-owned model prices and Inspect cost capture.

Direct Anthropic/OpenAI calls use Inspect's public model-cost registry.  OpenRouter
returns the actual billed cost in its raw response, but Inspect 0.3.239 does not expose
a public response-cost hook.  The small adapter here is therefore the one deliberate
private Inspect dependency in this module; it fails soft and leaves token counts plus
explicit estimate metadata if Inspect moves it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Mapping


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenPrice:
    """US dollars per million tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float
    source: str


def _price(
    input_price: float,
    output_price: float,
    cache_read: float | None = None,
    cache_write: float | None = None,
    source: str = "openrouter-derived",
) -> TokenPrice:
    return TokenPrice(
        input=input_price,
        output=output_price,
        cache_read=input_price if cache_read is None else cache_read,
        cache_write=input_price if cache_write is None else cache_write,
        source=source,
    )


# OpenRouter estimates were derived from billed project logs.  Direct prices are provider
# list rates.  Missing catalog models remain visibly unpriced rather than silently using 0.
PRICES: dict[str, TokenPrice] = {
    "deepseek/deepseek-chat": _price(0.20, 0.80, 0.20),
    "deepseek/deepseek-r1": _price(0.70, 2.50, 0.70),
    "deepseek/deepseek-v4-pro-20260423": _price(0.748, 1.496, 0.0624),
    "minimax/minimax-m2.7": _price(0.18, 1.00, 0.053),
    "moonshotai/kimi-k2.6-20260420": _price(0.95, 3.60, 0.125),
    "qwen/qwen-2.5-72b-instruct": _price(0.38, 0.40, 0.38),
    "qwen/qwen3-32b": _price(0.08, 0.30, 0.10),
    "qwen/qwen3.7-max": _price(1.25, 3.75, 0.25),
    "xiaomi/mimo-v2.5-pro": _price(0.60, 3.00, 0.26),
    "z-ai/glm-5.1": _price(0.966, 3.036, 0.179),
    "z-ai/glm-5.2": _price(1.1522, 3.6212, 0.214),
    "meta-llama/llama-3.3-70b-instruct": _price(
        0.10, 0.32, 0.10, source="openrouter-assumed"
    ),
    "x-ai/grok-4.20": _price(
        1.25, 2.50, 0.3125, source="openrouter-assumed"
    ),
    "anthropic/claude-opus-4-8": _price(
        5.0, 25.0, 0.50, 6.25, "anthropic-list"
    ),
    "anthropic/claude-opus-4-6": _price(
        5.0, 25.0, 0.50, 6.25, "anthropic-list"
    ),
    "anthropic/claude-sonnet-4-6": _price(
        3.0, 15.0, 0.30, 3.75, "anthropic-list"
    ),
    "openai/gpt-5.5-2026-04-23": _price(
        5.00, 30.00, 0.50, source="openai-list"
    ),
    "openai/gpt-5.4-mini-2026-03-17": _price(
        0.75, 4.50, 0.075, source="openai-list"
    ),
    "openai/gpt-5.6-sol": _price(
        5.00, 30.00, 0.50, 6.25, "openai-list"
    ),
    "openai/gpt-5.6-luna": _price(
        0.20, 1.20, 0.02, 0.25, "openai-list"
    ),
}

LONG_CONTEXT_PRICING_CAVEAT = (
    "GPT-5.5/5.6 calls above 272k prompt tokens can use a higher provider price tier; "
    "the flat fallback table underestimates those calls. A response-reported billed "
    "total still wins when available."
)

SUBSCRIPTION_USAGE_CAVEAT = (
    "Subscription agent calls (native model slugs prefixed 'subscription/' and "
    "host-bridge model slugs containing '/opencode-go/') draw on included subscription "
    "quota, not per-run billing. Their tokens are recorded, but they carry no dollar "
    "cost and are excluded from total_cost_usd; the exclusion is flagged via "
    "subscription_models_not_metered/total_excludes_subscription_usage."
)


CONTEXT_WINDOWS: dict[str, int] = {
    "anthropic/claude-opus-4-8": 1_000_000,
    "anthropic/claude-opus-4-6": 1_000_000,
    "anthropic/claude-sonnet-4-6": 1_000_000,
    "openai/gpt-5.5-2026-04-23": 1_050_000,
    "openai/gpt-5.6-sol": 1_050_000,
    "openai/gpt-5.6-luna": 1_050_000,
    "openai/gpt-5.4-mini-2026-03-17": 400_000,
    "deepseek/deepseek-chat": 131_072,
    "deepseek/deepseek-r1": 163_840,
    "deepseek/deepseek-v4-pro-20260423": 1_048_576,
    "google/gemma-3-27b-it": 131_072,
    "meta-llama/llama-3.3-70b-instruct": 131_072,
    "minimax/minimax-m2.7": 204_800,
    "mistralai/mistral-small-3.1-24b-instruct": 128_000,
    "moonshotai/kimi-k2.6-20260420": 262_144,
    "qwen/qwen-2.5-72b-instruct": 131_072,
    "qwen/qwen3-32b": 131_072,
    "qwen/qwen3.7-max": 1_000_000,
    "x-ai/grok-4.20": 2_000_000,
    "xiaomi/mimo-v2.5-pro": 1_048_576,
    "z-ai/glm-5.1": 202_752,
    "z-ai/glm-5.2": 1_048_576,
}


@dataclass(frozen=True)
class CostInstallReport:
    direct_prices_registered: tuple[str, ...]
    direct_prices_skipped: tuple[str, ...]
    openrouter_billed_cost_capture: bool
    context_windows_registered: tuple[str, ...]
    context_windows_skipped: tuple[str, ...]
    caveats: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_slug(slug: str) -> str:
    return slug.removeprefix("openrouter/")


def price_for(slug: str) -> TokenPrice | None:
    return PRICES.get(canonical_slug(slug))


def _direct_prices() -> dict[str, TokenPrice]:
    return {
        slug: price
        for slug, price in PRICES.items()
        if price.source.endswith("-list")
    }


def _register_direct_prices() -> tuple[list[str], list[str]]:
    """Use Inspect's public cost API for first-party provider list rates."""

    from inspect_ai.model import ModelCost, set_model_cost

    registered: list[str] = []
    skipped: list[str] = []
    for slug, price in _direct_prices().items():
        try:
            set_model_cost(
                slug,
                ModelCost(
                    input=price.input,
                    output=price.output,
                    input_cache_read=price.cache_read,
                    input_cache_write=price.cache_write,
                ),
            )
            registered.append(slug)
        except Exception as exc:  # one unknown future slug must not block a run
            skipped.append(f"{slug} ({type(exc).__name__}: {exc})")
    return registered, skipped


_openrouter_cost_installed = False


def _install_openrouter_billed_cost_capture() -> bool:
    """Copy OpenRouter's response-reported charge into Inspect's durable usage field.

    Inspect has no public hook for this in the pinned runtime.  Keep the private adapter
    isolated here so a version bump has one obvious compatibility point.
    """

    global _openrouter_cost_installed
    if _openrouter_cost_installed:
        return True
    try:
        from inspect_ai.model._providers.openrouter import OpenRouterAPI
    except Exception as exc:
        logger.warning(
            "could not install OpenRouter billed-cost capture (%s); token-based "
            "estimates remain available",
            exc,
        )
        return False

    original_generate = OpenRouterAPI.generate

    async def generate(self, *args, **kwargs):
        result = await original_generate(self, *args, **kwargs)
        try:
            if isinstance(result, tuple) and len(result) == 2:
                output, call = result
                usage = getattr(output, "usage", None)
                response = getattr(call, "response", None)
                response_usage = response.get("usage") if isinstance(response, dict) else None
                billed = (
                    response_usage.get("cost")
                    if isinstance(response_usage, dict)
                    else None
                )
                if usage is not None and isinstance(billed, (int, float)):
                    usage.total_cost = float(billed)
        except Exception:
            logger.debug("failed to retain OpenRouter billed cost", exc_info=True)
        return result

    OpenRouterAPI.generate = generate
    _openrouter_cost_installed = True
    return True


def _register_context_windows() -> tuple[list[str], list[str]]:
    """Use Inspect's public registry to correct missing or stale windows."""

    from inspect_ai.model import ModelInfo, get_model_info, set_model_info

    registered: list[str] = []
    skipped: list[str] = []
    for slug, window in CONTEXT_WINDOWS.items():
        try:
            existing = get_model_info(slug) or get_model_info(f"openrouter/{slug}")
            current = existing.context_length if existing is not None else None
            if current is not None and current >= window:
                continue
            if existing is None:
                info = ModelInfo(context_length=window, _input_tokens=window)
            else:
                values = existing.model_dump()
                values["context_length"] = window
                info = ModelInfo(**values, _input_tokens=window)
            keys = {slug}
            if not slug.startswith(("anthropic/", "openai/")):
                keys.add(f"openrouter/{slug}")
            for key in keys:
                set_model_info(key, info)
            registered.append(slug)
        except Exception as exc:
            skipped.append(f"{slug} ({type(exc).__name__}: {exc})")
    return registered, skipped


def install_cost_tracking() -> CostInstallReport:
    """Install all runtime accounting hooks before the first model call."""

    direct_registered, direct_skipped = _register_direct_prices()
    context_registered, context_skipped = _register_context_windows()
    exact_openrouter = _install_openrouter_billed_cost_capture()
    caveats = [LONG_CONTEXT_PRICING_CAVEAT]
    if direct_skipped:
        caveats.append("Some direct model prices could not be registered; see skipped list.")
    if not exact_openrouter:
        caveats.append(
            "OpenRouter exact billed-cost capture is unavailable; use the stored tokens "
            "and source-labeled price estimates."
        )
    return CostInstallReport(
        direct_prices_registered=tuple(direct_registered),
        direct_prices_skipped=tuple(direct_skipped),
        openrouter_billed_cost_capture=exact_openrouter,
        context_windows_registered=tuple(context_registered),
        context_windows_skipped=tuple(context_skipped),
        caveats=tuple(caveats),
    )


def _usage_value(usage: Mapping[str, Any] | Any, *names: str) -> float:
    for name in names:
        value = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _optional_usage_value(
    usage: Mapping[str, Any] | Any, *names: str
) -> float | None:
    for name in names:
        value = (
            usage.get(name)
            if isinstance(usage, Mapping)
            else getattr(usage, name, None)
        )
        if isinstance(value, (int, float)):
            return float(value)
    return None


def estimate_usage_cost(slug: str, usage: Mapping[str, Any] | Any) -> dict:
    """Price one stored usage record, preserving exact-vs-estimated provenance."""

    if slug.startswith("subscription/") or "/opencode-go/" in slug:
        return {
            "cost_usd": None,
            "exact": False,
            "source": "subscription_not_metered",
        }
    billed = _optional_usage_value(usage, "total_cost")
    if billed is not None:
        return {"cost_usd": billed, "exact": True, "source": "billed_or_direct_list"}
    price = price_for(slug)
    if price is None:
        return {"cost_usd": None, "exact": False, "source": "unpriced"}
    total = (
        _usage_value(usage, "input", "input_tokens") * price.input
        + _usage_value(usage, "output", "output_tokens") * price.output
        + _usage_value(usage, "cache_read", "input_tokens_cache_read")
        * price.cache_read
        + _usage_value(usage, "cache_write", "input_tokens_cache_write")
        * price.cache_write
    ) / 1_000_000
    return {"cost_usd": total, "exact": False, "source": price.source}


def cost_tracking_provenance() -> dict:
    """Static metadata to stamp before spend, including every known caveat."""

    return {
        "cost_tracking_version": "environments-cost-v1",
        "direct_cost_method": "inspect-public-model-cost-provider-list",
        "openrouter_cost_method": "response-reported-billed-cost-private-adapter",
        "subscription_cost_method": "native-cli-token-counts-included-quota-unbilled",
        "fallback_prices_usd_per_million_tokens": {
            slug: asdict(price) for slug, price in PRICES.items()
        },
        "caveats": [LONG_CONTEXT_PRICING_CAVEAT, SUBSCRIPTION_USAGE_CAVEAT],
    }
