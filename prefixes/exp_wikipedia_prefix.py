"""Build an approximately 100k-token prefix from 15 Wikipedia summaries.

The fixed order begins with the five articles used by the legacy approximately 40k
control, then appends ten pinned articles. Each user turn contains clean, readable
article text rather than HTML or JSON. The first turn requests an approximately
700-word summary; later turns request the same style. Every article is always asked so
agents see the same source material even when their tokenizer or answer length differs.

Usage (from the repository root):
  uv run prefixes/exp_wikipedia_prefix.py --model=deepseek-v4-pro \
      --harness=production
  uv run prefixes/exp_wikipedia_prefix.py --model=gpt-5.5 \
      --harness=subscription --dry-run

Flags:
  --model=<name>       REQUIRED. Catalog agent name.
  --harness=<mode>     REQUIRED: simple, production, or subscription.
  --reasoning=yes|no   Default yes.
  --name=<slug>        Optional payload name.
  --refresh-articles   Re-fetch the pinned revisions instead of using the local cache.
  --dry-run            FREE: fetch into a temporary cache and show article metadata and
                       prompt previews; no model calls, secrets, or retained files.

Non-dry runs cost money or subscription quota: one agent-provider call per article.
"""

from __future__ import annotations

import asyncio
import hashlib
import pathlib
import re
import sys
import tempfile
from datetime import datetime
from typing import Any


_ENVIRONMENTS = pathlib.Path(__file__).resolve().parents[1]
for _path in (_ENVIRONMENTS, _ENVIRONMENTS / "lib"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import tiktoken
from dotenv import load_dotenv

from cost_tracking import install_cost_tracking
from exp_real_continuation import (
    PREFIX_FORMAT,
    build_prefix_spec,
    store_prefix_payload,
    validate_prefix_payload,
)
from exp_target_harness import NATIVE_HARNESS_MODES, resolve_harness
from model_catalog import TARGET_CHOICES
from prefixes import exp_nq_prefix as nq
from prefixes.wikipedia_articles import (
    ARTICLE_CACHE_ROOT,
    ARTICLES,
    LICENSE_NAME,
    LICENSE_URL,
    load_articles,
)
from project_paths import ENV_FILE
from prompt_caching import install_inspect_warmup
from protocol_sources import resolve_reasoning


GENERATOR = "exp_wikipedia_prefix.py"
PREFIX_TYPE = "wikipedia_summaries_100k"
PREFIX_TYPE_LABEL = "Wikipedia summaries (~100k tokens)"
SCRIPT_VERSION = 2
SUMMARY_TARGET_WORDS = 700
TARGET_CONTEXT_TOKENS = 100_000
TARGET_CONTEXT_WARNING_BAND = (90_000, 110_000)
NO_TOKEN_STOP = sys.maxsize

FIRST_INSTRUCTION = (
    "Before we get into the coding task, I need you to help me summarize some stuff. "
    "First, can you summarize this article in about 700 words? Please make the summary "
    "clear and self-contained, cover the main points, and leave out minor details."
)
FOLLOWUP_INSTRUCTION = "Please summarize the following article in the same style."
SOURCE_CLEANUP_CAVEAT = (
    "Wikipedia articles were cleaned before prompting: readable headings, paragraphs, "
    "and ordinary lists were kept; webpage chrome, citations, tables, figures, and "
    "appendix sections were omitted."
)

_VALUE_FLAGS = {"--model", "--harness", "--reasoning", "--name"}
_SWITCH_FLAGS = {"--dry-run", "--refresh-articles"}


def _validate_cli_args() -> None:
    valid = sorted(_VALUE_FLAGS | _SWITCH_FLAGS)
    for argument in sys.argv[1:]:
        flag, separator, _value = argument.partition("=")
        if flag in _VALUE_FLAGS:
            if not separator:
                raise SystemExit(
                    f"{flag} requires a value in the form {flag}=<value>"
                )
            continue
        if flag in _SWITCH_FLAGS:
            if separator:
                raise SystemExit(f"{flag} is a switch and does not take a value")
            continue
        raise SystemExit(f"unknown argument {argument!r}; valid flags: {valid}")


def _arg(flag: str, default: str | None = None) -> str | None:
    return next(
        (
            argument.split("=", 1)[1]
            for argument in sys.argv
            if argument.startswith(flag + "=")
        ),
        default,
    )


def _model_slug_fragment(model_name: str) -> str:
    fragment = re.sub(r"[^a-z0-9]+", "-", model_name.casefold()).strip("-")
    return fragment or "model"


def default_name(model_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"wikipedia-summaries-100k-{_model_slug_fragment(model_name)}-{stamp}"


def parse_args() -> dict[str, Any]:
    _validate_cli_args()
    model = _arg("--model")
    if model is None:
        raise SystemExit(f"--model is required; choices: {sorted(TARGET_CHOICES)}")
    if model not in TARGET_CHOICES:
        raise SystemExit(
            f"unknown --model {model!r}; choices: {sorted(TARGET_CHOICES)}"
        )
    return {
        "model": model,
        "harness": resolve_harness(_arg("--harness")),
        "reasoning": resolve_reasoning(_arg("--reasoning")),
        "name": _arg("--name") or default_name(model),
        "dry_run": "--dry-run" in sys.argv,
        "refresh_articles": "--refresh-articles" in sys.argv,
        "tokens": NO_TOKEN_STOP,
        "generator": GENERATOR,
        "prefix_type": PREFIX_TYPE,
        "questions_are_complete_user_turns": True,
    }


def user_turn(article: dict[str, Any], position: int) -> str:
    instruction = FIRST_INSTRUCTION if position == 0 else FOLLOWUP_INSTRUCTION
    title = article["article"]["title"]
    return f"{instruction}\n\nWikipedia article: {title}\n\n{article['text']}"


def build_turns(articles: list[dict[str, Any]]) -> list[tuple[int, str]]:
    return [
        (position, user_turn(article, position))
        for position, article in enumerate(articles)
    ]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _turn_token_counts(turn: str) -> dict[str, int]:
    return {
        "o200k_base": len(tiktoken.get_encoding("o200k_base").encode(turn)),
        "cl100k_base": len(tiktoken.get_encoding("cl100k_base").encode(turn)),
    }


def _article_payload_record(
    article: dict[str, Any],
    turn: str,
    position: int,
) -> dict[str, Any]:
    return {
        "position": position + 1,
        "title": article["article"]["title"],
        "page_id": article["article"]["page_id"],
        "revision_id": article["article"]["revision_id"],
        "article_url": article["article_url"],
        "permanent_url": article["permanent_url"],
        "license": article["license"],
        "fetched_at": article["fetched_at"],
        "raw_html_sha256": article["raw_html_sha256"],
        "text_sha256": article["text_sha256"],
        "characters": article["characters"],
        "words": article["words"],
        "article_token_counts": article["token_counts"],
        "user_turn_sha256": _sha256(turn),
        "user_turn_token_counts": _turn_token_counts(turn),
        "extraction": article["extraction"],
    }


def _article_dialogue_token_estimate(
    messages: list[Any],
    turns: list[tuple[int, str]],
) -> dict[str, int]:
    # Keep this estimate comparable across harnesses by excluding native system,
    # developer, environment-context, and other scaffold-injected text. The exact
    # provider-context measurement remains stored separately.
    texts = [turn for _index, turn in turns]
    texts.extend(
        str(getattr(message, "text", "") or "")
        for message in messages
        if getattr(message, "role", None) == "assistant"
    )
    combined = "\n".join(texts)
    return _turn_token_counts(combined)


def _delivered_user_turns_are_exact(
    messages: list[Any], expected_turns: list[str]
) -> bool:
    delivered = [
        str(getattr(message, "text", "") or "")
        for message in messages
        if getattr(message, "role", None) == "user"
    ]
    position = 0
    for message in delivered:
        if position < len(expected_turns) and message == expected_turns[position]:
            position += 1
    return position == len(expected_turns)


def build_payload(
    cfg: dict[str, Any],
    result: dict[str, Any],
    articles: list[dict[str, Any]],
    turns: list[tuple[int, str]],
) -> dict[str, Any]:
    expected_turns = [turn for _position, turn in turns]
    asked_turns = [item.get("question") for item in result.get("asked") or []]
    completed = asked_turns == expected_turns
    if not completed:
        raise SystemExit(
            "conversation did not summarize the complete fixed Wikipedia article "
            "sequence in order; refusing to save it"
        )
    if not _delivered_user_turns_are_exact(result["messages"], expected_turns):
        raise SystemExit(
            "native harness changed at least one model-visible Wikipedia user turn "
            "(for example by adding wrapper quotes or escapes); refusing to save it"
        )

    article_records = [
        _article_payload_record(article, turn, position)
        for position, (article, (_index, turn)) in enumerate(zip(articles, turns))
    ]
    source_user_turn_tokens = {
        encoding: sum(
            record["user_turn_token_counts"][encoding]
            for record in article_records
        )
        for encoding in ("o200k_base", "cl100k_base")
    }
    payload = {
        "format": PREFIX_FORMAT,
        "name": cfg["name"],
        "target": cfg["model"],
        "reasoning": cfg["reasoning"],
        "source": {
            "kind": "external",
            "description": (
                "15 fixed Wikipedia articles summarized in one conversation before "
                "the main coding assignment"
            ),
            "generator": GENERATOR,
            "prefix_type": PREFIX_TYPE,
            "prefix_type_label": PREFIX_TYPE_LABEL,
            "source_label": PREFIX_TYPE_LABEL,
            "harness": cfg["harness"],
            "script_version": SCRIPT_VERSION,
            "fixed_article_order": True,
            "completed_script": True,
            "exact_user_turn_delivery": True,
            "summary_target_words": SUMMARY_TARGET_WORDS,
            "target_context_tokens": TARGET_CONTEXT_TOKENS,
            "target_context_warning_band": list(TARGET_CONTEXT_WARNING_BAND),
            "followup_instruction": FOLLOWUP_INSTRUCTION,
            "questions": [
                {
                    "position": record["position"],
                    "article_title": record["title"],
                    "permanent_url": record["permanent_url"],
                    "user_turn_sha256": record["user_turn_sha256"],
                }
                for record in article_records
            ],
            "articles": article_records,
            "article_license": {"name": LICENSE_NAME, "url": LICENSE_URL},
            "source_user_turn_tokens": source_user_turn_tokens,
            "estimated_article_dialogue_tokens": _article_dialogue_token_estimate(
                result["messages"], turns
            ),
            "lossy_processing": {
                "affected": True,
                "scope": "Wikipedia source article extraction",
                "visible_caveat": SOURCE_CLEANUP_CAVEAT,
                "per_article_records": "source.articles[*].extraction",
            },
            "generated_at": datetime.now().astimezone().isoformat(),
            "measured_context_tokens": result["context_tokens"],
            "token_measurement": result["measurement"],
            "generation_usage": result["usage"],
            "generation_cost": result["cost"],
        },
        "messages": [
            message.model_dump(mode="json", exclude_none=False)
            for message in result["messages"]
        ],
    }
    if cfg["harness"] in NATIVE_HARNESS_MODES:
        payload["source"]["native_harness"] = result["native_harness"]
        payload["native_resume"] = result["native_resume"]
    return validate_prefix_payload(payload, origin=GENERATOR)


def _cost_note(cost: dict[str, Any]) -> str:
    amount = cost.get("cost_usd")
    if isinstance(amount, (int, float)):
        return (
            f"${amount:.4f} "
            f"({'EXACT' if cost.get('exact') else 'estimate'}, "
            f"{cost.get('source') or 'unknown source'})"
        )
    if cost.get("source") == "subscription_not_metered":
        return "included subscription usage (tokens recorded; no per-run dollar cost)"
    return "unpriced model"


def _load_and_describe_articles(
    cfg: dict[str, Any],
    *,
    cache_root: pathlib.Path,
) -> list[dict[str, Any]]:
    articles = load_articles(
        cache_root=cache_root,
        refresh=bool(cfg["refresh_articles"] or cfg["dry_run"]),
    )
    print(f"[articles] loaded {len(articles)} pinned Wikipedia revisions")
    for position, article in enumerate(articles, start=1):
        tokens = article["token_counts"]["o200k_base"]
        print(
            f"  {position}. {article['article']['title']} "
            f"r{article['article']['revision_id']} · {article['words']:,} words · "
            f"~{tokens:,} article tokens"
        )
    return articles


def main() -> None:
    cfg = parse_args()
    print("=" * 76)
    print("WIKIPEDIA SUMMARY PREFIX  (15 readable articles -> ~100k context)")
    print("=" * 76)
    print(
        f"  model={cfg['model']} [reasoning:"
        f"{'on' if cfg['reasoning'] else 'off'}]  harness={cfg['harness']}"
    )
    print(f"  articles={len(ARTICLES)}  summary target≈700 words  name={cfg['name']}")

    if cfg["dry_run"]:
        with tempfile.TemporaryDirectory(
            prefix="environments-wikipedia-dry-run-"
        ) as temporary:
            articles = _load_and_describe_articles(
                cfg, cache_root=pathlib.Path(temporary)
            )
            turns = build_turns(articles)
            print("\n[prompt previews]")
            for position, (_index, turn) in enumerate(turns, start=1):
                preview = turn[:360].replace("\n", " ")
                print(f"  USER {position}: {preview} …")
            total = sum(
                _turn_token_counts(turn)["o200k_base"] for _index, turn in turns
            )
            print(f"\n  source user turns: ~{total:,} o200k_base tokens")
        print("\n[dry-run] no model calls, secrets loaded, or files retained.")
        return

    articles = _load_and_describe_articles(cfg, cache_root=ARTICLE_CACHE_ROOT)
    turns = build_turns(articles)
    load_dotenv(ENV_FILE)
    install_cost_tracking()
    if not install_inspect_warmup():
        raise SystemExit(
            "provider prompt-cache routing could not be installed; refusing to build "
            "a paid growing-conversation prefix without it"
        )
    print(f"\n[generate] asking all {len(turns)} article-summary turns ...")
    conversation = (
        nq.run_conversation
        if cfg["harness"] == "simple"
        else nq.run_production_conversation
    )
    result = asyncio.run(conversation(cfg, turns))
    payload = build_payload(cfg, result, articles, turns)
    spec = build_prefix_spec(payload, harness=cfg["harness"])
    path = store_prefix_payload(payload)
    transcript_tokens = payload["source"]["estimated_article_dialogue_tokens"]
    print(
        f"\n[done] {len(turns)} articles, "
        f"~{transcript_tokens['o200k_base']:,} article-dialogue tokens, "
        f"~{result['context_tokens']:,} provider-context tokens "
        f"({result['measurement']}), {len(spec.messages)} messages"
    )
    print(f"       generation cost: {_cost_note(result['cost'])}")
    print(f"       payload: {path}")
    low, high = TARGET_CONTEXT_WARNING_BAND
    if not low <= result["context_tokens"] <= high:
        print(
            f"WARNING: measured context {result['context_tokens']:,} is outside "
            f"the review band {low:,}-{high:,}; inspect before continuation runs"
        )
    print("\nUse it with:")
    print(
        "  uv run exp_continuation_pipeline.py --treatment=<slug> "
        f"--prefix-files={path} --harness={cfg['harness']} \\"
    )
    print("      --seed-dir=<family> --seeds=<members> --epochs=<n>")


if __name__ == "__main__":
    main()
