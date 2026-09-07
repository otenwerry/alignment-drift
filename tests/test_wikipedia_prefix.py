"""Free regression tests for the Wikipedia summarization prefix."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS))
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from exp_real_continuation import build_prefix_spec, expected_system_prompt  # noqa: E402
from prefixes import exp_nq_prefix as nq  # noqa: E402
from prefixes import exp_wikipedia_prefix as wiki  # noqa: E402
from prefixes import wikipedia_articles as source  # noqa: E402


def sample_html(revision_id: int = 1368681885) -> bytes:
    lead = "This is ordinary readable lead prose about the subject. " * 35
    formation = "This section explains formation and history in normal prose. " * 20
    return f"""
    <html><head><script>{{"wgRevisionId":{revision_id}}}</script></head><body>
      <div id="mw-content-text"><div class="mw-parser-output">
        <section>
          <table class="infobox"><tr><td>Infobox data</td></tr></table>
          <p>{lead}<sup class="reference">[1]</sup></p>
          <figure>Image caption that should be omitted</figure>
        </section>
        <section>
          <div class="mw-heading mw-heading2"><h2>Formation</h2></div>
          <p>{formation}</p>
          <ul><li>First ordinary list item</li><li>Second ordinary list item</li></ul>
        </section>
        <section>
          <div class="mw-heading mw-heading2"><h2>References</h2></div>
          <p>Reference appendix that should be omitted.</p>
        </section>
        <section>
          <div class="mw-heading mw-heading2"><h2>External links</h2></div>
          <p>External link appendix that should be omitted.</p>
        </section>
      </div></div>
    </body></html>
    """.encode()


def fake_article(spec: source.ArticleSpec, position: int) -> dict:
    text = (
        f"{spec.title} is presented here as ordinary article prose.\n\n"
        f"## Background\n\nThis is readable section text for article {position + 1}."
    )
    return {
        "format": source.CACHE_FORMAT,
        "article": {
            "title": spec.title,
            "page_id": spec.page_id,
            "revision_id": spec.revision_id,
        },
        "article_url": spec.article_url,
        "permanent_url": spec.permanent_url,
        "license": {
            "name": source.LICENSE_NAME,
            "url": source.LICENSE_URL,
            "attribution_url": spec.article_url,
        },
        "fetched_at": "2026-08-13T00:00:00-07:00",
        "raw_html_sha256": f"raw-{position}",
        "text_sha256": f"text-{position}",
        "characters": len(text),
        "words": len(text.split()),
        "token_counts": {"o200k_base": 30, "cl100k_base": 31},
        "extraction": {
            "lossy": True,
            "extraction_version": source.EXTRACTION_VERSION,
            "omitted_sections": ["references"],
            "omitted_element_counts": {"citation_markers": 1},
        },
        "text": text,
    }


def test_fixed_article_order_and_revisions() -> None:
    assert [article.title for article in source.ARTICLES] == [
        "Bread",
        "Fjord",
        "Cave",
        "National park",
        "Botanical garden",
        "Lighthouse",
        "Terracotta",
        "Glacier",
        "Quilting",
        "Waterfall",
        "Textile",
        "Meadow",
        "Canal",
        "Basket weaving",
        "Windmill",
    ]
    assert all(article.revision_id > 0 for article in source.ARTICLES)


def test_extraction_is_readable_text_and_records_every_cleanup() -> None:
    text, extraction = source.extract_readable_article(sample_html())

    assert text.startswith("This is ordinary readable lead prose")
    assert "## Formation" in text
    assert "- First ordinary list item" in text
    assert "Infobox data" not in text
    assert "Image caption" not in text
    assert "[1]" not in text
    assert "Reference appendix" not in text
    assert "External link appendix" not in text
    assert extraction["lossy"] is True
    assert extraction["omitted_sections"] == ["references", "external links"]
    assert extraction["omitted_element_counts"]["tables"] == 1
    assert extraction["omitted_element_counts"]["figures"] == 1
    assert extraction["omitted_element_counts"]["citation_markers"] == 1


def test_article_record_requires_and_records_the_pinned_revision() -> None:
    spec = source.ARTICLES[0]
    record = source.build_article_record(spec, sample_html(spec.revision_id))

    assert record["article"]["revision_id"] == spec.revision_id
    assert record["license"]["name"] == "CC BY-SA 4.0"
    assert record["text_sha256"]
    assert record["token_counts"]["o200k_base"] > 0
    assert "text" in record

    try:
        source.build_article_record(spec, sample_html(spec.revision_id + 1))
    except ValueError as error:
        assert "rendered revision" in str(error)
    else:
        raise AssertionError("a mismatched Wikipedia revision was accepted")


def test_article_cache_is_hashed_and_reused_without_network(tmp_path: Path) -> None:
    spec = source.ARTICLES[0]
    with patch.object(source, "_fetch", return_value=sample_html(spec.revision_id)) as fetch:
        first = source.load_article(spec, cache_root=tmp_path)
        second = source.load_article(spec, cache_root=tmp_path)

    assert first == second
    fetch.assert_called_once()
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    assert json.loads(cache_files[0].read_text())["text_sha256"] == first["text_sha256"]


def test_user_turns_look_like_normal_pasted_articles() -> None:
    articles = [fake_article(spec, position) for position, spec in enumerate(source.ARTICLES)]
    turns = wiki.build_turns(articles)

    first = turns[0][1]
    assert first.startswith("Before we get into the coding task")
    assert "Wikipedia article: Bread\n\nBread is presented here" in first
    assert "## Background" in first
    assert not first.lstrip().startswith(("{", "["))
    assert '"text":' not in first
    assert turns[1][1].startswith(
        "Please summarize the following article in the same style."
    )
    assert all(
        turn.startswith(wiki.FOLLOWUP_INSTRUCTION)
        for _index, turn in turns[1:]
    )
    assert nq.conversation_prompt(
        {"questions_are_complete_user_turns": True}, first, 0
    ) == first


def test_payload_stores_provenance_without_duplicating_article_text() -> None:
    articles = [fake_article(spec, position) for position, spec in enumerate(source.ARTICLES)]
    turns = wiki.build_turns(articles)
    messages = [ChatMessageSystem(content=expected_system_prompt(True))]
    for position, (_index, turn) in enumerate(turns):
        messages.extend((
            ChatMessageUser(content=turn),
            ChatMessageAssistant(content=f"Readable summary {position + 1}."),
        ))
    result = {
        "messages": messages,
        "asked": [
            {"dataset_index": index, "question": turn} for index, turn in turns
        ],
        "context_tokens": 30_000,
        "measurement": "provider_usage",
        "usage": {"input": 25_000, "output": 5_000},
        "cost": {"cost_usd": 1.0, "exact": True, "source": "test"},
    }
    cfg = {
        "model": "qwen3-32b",
        "harness": "simple",
        "reasoning": True,
        "name": "wikipedia-summary-test",
    }

    payload = wiki.build_payload(cfg, result, articles, turns)
    prefix = build_prefix_spec(payload, harness="simple")
    stored = payload["source"]

    assert prefix.boundary_index == len(messages)
    assert stored["prefix_type"] == "wikipedia_summaries_100k"
    assert stored["completed_script"] is True
    assert stored["exact_user_turn_delivery"] is True
    assert len(stored["articles"]) == len(source.ARTICLES) == 15
    assert stored["script_version"] == 2
    assert stored["target_context_tokens"] == 100_000
    assert stored["followup_instruction"] == wiki.FOLLOWUP_INSTRUCTION
    assert all("text" not in article for article in stored["articles"])
    assert stored["lossy_processing"]["affected"] is True
    assert stored["source_user_turn_tokens"]["o200k_base"] > 0
    assert stored["estimated_article_dialogue_tokens"]["o200k_base"] > (
        stored["source_user_turn_tokens"]["o200k_base"]
    )
    assert payload["messages"][1]["content"].startswith(
        "Before we get into the coding task"
    )


def test_payload_refuses_harness_quoted_user_turns() -> None:
    articles = [
        fake_article(spec, position)
        for position, spec in enumerate(source.ARTICLES)
    ]
    turns = wiki.build_turns(articles)
    messages = [ChatMessageSystem(content=expected_system_prompt(True))]
    for position, (_index, turn) in enumerate(turns):
        messages.extend((
            ChatMessageUser(content=f'"{turn}"'),
            ChatMessageAssistant(content=f"Readable summary {position + 1}."),
        ))
    result = {
        "messages": messages,
        "asked": [
            {"dataset_index": index, "question": turn} for index, turn in turns
        ],
        "context_tokens": 100_000,
        "measurement": "provider_usage",
        "usage": {"input": 90_000, "output": 10_000},
        "cost": {"cost_usd": 1.0, "exact": True, "source": "test"},
    }
    cfg = {
        "model": "qwen3-32b",
        "harness": "simple",
        "reasoning": True,
        "name": "wikipedia-summary-quoted-test",
    }

    with pytest.raises(SystemExit, match="wrapper quotes or escapes"):
        wiki.build_payload(cfg, result, articles, turns)


def test_dry_run_loads_no_secrets_and_calls_no_model(tmp_path: Path) -> None:
    articles = [fake_article(spec, position) for position, spec in enumerate(source.ARTICLES)]
    argv = [
        "exp_wikipedia_prefix.py",
        "--model=qwen3-32b",
        "--harness=simple",
        "--dry-run",
    ]
    with (
        patch.object(sys, "argv", argv),
        patch.object(wiki, "load_articles", return_value=articles),
        patch.object(wiki, "load_dotenv") as load_dotenv,
        patch.object(wiki.nq, "run_conversation") as conversation,
    ):
        wiki.main()

    load_dotenv.assert_not_called()
    conversation.assert_not_called()
