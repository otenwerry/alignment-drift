"""Pinned Wikipedia article extraction for the summarization prefix.

The extractor deliberately produces something close to copying the readable article
body in a browser: title-adjacent prose, section headings, paragraphs, and ordinary
lists. It removes webpage chrome and non-prose material. Every omission is counted and
returned as stored provenance rather than being hidden in logs.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken
from bs4 import BeautifulSoup, Tag


_ENVIRONMENTS = Path(__file__).resolve().parents[1]
_LIB = _ENVIRONMENTS / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from project_paths import WIKIPEDIA_ARTICLES_ROOT


CACHE_FORMAT = "environments-wikipedia-article-v1"
EXTRACTION_VERSION = 1
ARTICLE_CACHE_ROOT = WIKIPEDIA_ARTICLES_ROOT
LICENSE_NAME = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
USER_AGENT = "supermats-wikipedia-prefix/1.0 (research article cache)"

TERMINAL_SECTIONS = frozenset({
    "see also",
    "notes",
    "references",
    "sources",
    "bibliography",
    "further reading",
    "external links",
})

# Selectors are intentionally explicit so the stored omission counts are meaningful.
OMITTED_ELEMENT_SELECTORS = {
    "tables": "table",
    "figures": "figure",
    "styles": "style",
    "scripts": "script",
    "citation_markers": "sup.reference",
    "edit_controls": ".mw-editsection",
    "short_descriptions": ".shortdescription",
    "hatnotes": ".hatnote",
    "metadata_boxes": ".metadata",
    "navigation_boxes": ".navbox, .vertical-navbox",
    "sidebars": ".sidebar",
    "infoboxes": ".infobox",
    "thumbnails": ".thumb",
    "galleries": ".gallery",
    "non_print_elements": ".noprint",
    "reference_lists": ".reflist",
    "sister_site_boxes": ".sistersitebox",
    "authority_control": ".authority-control",
    "tables_of_contents": ".toc",
    "empty_elements": ".mw-empty-elt",
}


@dataclass(frozen=True)
class ArticleSpec:
    title: str
    page_id: int
    revision_id: int

    @property
    def article_url(self) -> str:
        title = urllib.parse.quote(self.title.replace(" ", "_"))
        return f"https://en.wikipedia.org/wiki/{title}"

    @property
    def permanent_url(self) -> str:
        query = urllib.parse.urlencode({
            "title": self.title.replace(" ", "_"),
            "oldid": self.revision_id,
        })
        return f"https://en.wikipedia.org/w/index.php?{query}"


LEGACY_40K_ARTICLES = (
    ArticleSpec("Bread", 36969, 1368681885),
    ArticleSpec("Fjord", 43598, 1364094600),
    ArticleSpec("Cave", 5778, 1364753553),
    ArticleSpec("National park", 21818, 1360969256),
    ArticleSpec("Botanical garden", 69427, 1361586657),
)

ARTICLES = (
    *LEGACY_40K_ARTICLES,
    ArticleSpec("Lighthouse", 17725, 1368127341),
    ArticleSpec("Terracotta", 174241, 1369643793),
    ArticleSpec("Glacier", 12463, 1371666943),
    ArticleSpec("Quilting", 25203, 1352026773),
    ArticleSpec("Waterfall", 69442, 1368766265),
    ArticleSpec("Textile", 51892, 1371791562),
    ArticleSpec("Meadow", 371406, 1368813703),
    ArticleSpec("Canal", 5623, 1364267324),
    ArticleSpec("Basket weaving", 3133158, 1364101187),
    ArticleSpec("Windmill", 43490, 1368600344),
)


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _element_text(element: Tag) -> str:
    # Empty separator preserves punctuation around links the same way browser text
    # selection does ("flour," rather than "flour ,"). Whitespace is normalized only
    # after the inline nodes have been joined.
    return re.sub(r"\s+", " ", element.get_text("", strip=False)).strip()


def _normalized_heading(heading: Tag) -> str:
    return _element_text(heading).casefold()


def _remove_terminal_sections(root: Tag) -> list[str]:
    omitted: list[str] = []
    for heading in list(root.select("h2")):
        title = _normalized_heading(heading)
        if title not in TERMINAL_SECTIONS:
            continue
        omitted.append(title)
        section = heading.find_parent("section")
        if isinstance(section, Tag) and root in section.parents:
            section.decompose()
            continue

        # Legacy Wikipedia HTML has headings and content as siblings rather than
        # wrapping each H2 section. Remove from this heading through the next H2.
        wrapper = heading.parent
        if not isinstance(wrapper, Tag):
            continue
        current: Tag | None = wrapper
        while current is not None:
            following = current.find_next_sibling()
            current.decompose()
            if isinstance(following, Tag) and following.find("h2") is not None:
                break
            current = following if isinstance(following, Tag) else None
    return omitted


def extract_readable_article(html: bytes) -> tuple[str, dict[str, Any]]:
    """Return readable article text plus exact, queryable cleanup metadata."""

    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one("#mw-content-text .mw-parser-output")
    if root is None:
        root = soup.select_one(".mw-parser-output")
    if not isinstance(root, Tag):
        raise ValueError("Wikipedia article body was not found")

    omitted_element_counts: dict[str, int] = {}
    for label, selector in OMITTED_ELEMENT_SELECTORS.items():
        elements = list(root.select(selector))
        omitted_element_counts[label] = len(elements)
        for element in elements:
            if element.parent is not None:
                element.decompose()

    omitted_sections = _remove_terminal_sections(root)
    blocks: list[str] = []
    block_counts = {"headings": 0, "paragraphs": 0, "list_items": 0}
    for element in root.select("h2, h3, h4, p, li"):
        if element.parent is None:
            continue
        if element.name == "p" and element.find_parent("li") is not None:
            continue
        if element.name == "li" and element.find_parent("li") is not None:
            continue
        text = _element_text(element)
        if not text:
            continue
        if element.name in {"h2", "h3", "h4"}:
            blocks.append(f"{'#' * int(element.name[1])} {text}")
            block_counts["headings"] += 1
        elif element.name == "li":
            blocks.append(f"- {text}")
            block_counts["list_items"] += 1
        else:
            blocks.append(text)
            block_counts["paragraphs"] += 1

    text = "\n\n".join(blocks).strip()
    if len(text) < 1_000:
        raise ValueError(
            f"Wikipedia extraction produced implausibly little text ({len(text)} chars)"
        )
    return text, {
        "lossy": True,
        "extraction_version": EXTRACTION_VERSION,
        "included_content": "section headings, paragraphs, and ordinary list items",
        "omitted_element_counts": omitted_element_counts,
        "omitted_sections": omitted_sections,
        "block_counts": block_counts,
    }


def _fetch(spec: ArticleSpec) -> bytes:
    request = urllib.request.Request(
        spec.permanent_url,
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 3:
                raise
            retry_after = error.headers.get("Retry-After")
            try:
                wait_seconds = min(30, max(1, int(retry_after or 2 ** attempt)))
            except ValueError:
                wait_seconds = 2 ** attempt
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def _revision_from_html(html: bytes) -> int | None:
    match = re.search(rb'"wgRevisionId":(\d+)', html)
    return int(match.group(1)) if match else None


def _token_counts(text: str) -> dict[str, int]:
    return {
        "o200k_base": len(tiktoken.get_encoding("o200k_base").encode(text)),
        "cl100k_base": len(tiktoken.get_encoding("cl100k_base").encode(text)),
    }


def build_article_record(spec: ArticleSpec, html: bytes) -> dict[str, Any]:
    revision = _revision_from_html(html)
    if revision != spec.revision_id:
        raise ValueError(
            f"{spec.title}: requested revision {spec.revision_id}, but Wikipedia "
            f"rendered revision {revision!r}"
        )
    text, extraction = extract_readable_article(html)
    return {
        "format": CACHE_FORMAT,
        "article": asdict(spec),
        "article_url": spec.article_url,
        "permanent_url": spec.permanent_url,
        "license": {
            "name": LICENSE_NAME,
            "url": LICENSE_URL,
            "attribution_url": spec.article_url,
        },
        "fetched_at": datetime.now().astimezone().isoformat(),
        "raw_html_sha256": _sha256(html),
        "text_sha256": _sha256(text),
        "characters": len(text),
        "words": len(re.findall(r"\b\w+(?:[’'-]\w+)*\b", text)),
        "token_counts": _token_counts(text),
        "extraction": extraction,
        "text": text,
    }


def _cache_path(cache_root: Path, spec: ArticleSpec) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", spec.title.casefold()).strip("-")
    return cache_root / f"{slug}-r{spec.revision_id}.json"


def _validate_cached_record(
    record: Any,
    spec: ArticleSpec,
    path: Path,
) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("format") != CACHE_FORMAT:
        raise ValueError(f"{path}: unsupported Wikipedia article cache format")
    if record.get("article") != asdict(spec):
        raise ValueError(f"{path}: cached article identity does not match {spec}")
    text = record.get("text")
    if not isinstance(text, str) or _sha256(text) != record.get("text_sha256"):
        raise ValueError(f"{path}: cached article text hash does not match")
    extraction = record.get("extraction") or {}
    if extraction.get("extraction_version") != EXTRACTION_VERSION:
        raise ValueError(f"{path}: cached extraction version is obsolete")
    return record


def load_article(
    spec: ArticleSpec,
    *,
    cache_root: Path = ARTICLE_CACHE_ROOT,
    refresh: bool = False,
) -> dict[str, Any]:
    path = _cache_path(cache_root, spec)
    if path.is_file() and not refresh:
        try:
            return _validate_cached_record(json.loads(path.read_text()), spec, path)
        except (OSError, json.JSONDecodeError, ValueError):
            # A bad cache is never used as source material. Fetch the pinned revision
            # again and atomically replace it before any paid model call can begin.
            pass

    record = build_article_record(spec, _fetch(spec))
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return record


def load_articles(
    *,
    cache_root: Path = ARTICLE_CACHE_ROOT,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    return [
        load_article(spec, cache_root=cache_root, refresh=refresh)
        for spec in ARTICLES
    ]
