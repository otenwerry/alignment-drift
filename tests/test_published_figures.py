"""Regression checks for the free, data-backed blog figure builder."""

from __future__ import annotations

import asyncio
import pytest
from pathlib import Path
import sys


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS))
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

import build_figures as builder
from post_figure_data import build_post_figure_data, point_lookup


def test_data_block_replacement_accepts_old_and_generated_pages() -> None:
    data = {
        "example": {
            "title": "Example",
            "models": [{
                "model": "glm-5.1",
                "points": [{"condition": "baseline", "k": 1, "n": 2}],
            }],
        },
    }
    old_page = (
        "<script>\n"
        f"{builder.OLD_DATA_START}\n"
        "    const DATA = {stale: true};\n\n"
        f"{builder.NEXT_FUNCTION}\n"
        "      return panels;\n"
        "    }\n"
        "</script>\n"
    )

    rendered = builder.replace_data_block(old_page, data)

    assert builder.GENERATED_START in rendered
    assert builder.GENERATED_END in rendered
    assert '"condition": "baseline"' in rendered
    assert "stale: true" not in rendered
    assert builder.replace_data_block(rendered, data) == rendered


def test_current_layout_can_be_refreshed_without_writing_it() -> None:
    page = (ENVIRONMENTS / "templates" / "figures.html").read_text()
    data = {
        "example": {
            "title": "Example",
            "models": [{"model": "gpt-5.5", "points": []}],
        },
    }

    rendered = builder.replace_data_block(page, data)

    assert rendered != page
    assert builder.NEXT_FUNCTION in rendered
    assert '<svg id="figure-1"' in rendered
    assert '<svg id="figure-1-revised"' not in rendered
    assert '<svg id="figure-6"' in rendered
    assert '<svg id="figure-6-revised"' not in rendered
    assert "Original" not in rendered
    assert "Updated" not in rendered
    assert "Possible new graphs" not in rendered
    assert 'id="possible-figure-' not in rendered
