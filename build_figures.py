#!/usr/bin/env python3
"""Refresh the existing blog-figure page from the canonical viewer data.

This is a free local build step.  It performs no experiment, model, judge, AWS,
or other network call.  The existing HTML remains the layout template; only its
generated ``DATA`` block is replaced.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

# Importing the shared viewer also imports Matplotlib. Keep its caches in a
# writable disposable location on machines where the user cache is read-only.
RUNTIME_CACHE = Path(tempfile.gettempdir()) / "supermats-post-figures-cache"
(RUNTIME_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
(RUNTIME_CACHE / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE / "xdg"))

import publication_analysis  # noqa: E402

from env_viewer_continuations import continuation_of  # noqa: E402
from env_viewer_load import assign_stable_ids, load_all, viewer_build_lock  # noqa: E402
from env_viewer_store import STORE_FILENAME, ViewerAuditStore  # noqa: E402
from old_runs import (  # noqa: E402
    old_run_names,
    old_trajectory_key_reasons,
    promoted_rejudge_run_names,
)
from post_figure_data import build_post_figure_data, point_lookup  # noqa: E402
from project_paths import FIGURES_VIEWER_ROOT, LOGS_ROOT  # noqa: E402


DEFAULT_OUTPUT = FIGURES_VIEWER_ROOT / "index.html"
GENERATED_START = "    // BEGIN GENERATED POST FIGURE DATA"
GENERATED_END = "    // END GENERATED POST FIGURE DATA"
OLD_DATA_START = "    // Exact counts from the poster viewer. Conditions remain separate and unpooled."
NEXT_FUNCTION = "    function aggregateByModel(panels) {"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--template",
        type=Path,
        help="layout source; defaults to the bundled templates/figures.html",
    )
    parser.add_argument("--data-output", type=Path, help="also save computed figure data as JSON")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="load and validate the data without rewriting the HTML",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def replace_data_block(page: str, data: dict[str, dict]) -> str:
    start = page.find(GENERATED_START)
    if start < 0:
        start = page.find(OLD_DATA_START)
    end = page.find(NEXT_FUNCTION, start if start >= 0 else 0)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("post figure template has no recognized DATA block")
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    generated = (
        GENERATED_START
        + "\n    const DATA = "
        + payload.replace("\n", "\n    ")
        + ";\n"
        + GENERATED_END
        + "\n\n"
    )
    return page[:start] + generated + page[end:]


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


async def load_curated_rows(
    *, use_cache: bool, all_records: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Apply the same loading, promotion, and current/old split as publication_analysis.py."""

    with viewer_build_lock(publication_analysis.CACHE_ROOT.parent), ViewerAuditStore(
        publication_analysis.CACHE_ROOT / STORE_FILENAME
    ) as audit_store:
        audits, errors = await load_all(
            LOGS_ROOT,
            cache_root=publication_analysis.CACHE_ROOT,
            use_cache=use_cache,
            audit_store=audit_store,
            progress=True,
            # Other analysis commands can share this cache.
            prune=False,
        )
        # Promoted rejudgments can refer to originals archived out of Current.
        # Load Old for those links, then exclude it from plotted cohorts below.
        old_audits, old_errors = await load_all(
            publication_analysis.OLD_LOGS_ROOT,
            cache_root=publication_analysis.CACHE_ROOT,
            use_cache=use_cache,
            audit_store=audit_store,
            store_mode_prefix=publication_analysis.OLD_STORE_PREFIX,
            prune=False,
        )
        for audit in old_audits:
            audit["viewer_scope"] = publication_analysis.OLD_SCOPE
        audits += old_audits
        errors += old_errors
        if all_records is not None:
            all_records.extend(
                audit for audit in audits
                if publication_analysis._is_prefix_generation_audit(audit)
                and audit.get("viewer_scope") != publication_analysis.OLD_SCOPE
            )
        audits = [
            audit for audit in audits
            if not publication_analysis._is_prefix_generation_audit(audit)
        ]
        assign_stable_ids(audits, publication_analysis.REGISTRY_FILE)
        publication_analysis._link_and_promote_audits(
            audits, audit_store, promoted_rejudge_run_names()
        )
        publication_analysis._apply_continuation_source_statuses(audits)
        current_methods = await publication_analysis._current_judge_methods()
        archived_modes = old_run_names()
        archived_key_reasons = old_trajectory_key_reasons()

        if all_records is not None:
            all_records.extend(
                audit for audit in audits
                if audit.get("viewer_scope") != publication_analysis.OLD_SCOPE
            )

        continuations: list[dict] = []
        originals: list[dict] = []
        for audit in audits:
            audit["current_judge_method"] = publication_analysis._is_current_judgment(
                audit, current_methods
            )
            archived = publication_analysis._archive_reason(
                audit,
                harness=publication_analysis._audit_harness(audit),
                archived_run_names=archived_modes,
                archived_key_reasons=archived_key_reasons,
            ) is not None
            if (
                audit.get("retrospective_rejudge")
                or archived
                or audit.get("viewer_scope") == publication_analysis.OLD_SCOPE
            ):
                continue
            if continuation_of(audit):
                continuations.append(audit)
                continue
            if publication_analysis._multi_agent_record(audit):
                continue
            if (
                audit["current_judge_method"]
                or (not audit.get("judgment") and not audit.get("judge_failure"))
            ):
                originals.append(audit)
        if errors:
            modes = ", ".join(str(error.get("mode") or "unknown") for error in errors)
            raise RuntimeError(f"viewer data had {len(errors)} load error(s): {modes}")
        return continuations, originals


def print_summary(data: dict[str, dict]) -> None:
    lookup = point_lookup(data)
    print(
        f"post figures: {len(data)} panels, {len(lookup)} plotted points, "
        f"max n={max(n for _k, n in lookup.values())}"
    )
    for panel, model, condition in sorted(lookup):
        k, n = lookup[(panel, model, condition)]
        print(f"  {panel:<9} {model:<18} {condition:<12} {k:>2}/{n:<2}")


async def build(args: argparse.Namespace) -> int:
    continuations, originals = await load_curated_rows(
        use_cache=not args.no_cache
    )
    data = build_post_figure_data(continuations, originals)
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print_summary(data)
    if args.check:
        print("Validated only; post figure HTML was not rewritten.")
        return 0

    if getattr(args, "data_output", None):
        write_atomic(args.data_output, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    output = args.output.resolve()
    template = (args.template or ROOT / "templates" / "figures.html").resolve()
    rendered = replace_data_block(template.read_text(), data)
    write_atomic(output, rendered)
    print(f"Wrote {output}")
    print("Free local build: no experiment, model, judge, AWS, or network call.")
    return 0


def main() -> int:
    return asyncio.run(build(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
