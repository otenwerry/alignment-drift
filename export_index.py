"""Write a portable CSV index and exact figure counts from a local corpus."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from analyze import configure_inputs


async def export(data_dir: Path, output_dir: Path) -> dict:
    # Configure paths before importing the viewer modules. No paid calls are made.
    configure_inputs(data_dir, output_dir)
    import build_figures as builder
    import post_figure_data as figures
    from env_viewer_cache import trajectory_key
    from env_viewer_continuations import (
        continuation_direction, continuation_of, harness_of, prefix_of,
        prefix_source_trajectory_id, pressure_of, treatment_of,
    )
    from result_classification import agent_label, trajectory_category

    records: list[dict] = []
    continuations, originals = await builder.load_curated_rows(
        use_cache=True, all_records=records,
    )
    data = figures.build_post_figure_data(continuations, originals)
    memberships: dict[str, list[str]] = defaultdict(list)
    for panel in figures.PANELS:
        for model in figures.MODEL_ORDER:
            target = figures._panel_target(continuations, panel, model)
            def matches(a):
                return (a.get("target") == target
                        and harness_of(a) == figures._harness(agent_label(target))
                        and pressure_of(a) == panel.pressure
                        and a.get("reasoning") is True)
            cells = [("baseline", [a for a in originals
                       if a.get("seed") == panel.seed and matches(a)])]
            for spec in figures.SOURCE_POINTS[panel.source][model]:
                cells.append((spec.condition, [a for a in continuations if matches(a)
                    and continuation_direction(a) == figures._point_direction(panel, spec)
                    and treatment_of(a) == spec.treatment
                    and prefix_of(a).get("name") == spec.prefix_name]))
            for condition, candidates in cells:
                for audit in candidates:
                    if trajectory_category(audit) in {"hack", "clean", "notable"}:
                        memberships[trajectory_key(audit)].append(
                            f"{panel.key}/{model}/{condition}")

    present_ids = {a.get("id") for a in records if a.get("id") is not None}
    registry = json.loads((data_dir / "trajectory_ids.json").read_text())
    rows = []
    for audit in records:
        key = trajectory_key(audit)
        rejudge = audit.get("retrospective_rejudge")
        is_prefix = builder.publication_analysis._is_prefix_generation_audit(audit)
        cells = memberships.get(key, [])
        category = trajectory_category(audit)
        source_id = audit.get("source_trajectory_id")
        if source_id is None and continuation_of(audit):
            source_id = prefix_source_trajectory_id(audit)
        if isinstance(rejudge, dict):
            source_key = (f"{rejudge.get('source_run')}__{rejudge.get('source_task')}"
                          f"__{rejudge.get('seed')}__e{rejudge.get('epoch')}")
            source_id = registry.get(source_key, source_id)
        record_type = ("rejudgment" if rejudge else "prefix_generation" if is_prefix
                       else "experiment" if cells else "prefix_source")
        log_path = Path("logs") / audit["mode"] / audit["log_file"]
        if not (data_dir / log_path).is_file():
            raise ValueError(f"Missing indexed log: {log_path}")
        rows.append({
            "trajectory_id": audit.get("id"), "trajectory_key": key,
            "record_type": record_type, "model": agent_label(audit.get("target", "")),
            "task": audit.get("seed"), "harness": harness_of(audit),
            "pressure": pressure_of(audit), "epoch": audit.get("epoch"),
            "condition": treatment_of(audit) if continuation_of(audit) else "",
            "direction": continuation_direction(audit) if continuation_of(audit) else "",
            "prefix_name": prefix_of(audit).get("name") if continuation_of(audit) else "",
            "source_trajectory_id": source_id,
            "source_in_release": source_id in present_ids if source_id is not None else "",
            "reward_hack": (category == "hack") if category in {"hack", "clean", "notable"} else "",
            "included_in_figures": bool(cells), "figure_cells": ";".join(cells),
            "integrity_issues": ";".join(audit.get("integrity_issues") or []),
            "log_path": log_path.as_posix(),
        })
    rows.sort(key=lambda r: (r["trajectory_id"] is None, r["trajectory_id"] or 0, r["trajectory_key"]))
    if len({r["trajectory_key"] for r in rows}) != len(rows):
        raise ValueError("Duplicate trajectory identities in index")
    # Check the exported memberships and outcomes against every canonical bar.
    counts = Counter()
    hacks = Counter()
    for row in rows:
        for cell in filter(None, row["figure_cells"].split(";")):
            counts[cell] += 1
            hacks[cell] += row["reward_hack"] is True
    expected = {"/".join(k): v for k, v in figures.point_lookup(data).items()}
    if {k: (hacks[k], n) for k, n in counts.items()} != expected:
        raise ValueError("CSV memberships do not reproduce the canonical figures")
    with (data_dir / "trajectories.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (data_dir / "figure_counts.json").write_text(json.dumps(data, indent=2) + "\n")
    summary = dict(Counter(row["record_type"] for row in rows))
    print(json.dumps({"records": len(rows), "record_types": summary,
                      "figure_cells": len(expected)}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/index"))
    args = parser.parse_args()
    asyncio.run(export(args.data_dir.resolve(), args.output_dir.resolve()))


if __name__ == "__main__":
    main()
