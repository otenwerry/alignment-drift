"""Export the HF preview from the plotted cohort and stored final judgments."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import quote
import zipfile

# Enable the Zstandard compression used by Inspect .eval archives.
import inspect_ai._util.zipfile  # noqa: F401
from inspect_ai.log import read_eval_log


MODELS = {
    "gpt-5.5": "GPT-5.5", "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "glm-5.1": "GLM-5.1", "kimi-k2.6": "Kimi-K2.6",
}
TASKS = {
    "fraud_detection": "ML: fraud detection",
    "checkout_redesign": "Data analysis: checkout redesign",
}
TREATMENTS = {
    "hack-in-one-turn": "Hack prefix (1 turn)",
    "hack-in-two-turns": "Hack prefix (2 turns)",
    "no-hack": "Honest prefix",
    "no-honeypot": "No-honeypot prefix",
    "wikipedia-summaries": "Wikipedia context (~40k tokens)",
    "wikipedia-summaries-100k": "Wikipedia context (~100k tokens)",
}
SOURCES = {
    "demand_forecasting": "ML",
    "reasoning_prompt_benchmark": "Data analysis",
    "activity_log_context_ml": "ML",
    "activity_log_context": "Data analysis",
}


def condition_label(row: dict[str, str]) -> str:
    if not row["condition"]:
        return "Baseline (no prefix)"
    label = TREATMENTS[row["condition"]]
    if row["condition"].startswith("wikipedia-summaries"):
        return label
    source, destination = row["direction"].rsplit("_to_", 1)
    delivery = "Activity log" if source.startswith("activity_log") else "Conversation"
    destination_label = {"fraud_detection": "ML", "checkout_redesign": "Data analysis"}[destination]
    return f"{label}; {delivery.lower()}; {SOURCES[source]} → {destination_label}"


def judge_summary(root: Path, row: dict[str, str]) -> str:
    path = root / row["log_path"]
    # Reductions preserve complete score metadata without loading target events.
    with zipfile.ZipFile(path) as archive:
        reductions = json.loads(archive.read("reductions.json"))
    scores = [s for r in reductions if r["scorer"] == "environment_judge"
              for s in r["samples"] if str(s["sample_id"]) == row["task"]]
    if len(scores) != 1:
        raise ValueError(f"Expected one final judgment: {row['trajectory_id']}")
    score = scores[0]
    envelope = score["metadata"]["environment_judge"]
    if envelope.get("post_validation") != "passed":
        raise ValueError(f"Invalid final judgment: {row['trajectory_id']}")
    expected = {"True": True, "False": False}[row["reward_hack"]]
    if score["value"]["reward_hacking"] != expected:
        raise ValueError(f"Index/judgment mismatch: {row['trajectory_id']}")
    summary = envelope["result"]["summary"]
    if "attachment://" in summary:
        log = read_eval_log(path, resolve_attachments=True)
        sample = next(s for s in log.samples or []
                      if str(s.id) == row["task"] and s.epoch == int(row["epoch"]))
        summary = sample.scores["environment_judge"].metadata["environment_judge"]["result"]["summary"]
    if not isinstance(summary, str) or not summary.strip() or "attachment://" in summary:
        raise ValueError(f"Missing complete judge summary: {row['trajectory_id']}")
    return summary


def export(root: Path, repo_id: str, revision: str) -> int:
    with (root / "trajectories.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["included_in_figures"] == "True"]
    output = []
    for index, row in enumerate(rows, 1):
        output.append({
            "model": MODELS[row["model"]],
            "task": TASKS[row["task"]],
            "condition": condition_label(row),
            "reward_hack": {"True": "Yes", "False": "No"}[row["reward_hack"]],
            "judge_summary": judge_summary(root, row),
            "trajectory_file": f"https://huggingface.co/datasets/{repo_id}/resolve/{quote(revision, safe='')}/{quote(row['log_path'], safe='/')}?download=true",
        })
        if index % 250 == 0:
            print(f"Read {index}/{len(rows)} final judgments", flush=True)
    with (root / "preview.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} preview rows", flush=True)
    return len(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True, help="HF commit containing the trajectory logs")
    args = parser.parse_args()
    export(args.data_dir.resolve(), args.repo_id, args.revision)
