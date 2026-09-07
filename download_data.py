"""Download a pinned Hugging Face dataset, optionally just its reusable prefixes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "experiments" / "published" / "dataset.json"
PREFIX_PATTERNS = ["continuation_prefixes/*", "activity_logs/*", "wikipedia_articles/*",
                   "prefix_selections.json", "trajectory_ids.json", "README.md", "manifest.json"]


def download(*, repo_id: str, revision: str, output: Path, prefixes_only: bool, fetch=None) -> Path:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("Use a full 40-character HF commit hash so the download is reproducible.")
    output = output.expanduser().resolve()
    receipt = output / "download.json"
    identity = {"repo_id": repo_id, "revision": revision.lower()}
    previous = json.loads(receipt.read_text()) if receipt.exists() else None
    if previous and any(previous.get(k) != v for k, v in identity.items()):
        raise ValueError("This folder contains a different dataset revision; choose another --output.")
    if output.exists() and any(output.iterdir()) and previous is None:
        raise ValueError("Destination is not empty and has no download receipt; choose another --output.")
    output.mkdir(parents=True, exist_ok=True)
    # Record identity first so interrupted downloads can resume.
    receipt.write_text(json.dumps({**identity, "status": "downloading", "prefixes_only": prefixes_only}, indent=2) + "\n")
    if fetch is None:
        from huggingface_hub import snapshot_download
        fetch = snapshot_download
    fetch(repo_id=repo_id, repo_type="dataset", revision=revision, local_dir=output,
          allow_patterns=PREFIX_PATTERNS if prefixes_only else None,
          ignore_patterns=["download.json"])
    complete_full = bool(previous and previous.get("status") == "complete" and not previous.get("prefixes_only"))
    receipt.write_text(json.dumps({**identity, "status": "complete", "prefixes_only": prefixes_only and not complete_full}, indent=2) + "\n")
    return output


def main():
    manifest = json.loads(MANIFEST.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=manifest.get("repo_id"))
    parser.add_argument("--revision", default=manifest.get("revision"))
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "published")
    parser.add_argument("--prefixes-only", action="store_true")
    args = parser.parse_args()
    if not args.repo_id or not args.revision:
        parser.error("HF publication is pending. Supply --repo-id OWNER/DATASET and --revision COMMIT, or fill experiments/published/dataset.json after publication.")
    try:
        path = download(repo_id=args.repo_id, revision=args.revision, output=args.output, prefixes_only=args.prefixes_only)
    except ValueError as error:
        parser.error(str(error))
    print(f"Downloaded {'prefix inputs' if args.prefixes_only else 'dataset'} to {path}")


if __name__ == "__main__":
    main()
