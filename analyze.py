"""Rebuild published figures and counts without model or AWS calls."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def configure_inputs(data_dir: Path, output_dir: Path) -> None:
    data_dir = data_dir.expanduser().resolve()
    receipt = data_dir / "download.json"
    if receipt.is_file():
        download = json.loads(receipt.read_text())
        if download.get("status") != "complete" or download.get("prefixes_only"):
            raise SystemExit("Figure reproduction requires a completed full download; rerun download_data.py without --prefixes-only.")
    if not (data_dir / "logs").is_dir() or not (data_dir / "trajectory_ids.json").is_file():
        raise SystemExit(f"Incomplete dataset at {data_dir}: need logs/ and trajectory_ids.json. Run download_data.py first.")
    os.environ["ENVIRONMENTS_DATA_ROOT"] = str(data_dir)
    os.environ["ENVIRONMENTS_OUTPUT_ROOT"] = str(output_dir.expanduser().resolve())
    os.environ["ENVIRONMENTS_READ_ONLY_DATA"] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "published")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "published")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    configure_inputs(args.data_dir, args.output_dir)
    sys.argv = [str(ROOT / "build_figures.py"), "--output", str(args.output_dir / "figures.html"),
                "--data-output", str(args.output_dir / "counts.json")]
    if args.check:
        sys.argv.append("--check")
    if args.no_cache:
        sys.argv.append("--no-cache")
    runpy.run_path(str(ROOT / "build_figures.py"), run_name="__main__")


if __name__ == "__main__":
    main()
