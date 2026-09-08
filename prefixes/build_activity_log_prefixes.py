"""Build free continuation prefixes from stored observable activity logs.

Each output contains one neutral user message with the complete line-addressable
observable log from a prior trajectory. It contains no fabricated assistant history,
no workspace file, and no native session state. Production and subscription
continuations therefore start fresh and deliver the text directly in the opening
user context.

Usage (from the repository root):
  uv run prefixes/build_activity_log_prefixes.py --trajectories=711,715,723
  uv run prefixes/build_activity_log_prefixes.py \
      --source-prefix-files='data/runs/continuation_prefixes/run-*.json'
  uv run prefixes/build_activity_log_prefixes.py --trajectories=711 --dry-run

This command is free: it reads local trajectories and optionally writes deterministic,
content-addressed JSON payloads. It makes no model, judge, AWS, or network calls.
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path


_ENVIRONMENTS = Path(__file__).resolve().parents[1]
for _path in (_ENVIRONMENTS, _ENVIRONMENTS / "lib"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from exp_real_continuation import (  # noqa: E402
    activity_log_prefix_payload_from_trajectory,
    activity_log_prefix_payload_from_source,
    build_prefix_spec,
    load_prefix_payload_file,
    load_prefix_specs,
    reconstruct_prefix_payloads,
    store_prefix_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectories",
        default="",
        help="comma-separated viewer trajectory IDs",
    )
    parser.add_argument(
        "--source-prefix-files",
        default="",
        help=(
            "comma-separated saved trajectory-prefix payloads; quoted glob patterns "
            "are expanded by this command"
        ),
    )
    parser.add_argument(
        "--expected",
        type=int,
        help="fail unless exactly this many total sources resolve",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        args.trajectories = [
            int(value.strip())
            for value in args.trajectories.split(",")
            if value.strip()
        ]
    except ValueError as error:
        raise SystemExit(
            "--trajectories must be comma-separated viewer IDs"
        ) from error
    if len(set(args.trajectories)) != len(args.trajectories):
        raise SystemExit("--trajectories contains duplicate viewer IDs")
    source_files: list[Path] = []
    for value in args.source_prefix_files.split(","):
        pattern = value.strip()
        if not pattern:
            continue
        matches = sorted(Path(match).resolve() for match in glob.glob(pattern))
        if not matches and not glob.has_magic(pattern):
            matches = [Path(pattern).resolve()]
        if not matches:
            raise SystemExit(
                f"--source-prefix-files pattern matched no files: {pattern}"
            )
        missing = [path for path in matches if not path.is_file()]
        if missing:
            raise SystemExit(f"source prefix file not found: {missing[0]}")
        source_files.extend(matches)
    if len(set(source_files)) != len(source_files):
        raise SystemExit("--source-prefix-files resolves duplicate files")
    args.source_prefix_files = source_files
    total = len(args.trajectories) + len(args.source_prefix_files)
    if total == 0:
        raise SystemExit(
            "supply --trajectories and/or --source-prefix-files"
        )
    if args.expected is not None:
        if args.expected < 1:
            raise SystemExit("--expected must be at least 1")
        if total != args.expected:
            raise SystemExit(
                f"resolved {total} source(s), expected exactly {args.expected}"
            )
    return args


def main() -> None:
    args = parse_args()
    pending: list[tuple[dict, str, str]] = []
    outputs: list[tuple[dict, Path | None, str, str]] = []
    reconstructed = reconstruct_prefix_payloads(
        args.trajectories,
        source_use="activity_log",
    )
    for trajectory_id, source_payload in zip(
        args.trajectories, reconstructed, strict=True
    ):
        payload = activity_log_prefix_payload_from_trajectory(source_payload)
        harness = str(payload["source"].get("harness") or "simple")
        # Validate the exact delivery path the source model normally uses. This is
        # free and proves that no native resume bundle is required.
        build_prefix_spec(payload, harness=harness)
        pending.append((payload, f"trajectory #{trajectory_id}", harness))

    for source_path in args.source_prefix_files:
        source_payload = load_prefix_payload_file(source_path)
        payload = activity_log_prefix_payload_from_source(source_payload)
        harness = str(payload["source"].get("harness") or "simple")
        build_prefix_spec(payload, harness=harness)
        pending.append((payload, source_path.name, harness))

    names = [payload["name"] for payload, _source, _harness in pending]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise SystemExit(
            "resolved sources would create duplicate activity-log prefix names: "
            + ", ".join(duplicate_names)
        )

    for payload, source_name, harness in pending:
        path = None if args.dry_run else store_prefix_payload(payload)
        if path is not None:
            # Exercise the exact file-loading/revalidation path used by the ordinary
            # continuation controller, including current source-integrity checks.
            load_prefix_specs([], [str(path)], harness=harness)
        outputs.append((payload, path, source_name, harness))

    verb = "would build" if args.dry_run else "built"
    print(f"{verb} {len(outputs)} inline activity-log prefix(es):")
    by_harness: dict[str, list[str]] = defaultdict(list)
    for payload, path, source_name, harness in outputs:
        metadata = payload["source"]["activity_log_metadata"]
        destination = str(path) if path is not None else "(dry run; no file written)"
        print(
            f"  {source_name} -> {payload['name']} · "
            f"{payload['target']} · {metadata['line_count']:,} lines · "
            f"{metadata['byte_count']:,} bytes -> {destination}"
        )
        if path is not None:
            by_harness[harness].append(str(path))

    if by_harness:
        print("\nUse the generated files with the ordinary continuation pipeline:")
        for harness, paths in sorted(by_harness.items()):
            print(
                "  uv run exp_continuation_pipeline.py "
                "--treatment=activity-log-context "
                f"--prefix-files={','.join(paths)} --harness={harness} "
                "--seed-dir=<family> --seeds=<members> --epochs=<n>"
            )
    print("\nNo model, judge, AWS, credential, or network call was made.")


if __name__ == "__main__":
    main()
