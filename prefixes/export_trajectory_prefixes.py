"""Export stored trajectories into the shared continuation-prefix catalog.

This command is free: it reads saved trajectories and writes self-contained prefix
payloads. With ``--cutoff-before-user-turn=2`` it also removes the second experiment
turn and every later message, and rewinds supported native OpenCode/Codex session
state to the same boundary.

Usage (from mats/environments/):
  uv run prefixes/export_trajectory_prefixes.py \
      --prefixes=2852,2856 --prefix-type=ml_fraud_detection \
      --prefix-type-label='ML: fraud_detection'
  uv run prefixes/export_trajectory_prefixes.py \
      --prefixes=2852 --prefix-type=ml_fraud_detection_cutoff \
      --prefix-type-label='ML: fraud_detection (clean, 1-turn cutoff)' \
      --cutoff-before-user-turn=2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


_ENVIRONMENTS = Path(__file__).resolve().parents[1]
for _path in (_ENVIRONMENTS, _ENVIRONMENTS / "lib"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from exp_real_continuation import (  # noqa: E402
    build_prefix_spec,
    cutoff_prefix_payload,
    reconstruct_prefix_payloads,
    store_prefix_payload,
)


_TYPE_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", required=True)
    parser.add_argument("--prefix-type", required=True)
    parser.add_argument("--prefix-type-label", required=True)
    parser.add_argument("--cutoff-before-user-turn", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        args.prefixes = [
            int(value.strip())
            for value in args.prefixes.split(",")
            if value.strip()
        ]
    except ValueError as error:
        raise SystemExit("--prefixes must be comma-separated viewer IDs") from error
    if not args.prefixes:
        raise SystemExit("--prefixes cannot be empty")
    if len(set(args.prefixes)) != len(args.prefixes):
        raise SystemExit("--prefixes contains duplicate viewer IDs")
    if not _TYPE_RE.fullmatch(args.prefix_type):
        raise SystemExit(
            "--prefix-type must use lowercase letters, digits, and single underscores"
        )
    if not args.prefix_type_label.strip():
        raise SystemExit("--prefix-type-label cannot be blank")
    return args


def main() -> None:
    args = parse_args()
    outputs = []
    for payload in reconstruct_prefix_payloads(args.prefixes):
        payload["source"]["prefix_type"] = args.prefix_type
        payload["source"]["prefix_type_label"] = args.prefix_type_label.strip()
        if args.cutoff_before_user_turn is not None:
            payload = cutoff_prefix_payload(
                payload,
                before_experiment_user_turn=args.cutoff_before_user_turn,
            )
        # This validates the transcript and native archive using the exact harness
        # the future continuation must use, without calling a model.
        build_prefix_spec(
            payload,
            harness=str(payload["source"].get("harness") or "simple"),
        )
        path = None if args.dry_run else store_prefix_payload(payload)
        condition = payload["source"].get("prefix_condition") or {}
        outputs.append((payload, path, condition))

    verb = "would export" if args.dry_run else "exported"
    print(f"{verb} {len(outputs)} trajectory prefix(es):")
    for payload, path, condition in outputs:
        destination = str(path) if path is not None else "(dry run; no file written)"
        print(
            f"  {payload['name']}: {condition.get('label') or 'outcome unavailable'} "
            f"· {len(payload['messages'])} messages -> {destination}"
        )


if __name__ == "__main__":
    main()
