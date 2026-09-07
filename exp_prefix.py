"""Build or export prefixes with the existing experiment implementations."""
import argparse
import sys
from lib.release_cli import dispatch

BUILDERS = {
    "ml": "exp_ml_prefix.py",
    "p-hacking": "exp_p_hacking_prefix.py",
    "wikipedia": "exp_wikipedia_prefix.py",
    "nq": "exp_nq_prefix.py",
    "science-ethics": "exp_science_ethics_prefix.py",
    "general-ethics": "exp_general_ethics_prefix.py",
    "move-fast": "exp_move_fast_prefix.py",
    "export": "export_trajectory_prefixes.py",
    "activity-log": "build_activity_log_prefixes.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, usage="%(prog)s KIND [builder options]")
    parser.add_argument("kind", choices=BUILDERS)
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    dispatch("prefixes/" + BUILDERS[args.kind], paid=args.kind not in {"export", "activity-log"})


if __name__ == "__main__":
    main()
