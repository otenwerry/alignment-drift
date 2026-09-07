"""Run the current judge on the saved fixed 20-source cohort."""

import sys


if not any(argument.startswith("--source-runs=") for argument in sys.argv[1:]):
    sys.argv.append("--source-runs=judge-tests")

from exp_rejudge import main


if __name__ == "__main__":
    main()
