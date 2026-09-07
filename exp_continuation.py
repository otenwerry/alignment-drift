"""Continue saved prefixes on a new task."""
from lib.release_cli import dispatch

if __name__ == "__main__":
    dispatch("exp_continuation_pipeline.py", paid=True)
