"""Generate baseline trajectories using the existing audit pipeline."""
from lib.release_cli import dispatch

if __name__ == "__main__":
    dispatch("exp_real_audit_pipeline.py", paid=True)
