"""Run the activity-log transfer condition."""
from lib.release_cli import dispatch

if __name__ == "__main__":
    dispatch("exp_multi_agent_pipeline.py", paid=True)
