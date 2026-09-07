"""Missing release sources require explicit archival provenance."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from env_viewer_load import promote_rejudge_judgments


def test_missing_source_requires_exact_archive_key():
    row = {"id": 10, "mode": "rejudge", "retrospective_rejudge": {
        "source_run": "run", "source_task": "task", "seed": "seed", "epoch": 2}}
    for allowed in [(), ("run__task__seed__e1",)]:
        with pytest.raises(ValueError, match="no loaded source"):
            promote_rejudge_judgments([row], {"rejudge"}, archived_source_keys=allowed)
    assert promote_rejudge_judgments(
        [row], {"rejudge"}, archived_source_keys={"run__task__seed__e2"}) == 0
    assert row["load_issues"] == [{"kind": "archived_rejudge_source_not_in_release",
                                   "source_trajectory_key": "run__task__seed__e2"}]
