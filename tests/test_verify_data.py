import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_data import verify


def test_manifest_detects_corruption_and_unsafe_paths(tmp_path):
    payload = tmp_path / "trace.txt"
    payload.write_bytes(b"original")
    entry = {"path": "trace.txt", "bytes": 8,
             "sha256": hashlib.sha256(b"original").hexdigest()}
    manifest = {"format": "trajectory-release-files-v1", "files": [entry]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert verify(tmp_path) == 1
    payload.write_bytes(b"modified")
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify(tmp_path)
    entry["path"] = "../outside"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outside the dataset"):
        verify(tmp_path)
