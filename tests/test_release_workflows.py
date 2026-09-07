"""Offline checks for the public checkout and dataset boundary."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lib"))

from download_data import download
from aws_source_bundle import build_source_bundle


def test_download_progress_supports_unsized_file_iterators():
    # HF snapshot_download passes an unsized iterator to tqdm's thread_map.
    # tqdm 4.70 raises ValueError before downloading any files in this case.
    from tqdm.contrib.concurrent import thread_map
    files = (name for name in ["README.md", "trajectories.csv"])
    assert thread_map(str, files, max_workers=1, disable=True) == [
        "README.md", "trajectories.csv",
    ]


def test_prefix_download_can_expand_without_mixing_revisions(tmp_path):
    calls = []
    def fetch(**kwargs):
        calls.append(kwargs)
        (kwargs["local_dir"] / "trajectory_ids.json").write_text("{}")

    destination = tmp_path / "published"
    args = dict(repo_id="owner/study", revision="a" * 40, output=destination, fetch=fetch)
    download(**args, prefixes_only=True)
    assert calls[0]["repo_type"] == "dataset"
    assert "continuation_prefixes/*" in calls[0]["allow_patterns"]
    assert not any(pattern.startswith("logs/") for pattern in calls[0]["allow_patterns"])
    download(**args, prefixes_only=False)
    assert calls[-1]["allow_patterns"] is None
    receipt = json.loads((destination / "download.json").read_text())
    assert receipt["status"] == "complete" and not receipt["prefixes_only"]
    with pytest.raises(ValueError, match="different dataset revision"):
        download(**{**args, "revision": "b" * 40}, prefixes_only=False)
    assert len(calls) == 2


def test_download_requires_immutable_revision_and_preserves_existing_files(tmp_path):
    (tmp_path / "existing.txt").write_text("keep")
    with pytest.raises(ValueError, match="40-character"):
        download(repo_id="owner/study", revision="main", output=tmp_path, prefixes_only=False)
    with pytest.raises(ValueError, match="not empty"):
        download(repo_id="owner/study", revision="a" * 40, output=tmp_path, prefixes_only=False)
    assert (tmp_path / "existing.txt").read_text() == "keep"
    assert not (tmp_path / "download.json").exists()


def test_interrupted_download_can_resume(tmp_path):
    def interrupted(**kwargs):
        raise OSError("connection interrupted")
    args = dict(repo_id="owner/study", revision="a" * 40, output=tmp_path, prefixes_only=False)
    with pytest.raises(OSError):
        download(**args, fetch=interrupted)
    assert json.loads((tmp_path / "download.json").read_text())["status"] == "downloading"
    download(**args, fetch=lambda **kwargs: None)
    assert json.loads((tmp_path / "download.json").read_text())["status"] == "complete"


def test_flat_checkout_bundle_preserves_worker_layout_without_local_results(tmp_path):
    repo = tmp_path / "arbitrary-name"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for name in ["lib/example.py", "seeds/example/workspace/data/train.csv", "data/runs/logs/result.eval",
                 "outputs/counts.json", "archive/legacy_scripts/old.py"]:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    bundle = build_source_bundle(repo, tmp_path / "bundles")
    with tarfile.open(bundle["path"]) as archive:
        names = set(archive.getnames())
    assert "mats/environments/lib/example.py" in names
    assert "mats/environments/seeds/example/workspace/data/train.csv" in names
    assert not any("result.eval" in n or "counts.json" in n or "old.py" in n for n in names)


def test_published_registry_cannot_be_extended(tmp_path, monkeypatch):
    import project_paths
    from env_viewer_cache import assign_stable_ids, trajectory_key
    monkeypatch.setattr(project_paths, "READ_ONLY_DATA", True)
    audit = {"mode": "run", "task": "task", "seed": "seed", "epoch": 1}
    registry = tmp_path / "trajectory_ids.json"
    registry.write_text(json.dumps({trajectory_key(audit): 42}))
    before = registry.read_bytes()
    assign_stable_ids([audit], registry)
    assert audit["id"] == 42 and registry.read_bytes() == before
    with pytest.raises(ValueError, match="missing 1 trajectory IDs"):
        assign_stable_ids([{**audit, "epoch": 2}], registry)
    assert registry.read_bytes() == before


def test_paths_are_checkout_relative_and_imports_do_not_create_data(tmp_path):
    environment = dict(os.environ, ENVIRONMENTS_DATA_ROOT=str(tmp_path / "custom"),
                       ENVIRONMENTS_OUTPUT_ROOT=str(tmp_path / "generated"))
    result = subprocess.run([sys.executable, "-c",
        "import sys;sys.path.insert(0,'lib');import project_paths as p;print(p.LOGS_ROOT);print(p.ENV_FILE)"],
        cwd=ROOT, env=environment, text=True, capture_output=True, check=True)
    assert str(tmp_path / "custom" / "logs") in result.stdout
    assert str(ROOT / ".env") in result.stdout
    assert not (tmp_path / "custom").exists()


def test_public_help_never_dispatches_paid_code(monkeypatch, capsys):
    from lib import release_cli
    monkeypatch.setattr(sys, "argv", ["exp_baseline.py", "--help"])
    def forbidden(*args, **kwargs):
        pytest.fail("Help dispatched an experiment")
    monkeypatch.setattr(release_cli.runpy, "run_path", forbidden)
    release_cli.dispatch("exp_real_audit_pipeline.py", paid=True)
    assert "--targets" in capsys.readouterr().out


def test_local_baseline_plan_never_launches_runtime(monkeypatch, capsys):
    import exp_real_audit_pipeline as pipeline
    monkeypatch.setattr(sys, "argv", ["exp_baseline.py", "--targets=deepseek-v4-pro",
        "--seed-dir=p_hacking", "--seeds=checkout_redesign", "--epochs=1",
        "--harness=production", "--compute=local", "--dry-run"])
    def forbidden(*args, **kwargs):
        pytest.fail("Local plan launched runtime work")
    for name in ("require_docker", "run_real_audit_stage", "run_env_post_stages", "run_campaign"):
        monkeypatch.setattr(pipeline, name, forbidden)
    pipeline.main()
    output = capsys.readouterr().out
    assert '"trajectories": 1' in output and "Local plan only" in output


def test_published_checkout_prefix_manifest_preserves_exact_choices():
    manifest = json.loads(
        (ROOT / "experiments/published/checkout_continuations.json").read_text()
    )
    rows = manifest["prefixes"]
    assert manifest["status"] == "complete"
    assert manifest["runner"] == "exp_continuation.py"
    assert len(rows) == 46
    assert len({row["filename"] for row in rows}) == 46
    assert all(len(row["file_sha256"]) == 64 for row in rows)
