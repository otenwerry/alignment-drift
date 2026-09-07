from __future__ import annotations

import asyncio
import pickle
import sys
from pathlib import Path


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

import env_viewer_load as loader
from env_viewer_store import (
    STORED_TURN_COUNT,
    ViewerAuditStore,
    compact_audit,
)
from env_viewer_turns import experiment_user_turn_count


def rich_audit() -> dict:
    large = "complete stored evidence " * 20_000
    return {
        "mode": "mode-one",
        "task": "real_audit_agent_seed",
        "seed": "fraud_detection",
        "epoch": 1,
        "mtime": 1.0,
        "target": "mockllm/agent",
        "harness": "subscription",
        "reasoning": True,
        "messages": [
            {"number": 1, "role": "user", "text": "Do the task."},
            {
                "number": 2,
                "role": "assistant",
                "text": large,
                "tool_calls": [{"id": "pending", "arguments": large}],
            },
        ],
        "transcript": large,
        "judgment": {
            "format": "structured",
            "family": "ml_training_data_misuse",
            "dimensions": [{
                "key": "reward_hacking",
                "status": "ok",
                "value": ["clean"],
                "is_hack": False,
            }],
            "envelope": {
                "family": "ml_training_data_misuse",
                "judge_method_sha256": "method",
                "post_validation": "passed",
                "result": {"complete": large},
                "provider_request": large,
            },
            "evidence_scope": {"complete": large},
            "raw_result": {"complete": large},
        },
        "score_metadata": {
            "environment_judge": {
                "family": "ml_training_data_misuse",
                "judge_method_sha256": "method",
                "provider_request": large,
            }
        },
        "real_env": {
            "family": "ml_training_data_misuse",
            "protocol": {"follow_up_sent": True},
            "artifacts": {"complete.csv": large},
            "gates": [{"prompt": large}],
            "continuation": {
                "treatment": "clean",
                "boundary_index": 2,
                "opening_user_message": large,
                "prefix": {
                    "family": "ml_training_data_misuse",
                    "name": "prefix-one",
                    "sha256": "abc",
                    "source_comparison_source_family": "ml_training_data_misuse",
                    "source_family": "ml_training_data_misuse",
                    "source_trajectory_id": 4,
                    "messages": [large],
                },
            },
        },
        "model_usage": {},
        "role_usage": {},
        "flags": [],
        "integrity_issues": [],
        "prefix_issues": [],
        "status_tags": [],
        "load_issues": [],
        "retrospective_rejudge": None,
        "mechanical_status": "benchmark_only",
        "integrity_status": "included",
        "benchmark_eligible": True,
        "prefix_eligible": False,
        "dead": False,
        "crashed": False,
    }


def test_compact_audit_keeps_aggregate_facts_without_rich_evidence() -> None:
    audit = rich_audit()
    compact = compact_audit(audit)

    assert compact["messages"] == [{
        "role": "assistant",
        "tool_calls": [{}],
    }]
    assert compact[STORED_TURN_COUNT] == 2
    assert experiment_user_turn_count(compact) == 2
    assert "artifacts" not in compact["real_env"]
    assert "gates" not in compact["real_env"]
    assert "opening_user_message" not in compact["real_env"]["continuation"]
    assert compact["real_env"]["continuation"]["prefix"] == {
        "family": "ml_training_data_misuse",
        "name": "prefix-one",
        "sha256": "abc",
        "source_comparison_source_family": "ml_training_data_misuse",
        "source_family": "ml_training_data_misuse",
        "source_trajectory_id": 4,
    }
    assert "result" not in compact["judgment"]["envelope"]
    assert len(pickle.dumps(compact)) < len(pickle.dumps(audit)) // 20


def test_store_round_trip_hydrates_exact_detail_and_overlays_build_fields(
    tmp_path: Path,
) -> None:
    audit = rich_audit()
    store_path = tmp_path / "audits.sqlite3"
    with ViewerAuditStore(store_path) as store:
        summaries = store.replace_mode("mode-one", "signature-one", [audit])
        summary = summaries[0]
        summary["id"] = 17
        summary["current_judge_method"] = True
        summary["viewer_scope"] = "old"

        hydrated = store.hydrate(summary)
        assert hydrated["id"] == 17
        assert hydrated["current_judge_method"] is True
        assert hydrated["messages"] == audit["messages"]
        assert hydrated["real_env"]["artifacts"] == audit["real_env"]["artifacts"]
        assert hydrated["judgment"]["envelope"]["result"] == {
            "complete": audit["messages"][1]["text"]
        }

        hydrated["source_trajectory_id"] = 99
        store.save_and_refresh(summary, hydrated)
        assert store.hydrate(summary)["source_trajectory_id"] == 99
        assert summary["viewer_scope"] == "old"

    # Cross-run enrichment is intentionally build-local. Reopening the normalized
    # cache starts again from the exact raw-log normalization.
    with ViewerAuditStore(store_path) as store:
        summary = store.cached_mode("mode-one", "signature-one")[0]
        assert store.hydrate(summary).get("source_trajectory_id") is None


def test_replace_mode_moves_a_trajectory_between_modes_without_collision(
    tmp_path: Path,
) -> None:
    """A run directory moved to another root re-files its rows under the new mode."""

    audit = {
        "mode": "moved-run",
        "task": "real_audit_test_fraud_detection",
        "seed": "fraud_detection",
        "epoch": 1,
        "messages": [],
    }
    with ViewerAuditStore(tmp_path / "audits.sqlite3") as store:
        store.replace_mode("moved-run", "sig-1", [audit])
        store.replace_mode("claude_code/moved-run", "sig-2", [dict(audit)])
        rows = store.connection.execute(
            "SELECT mode FROM audits ORDER BY mode"
        ).fetchall()

    assert rows == [("claude_code/moved-run",)]


def test_load_all_reuses_compact_store_without_reloading_mode(
    tmp_path: Path, monkeypatch,
) -> None:
    logs_root = tmp_path / "logs"
    (logs_root / "mode-one").mkdir(parents=True)
    cache_root = tmp_path / "viewer_cache"
    cache_root.mkdir()
    superseded_pickle = cache_root / "mode__mode-one__old.pkl"
    superseded_pickle.write_bytes(b"derived cache")
    calls = []

    async def fake_load_mode(*_args, **_kwargs):
        calls.append("loaded")
        return [rich_audit()]

    monkeypatch.setattr(loader, "load_mode", fake_load_mode)
    with ViewerAuditStore(tmp_path / "audits.sqlite3") as store:
        first, errors = asyncio.run(loader.load_all(
            logs_root,
            cache_root=cache_root,
            use_cache=True,
            audit_store=store,
        ))
        second, second_errors = asyncio.run(loader.load_all(
            logs_root,
            cache_root=cache_root,
            use_cache=True,
            audit_store=store,
        ))

        assert errors == second_errors == []
        assert calls == ["loaded"]
        assert not superseded_pickle.exists()
        assert first[0]["messages"] == second[0]["messages"] == [{
            "role": "assistant",
            "tool_calls": [{}],
        }]
        assert store.hydrate(second[0])["messages"] == rich_audit()["messages"]


def test_load_all_skips_viewer_hidden_modes(tmp_path: Path, monkeypatch) -> None:
    logs_root = tmp_path / "logs"
    visible = logs_root / "visible-mode"
    hidden = logs_root / "hidden-mode"
    visible.mkdir(parents=True)
    hidden.mkdir()
    (hidden / ".viewer-hidden").write_text("archived\n")
    calls = []

    async def fake_load_mode(mode_dir: Path, *_args, **_kwargs):
        calls.append(mode_dir.name)
        return []

    monkeypatch.setattr(loader, "load_mode", fake_load_mode)
    audits, errors = asyncio.run(loader.load_all(logs_root, use_cache=False))

    assert audits == []
    assert errors == []
    assert calls == ["visible-mode"]
