"""Bounded-memory storage for normalized viewer audits.

The normalizer produces rich dictionaries containing complete transcripts, artifacts,
and exact stored judge requests.  Keeping every one of those dictionaries alive for a
whole static-site build scales with the complete log corpus and can exhaust a laptop.

This module stores the rich dictionaries as individual SQLite blobs while the builder
keeps only compact records needed by indexes and aggregate figures.  A detail record is
hydrated only while its trajectory and judge pages are being rendered.  The split is
lossless: the compact record is an in-memory build aid, not a replacement for the full
stored audit.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import pickle
import sqlite3
from pathlib import Path
from typing import Iterable

from env_viewer_cache import trajectory_key
from env_viewer_turns import experiment_user_turn_count


STORE_FORMAT = "environments-viewer-audit-store-v1"
STORE_FILENAME = "audits-v1.sqlite3"
STORE_KEY = "_viewer_audit_store_key"
STORED_TURN_COUNT = "_viewer_experiment_user_turn_count"


_SUMMARY_FIELDS = (
    "benchmark_eligible",
    "condition",
    "crashed",
    "dead",
    "dimension_scope",
    "epoch",
    "flags",
    "harness",
    "id",
    "integrity_issues",
    "integrity_status",
    "interrupted_native_transcript",
    "judge",
    "judge_dimensions",
    "judge_failure",
    "judgment_role",
    "judgment_transcript_coverage",
    "load_issues",
    "log_file",
    "mechanical_status",
    "mode",
    "model_usage",
    "mtime",
    "native_harness",
    "prefix_eligible",
    "prefix_issues",
    "pressure",
    "production_harness",
    "real_ended_reason",
    "reasoning",
    "real_env_manifest_sha",
    "retrospective_rejudge",
    "role_usage",
    "score_key",
    "score_value",
    "scores",
    "seed",
    "stored_seed",
    "source_trajectory_id",
    "status_tags",
    "stored_integrity",
    "target",
    "target_context_usage",
    "target_tools_mode",
    "task",
    "tool_truncations",
    "viewer_scope",
)

_SUMMARY_REAL_ENV_FIELDS = (
    "clock",
    "compute",
    "family",
    "grade",
    "harness",
    "judging",
    "multi_agent",
    "pressure",
    "protocol",
    "turns_used",
)

_SUMMARY_JUDGMENT_FIELDS = (
    "dimensions",
    "family",
    "format",
    "issues",
    "schema_version",
)

_SUMMARY_ENVELOPE_FIELDS = (
    "family",
    "judge_method_sha256",
    "post_validation",
)

_SUMMARY_CONTINUATION_FIELDS = (
    "boundary_index",
    "cross_family",
    "delivery_mode",
    "pivot_folded_into_trailing_user",
    "prefix_message_count",
    "pressure",
    "treatment",
    "version",
)

_SUMMARY_PREFIX_FIELDS = (
    "family",
    "name",
    "sha256",
    "source_comparison_source_family",
    "source_comparison_source_seed",
    "source_cutoff",
    "source_dataset",
    "source_family",
    "source_generator",
    "source_prefix_condition",
    "source_prefix_type",
    "source_seed",
    "source_trajectory_id",
)


def _compact_envelope(envelope: object) -> dict:
    if not isinstance(envelope, dict):
        return {}
    return {
        field: deepcopy(envelope[field])
        for field in _SUMMARY_ENVELOPE_FIELDS
        if field in envelope
    }


def _compact_judgment(judgment: object) -> dict | None:
    if not isinstance(judgment, dict):
        return None
    compact = {
        field: deepcopy(judgment[field])
        for field in _SUMMARY_JUDGMENT_FIELDS
        if field in judgment
    }
    envelope = _compact_envelope(judgment.get("envelope"))
    if envelope:
        compact["envelope"] = envelope
    return compact


def _compact_score_metadata(metadata: object) -> dict:
    if not isinstance(metadata, dict):
        return {}
    environment_judge = _compact_envelope(metadata.get("environment_judge"))
    return {"environment_judge": environment_judge} if environment_judge else {}


def _compact_real_env(real_env: object) -> dict:
    if not isinstance(real_env, dict):
        return {}
    compact = {
        field: deepcopy(real_env[field])
        for field in _SUMMARY_REAL_ENV_FIELDS
        if field in real_env
    }
    continuation = real_env.get("continuation")
    if isinstance(continuation, dict):
        compact_continuation = {
            field: deepcopy(continuation[field])
            for field in _SUMMARY_CONTINUATION_FIELDS
            if field in continuation
        }
        prefix = continuation.get("prefix")
        if isinstance(prefix, dict):
            compact_continuation["prefix"] = {
                field: deepcopy(prefix[field])
                for field in _SUMMARY_PREFIX_FIELDS
                if field in prefix
            }
        compact["continuation"] = compact_continuation
    return compact


def _last_message_shape(messages: object) -> list[dict]:
    """Retain only what the mechanical unfinished-action check needs."""

    if not isinstance(messages, list) or not messages:
        return []
    last = messages[-1]
    if not isinstance(last, dict):
        return []
    return [{
        "role": last.get("role"),
        "tool_calls": [{}] if last.get("tool_calls") else [],
    }]


def compact_audit(audit: dict) -> dict:
    """Return the lossless audit's small index/aggregate projection."""

    compact = {
        field: deepcopy(audit[field])
        for field in _SUMMARY_FIELDS
        if field in audit
    }
    compact["judgment"] = _compact_judgment(audit.get("judgment"))
    compact["score_metadata"] = _compact_score_metadata(
        audit.get("score_metadata")
    )
    compact["real_env"] = _compact_real_env(audit.get("real_env"))
    compact["messages"] = _last_message_shape(audit.get("messages"))
    compact[STORED_TURN_COUNT] = experiment_user_turn_count(audit)
    return compact


def _store_key(audit: dict) -> str:
    identity = trajectory_key(audit)
    return hashlib.sha256(identity.encode()).hexdigest()


class ViewerAuditStore:
    """Persistent per-audit cache with a deliberately small SQLite page cache."""

    def __init__(self, path: Path):
        self.path = path
        self._runtime_overrides: dict[str, bytes] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA cache_size = -8192")
        self.connection.execute("PRAGMA temp_store = FILE")
        self.connection.execute("PRAGMA mmap_size = 0")
        self.connection.execute("PRAGMA journal_mode = TRUNCATE")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS modes (
                mode TEXT PRIMARY KEY,
                signature TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audits (
                store_key TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                position INTEGER NOT NULL,
                summary BLOB NOT NULL,
                detail BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS audits_mode_position
                ON audits(mode, position);
            """
        )
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'format'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES('format', ?)",
                (STORE_FORMAT,),
            )
            self.connection.commit()
        elif row[0] != STORE_FORMAT:
            raise ValueError(
                f"unsupported viewer audit store format in {path}: {row[0]!r}"
            )

    def __enter__(self) -> "ViewerAuditStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def cached_mode(self, mode: str, signature: str) -> list[dict] | None:
        row = self.connection.execute(
            "SELECT signature FROM modes WHERE mode = ?", (mode,)
        ).fetchone()
        if row is None or row[0] != signature:
            return None
        summaries = []
        for store_key, payload in self.connection.execute(
            "SELECT store_key, summary FROM audits WHERE mode = ? "
            "ORDER BY position",
            (mode,),
        ):
            summary = pickle.loads(payload)
            summary[STORE_KEY] = store_key
            summaries.append(summary)
        return summaries

    def replace_mode(
        self, mode: str, signature: str, audits: Iterable[dict]
    ) -> list[dict]:
        summaries = []
        with self.connection:
            self.connection.execute("DELETE FROM audits WHERE mode = ?", (mode,))
            for position, audit in enumerate(audits):
                store_key = _store_key(audit)
                summary = compact_audit(audit)
                # A trajectory identity is unique across the whole store. When a run
                # directory moves to another root (or is renamed), its rows may still
                # sit under the old mode until pruning; replace them here.
                self.connection.execute(
                    "DELETE FROM audits WHERE store_key = ?", (store_key,)
                )
                self.connection.execute(
                    "INSERT INTO audits(store_key, mode, position, summary, detail) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (
                        store_key,
                        mode,
                        position,
                        pickle.dumps(summary, protocol=pickle.HIGHEST_PROTOCOL),
                        pickle.dumps(audit, protocol=pickle.HIGHEST_PROTOCOL),
                    ),
                )
                summary[STORE_KEY] = store_key
                summaries.append(summary)
            self.connection.execute(
                "INSERT INTO modes(mode, signature) VALUES(?, ?) "
                "ON CONFLICT(mode) DO UPDATE SET signature = excluded.signature",
                (mode, signature),
            )
        return summaries

    def prune_modes(self, active_modes: set[str]) -> None:
        stored = {
            str(row[0])
            for row in self.connection.execute("SELECT mode FROM modes")
        }
        removed = stored - active_modes
        if not removed:
            return
        with self.connection:
            for mode in removed:
                self.connection.execute("DELETE FROM audits WHERE mode = ?", (mode,))
                self.connection.execute("DELETE FROM modes WHERE mode = ?", (mode,))

    def is_backed(self, audit: dict) -> bool:
        return isinstance(audit.get(STORE_KEY), str)

    def hydrate(self, summary: dict) -> dict:
        store_key = summary.get(STORE_KEY)
        if not isinstance(store_key, str):
            return summary
        payload = self._runtime_overrides.get(store_key)
        if payload is None:
            row = self.connection.execute(
                "SELECT detail FROM audits WHERE store_key = ?", (store_key,)
            ).fetchone()
            if row is None:
                raise KeyError(f"viewer audit detail is missing: {store_key}")
            payload = row[0]
        audit = pickle.loads(payload)
        # Build-time annotations (stable ID, routing, propagated source status, and
        # current-method status) live on the compact record.  Rich nested values and
        # the complete transcript always come from the lossless detail blob.
        for key, value in summary.items():
            if key in {
                STORE_KEY,
                STORED_TURN_COUNT,
                "judgment",
                "messages",
                "real_env",
                "score_metadata",
            }:
                continue
            audit[key] = deepcopy(value)
        audit[STORE_KEY] = store_key
        audit[STORED_TURN_COUNT] = summary.get(STORED_TURN_COUNT)
        return audit

    def save_and_refresh(self, summary: dict, audit: dict) -> None:
        """Keep a build-time enriched detail without changing normalized cache data."""

        store_key = summary.get(STORE_KEY) or audit.get(STORE_KEY)
        if not isinstance(store_key, str):
            return
        detail = {
            key: value
            for key, value in audit.items()
            if key not in {STORE_KEY, STORED_TURN_COUNT}
        }
        refreshed = compact_audit(detail)
        self._runtime_overrides[store_key] = pickle.dumps(
            detail, protocol=pickle.HIGHEST_PROTOCOL
        )
        summary.clear()
        summary.update(refreshed)
        summary[STORE_KEY] = store_key
