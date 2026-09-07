"""Paths owned by the real-environments project.

This module deliberately has no Petri paths.  Importing it is safe in free tooling: it
does not read secrets, create directories, or import Inspect.
"""

from __future__ import annotations

from pathlib import Path
import os
import hashlib


ENVIRONMENTS_ROOT = Path(__file__).resolve().parent.parent
MATS_ROOT = ENVIRONMENTS_ROOT  # compatibility name for existing imports

# API clients may load this explicitly at an experiment endpoint.  Helpers should not
# read it merely because they were imported.
ENV_FILE = MATS_ROOT / ".env"

SEEDS_ROOT = ENVIRONMENTS_ROOT / "seeds"
SYSTEM_PROMPT_PATH = SEEDS_ROOT / "SYSTEM_PROMPT.txt"

DATA_ROOT = Path(os.environ.get("ENVIRONMENTS_DATA_ROOT", ENVIRONMENTS_ROOT / "data" / "runs")).expanduser().resolve()
PUBLISHED_DATA_ROOT = ENVIRONMENTS_ROOT / "data" / "published"
READ_ONLY_DATA = os.environ.get("ENVIRONMENTS_READ_ONLY_DATA") == "1"
OUTPUTS_ROOT = Path(os.environ.get("ENVIRONMENTS_OUTPUT_ROOT", ENVIRONMENTS_ROOT / "outputs")).expanduser().resolve()
# Different input corpora must never share a viewer database or stable-ID cache.
CORPUS_OUTPUT_ROOT = OUTPUTS_ROOT / hashlib.sha256(str(DATA_ROOT).encode()).hexdigest()[:12]

# Preserve file names inside a corpus; new runs and HF downloads are separate roots.
FINAL_DATA_ROOT = DATA_ROOT  # compatibility alias; no "final_data" nesting
OLD_DATA_ROOT = DATA_ROOT / "old"
OTHER_DATA_ROOT = CORPUS_OUTPUT_ROOT

LOGS_ROOT = FINAL_DATA_ROOT / "logs"
CONTINUATION_PREFIXES_ROOT = FINAL_DATA_ROOT / "continuation_prefixes"
ACTIVITY_LOGS_ROOT = FINAL_DATA_ROOT / "activity_logs"
WIKIPEDIA_ARTICLES_ROOT = FINAL_DATA_ROOT / "wikipedia_articles"
TRAJECTORY_IDS_FILE = FINAL_DATA_ROOT / "trajectory_ids.json"
PREFIX_SELECTIONS_FILE = FINAL_DATA_ROOT / "prefix_selections.json"

# Archived and unpublished rows are separate inputs, excluded from published figures.
OLD_LOGS_ROOT = OLD_DATA_ROOT / "logs"
CLAUDE_CODE_LOGS_ROOT = OLD_DATA_ROOT / "logs_claude_code"
# Prefix payloads whose experiments are not in the write-up; shown only in Old.
OLD_CONTINUATION_PREFIXES_ROOT = OLD_DATA_ROOT / "continuation_prefixes"

VIEWER_ROOT = OTHER_DATA_ROOT / "viewer"
VIEWER_CACHE_ROOT = OTHER_DATA_ROOT / "viewer_cache"
FIGURES_VIEWER_ROOT = OTHER_DATA_ROOT / "figures_viewer"
FINAL_VIEWER_ROOT = OTHER_DATA_ROOT / "viewer_final"
REMOTE_CAMPAIGNS_ROOT = OTHER_DATA_ROOT / "remote_campaigns"
OVERNIGHT_RUNS_ROOT = OTHER_DATA_ROOT / "overnight_runs"
PREFIX_BATCHES_ROOT = OTHER_DATA_ROOT / "prefix_batches"
CONTINUATION_BATCHES_ROOT = OTHER_DATA_ROOT / "continuation_batches"
HF_CACHE_ROOT = OTHER_DATA_ROOT / "hf_cache"
JUDGE_TEST_SOURCES_FILE = OTHER_DATA_ROOT / "judge_test_sources.json"
ANNOTATIONS_PATH = OTHER_DATA_ROOT / "annotations.json"

# Short-lived AWS session credentials cached by tools/aws_credential_process.py.
# This is ephemeral, auto-refreshed state, not configuration, so it lives in the
# user cache directory rather than under DATA_ROOT, which holds publishable results.
AWS_AUTH_CACHE_ROOT = Path.home() / ".cache" / "trajectory-experiments" / "aws_auth"
