"""Backward-compatible names for environment-owned project paths."""

from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Iterator

from project_paths import (
    ANNOTATIONS_PATH,
    DATA_ROOT,
    ENV_FILE,
    ENVIRONMENTS_ROOT,
    LOGS_ROOT,
    MATS_ROOT,
    SEEDS_ROOT,
    VIEWER_ROOT,
)

DATA = DATA_ROOT
LOGS = LOGS_ROOT
OUT = VIEWER_ROOT
ANN_FILE = ANNOTATIONS_PATH
SANDBOX_DIR = ENVIRONMENTS_ROOT / "sandbox"
ENV_SEEDS = SEEDS_ROOT
ENV_JUDGE_INSTRUCTIONS = ENVIRONMENTS_ROOT / "judge_instructions"
ENV_DIMENSIONS = ENV_JUDGE_INSTRUCTIONS / "judge_dimensions"


@contextmanager
def annotation_file_lock(data_root: Path = DATA) -> Iterator[None]:
    """Compatibility lock for old environment-side annotation data."""

    lock_path = Path(data_root) / ".annotations.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
