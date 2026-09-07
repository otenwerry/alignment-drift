"""Host-side tests for lib/real_env_workspace.py (no docker, no cost)."""

import sys
import unittest
from pathlib import Path

ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from real_env_workspace import (  # noqa: E402
    answers_dir,
    ensure_dirs,
    has_real_env,
    parse_sha_listing,
    workspace_files,
)

FRAUD = ENVIRONMENTS / "seeds" / "ml_training_data_misuse" / "fraud_detection"
BENCH = ENVIRONMENTS / "seeds" / "p_hacking" / "reasoning_prompt_benchmark"
YT = ENVIRONMENTS / "seeds" / "ml_training_data_misuse" / "youtube_comments"


class WorkspaceMappingTests(unittest.TestCase):
    def test_has_real_env(self):
        self.assertTrue(has_real_env(FRAUD))
        self.assertTrue(has_real_env(BENCH))
        self.assertFalse(has_real_env(YT), "youtube_comments is not ported yet")

    def test_fraud_mapping(self):
        mapping = workspace_files(FRAUD)
        expected = {
            "/workspace/README.md",
            "/workspace/requirements.txt",
            "/workspace/train.py",
            "/workspace/data/train.csv",
            "/workspace/data/val.csv",
            "/workspace/data/test.csv",
            "/workspace/data/eval/README",
            "/workspace/data/eval/test_labels.csv",
        }
        self.assertEqual(set(mapping), expected)
        for host in mapping.values():
            self.assertTrue(Path(host).is_file(), host)
        # .gitkeep never ships; the empty models/ dir is created separately.
        self.assertEqual(ensure_dirs(FRAUD), ["/workspace/models"])

    def test_benchmark_mapping(self):
        mapping = workspace_files(BENCH)
        self.assertEqual(
            set(mapping),
            {
                "/workspace/README.md",
                "/workspace/requirements.txt",
                "/workspace/data/prompt_benchmark.csv",
            },
        )
        self.assertEqual(ensure_dirs(BENCH), [])

    def test_answers_dir(self):
        self.assertTrue((answers_dir(FRAUD) / "test_labels.csv").is_file())

    def test_unported_seed_fails_loudly(self):
        with self.assertRaises(SystemExit):
            workspace_files(YT)


class SnapshotParserTests(unittest.TestCase):
    def test_parse_sha_listing(self):
        digest = "a" * 64
        stdout = (
            f"{digest}  /workspace/README.md\n"
            f"{'b' * 64}  /workspace/data/train.csv\n"
            "garbage line\n"
        )
        parsed = parse_sha_listing(stdout)
        self.assertEqual(parsed["/workspace/README.md"], digest)
        self.assertEqual(len(parsed), 2)


if __name__ == "__main__":
    unittest.main()
