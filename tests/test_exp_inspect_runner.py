"""Free tests for the shared Inspect runner."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

import exp_inspect_runner as runner


class InspectRunnerTests(unittest.TestCase):
    def test_run_eval_returns_complete_saved_logs(self):
        summary_log = SimpleNamespace(samples=[], location="saved.eval")
        saved_log = SimpleNamespace(
            samples=[SimpleNamespace(id="completed-sample")],
            stats=SimpleNamespace(model_usage={}),
        )
        install = SimpleNamespace(to_dict=lambda: {"installed": True})

        with tempfile.TemporaryDirectory() as temporary, patch(
            "inspect_ai.eval_set", return_value=(True, [summary_log])
        ), patch(
            "inspect_ai.log.list_eval_logs", return_value=["saved.eval"]
        ), patch(
            "inspect_ai.log.read_eval_log", return_value=saved_log
        ), patch.object(
            runner, "install_cost_tracking", return_value=install
        ), patch.object(
            runner, "install_inspect_warmup", return_value=True
        ), patch.object(
            runner, "cost_tracking_provenance", return_value={}
        ), patch.object(
            runner,
            "write_cache_report",
            return_value={"verified_models": [], "unverified_models": []},
        ):
            success, logs = runner.run_eval(
                tasks=[object()],
                epochs=1,
                concurrency=1,
                log_dir=temporary,
            )
            manifest = json.loads(
                (Path(temporary) / runner.RUNTIME_MANIFEST_FILENAME).read_text()
            )

        self.assertTrue(success)
        self.assertEqual(logs, [saved_log])
        self.assertEqual(logs[0].samples[0].id, "completed-sample")
        self.assertEqual(manifest["status"], "complete")

    def test_dead_targets_accepts_undated_included_usage_names(self):
        log = SimpleNamespace(stats=SimpleNamespace(model_usage={
            "openai-api/opencode-go/deepseek-v4-pro": SimpleNamespace(
                output_tokens=24_158
            ),
            "openai-api/opencode-go/kimi-k2.6": SimpleNamespace(
                output_tokens=18_634
            ),
            "subscription/openai/gpt-5.5": SimpleNamespace(output_tokens=10_000),
        }))

        dead = runner.dead_targets(
            [log],
            [
                "openrouter/deepseek/deepseek-v4-pro-20260423",
                "openrouter/moonshotai/kimi-k2.6-20260420",
                "openai/gpt-5.5-2026-04-23",
            ],
        )

        self.assertEqual(dead, [])

    def test_dead_targets_still_reports_a_genuinely_silent_target(self):
        log = SimpleNamespace(stats=SimpleNamespace(model_usage={
            "openai-api/opencode-go/kimi-k2.6": SimpleNamespace(output_tokens=1),
        }))

        dead = runner.dead_targets(
            [log],
            [
                "openrouter/deepseek/deepseek-v4-pro-20260423",
                "openrouter/moonshotai/kimi-k2.6-20260420",
            ],
        )

        self.assertEqual(dead, ["deepseek-v4-pro-20260423"])

    def test_dead_targets_does_not_guess_between_two_snapshot_versions(self):
        log = SimpleNamespace(stats=SimpleNamespace(model_usage={
            "provider/example": SimpleNamespace(output_tokens=1),
        }))

        dead = runner.dead_targets(
            [log],
            ["provider/example-20260101", "provider/example-20260202"],
        )

        self.assertEqual(dead, ["example-20260101", "example-20260202"])


if __name__ == "__main__":
    unittest.main()
