import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


ENVIRONMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS_ROOT))

import exp_real_audit_pipeline


class RealAuditPipelineArgTests(unittest.TestCase):
    def setUp(self):
        self.base_argv = [
            "exp_real_audit_pipeline.py",
            "--targets=qwen3-32b",
            "--seed-dir=ml_training_data_misuse",
            "--seeds=fraud_detection",
            "--epochs=1",
            "--harness=simple",
            "--judge=gpt-5.6-luna",
        ]

    def test_default_runtime_selection(self):
        with patch.object(sys, "argv", self.base_argv):
            config = exp_real_audit_pipeline._parse_args()

        self.assertEqual(config["compute"], "aws")
        self.assertEqual(config["time_limit"], 4200)
        self.assertEqual(config["vm_concurrency"], 250)
        self.assertEqual(config["gate_model"], config["judge_resolved"])
        self.assertEqual(config["harness"], "simple")

    def test_harness_is_required_and_has_no_default(self):
        argv = [arg for arg in self.base_argv if not arg.startswith("--harness=")]
        with patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(SystemExit, "--harness is required"):
                exp_real_audit_pipeline._parse_args()

    def test_production_harness_is_explicitly_selectable(self):
        argv = [
            "--harness=production" if arg == "--harness=simple" else arg
            for arg in self.base_argv
        ]
        with patch.object(sys, "argv", argv):
            config = exp_real_audit_pipeline._parse_args()
        self.assertEqual(config["harness"], "production")

    def test_subscription_harness_is_explicitly_selectable(self):
        argv = [
            "--harness=subscription" if arg == "--harness=simple" else arg
            for arg in self.base_argv
        ]
        with patch.object(sys, "argv", argv):
            config = exp_real_audit_pipeline._parse_args()
        self.assertEqual(config["harness"], "subscription")

    def test_judge_defaults_to_luna(self):
        argv = [arg for arg in self.base_argv if not arg.startswith("--judge=")]
        with (
            patch.object(sys, "argv", argv),
            patch.dict(
                "os.environ",
                {"ENVIRONMENTS_JUDGE": ""},
            ),
        ):
            config = exp_real_audit_pipeline._parse_args()

        self.assertIsNone(config["judge"])
        self.assertEqual(config["judge_resolved"], "openai/gpt-5.6-luna")
        self.assertEqual(config["gate_model"], config["judge_resolved"])

    def test_local_debugging_remains_explicitly_available(self):
        with patch.object(sys, "argv", [*self.base_argv, "--compute=local"]):
            config = exp_real_audit_pipeline._parse_args()

        self.assertEqual(config["compute"], "local")

    def test_aws_ml_limit_cannot_drift_from_seventy_minutes(self):
        with patch.object(sys, "argv", [*self.base_argv, "--time-limit=3600"]):
            with self.assertRaisesRegex(SystemExit, "fixed 70-minute"):
                exp_real_audit_pipeline._parse_args()

    def test_p_hacking_defaults_to_aws_with_its_shorter_time_limit(self):
        argv = [
            "exp_real_audit_pipeline.py",
            "--targets=qwen3-32b",
            "--seed-dir=p_hacking",
            "--seeds=reasoning_prompt_benchmark",
            "--epochs=1",
            "--harness=simple",
            "--judge=gpt-5.6-luna",
        ]
        with patch.object(sys, "argv", argv):
            config = exp_real_audit_pipeline._parse_args()

        self.assertEqual(config["compute"], "aws")
        self.assertEqual(config["time_limit"], 1800)
        self.assertEqual(config["pressure"], "low")
        self.assertEqual(config["family"], "p_hacking")

    def test_p_hacking_local_debugging_remains_explicitly_available(self):
        argv = [
            "exp_real_audit_pipeline.py",
            "--targets=qwen3-32b",
            "--seed-dir=p_hacking",
            "--seeds=reasoning_prompt_benchmark",
            "--epochs=1",
            "--harness=simple",
            "--pressure=high",
            "--judge=gpt-5.6-luna",
            "--compute=local",
        ]
        with patch.object(sys, "argv", argv):
            config = exp_real_audit_pipeline._parse_args()

        self.assertEqual(config["compute"], "local")

    def test_p_hacking_defaults_to_low_pressure(self):
        argv = [
            "exp_real_audit_pipeline.py",
            "--targets=qwen3-32b",
            "--seed-dir=p_hacking",
            "--seeds=reasoning_prompt_benchmark",
            "--epochs=1",
            "--harness=simple",
            "--judge=gpt-5.6-luna",
        ]
        with patch.object(sys, "argv", argv):
            config = exp_real_audit_pipeline._parse_args()

        self.assertEqual(config["pressure"], "low")

    def test_ml_rejects_pressure(self):
        with patch.object(sys, "argv", [*self.base_argv, "--pressure=low"]):
            with self.assertRaisesRegex(SystemExit, "applies only to p_hacking"):
                exp_real_audit_pipeline._parse_args()

    def test_prefix_only_family_is_rejected_before_aws_planning(self):
        argv = [
            "--seed-dir=ml_prefix_only" if argument.startswith("--seed-dir=")
            else "--seeds=demand_forecasting" if argument.startswith("--seeds=")
            else argument
            for argument in self.base_argv
        ]
        with patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(SystemExit, "prefixes/exp_ml_prefix.py"):
                exp_real_audit_pipeline._parse_args()

    def test_aws_setup_does_not_require_experiment_args(self):
        argv = ["exp_real_audit_pipeline.py", "--aws-setup", "--confirm-personal-account"]
        with (
            patch.object(sys, "argv", argv),
            patch.object(exp_real_audit_pipeline, "setup_aws") as setup,
        ):
            exp_real_audit_pipeline.main()

        setup.assert_called_once()
        self.assertTrue(setup.call_args.args[0]["confirm_personal_account"])
        self.assertFalse(setup.call_args.args[0]["confirm_approved_account"])

    def test_aws_setup_rejects_ignored_dry_run(self):
        argv = [
            "exp_real_audit_pipeline.py", "--aws-setup",
            "--confirm-approved-account", "--dry-run",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(exp_real_audit_pipeline, "setup_aws") as setup,
        ):
            with self.assertRaisesRegex(SystemExit, "not supported"):
                exp_real_audit_pipeline.main()

        setup.assert_not_called()

    def test_retired_annotation_flags_are_rejected(self):
        for flag in ("--annotate", "--force-annotate", "--skip-annotate"):
            with self.subTest(flag=flag), patch.object(
                sys, "argv", [*self.base_argv, flag]
            ):
                with self.assertRaisesRegex(SystemExit, "unknown argument"):
                    exp_real_audit_pipeline._parse_args()


class RealAuditPipelinePostStageTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_pipeline_does_not_launch_a_viewer(self):
        with patch.object(exp_real_audit_pipeline.asyncio, "create_subprocess_exec", new=AsyncMock()) as launch:
            for skip in (False, True):
                self.assertTrue(await exp_real_audit_pipeline.run_env_post_stages({"skip_viewer": skip}))
        launch.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
