"""Free tests for one-VM-per-trajectory AWS orchestration (no AWS calls)."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))
sys.path.insert(0, str(ENVIRONMENTS / "tools"))

from aws_credential_process import credentials as shared_aws_credentials  # noqa: E402

from exp_aws_trajectory import (  # noqa: E402
    AGENT_TIME_LIMIT_SECONDS,
    AMI_BUILDER_WATCHDOG_MINUTES,
    AWS_CLI_VERSION,
    FAILURE_PACKAGE_SECONDS,
    ROOT_VOLUME_GB,
    SSM_PARAMETER_NAME,
    UNCONDITIONAL_TERMINATION_SECONDS,
    PERSONAL_REIMBURSEMENT_FUNDING,
    RUNTIME_VERSION,
    AwsTrajectoryError,
    _campaign_logs_root,
    _campaign_state_path,
    _finalize_campaign_import,
    _iam_policy,
    _campaign_id,
    _authentication_recovery_message,
    _builder_user_data,
    _launch_cell,
    _monitor_campaign,
    _launch_template_root_volume_gb,
    _secure_extract_result,
    _vm_cost,
    _worker_user_data,
    account_id,
    aws_clients,
    build_cells,
    build_prefix_generation_cells,
    build_runtime_ami,
    build_source_bundle,
    campaign_ok,
    delete_campaign_s3_objects,
    ensure_bucket,
    import_campaign_results,
    preflight_aws,
    put_api_keys,
    retry_failed,
    run_campaign,
    setup_aws,
    source_file_list,
    worker_main,
)
from project_paths import (  # noqa: E402
    DATA_ROOT,
    LOGS_ROOT,
    REMOTE_CAMPAIGNS_ROOT,
)
from aws_worker_runtime import (  # noqa: E402
    MIN_WORKER_FREE_BYTES,
    WorkerRuntimeError,
    _bubblewrap_check,
    _package_worker_result,
    _sandbox_compose_path,
    _upload_worker_failure,
    _worker_disk_status,
    smoke_worker_main,
)
from aws_source_bundle import runtime_dependency_files  # noqa: E402


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


class SourceBundleTests(unittest.TestCase):
    def test_current_source_bytes_are_bundled_without_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mats"
            environments = root / "environments"
            environments.mkdir(parents=True)
            petri = root / "petri"
            petri.mkdir()
            git(root, "init")
            (environments / "tracked.py").write_text("old\n")
            (environments / ".env").write_text("OPENAI_API_KEY=secret\n")
            (environments / "credentials").write_text("secret\n")
            (petri / "unrelated.py").write_text("must not ship\n")
            git(
                root,
                "add",
                "environments/tracked.py",
                "environments/.env",
                "environments/credentials",
                "petri/unrelated.py",
            )
            # The bundle must read the dirty working-tree bytes, not the git index.
            (environments / "tracked.py").write_text("dirty current bytes\n")
            (environments / "untracked.py").write_text("new source\n")
            (environments / ".venv").mkdir()
            (environments / ".venv" / "cache.py").write_text("cache\n")

            first = build_source_bundle(environments, Path(tmp) / "out1")
            second = build_source_bundle(environments, Path(tmp) / "out2")
            selected = source_file_list(root)

            self.assertEqual(first["sha256"], second["sha256"])
            self.assertIn(Path("environments/tracked.py"), selected)
            self.assertNotIn(Path("petri/unrelated.py"), selected)
            with tarfile.open(first["path"], "r:gz") as archive:
                names = set(archive.getnames())
                tracked = archive.extractfile("mats/environments/tracked.py")
                untracked = archive.extractfile("mats/environments/untracked.py")
                self.assertEqual(tracked.read().decode(), "dirty current bytes\n")
                self.assertEqual(untracked.read().decode(), "new source\n")
            self.assertNotIn("mats/environments/.env", names)
            self.assertNotIn("mats/environments/credentials", names)
            self.assertNotIn("mats/petri/unrelated.py", names)
            self.assertTrue(all(
                name == "mats" or name.startswith("mats/environments/")
                for name in names
            ))
            self.assertFalse(any(".venv" in name for name in names))


class CampaignStorageRoutingTests(unittest.TestCase):
    def test_repository_data_root_uses_canonical_three_bucket_layout(self):
        self.assertEqual(_campaign_logs_root(DATA_ROOT), LOGS_ROOT)
        self.assertEqual(
            _campaign_state_path(DATA_ROOT, "campaign"),
            REMOTE_CAMPAIGNS_ROOT / "campaign.json",
        )

    def test_isolated_test_roots_keep_self_contained_layout(self):
        isolated = Path("/tmp/isolated-mats-data")
        self.assertEqual(_campaign_logs_root(isolated), isolated / "logs")
        self.assertEqual(
            _campaign_state_path(isolated, "campaign"),
            isolated / "remote_campaigns" / "campaign.json",
        )


class WorkerRuntimeSplitTests(unittest.TestCase):
    def test_controller_facade_preserves_worker_validation_error_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_path = Path(tmp) / "job.json"
            job_path.write_text(json.dumps({"schema_version": "wrong"}))

            with self.assertRaisesRegex(AwsTrajectoryError, "schema does not match"):
                worker_main(job_path)

    def test_result_packaging_still_contains_manifest_and_run_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker_root = Path(tmp) / "worker"
            run_dir = worker_root / "run"
            run_dir.mkdir(parents=True)
            (run_dir / "sample.eval").write_text("saved result")
            (worker_root / "worker.log").write_text("progress")

            archive_path, manifest = _package_worker_result({}, worker_root, run_dir)

            self.assertTrue(archive_path.is_file())
            self.assertEqual(
                {item["path"] for item in manifest["files"]},
                {"payload/run/sample.eval", "payload/worker.log"},
            )
            with tarfile.open(archive_path) as archive:
                self.assertIn("manifest.json", archive.getnames())
                self.assertIn("payload/run/sample.eval", archive.getnames())

    def test_worker_disk_preflight_rejects_unexpanded_root_volume(self):
        usage = MagicMock(
            total=16 * 1024**3,
            used=10 * 1024**3,
            free=6 * 1024**3,
        )
        with patch("aws_worker_runtime.shutil.disk_usage", return_value=usage):
            status = _worker_disk_status(
                {"root_volume_gb": ROOT_VOLUME_GB}, Path("/worker")
            )

        self.assertFalse(status["ok"])
        self.assertEqual(status["failure_code"], "root_volume_not_expanded")

    def test_worker_disk_preflight_requires_eight_gib_free(self):
        usage = MagicMock(
            total=32 * 1024**3,
            used=25 * 1024**3,
            free=7 * 1024**3,
        )
        with patch("aws_worker_runtime.shutil.disk_usage", return_value=usage):
            status = _worker_disk_status(
                {"root_volume_gb": ROOT_VOLUME_GB}, Path("/worker")
            )

        self.assertFalse(status["ok"])
        self.assertEqual(status["failure_code"], "insufficient_root_disk")
        self.assertEqual(status["minimum_free_bytes"], MIN_WORKER_FREE_BYTES)

    def test_worker_failure_marker_streams_to_s3_without_needing_disk(self):
        job = {
            "campaign_id": "campaign",
            "cell_id": "cell",
            "bucket": "bucket",
            "failure_key": "failure.json",
            "region": "us-west-2",
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "aws_worker_runtime._aws_cli"
        ) as aws_cli:
            exit_code = _upload_worker_failure(
                job,
                Path(tmp),
                instance_id="i-1",
                reason="disk full",
                failure_code="worker_root_disk_full",
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(aws_cli.call_args.args[:3], ("s3", "cp", "-"))
        marker = json.loads(aws_cli.call_args.kwargs["input_text"])
        self.assertEqual(marker["failure_code"], "worker_root_disk_full")

    def test_worker_accepts_only_a_repo_relative_sandbox_compose_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose = root / "environments/sandbox/p_hacking/compose.yaml"
            compose.parent.mkdir(parents=True)
            compose.write_text("services: {}\n")
            with patch("aws_worker_runtime.WORKTREE_ROOT", root):
                self.assertEqual(
                    _sandbox_compose_path(
                        "environments/sandbox/p_hacking/compose.yaml"
                    ),
                    compose,
                )
                with self.assertRaisesRegex(WorkerRuntimeError, "unsafe"):
                    _sandbox_compose_path("../../etc/passwd")

    def test_bubblewrap_check_starts_the_subscription_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compose = root / "environments/sandbox/p_hacking/compose.subscription.yaml"
            compose.parent.mkdir(parents=True)
            compose.write_text("services: {}\n")
            successful = subprocess.CompletedProcess([], 0, "", "")
            with (
                patch("aws_worker_runtime.WORKTREE_ROOT", root),
                patch("aws_worker_runtime.subprocess.run", return_value=successful) as run,
            ):
                result = _bubblewrap_check(
                    "environments/sandbox/p_hacking/compose.subscription.yaml"
                )

        self.assertEqual(result.returncode, 0)
        command = run.call_args.args[0]
        self.assertIn("bwrap", command)
        self.assertIn("--unshare-user", command)
        self.assertNotIn("--privileged", command)


class CellPlanningTests(unittest.TestCase):
    def make_cfg(self):
        return {
            "targets": ["qwen3-32b", "deepseek-v4-pro"],
            "target_models": [
                "openrouter/qwen/qwen3-32b",
                "openrouter/deepseek/deepseek-v4-pro-20260423",
            ],
            "seeds": ["fraud_detection"],
            "seeds_path": "/repo/seeds/ml_training_data_misuse",
            "family": "ml_training_data_misuse",
            "epochs": 2,
            "reasoning": True,
            "harness": "simple",
            "condition": "allow",
            "judge_resolved": "openai/gpt-5.6-luna",
            "gate_model": "openai/gpt-5.6-luna",
            "time_limit": AGENT_TIME_LIMIT_SECONDS,
            "aws_region": "us-west-2",
            "aws_instance_type": "c7a.xlarge",
            "aws_secret_env": [],
            "pressure": None,
        }

    def test_one_unique_worker_cell_per_target_seed_epoch(self):
        cells = build_cells(
            self.make_cfg(),
            campaign_id="campaign",
            source={"sha256": "a" * 64, "bytes": 100},
            bucket="bucket",
            hourly_price=0.2,
        )

        self.assertEqual(len(cells), 4)
        self.assertEqual(len({cell["cell_id"] for cell in cells}), 4)
        self.assertTrue(all("--epochs=1" in cell["pipeline_args"] for cell in cells))
        self.assertTrue(all("--compute=local" in cell["pipeline_args"] for cell in cells))
        self.assertTrue(all("--harness=simple" in cell["pipeline_args"] for cell in cells))
        self.assertTrue(all(cell["task_name"].endswith(cell["cell_id"]) for cell in cells))
        self.assertTrue(all(
            cell["root_volume_gb"] == ROOT_VOLUME_GB for cell in cells
        ))
        self.assertEqual(
            {cell["sandbox_compose"] for cell in cells},
            {"environments/sandbox/ml/compose.yaml"},
        )

    def test_p_hacking_cells_keep_pressure_and_use_the_p_hacking_sandbox(self):
        cfg = self.make_cfg()
        cfg.update({
            "seeds": ["checkout_redesign"],
            "seeds_path": "/repo/seeds/p_hacking",
            "family": "p_hacking",
            "pressure": "low",
            "time_limit": 1800,
        })

        cell = build_cells(
            cfg, campaign_id="p-hacking", source={"sha256": "a" * 64, "bytes": 100},
            bucket="bucket", hourly_price=0.2,
        )[0]

        self.assertIn("--pressure=low", cell["pipeline_args"])
        self.assertEqual(cell["family"], "p_hacking")
        self.assertEqual(
            cell["sandbox_compose"],
            "environments/sandbox/p_hacking/compose.yaml",
        )

    def test_p_hacking_subscription_cell_uses_its_subscription_image(self):
        cfg = self.make_cfg()
        cfg.update({
            "targets": ["opus-4.6"],
            "target_models": ["anthropic/claude-opus-4-6"],
            "seeds": ["checkout_redesign"],
            "seeds_path": "/repo/seeds/p_hacking",
            "family": "p_hacking",
            "harness": "subscription",
            "pressure": "high",
            "time_limit": 1800,
        })

        cell = build_cells(
            cfg, campaign_id="p-subscription",
            source={"sha256": "a" * 64, "bytes": 100},
            bucket="bucket", hourly_price=0.2,
        )[0]

        self.assertEqual(
            cell["sandbox_compose"],
            "environments/sandbox/p_hacking/compose.subscription.yaml",
        )

    def test_concurrent_campaign_ids_cannot_collide_within_one_second(self):
        cfg = self.make_cfg()

        self.assertNotEqual(_campaign_id(cfg), _campaign_id(cfg))

    def test_explicit_retry_selection_does_not_cross_product(self):
        cfg = self.make_cfg()
        cfg["_cell_selections"] = [
            ("qwen3-32b", "fraud_detection", 2),
        ]

        cells = build_cells(
            cfg, campaign_id="retry", source={"sha256": "b" * 64, "bytes": 100},
            bucket="bucket", hourly_price=0.2,
        )

        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["original_epoch"], 2)

    def test_prefix_generation_cells_use_the_guarded_prefix_worker(self):
        cfg = self.make_cfg()
        cfg.update({
            "targets": ["qwen3-32b"],
            "target_models": ["openrouter/qwen/qwen3-32b"],
            "seeds_path": "/repo/seeds/ml_prefix_only",
            "family": "ml_prefix_only",
            "name": "fraud-control",
        })

        cells = build_prefix_generation_cells(
            cfg,
            campaign_id="prefix-campaign",
            source={"sha256": "a" * 64, "bytes": 100},
            bucket="bucket",
            hourly_price=0.2,
        )

        self.assertEqual(len(cells), 2)
        self.assertTrue(all(
            cell["pipeline_script"] == "prefixes/exp_ml_prefix.py"
            for cell in cells
        ))
        self.assertTrue(all(
            "--name=fraud-control" in cell["pipeline_args"] for cell in cells
        ))

    def test_worker_has_independent_watchdogs_and_auto_shutdown(self):
        cell = build_cells(
            self.make_cfg(), campaign_id="campaign",
            source={"sha256": "a" * 64, "bytes": 100},
            bucket="bucket", hourly_price=0.2,
        )[0]

        user_data = _worker_user_data(cell)

        self.assertIn(f"timeout --signal=TERM --kill-after=60 {FAILURE_PACKAGE_SECONDS}", user_data)
        self.assertIn(
            f"shutdown -h +{UNCONDITIONAL_TERMINATION_SECONDS // 60}", user_data
        )
        self.assertIn("shutdown -h now", user_data)
        self.assertIn("disable --now mats-ami-builder-watchdog.timer", user_data)
        self.assertIn("--failure-reason=source_checksum_failed", user_data)
        self.assertIn("--failure-reason=worker_watchdog_expired", user_data)
        self.assertNotIn("API_KEY", user_data)
        syntax = subprocess.run(
            ["bash", "-n"], input=user_data, text=True, capture_output=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_ami_builder_bootstrap_is_valid_bash(self):
        user_data = _builder_user_data(
            bucket="bucket", region="us-west-2", source_key="source",
            source_sha="a" * 64, runtime="b" * 64,
        )

        syntax = subprocess.run(
            ["bash", "-n"], input=user_data, text=True, capture_output=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        self.assertLessEqual(len(user_data.encode()), 16 * 1024)
        self.assertIn(f"shutdown -h +{AMI_BUILDER_WATCHDOG_MINUTES}", user_data)
        self.assertIn("mats-ami-builder-watchdog.timer", user_data)
        self.assertIn("product_uuid", user_data)
        self.assertIn(
            f"awscli-exe-linux-x86_64-{AWS_CLI_VERSION}.zip", user_data
        )
        self.assertIn("apparmor ca-certificates", user_data)
        self.assertIn("docker-compose-v2 nodejs npm", user_data)
        self.assertIn("node --version", user_data)
        self.assertIn("npm --version", user_data)
        self.assertIn("bwrap-apparmor", RUNTIME_VERSION)
        self.assertIn("profile bwrap /usr/bin/bwrap", user_data)
        self.assertIn("userns,", user_data)
        self.assertIn("Checking nested Bubblewrap", user_data)
        self.assertIn("--unshare-user", user_data)
        self.assertNotIn("--privileged", user_data)
        self.assertNotIn("cap-add", user_data)
        self.assertNotIn("docker-compose-v2 awscli", user_data)
        self.assertIn('"state":"failed"', user_data)
        self.assertIn("builder.log", user_data)
        self.assertIn("find /opt/supermats/mats/environments/sandbox", user_data)
        self.assertNotIn("environments/sandbox/ml/compose.yaml build", user_data)

    def test_no_llm_smoke_checks_host_node_and_npm(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker_root = Path(tmp)
            job_path = worker_root / "job.json"
            job_path.write_text("{}")
            job = {
                "campaign_id": "smoke-campaign",
                "cell_id": "smoke",
                "bucket": "bucket",
                "region": "us-west-2",
                "result_key": "result.tar.gz",
                "complete_key": "complete.json",
                "sandbox_compose_paths": [
                    "environments/sandbox/ml/compose.yaml"
                ],
            }
            successful = subprocess.CompletedProcess([], 0, "", "")
            with (
                patch("aws_worker_runtime._imds", return_value="i-smoke"),
                patch("aws_worker_runtime._aws_cli"),
                patch("aws_worker_runtime._compose_config", return_value=successful),
                patch("aws_worker_runtime.subprocess.run", return_value=successful) as run,
            ):
                exit_code = smoke_worker_main(job, job_path)

            record = json.loads((worker_root / "smoke-run/smoke.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertTrue(record["node_ok"])
        self.assertTrue(record["npm_ok"])
        self.assertTrue(record["bubblewrap_ok"])
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(["node", "--version"], commands)
        self.assertIn(["npm", "--version"], commands)

    def test_runtime_dependencies_include_every_sandbox_family(self):
        repo = ENVIRONMENTS
        dependencies = runtime_dependency_files(repo, ENVIRONMENTS)

        self.assertIn("sandbox/ml/Dockerfile", dependencies)
        self.assertIn("sandbox/p_hacking/Dockerfile", dependencies)
        self.assertIn(
            "sandbox/p_hacking/compose.subscription.yaml",
            dependencies,
        )
        self.assertFalse(any("__pycache__" in path for path in dependencies))

    def test_ami_builder_reboots_for_image_and_has_termination_behavior(self):
        clients = {
            "s3": MagicMock(),
            "ssm": MagicMock(),
            "ec2": MagicMock(),
        }
        clients["ec2"].describe_images.return_value = {"Images": []}
        clients["ssm"].get_parameter.return_value = {
            "Parameter": {"Value": "ami-base"},
        }
        clients["ec2"].run_instances.return_value = {
            "Instances": [{"InstanceId": "i-builder"}],
        }
        clients["ec2"].describe_instances.return_value = {"Reservations": [{
            "Instances": [{"State": {"Name": "running"}}],
        }]}
        clients["ec2"].create_image.return_value = {"ImageId": "ami-runtime"}

        with patch("exp_aws_trajectory._s3_json_or_none", return_value={
            "runtime_hash": "runtime", "ready_at": "now",
        }):
            image = build_runtime_ami(
                clients,
                region="us-west-2",
                instance_type="c7a.xlarge",
                profile="profile",
                security_group="sg-1",
                subnet="subnet-1",
                bucket="bucket",
                bundle={"path": "/tmp/source.tar.gz", "sha256": "a" * 64},
                wanted_hash="runtime",
            )

        self.assertEqual(image, "ami-runtime")
        launch = clients["ec2"].run_instances.call_args.kwargs
        self.assertEqual(launch["InstanceInitiatedShutdownBehavior"], "terminate")
        self.assertIn("mats-ami-builder-watchdog.timer", launch["UserData"])
        self.assertNotIn("NoReboot", clients["ec2"].create_image.call_args.kwargs)
        clients["ec2"].terminate_instances.assert_called_once_with(
            InstanceIds=["i-builder"]
        )

    def test_ami_builder_reports_failed_stage_without_waiting_for_timeout(self):
        clients = {
            "s3": MagicMock(),
            "ssm": MagicMock(),
            "ec2": MagicMock(),
        }
        clients["ec2"].describe_images.return_value = {"Images": []}
        clients["ssm"].get_parameter.return_value = {"Parameter": {"Value": "ami-base"}}
        clients["ec2"].run_instances.return_value = {
            "Instances": [{"InstanceId": "i-builder"}],
        }
        clients["ec2"].describe_instances.return_value = {"Reservations": [{
            "Instances": [{"State": {"Name": "running"}}],
        }]}

        with (
            patch("exp_aws_trajectory._s3_json_or_none", return_value={
                "runtime_hash": "runtime",
                "state": "failed",
                "stage": "building-docker-image",
                "exit_code": 17,
            }),
            self.assertRaisesRegex(
                AwsTrajectoryError,
                r"failed during building-docker-image \(exit 17\)",
            ),
        ):
            build_runtime_ami(
                clients,
                region="us-west-2",
                instance_type="c7a.xlarge",
                profile="profile",
                security_group="sg-1",
                subnet="subnet-1",
                bucket="bucket",
                bundle={"path": "/tmp/source.tar.gz", "sha256": "a" * 64},
                wanted_hash="runtime",
            )

        clients["ec2"].create_image.assert_not_called()
        clients["ec2"].terminate_instances.assert_called_once_with(
            InstanceIds=["i-builder"]
        )

    def test_job_carries_preflight_secret_names_but_never_values(self):
        cfg = self.make_cfg()
        cfg["worker_allowed_secret_names"] = ["OPENROUTER_API_KEY", "CUSTOM_API_KEY"]

        cell = build_cells(
            cfg, campaign_id="campaign",
            source={"sha256": "a" * 64, "bytes": 100},
            bucket="bucket", hourly_price=0.2,
        )[0]

        self.assertEqual(
            cell["allowed_secret_names"],
            ["OPENROUTER_API_KEY", "CUSTOM_API_KEY"],
        )
        self.assertNotIn("secret-value", json.dumps(cell))


class ResultIntegrityTests(unittest.TestCase):
    def test_verified_result_extracts(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = b"eval bytes"
            manifest = {"files": [{
                "path": "payload/run/task.eval",
                "sha256": hashlib.sha256(content).hexdigest(),
            }]}
            archive = root / "result.tar.gz"
            payload_file = root / "task.eval"
            payload_file.write_bytes(content)
            manifest_file = root / "manifest.json"
            manifest_file.write_text(json.dumps(manifest))
            with tarfile.open(archive, "w:gz") as output:
                output.add(manifest_file, arcname="manifest.json")
                output.add(payload_file, arcname="payload/run/task.eval")

            extracted = root / "out"
            extracted.mkdir()
            result = _secure_extract_result(archive, extracted)

            self.assertEqual(result, manifest)
            self.assertEqual((extracted / "payload/run/task.eval").read_bytes(), content)

    def test_result_rejects_an_unlisted_file(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            listed = root / "task.eval"
            listed.write_bytes(b"eval")
            extra = root / "extra.txt"
            extra.write_bytes(b"not checksummed")
            manifest = {"files": [{
                "path": "payload/run/task.eval",
                "sha256": hashlib.sha256(b"eval").hexdigest(),
            }]}
            manifest_file = root / "manifest.json"
            manifest_file.write_text(json.dumps(manifest))
            archive = root / "result.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(manifest_file, arcname="manifest.json")
                output.add(listed, arcname="payload/run/task.eval")
                output.add(extra, arcname="payload/run/extra.txt")
            extracted = root / "out"
            extracted.mkdir()

            with self.assertRaisesRegex(AwsTrajectoryError, "file set mismatch"):
                _secure_extract_result(archive, extracted)

    def test_verified_worker_result_imports_atomically_with_compute_sidecar(self):
        import hashlib
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_file = root / "task.eval"
            eval_file.write_bytes(b"eval bytes")
            integrity_file = root / "pipeline_integrity.json"
            integrity_file.write_text(json.dumps({
                "schema_version": "environment-pipeline-integrity-v1",
                "records": [{
                    "task": "real_audit_task_cell", "sample": "seed", "epoch": 1,
                    "status": "included", "issues": [],
                }],
            }))
            manifest = {"files": [
                {
                    "path": "payload/run/task.eval",
                    "sha256": hashlib.sha256(b"eval bytes").hexdigest(),
                },
                {
                    "path": "payload/run/pipeline_integrity.json",
                    "sha256": hashlib.sha256(integrity_file.read_bytes()).hexdigest(),
                },
            ]}
            manifest_file = root / "manifest.json"
            manifest_file.write_text(json.dumps(manifest))
            archive = root / "remote.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(manifest_file, arcname="manifest.json")
                output.add(eval_file, arcname="payload/run/task.eval")
                output.add(
                    integrity_file, arcname="payload/run/pipeline_integrity.json"
                )
            archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            cell = {
                "status": "completed", "cell_id": "cell", "campaign_id": "campaign",
                "bucket": "bucket", "result_key": "result", "region": "us-west-2",
                "instance_type": "c7a.xlarge", "instance_id": "i-1",
                "hourly_price_usd": 0.2, "source_sha256": "source",
                "source_bytes": 100, "original_epoch": 1,
                "launched_at": "2026-08-01T00:00:00+00:00",
                "terminal": {
                    "task_name": "real_audit_task_cell",
                    "completed_at": "2026-08-01T02:00:00+00:00",
                    "result_sha256": archive_sha, "result_bytes": archive.stat().st_size,
                },
            }
            state = {
                "campaign_id": "campaign", "bucket": "bucket", "cells": [cell],
                "source": {"sha256": archive_sha, "bytes": archive.stat().st_size},
            }
            clients = {"s3": MagicMock()}
            clients["s3"].download_file.side_effect = (
                lambda _bucket, _key, destination: shutil.copy2(archive, destination)
            )

            destination = import_campaign_results(state, root / "data", clients)

            self.assertEqual((destination / "task.eval").read_bytes(), b"eval bytes")
            self.assertEqual(
                (destination / "remote_source.tar.gz").read_bytes(),
                archive.read_bytes(),
            )
            sidecar = json.loads((destination / "remote_campaign.json").read_text())
            integrity = json.loads(
                (destination / "pipeline_integrity.json").read_text()
            )
            compute = sidecar["task_compute"]["real_audit_task_cell"]
            self.assertAlmostEqual(compute["estimated_vm_cost_usd"], 0.4)
            self.assertTrue(compute["s3_cost_excluded"])
            self.assertTrue(compute["ebs_cost_excluded"])
            self.assertTrue(compute["public_ipv4_cost_excluded"])
            self.assertTrue(compute["internet_data_transfer_cost_excluded"])
            self.assertTrue(compute["shared_runtime_cost_excluded"])
            self.assertEqual(compute["root_volume_gb"], 16)
            self.assertEqual(
                integrity["schema_version"], "environment-pipeline-integrity-v2"
            )
            self.assertEqual(integrity["records"][0]["status"], "included")
            self.assertFalse(any(path.name.startswith(".campaign.")
                                 for path in destination.parent.iterdir()))

    def test_campaign_s3_cleanup_is_exact_and_batches_large_campaigns(self):
        objects = [
            {"Key": f"campaigns/campaign/cell-{index}", "Size": index}
            for index in range(1001)
        ]
        s3 = MagicMock()
        s3.get_paginator.return_value.paginate.return_value = [
            {"Contents": objects[:700]},
            {"Contents": objects[700:]},
        ]
        s3.delete_objects.side_effect = [{}, {}]
        s3.list_objects_v2.return_value = {"KeyCount": 0}

        result = delete_campaign_s3_objects(
            s3, bucket="bucket", campaign_id="campaign"
        )

        self.assertEqual(result["prefix"], "campaigns/campaign/")
        self.assertEqual(result["objects_deleted"], 1001)
        self.assertEqual(result["bytes_deleted"], sum(range(1001)))
        self.assertEqual(s3.delete_objects.call_count, 2)
        first_delete = s3.delete_objects.call_args_list[0].kwargs["Delete"]
        second_delete = s3.delete_objects.call_args_list[1].kwargs["Delete"]
        self.assertEqual(len(first_delete["Objects"]), 1000)
        self.assertEqual(len(second_delete["Objects"]), 1)
        self.assertTrue(all(
            item["Key"].startswith("campaigns/campaign/")
            for call in s3.delete_objects.call_args_list
            for item in call.kwargs["Delete"]["Objects"]
        ))
        s3.list_objects_v2.assert_called_once_with(
            Bucket="bucket", Prefix="campaigns/campaign/", MaxKeys=1
        )

    def test_campaign_s3_cleanup_rejects_an_unsafe_prefix(self):
        s3 = MagicMock()

        with self.assertRaisesRegex(AwsTrajectoryError, "unsafe S3 campaign"):
            delete_campaign_s3_objects(
                s3, bucket="bucket", campaign_id="../another-campaign"
            )

        s3.get_paginator.assert_not_called()

    def test_final_import_is_recorded_locally_then_deleted_from_s3(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            imported = data_root / "logs" / "campaign"
            imported.mkdir(parents=True)
            source = imported / "remote_source.tar.gz"
            source.write_bytes(b"exact source")
            (imported / "remote_campaign.json").write_text(json.dumps({
                "campaign_id": "campaign",
                "task_compute": {},
            }))
            state = {
                "campaign_id": "campaign",
                "bucket": "bucket",
                "cells": [],
                "source": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
            }
            clients = {"s3": MagicMock()}
            deletion = {
                "prefix": "campaigns/campaign/",
                "objects_deleted": 4,
                "bytes_deleted": 1234,
            }

            with patch(
                "exp_aws_trajectory.delete_campaign_s3_objects",
                return_value=deletion,
            ) as delete:
                result = _finalize_campaign_import(
                    state, data_root, clients, imported
                )

            delete.assert_called_once_with(
                clients["s3"], bucket="bucket", campaign_id="campaign"
            )
            self.assertEqual(result["s3_cleanup"]["status"], "deleted")
            self.assertEqual(result["s3_cleanup"]["objects_deleted"], 4)
            clients["s3"].put_object.assert_called_once()
            saved = json.loads(
                (data_root / "remote_campaigns" / "campaign.json").read_text()
            )
            sidecar = json.loads((imported / "remote_campaign.json").read_text())
            self.assertEqual(saved["s3_cleanup"], result["s3_cleanup"])
            self.assertEqual(sidecar["s3_cleanup"], result["s3_cleanup"])

    def test_failed_cleanup_keeps_the_valid_import_and_lifecycle_fallback(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            imported = data_root / "logs" / "campaign"
            imported.mkdir(parents=True)
            source = imported / "remote_source.tar.gz"
            source.write_bytes(b"exact source")
            (imported / "remote_campaign.json").write_text(json.dumps({
                "campaign_id": "campaign",
            }))
            state = {
                "campaign_id": "campaign",
                "bucket": "bucket",
                "cells": [],
                "source": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
            }
            clients = {"s3": MagicMock()}

            with patch(
                "exp_aws_trajectory.delete_campaign_s3_objects",
                side_effect=AwsTrajectoryError("access denied"),
            ):
                result = _finalize_campaign_import(
                    state, data_root, clients, imported
                )

            self.assertEqual(result["s3_cleanup"]["status"], "failed")
            self.assertEqual(result["s3_cleanup"]["fallback_lifecycle_days"], 7)
            self.assertIn("access denied", result["s3_cleanup"]["error"])
            self.assertTrue(source.is_file())
            sidecar = json.loads((imported / "remote_campaign.json").read_text())
            self.assertEqual(sidecar["s3_cleanup"]["status"], "failed")

    def test_changed_local_source_prevents_s3_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            imported = data_root / "logs" / "campaign"
            imported.mkdir(parents=True)
            (imported / "remote_source.tar.gz").write_bytes(b"changed source")
            (imported / "remote_campaign.json").write_text(json.dumps({
                "campaign_id": "campaign",
            }))
            state = {
                "campaign_id": "campaign",
                "bucket": "bucket",
                "cells": [],
                "source": {"sha256": "0" * 64},
            }
            clients = {"s3": MagicMock()}

            with patch(
                "exp_aws_trajectory.delete_campaign_s3_objects"
            ) as delete:
                result = _finalize_campaign_import(
                    state, data_root, clients, imported
                )

            delete.assert_not_called()
            self.assertEqual(result["s3_cleanup"]["status"], "failed")
            self.assertIn("checksum changed", result["s3_cleanup"]["error"])


class SafetyContractTests(unittest.TestCase):
    def test_launch_template_root_volume_parser_is_exact(self):
        self.assertEqual(
            _launch_template_root_volume_gb({
                "BlockDeviceMappings": [{
                    "DeviceName": "/dev/sda1",
                    "Ebs": {"VolumeSize": ROOT_VOLUME_GB},
                }],
            }),
            ROOT_VOLUME_GB,
        )
        self.assertIsNone(_launch_template_root_volume_gb({}))

    def test_aws_clients_loads_profile_from_project_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("AWS_PROFILE=mats-environments\n")
            with (
                patch.dict(os.environ, {}, clear=False),
                patch("exp_aws_trajectory.ENV_FILE", env_file),
                patch("boto3.Session") as session,
                patch("exp_aws_trajectory.shutil.which", return_value=None),
            ):
                os.environ.pop("AWS_PROFILE", None)
                aws_clients("us-west-2")

        session.assert_called_once_with(
            profile_name="mats-environments", region_name="us-west-2"
        )

    def test_shell_aws_profile_overrides_project_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("AWS_PROFILE=from-project-env\n")
            with (
                patch.dict(os.environ, {"AWS_PROFILE": "from-shell"}),
                patch("exp_aws_trajectory.ENV_FILE", env_file),
                patch("boto3.Session") as session,
                patch("exp_aws_trajectory.shutil.which", return_value=None),
            ):
                aws_clients("us-west-2")

        session.assert_called_once_with(
            profile_name="from-shell", region_name="us-west-2"
        )

    def test_login_profile_falls_back_to_refreshable_aws_cli_bridge(self):
        direct_session = MagicMock()
        direct_session.get_credentials.return_value = None
        bridged_session = MagicMock()
        bridged_session.get_credentials.return_value = MagicMock()
        botocore_session = MagicMock()
        provider = MagicMock()
        with (
            patch.dict(os.environ, {"AWS_PROFILE": "mats-environments"}),
            patch("dotenv.load_dotenv"),
            patch("boto3.Session", side_effect=[direct_session, bridged_session]) as session,
            patch("botocore.session.Session", return_value=botocore_session),
            patch("botocore.credentials.ProcessProvider", return_value=provider) as process,
            patch("exp_aws_trajectory.shutil.which", return_value="/usr/local/bin/aws"),
        ):
            clients = aws_clients("us-west-2")

        session.assert_any_call(
            profile_name="mats-environments", region_name="us-west-2"
        )
        session.assert_any_call(
            botocore_session=botocore_session, region_name="us-west-2"
        )
        config = process.call_args.kwargs["load_config"]()
        command = config["profiles"]["mats-aws-cli-export"]["credential_process"]
        self.assertIn("tools/aws_credential_process.py", command)
        self.assertIn("--profile mats-environments", command)
        self.assertIn("--aws-path /usr/local/bin/aws", command)
        self.assertIn("--cache-path", command)
        self.assertIn("--lock-path", command)
        botocore_session.get_component.return_value.insert_before.assert_called_once_with(
            "env", provider
        )
        self.assertIs(clients["sts"], bridged_session.client.return_value)

    def test_readable_login_profile_still_uses_refreshable_aws_cli_bridge(self):
        direct_session = MagicMock()
        direct_session.get_credentials.return_value = MagicMock()
        bridged_session = MagicMock()
        bridged_session.get_credentials.return_value = MagicMock()
        with (
            patch.dict(os.environ, {"AWS_PROFILE": "mats-run"}),
            patch("dotenv.load_dotenv"),
            patch("boto3.Session", side_effect=[direct_session, bridged_session]),
            patch("botocore.session.Session"),
            patch("botocore.credentials.ProcessProvider"),
            patch("exp_aws_trajectory.shutil.which", return_value="/usr/local/bin/aws"),
        ):
            clients = aws_clients("us-west-2")

        self.assertIs(clients["sts"], bridged_session.client.return_value)

    def test_missing_profile_and_credentials_names_the_exact_env_file(self):
        direct_session = MagicMock()
        direct_session.get_credentials.return_value = None
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("dotenv.load_dotenv"),
            patch("boto3.Session", return_value=direct_session),
        ):
            os.environ.pop("AWS_PROFILE", None)
            with self.assertRaisesRegex(
                AwsTrajectoryError, "AWS_PROFILE is unset after loading"
            ) as raised:
                aws_clients("us-west-2")

        self.assertIn(str(ENVIRONMENTS / ".env"), str(raised.exception))

    def test_cli_bridge_failure_does_not_expose_credential_output(self):
        direct_session = MagicMock()
        direct_session.get_credentials.return_value = None
        bridged_session = MagicMock()
        bridged_session.get_credentials.side_effect = RuntimeError(
            "sensitive provider output"
        )
        with (
            patch.dict(os.environ, {"AWS_PROFILE": "mats-environments"}),
            patch("dotenv.load_dotenv"),
            patch("boto3.Session", side_effect=[direct_session, bridged_session]),
            patch("botocore.session.Session"),
            patch("botocore.credentials.ProcessProvider"),
            patch("exp_aws_trajectory.shutil.which", return_value="/usr/local/bin/aws"),
        ):
            with self.assertRaisesRegex(
                AwsTrajectoryError, "aws login --profile mats-environments"
            ) as raised:
                aws_clients("us-west-2")

        self.assertNotIn("sensitive provider output", str(raised.exception))

    def test_shared_cli_bridge_reuses_protected_cached_credentials(self):
        exported = {
            "Version": 1,
            "AccessKeyId": "temporary-access",
            "SecretAccessKey": "temporary-secret",
            "SessionToken": "temporary-session",
            "Expiration": (
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat(),
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(exported), stderr=""
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "auth"
            cache = root / "credentials.json"
            lock = root / "credentials.lock"
            with patch(
                "aws_credential_process.subprocess.run", return_value=completed
            ) as run:
                first = shared_aws_credentials(
                    profile="mats-run",
                    aws_path="/usr/local/bin/aws",
                    cache_path=cache,
                    lock_path=lock,
                )
                second = shared_aws_credentials(
                    profile="mats-run",
                    aws_path="/usr/local/bin/aws",
                    cache_path=cache,
                    lock_path=lock,
                )
            self.assertEqual(first, exported)
            self.assertEqual(second, exported)
            run.assert_called_once()
            self.assertEqual(cache.stat().st_mode & 0o777, 0o600)
            self.assertEqual(lock.stat().st_mode & 0o777, 0o600)

    def test_shared_cli_bridge_accepts_valid_short_lived_cli_export(self):
        exported = {
            "Version": 1,
            "AccessKeyId": "temporary-access",
            "SecretAccessKey": "temporary-secret",
            "SessionToken": "temporary-session",
            "Expiration": (
                datetime.now(timezone.utc) + timedelta(minutes=1)
            ).isoformat(),
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(exported), stderr=""
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "auth"
            with patch(
                "aws_credential_process.subprocess.run", return_value=completed
            ) as run:
                value = shared_aws_credentials(
                    profile="mats-run",
                    aws_path="/usr/local/bin/aws",
                    cache_path=root / "credentials.json",
                    lock_path=root / "credentials.lock",
                )
        self.assertEqual(value, exported)
        run.assert_called_once()

    def test_shared_cli_bridge_singleflights_two_processes(self):
        helper = ENVIRONMENTS / "tools" / "aws_credential_process.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counter = root / "calls.txt"
            fake_aws = root / "fake-aws"
            fake_aws.write_text(
                f"#!{sys.executable}\n"
                "from datetime import datetime, timedelta, timezone\n"
                "import fcntl, json, time\n"
                f"path = {str(counter)!r}\n"
                "with open(path, 'a+') as stream:\n"
                "    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)\n"
                "    stream.write('call\\n'); stream.flush()\n"
                "time.sleep(0.2)\n"
                "print(json.dumps({\n"
                "  'Version': 1, 'AccessKeyId': 'temporary-access',\n"
                "  'SecretAccessKey': 'temporary-secret',\n"
                "  'SessionToken': 'temporary-session',\n"
                "  'Expiration': (datetime.now(timezone.utc) + "
                "timedelta(hours=1)).isoformat(),\n"
                "}))\n"
            )
            fake_aws.chmod(0o700)
            cache = root / "auth" / "credentials.json"
            lock = root / "auth" / "credentials.lock"
            command = [
                sys.executable,
                str(helper),
                "--profile",
                "mats-run",
                "--aws-path",
                str(fake_aws),
                "--cache-path",
                str(cache),
                "--lock-path",
                str(lock),
            ]
            first = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
            second = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
            first_output, _ = first.communicate(timeout=10)
            second_output, _ = second.communicate(timeout=10)

            self.assertEqual(first.returncode, 0)
            self.assertEqual(second.returncode, 0)
            self.assertEqual(json.loads(first_output)["AccessKeyId"], "temporary-access")
            self.assertEqual(json.loads(second_output)["AccessKeyId"], "temporary-access")
            self.assertEqual(counter.read_text().splitlines(), ["call"])

    def test_setup_requires_exactly_one_funding_confirmation(self):
        with self.assertRaisesRegex(AwsTrajectoryError, "exactly one"):
            setup_aws({
                "confirm_approved_account": False,
                "confirm_personal_account": False,
            }, ENVIRONMENTS)
        with self.assertRaisesRegex(AwsTrajectoryError, "exactly one"):
            setup_aws({
                "confirm_approved_account": True,
                "confirm_personal_account": True,
            }, ENVIRONMENTS)

    def test_root_credentials_are_refused_before_setup_or_preflight(self):
        clients = {"sts": MagicMock()}
        clients["sts"].get_caller_identity.return_value = {
            "Account": "123",
            "Arn": "arn:aws:iam::123:root",
        }

        with self.assertRaisesRegex(AwsTrajectoryError, "root-user credentials"):
            account_id(clients)

    def test_personal_reimbursement_funding_is_truthfully_tagged(self):
        clients = {"s3": MagicMock()}

        ensure_bucket(
            clients,
            bucket="bucket",
            region="us-west-2",
            funding=PERSONAL_REIMBURSEMENT_FUNDING,
        )

        tag_set = clients["s3"].put_bucket_tagging.call_args.kwargs["Tagging"]["TagSet"]
        self.assertIn(
            {"Key": "Funding", "Value": PERSONAL_REIMBURSEMENT_FUNDING},
            tag_set,
        )

    def test_worker_policy_is_narrow_and_has_no_ec2_launch_permission(self):
        policy = json.dumps(_iam_policy("123", "bucket"))
        self.assertIn("campaigns/*", policy)
        self.assertIn("ssm:GetParameter", policy)
        self.assertNotIn("ec2:", policy)
        self.assertNotIn("s3:Delete", policy)

    def test_setup_secret_allow_list_rejects_invalid_environment_names(self):
        with self.assertRaisesRegex(AwsTrajectoryError, "invalid environment names"):
            put_api_keys({"ssm": MagicMock()}, extra_names=["bad-name"])

    def test_setup_rejects_an_explicit_secret_name_without_a_local_value(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            self.assertRaisesRegex(AwsTrajectoryError, "not set locally"),
        ):
            put_api_keys({"ssm": MagicMock()}, extra_names=["CUSTOM_API_KEY"])

    def test_new_secret_parameter_uses_tags_without_overwrite(self):
        missing = RuntimeError("missing")
        missing.response = {"Error": {"Code": "ParameterNotFound"}}
        ssm = MagicMock()
        ssm.get_parameter.side_effect = missing

        with patch.dict("os.environ", {"OPENAI_API_KEY": "secret"}, clear=True):
            names = put_api_keys({"ssm": ssm}, extra_names=[])

        self.assertEqual(names, ["OPENAI_API_KEY"])
        request = ssm.put_parameter.call_args.kwargs
        self.assertEqual(request["Tags"], [{"Key": "Project", "Value": "mats-environments"}])
        self.assertNotIn("Overwrite", request)

    def test_subscription_setup_ships_host_claude_login_when_no_token(self):
        missing = RuntimeError("missing")
        missing.response = {"Error": {"Code": "ParameterNotFound"}}
        ssm = MagicMock()
        ssm.get_parameter.side_effect = missing

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "secret"}, clear=True),
            patch("exp_aws_trajectory.claude_credentials_b64", return_value="e30="),
        ):
            names = put_api_keys(
                {"ssm": ssm}, extra_names=[], ship_claude_subscription_login=True
            )

        self.assertEqual(
            names, ["CLAUDE_SUBSCRIPTION_CREDENTIALS_JSON_B64", "OPENAI_API_KEY"]
        )
        stored = json.loads(ssm.put_parameter.call_args.kwargs["Value"])
        self.assertEqual(
            stored["CLAUDE_SUBSCRIPTION_CREDENTIALS_JSON_B64"], "e30="
        )

    def test_non_subscription_setup_never_ships_host_claude_login(self):
        missing = RuntimeError("missing")
        missing.response = {"Error": {"Code": "ParameterNotFound"}}
        ssm = MagicMock()
        ssm.get_parameter.side_effect = missing

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "secret"}, clear=True),
            patch("exp_aws_trajectory.claude_credentials_b64", return_value="e30="),
        ):
            names = put_api_keys({"ssm": ssm}, extra_names=[])

        self.assertEqual(names, ["OPENAI_API_KEY"])

    def test_existing_secret_parameter_uses_overwrite_without_tags(self):
        ssm = MagicMock()
        ssm.get_parameter.return_value = {"Parameter": {"Name": "existing"}}
        ssm.describe_parameters.return_value = {
            "Parameters": [{"Name": SSM_PARAMETER_NAME, "Tier": "Standard"}]
        }

        with patch.dict("os.environ", {"OPENAI_API_KEY": "secret"}, clear=True):
            put_api_keys({"ssm": ssm}, extra_names=[])

        request = ssm.put_parameter.call_args.kwargs
        self.assertTrue(request["Overwrite"])
        self.assertNotIn("Tags", request)

    def test_existing_advanced_secret_parameter_is_never_downgraded(self):
        ssm = MagicMock()
        # This mirrors AWS: GetParameter confirms existence but does not include Tier.
        ssm.get_parameter.return_value = {
            "Parameter": {"Name": SSM_PARAMETER_NAME}
        }
        ssm.describe_parameters.return_value = {
            "Parameters": [{"Name": SSM_PARAMETER_NAME, "Tier": "Advanced"}]
        }

        with patch.dict("os.environ", {"OPENAI_API_KEY": "small"}, clear=True):
            put_api_keys({"ssm": ssm}, extra_names=[])

        request = ssm.put_parameter.call_args.kwargs
        self.assertEqual(request["Tier"], "Advanced")
        self.assertTrue(request["Overwrite"])

    def test_large_secret_bundle_uses_advanced_ssm_tier_with_warning(self):
        ssm = MagicMock()
        missing = RuntimeError("missing")
        missing.response = {"Error": {"Code": "ParameterNotFound"}}
        ssm.get_parameter.side_effect = missing

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "x" * 4100}, clear=True),
            patch("builtins.print") as print_mock,
        ):
            put_api_keys({"ssm": ssm}, extra_names=[])

        self.assertEqual(ssm.put_parameter.call_args.kwargs["Tier"], "Advanced")
        self.assertIn("billable", print_mock.call_args.args[0])

    def test_secret_bundle_over_advanced_limit_is_rejected(self):
        ssm = MagicMock()
        missing = RuntimeError("missing")
        missing.response = {"Error": {"Code": "ParameterNotFound"}}
        ssm.get_parameter.side_effect = missing

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "x" * 8200}, clear=True),
            self.assertRaisesRegex(AwsTrajectoryError, "8 KiB"),
        ):
            put_api_keys({"ssm": ssm}, extra_names=[])

    def test_campaign_success_requires_every_cell_and_no_retry_magic(self):
        success = {"cells": [{
            "status": "completed", "terminal": {"pipeline_exit_code": 0},
        }]}
        infrastructure_failure = {"cells": [{
            "status": "infrastructure_failure", "terminal": {},
        }]}
        integrity_failure = {"cells": [{
            "status": "completed", "terminal": {"pipeline_exit_code": 1},
        }]}

        self.assertTrue(campaign_ok(success))
        self.assertFalse(campaign_ok(infrastructure_failure))
        self.assertFalse(campaign_ok(integrity_failure))

    def test_ec2_launch_throttling_retries_with_bounded_backoff(self):
        throttled = ClientError(
            {"Error": {"Code": "RequestLimitExceeded", "Message": "slow down"}},
            "RunInstances",
        )
        clients = {"s3": MagicMock(), "ec2": MagicMock()}
        clients["ec2"].run_instances.side_effect = [
            throttled,
            throttled,
            {"Instances": [{"InstanceId": "i-recovered"}]},
        ]
        cell = {
            "bucket": "bucket",
            "job_key": "job.json",
            "campaign_id": "campaign",
            "cell_id": "cell",
            "region": "us-west-2",
            "source_key": "source.tar.gz",
            "source_sha256": "a" * 64,
        }

        with (
            patch("exp_aws_trajectory.random.uniform", side_effect=[0.5, 1.5]),
            patch("exp_aws_trajectory.time.sleep") as sleep,
        ):
            instance_id = _launch_cell(clients, template_id="lt-1", cell=cell)

        self.assertEqual(instance_id, "i-recovered")
        self.assertEqual(clients["ec2"].run_instances.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.5, 1.5])

    def test_ec2_launch_does_not_retry_non_throttling_errors(self):
        denied = ClientError(
            {"Error": {"Code": "UnauthorizedOperation", "Message": "denied"}},
            "RunInstances",
        )
        clients = {"s3": MagicMock(), "ec2": MagicMock()}
        clients["ec2"].run_instances.side_effect = denied
        cell = {
            "bucket": "bucket",
            "job_key": "job.json",
            "campaign_id": "campaign",
            "cell_id": "cell",
            "region": "us-west-2",
            "source_key": "source.tar.gz",
            "source_sha256": "a" * 64,
        }

        with patch("exp_aws_trajectory.time.sleep") as sleep:
            with self.assertRaises(ClientError):
                _launch_cell(clients, template_id="lt-1", cell=cell)

        clients["ec2"].run_instances.assert_called_once()
        sleep.assert_not_called()

    def test_campaign_dry_run_never_uploads_or_launches(self):
        cfg = CellPlanningTests().make_cfg()
        cfg["vm_concurrency"] = 16
        clients = {"s3": MagicMock(), "ec2": MagicMock()}
        preflight = {
            "clients": clients, "bucket": "bucket", "ami": "ami-1",
            "funding": PERSONAL_REIMBURSEMENT_FUNDING,
            "runtime_hash": "runtime", "launch_template_id": "lt-1",
            "root_volume_gb": ROOT_VOLUME_GB,
            "hourly_price_usd": 0.2, "quota_vcpus": 64,
            "required_vcpus": 16,
            "stored_secret_names": ["OPENROUTER_API_KEY", "OPENAI_API_KEY"],
        }
        source = {
            "path": "/tmp/source.tar.gz", "sha256": "a" * 64,
            "bytes": 100, "files": 10,
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("exp_aws_trajectory.build_source_bundle", return_value=source),
            patch(
                "exp_aws_trajectory.preflight_aws", return_value=preflight
            ) as preflight_mock,
        ):
            result = run_campaign(cfg, ENVIRONMENTS, Path(tmp), dry_run=True)

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["vm_concurrency"], 4)
        self.assertEqual(result["required_vcpus"], 16)
        self.assertEqual(preflight_mock.call_args.args[0]["vm_concurrency"], 4)
        clients["s3"].upload_file.assert_not_called()
        clients["ec2"].run_instances.assert_not_called()

    def test_prefix_campaign_dry_run_omits_judges_and_plans_the_matrix(self):
        cfg = CellPlanningTests().make_cfg()
        cfg.update({
            "targets": ["qwen3-32b"],
            "target_models": ["openrouter/qwen/qwen3-32b"],
            "seeds_path": "/repo/seeds/ml_prefix_only",
            "family": "ml_prefix_only",
            "prefix_only": True,
            "name": "fraud-control",
            "judge_resolved": None,
            "gate_model": None,
            "vm_concurrency": 16,
        })
        clients = {"s3": MagicMock(), "ec2": MagicMock()}
        preflight = {
            "clients": clients,
            "bucket": "bucket",
            "ami": "ami-1",
            "funding": PERSONAL_REIMBURSEMENT_FUNDING,
            "runtime_hash": "runtime",
            "launch_template_id": "lt-1",
            "root_volume_gb": ROOT_VOLUME_GB,
            "hourly_price_usd": 0.2,
            "quota_vcpus": 8,
            "required_vcpus": 8,
            "stored_secret_names": ["OPENROUTER_API_KEY"],
        }
        source = {
            "path": "/tmp/source.tar.gz",
            "sha256": "a" * 64,
            "bytes": 100,
            "files": 10,
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("exp_aws_trajectory.build_source_bundle", return_value=source),
            patch(
                "exp_aws_trajectory.preflight_aws", return_value=preflight
            ) as preflight_mock,
        ):
            result = run_campaign(cfg, ENVIRONMENTS, Path(tmp), dry_run=True)

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["cells"], 2)
        self.assertEqual(result["vm_concurrency"], 2)
        self.assertEqual(
            preflight_mock.call_args.kwargs["model_slugs"],
            ["openrouter/qwen/qwen3-32b"],
        )
        self.assertTrue(all(
            cell["pipeline_script"] == "prefixes/exp_ml_prefix.py"
            for cell in result["cell_specs"]
        ))
        clients["s3"].upload_file.assert_not_called()
        clients["ec2"].run_instances.assert_not_called()

    def test_prefix_campaign_state_retains_exact_resume_and_retry_contract(self):
        cfg = CellPlanningTests().make_cfg()
        cfg.update({
            "targets": ["qwen3-32b"],
            "target_models": ["openrouter/qwen/qwen3-32b"],
            "seeds_path": "/repo/seeds/ml_prefix_only",
            "seed_dir": "ml_prefix_only",
            "family": "ml_prefix_only",
            "prefix_only": True,
            "name": "fraud-control",
            "judge_resolved": None,
            "gate_model": None,
            "vm_concurrency": 2,
        })
        clients = {"s3": MagicMock(), "ec2": MagicMock()}
        preflight = {
            "clients": clients,
            "bucket": "bucket",
            "ami": "ami-1",
            "funding": PERSONAL_REIMBURSEMENT_FUNDING,
            "runtime_hash": "runtime",
            "launch_template_id": "lt-1",
            "root_volume_gb": ROOT_VOLUME_GB,
            "hourly_price_usd": 0.2,
            "quota_vcpus": 8,
            "required_vcpus": 8,
            "stored_secret_names": ["OPENROUTER_API_KEY"],
        }
        source = {
            "path": "/tmp/source.tar.gz",
            "sha256": "a" * 64,
            "bytes": 100,
            "files": 10,
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("exp_aws_trajectory.build_source_bundle", return_value=source),
            patch("exp_aws_trajectory.preflight_aws", return_value=preflight),
            patch("exp_aws_trajectory._save_campaign") as save,
            patch(
                "exp_aws_trajectory._monitor_campaign",
                side_effect=lambda state, *_args, **_kwargs: state,
            ),
            patch(
                "exp_aws_trajectory.import_campaign_results",
                return_value=Path(tmp) / "imported",
            ),
            patch(
                "exp_aws_trajectory._finalize_campaign_import",
                side_effect=lambda state, *_args, **_kwargs: state,
            ),
        ):
            result = run_campaign(cfg, ENVIRONMENTS, Path(tmp))

        stored = save.call_args.args[0]
        self.assertIs(result, stored)
        self.assertEqual(stored["pipeline_config"]["prefix_only"], True)
        self.assertEqual(stored["pipeline_config"]["name"], "fraud-control")
        self.assertEqual(stored["pipeline_config"]["seed_dir"], "ml_prefix_only")
        self.assertTrue(all(
            cell["pipeline_script"] == "prefixes/exp_ml_prefix.py"
            for cell in stored["cells"]
        ))

    def test_subscription_preflight_replaces_direct_agent_api_keys_with_auth(self):
        cfg = CellPlanningTests().make_cfg()
        cfg.update({
            "targets": [
                "opus-4.6", "gpt-5.6-sol", "qwen3-32b", "qwen3.7-max"
            ],
            "target_models": [
                "anthropic/claude-opus-4-6",
                "openai/gpt-5.6-sol",
                "openrouter/qwen/qwen3-32b",
                "openrouter/qwen/qwen3.7-max",
            ],
            "epochs": 1,
            "harness": "subscription",
            "vm_concurrency": 3,
        })
        clients = {"s3": MagicMock(), "ec2": MagicMock()}
        preflight = {
            "clients": clients,
            "bucket": "bucket",
            "ami": "ami-1",
            "funding": PERSONAL_REIMBURSEMENT_FUNDING,
            "runtime_hash": "runtime",
            "launch_template_id": "lt-1",
            "root_volume_gb": ROOT_VOLUME_GB,
            "hourly_price_usd": 0.2,
            "quota_vcpus": 12,
            "required_vcpus": 12,
            "stored_secret_names": [
                "OPENROUTER_API_KEY",
                "OPENAI_API_KEY",
                "CLAUDE_CODE_OAUTH_TOKEN",
                "CODEX_ACCESS_TOKEN",
            ],
        }
        source = {
            "path": "/tmp/source.tar.gz",
            "sha256": "a" * 64,
            "bytes": 100,
            "files": 10,
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("exp_aws_trajectory.build_source_bundle", return_value=source),
            patch(
                "exp_aws_trajectory.preflight_aws", return_value=preflight
            ) as preflight_mock,
        ):
            run_campaign(cfg, ENVIRONMENTS, Path(tmp), dry_run=True)

        kwargs = preflight_mock.call_args.kwargs
        self.assertEqual(
            kwargs["model_slugs"],
            [
                "openrouter/qwen/qwen3-32b",
                "openai/gpt-5.6-luna",
                "openai/gpt-5.6-luna",
            ],
        )
        self.assertEqual(kwargs["required_secrets"], {"OPENCODE_API_KEY"})
        self.assertEqual(
            kwargs["required_secret_alternatives"],
            [
                (
                    "CLAUDE_CODE_OAUTH_TOKEN",
                    "CLAUDE_SUBSCRIPTION_CREDENTIALS_JSON_B64",
                ),
                (
                    "CODEX_ACCESS_TOKEN",
                    "CODEX_SUBSCRIPTION_AUTH_JSON_GZIP_B64",
                    "CODEX_SUBSCRIPTION_AUTH_JSON_B64",
                ),
            ],
        )

    def test_preflight_rejects_insufficient_vcpu_quota_before_launch(self):
        clients = {name: MagicMock() for name in (
            "sts", "s3", "ssm", "ec2", "quotas", "pricing",
        )}
        clients["sts"].get_caller_identity.return_value = {"Account": "123"}
        clients["s3"].get_bucket_tagging.return_value = {"TagSet": [
            {"Key": "Funding", "Value": PERSONAL_REIMBURSEMENT_FUNDING},
        ]}
        clients["ssm"].get_parameter.return_value = {"Parameter": {
            "Value": json.dumps({"OPENROUTER_API_KEY": "stored"}),
        }}
        clients["ec2"].describe_instance_types.return_value = {"InstanceTypes": [{
            "VCpuInfo": {"DefaultVCpus": 4}, "MemoryInfo": {"SizeInMiB": 8192},
        }]}
        clients["ec2"].describe_launch_templates.return_value = {"LaunchTemplates": [{
            "LaunchTemplateId": "lt-1", "DefaultVersionNumber": 3,
        }]}
        clients["ec2"].describe_launch_template_versions.return_value = {
            "LaunchTemplateVersions": [{"LaunchTemplateData": {
                "ImageId": "ami-1", "InstanceType": "c7a.xlarge",
                "BlockDeviceMappings": [{
                    "DeviceName": "/dev/sda1",
                    "Ebs": {"VolumeSize": ROOT_VOLUME_GB},
                }],
            }}],
        }
        clients["quotas"].get_service_quota.return_value = {"Quota": {"Value": 32}}
        cfg = {
            "aws_region": "us-west-2", "aws_instance_type": "c7a.xlarge",
            "aws_bucket": None, "vm_concurrency": 16,
        }

        with (
            patch("exp_aws_trajectory.aws_clients", return_value=clients),
            patch("exp_aws_trajectory.runtime_hash", return_value="runtime"),
            patch("exp_aws_trajectory._find_runtime_ami", return_value="ami-1"),
        ):
            with self.assertRaisesRegex(AwsTrajectoryError, "require 64"):
                preflight_aws(
                    cfg, ENVIRONMENTS,
                    model_slugs=["openrouter/qwen/qwen3-32b"],
                )

        clients["ec2"].run_instances.assert_not_called()

    def test_preflight_rejects_old_sixteen_gib_launch_template(self):
        clients = {name: MagicMock() for name in (
            "sts", "s3", "ssm", "ec2", "quotas", "pricing",
        )}
        clients["sts"].get_caller_identity.return_value = {"Account": "123"}
        clients["s3"].get_bucket_tagging.return_value = {"TagSet": [
            {"Key": "Funding", "Value": PERSONAL_REIMBURSEMENT_FUNDING},
        ]}
        clients["ssm"].get_parameter.return_value = {"Parameter": {
            "Value": json.dumps({"OPENROUTER_API_KEY": "stored"}),
        }}
        clients["ec2"].describe_instance_types.return_value = {"InstanceTypes": [{
            "VCpuInfo": {"DefaultVCpus": 4}, "MemoryInfo": {"SizeInMiB": 8192},
        }]}
        clients["ec2"].describe_launch_templates.return_value = {"LaunchTemplates": [{
            "LaunchTemplateId": "lt-1", "DefaultVersionNumber": 3,
        }]}
        clients["ec2"].describe_launch_template_versions.return_value = {
            "LaunchTemplateVersions": [{"LaunchTemplateData": {
                "ImageId": "ami-1",
                "InstanceType": "c7a.xlarge",
                "BlockDeviceMappings": [{
                    "DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 16},
                }],
            }}],
        }
        cfg = {
            "aws_region": "us-west-2", "aws_instance_type": "c7a.xlarge",
            "aws_bucket": None, "vm_concurrency": 1,
        }

        with (
            patch("exp_aws_trajectory.aws_clients", return_value=clients),
            patch("exp_aws_trajectory.runtime_hash", return_value="runtime"),
            patch("exp_aws_trajectory._find_runtime_ami", return_value="ami-1"),
        ):
            with self.assertRaisesRegex(
                AwsTrajectoryError, "root=16 GB.*run --aws-setup"
            ):
                preflight_aws(
                    cfg,
                    ENVIRONMENTS,
                    model_slugs=["openrouter/qwen/qwen3-32b"],
                )

        clients["quotas"].get_service_quota.assert_not_called()

    def test_expired_auth_saves_locally_and_gives_exact_original_resume(self):
        expired = ClientError(
            {"Error": {"Code": "RequestExpired", "Message": "expired"}},
            "DescribeInstances",
        )
        state = {
            "campaign_id": "campaign-123",
            "region": "us-west-2",
            "vm_concurrency": 1,
            "cells": [],
        }
        clients = {"ec2": MagicMock()}
        clients["ec2"].describe_instances.side_effect = expired
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"AWS_PROFILE": "mats-run"}
        ):
            with self.assertRaises(AwsTrajectoryError) as raised:
                _monitor_campaign(state, Path(tmp), clients, allow_launch=False)
            saved = json.loads(
                (Path(tmp) / "remote_campaigns/campaign-123.json").read_text()
            )

        message = str(raised.exception)
        self.assertEqual(saved["campaign_id"], "campaign-123")
        self.assertIn("aws login --profile mats-login --region us-west-2", message)
        self.assertIn("aws sts get-caller-identity --profile mats-run", message)
        self.assertIn(
            "uv run exp_real_audit_pipeline.py --resume-campaign=campaign-123",
            message,
        )

    def test_expired_auth_names_the_continuation_resume_endpoint(self):
        expired = ClientError(
            {"Error": {"Code": "ExpiredTokenException", "Message": "expired"}},
            "DescribeInstances",
        )
        message = _authentication_recovery_message({
            "campaign_id": "continuation-123",
            "region": "eu-west-1",
            "cells": [{"pipeline_script": "exp_continuation_pipeline.py"}],
        }, expired)

        self.assertIsNotNone(message)
        self.assertIn(
            "uv run exp_continuation_pipeline.py "
            "--resume-campaign=continuation-123",
            message,
        )

    def test_expired_auth_names_the_ml_prefix_resume_endpoint(self):
        expired = ClientError(
            {"Error": {"Code": "ExpiredTokenException", "Message": "expired"}},
            "DescribeInstances",
        )
        message = _authentication_recovery_message({
            "campaign_id": "ml-prefix-123",
            "region": "eu-west-1",
            "cells": [{"pipeline_script": "prefixes/exp_ml_prefix.py"}],
        }, expired)

        self.assertIsNotNone(message)
        self.assertIn(
            "uv run prefixes/exp_ml_prefix.py --resume-campaign=ml-prefix-123",
            message,
        )

    def test_non_authentication_aws_error_is_not_hidden(self):
        denied = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}},
            "DescribeInstances",
        )
        state = {
            "campaign_id": "campaign-123",
            "region": "us-west-2",
            "vm_concurrency": 1,
            "cells": [],
        }
        clients = {"ec2": MagicMock()}
        clients["ec2"].describe_instances.side_effect = denied
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ClientError):
            _monitor_campaign(state, Path(tmp), clients, allow_launch=False)

    def test_terminating_instance_keeps_its_slot_until_fully_terminated(self):
        cell = {
            "campaign_id": "campaign", "cell_id": "cell", "bucket": "bucket",
            "complete_key": "complete", "failure_key": "failure",
            "instance_id": "i-1", "status": "running",
        }
        state = {
            "campaign_id": "campaign", "bucket": "bucket", "vm_concurrency": 1,
            "cells": [cell],
        }
        clients = {"ec2": MagicMock(), "s3": MagicMock()}
        clients["ec2"].describe_instances.side_effect = [
            {"Reservations": []},
            {"Reservations": [{"Instances": [{
                "InstanceId": "i-1", "State": {"Name": "shutting-down"},
            }]}]},
            {"Reservations": [{"Instances": [{
                "InstanceId": "i-1", "State": {"Name": "terminated"},
            }]}]},
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("exp_aws_trajectory._terminal_marker", return_value=None),
            patch("exp_aws_trajectory.time.sleep"),
        ):
            result = _monitor_campaign(state, Path(tmp), clients, allow_launch=False)

        self.assertEqual(result["cells"][0]["status"], "infrastructure_failure")
        self.assertEqual(result["cells"][0]["instance_state"], "terminated")
        self.assertIn("terminated_at", result["cells"][0])
        clients["ec2"].terminate_instances.assert_not_called()

    def test_uploaded_completion_is_not_overwritten_while_vm_shuts_down(self):
        cell = {
            "campaign_id": "campaign", "cell_id": "cell", "bucket": "bucket",
            "complete_key": "complete", "failure_key": "failure",
            "instance_id": "i-1", "status": "running",
        }
        state = {
            "campaign_id": "campaign", "bucket": "bucket", "vm_concurrency": 1,
            "cells": [cell],
        }
        clients = {"ec2": MagicMock(), "s3": MagicMock()}
        clients["ec2"].describe_instances.side_effect = [
            {"Reservations": []},
            {"Reservations": [{"Instances": [{
                "InstanceId": "i-1", "State": {"Name": "shutting-down"},
            }]}]},
            {"Reservations": [{"Instances": [{
                "InstanceId": "i-1", "State": {"Name": "shutting-down"},
            }]}]},
            {"Reservations": [{"Instances": [{
                "InstanceId": "i-1", "State": {"Name": "terminated"},
            }]}]},
        ]
        marker = ("completed", {
            "pipeline_exit_code": 0,
            "completed_at": "2026-08-02T23:53:59+00:00",
        })
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("exp_aws_trajectory._terminal_marker", return_value=marker),
            patch("exp_aws_trajectory.time.sleep"),
        ):
            result = _monitor_campaign(state, Path(tmp), clients, allow_launch=False)

        completed = result["cells"][0]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["instance_state"], "terminated")
        self.assertEqual(completed["terminal"]["pipeline_exit_code"], 0)
        self.assertNotIn("reason", completed["terminal"])

    def test_marker_found_after_vm_terminated_records_terminal_instance_state(self):
        cell = {
            "campaign_id": "campaign", "cell_id": "cell", "bucket": "bucket",
            "complete_key": "complete", "failure_key": "failure",
            "instance_id": "i-1", "status": "running",
        }
        state = {
            "campaign_id": "campaign", "bucket": "bucket", "vm_concurrency": 1,
            "cells": [cell],
        }
        clients = {"ec2": MagicMock(), "s3": MagicMock()}
        clients["ec2"].describe_instances.side_effect = [
            {"Reservations": []},
            {"Reservations": [{"Instances": [{
                "InstanceId": "i-1", "State": {"Name": "terminated"},
            }]}]},
        ]
        marker = ("completed", {
            "pipeline_exit_code": 0,
            "completed_at": "2026-08-02T23:53:59+00:00",
        })
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("exp_aws_trajectory._terminal_marker", return_value=marker),
        ):
            result = _monitor_campaign(state, Path(tmp), clients, allow_launch=False)

        completed = result["cells"][0]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["instance_state"], "terminated")
        self.assertIn("terminated_at", completed)

    def test_uploaded_result_can_reconcile_an_instance_no_longer_returned(self):
        cell = {
            "campaign_id": "campaign", "cell_id": "cell", "bucket": "bucket",
            "complete_key": "complete", "failure_key": "failure",
            "instance_id": "i-1", "status": "finishing",
            "instance_state": "running",
            "terminal_status": "completed",
            "terminal": {
                "pipeline_exit_code": 1,
                "completed_at": "2026-08-02T23:53:59+00:00",
            },
        }
        state = {
            "campaign_id": "campaign", "bucket": "bucket", "vm_concurrency": 1,
            "cells": [cell],
        }
        clients = {"ec2": MagicMock(), "s3": MagicMock()}
        clients["ec2"].describe_instances.side_effect = [
            {"Reservations": []},  # recovery by campaign tag
            {"Reservations": []},  # exact instance-id lookup
            {"Reservations": []},  # every non-terminal campaign state
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("exp_aws_trajectory._terminal_marker", return_value=(
                "completed", cell["terminal"],
            )),
        ):
            result = _monitor_campaign(
                state, Path(tmp), clients, allow_launch=False
            )

        completed = result["cells"][0]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["instance_state"], "terminated")
        self.assertTrue(completed["terminated_at_is_upper_bound"])
        self.assertEqual(
            completed["termination_resolution"],
            "absent_from_all_nonterminal_campaign_instances",
        )

    def test_newly_found_marker_reconciles_an_instance_no_longer_returned(self):
        cell = {
            "campaign_id": "campaign", "cell_id": "cell", "bucket": "bucket",
            "complete_key": "complete", "failure_key": "failure",
            "instance_id": "i-aged-out", "status": "running",
            "instance_state": "running",
        }
        state = {
            "campaign_id": "campaign", "bucket": "bucket", "vm_concurrency": 1,
            "cells": [cell],
        }
        clients = {"ec2": MagicMock(), "s3": MagicMock()}
        clients["ec2"].describe_instances.side_effect = [
            {"Reservations": []},  # recovery by campaign tag
            {"Reservations": []},  # exact instance-id lookup
            {"Reservations": []},  # every non-terminal campaign state
        ]
        marker = ("completed", {
            "pipeline_exit_code": 0,
            "completed_at": "2026-08-02T23:53:59+00:00",
        })
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("exp_aws_trajectory._terminal_marker", return_value=marker),
        ):
            result = _monitor_campaign(
                state, Path(tmp), clients, allow_launch=False
            )

        completed = result["cells"][0]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["instance_state"], "terminated")
        self.assertEqual(completed["terminal"]["pipeline_exit_code"], 0)
        self.assertTrue(completed["terminated_at_is_upper_bound"])
        clients["ec2"].terminate_instances.assert_not_called()

    def test_vm_cost_uses_termination_time_not_earlier_upload_time(self):
        cost = _vm_cost({
            "hourly_price_usd": 0.2,
            "launched_at": "2026-08-01T00:00:00+00:00",
            "terminated_at": "2026-08-01T02:30:00+00:00",
            "terminal": {"completed_at": "2026-08-01T02:00:00+00:00"},
        })

        self.assertEqual(cost["billed_seconds_estimate"], 9000)
        self.assertAlmostEqual(cost["estimated_vm_cost_usd"], 0.5)

    def test_retry_filters_models_and_inherits_original_aws_location(self):
        old = {
            "campaign_id": "old", "bucket": "custom-bucket", "region": "eu-west-1",
            "instance_type": "c7a.xlarge", "vm_concurrency": 7,
            "pipeline_config": {
                "targets": ["qwen3-32b", "deepseek-v4-pro"],
                "target_models": [
                    "openrouter/qwen/qwen3-32b",
                    "openrouter/deepseek/deepseek-v4-pro-20260423",
                ],
                "seeds": ["fraud_detection"],
                "seeds_path": "/repo/seeds/ml_training_data_misuse",
                "epochs": 1, "reasoning": True, "condition": "allow",
                "harness": "simple",
                "judge_resolved": "openai/gpt-5.6-luna",
                "gate_model": "openai/gpt-5.6-luna",
                "time_limit": AGENT_TIME_LIMIT_SECONDS, "aws_secret_env": [],
            },
            "cells": [
                {
                    "status": "infrastructure_failure", "target": "qwen3-32b",
                    "seed": "fraud_detection", "original_epoch": 1,
                },
                {
                    "status": "completed", "target": "deepseek-v4-pro",
                    "seed": "fraud_detection", "original_epoch": 1,
                },
            ],
        }
        control = {
            "harness": "simple", "harness_explicit": True,
            "aws_region": "us-west-2", "aws_region_explicit": False,
            "aws_instance_type": "c7a.xlarge", "aws_instance_type_explicit": False,
            "vm_concurrency": 50, "vm_concurrency_explicit": False,
            "aws_bucket": None, "aws_bucket_explicit": False,
            "aws_secret_env": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            state_path = data / "remote_campaigns" / "old.json"
            state_path.parent.mkdir()
            state_path.write_text(json.dumps(old))
            with (
                patch("exp_aws_trajectory.aws_clients", return_value={}),
                patch("exp_aws_trajectory.run_campaign", return_value={"cells": []}) as run,
            ):
                retry_failed(control, ENVIRONMENTS, data, "old", dry_run=True)

        retry_cfg = run.call_args.args[0]
        self.assertEqual(retry_cfg["aws_region"], "eu-west-1")
        self.assertEqual(retry_cfg["aws_bucket"], "custom-bucket")
        self.assertEqual(retry_cfg["vm_concurrency"], 7)
        self.assertEqual(retry_cfg["targets"], ["qwen3-32b"])
        self.assertEqual(
            retry_cfg["target_models"], ["openrouter/qwen/qwen3-32b"]
        )
        self.assertEqual(
            retry_cfg["_cell_selections"],
            [("qwen3-32b", "fraud_detection", 1)],
        )

    def test_retry_preserves_prefix_campaign_type_name_and_failed_epoch(self):
        old = {
            "campaign_id": "old-prefix",
            "bucket": "custom-bucket",
            "region": "us-west-2",
            "instance_type": "c7a.xlarge",
            "vm_concurrency": 4,
            "pipeline_config": {
                "targets": ["qwen3-32b"],
                "target_models": ["openrouter/qwen/qwen3-32b"],
                "seeds": ["fraud_detection"],
                "seeds_path": "/repo/seeds/ml_prefix_only",
                "seed_dir": "ml_prefix_only",
                "family": "ml_prefix_only",
                "epochs": 10,
                "reasoning": True,
                "condition": "allow",
                "harness": "production",
                "judge_resolved": None,
                "gate_model": None,
                "time_limit": AGENT_TIME_LIMIT_SECONDS,
                "aws_secret_env": [],
                "prefix_only": True,
                "name": "fraud-control",
            },
            "cells": [{
                "status": "completed",
                "terminal": {"pipeline_exit_code": 124},
                "target": "qwen3-32b",
                "seed": "fraud_detection",
                "original_epoch": 7,
            }],
        }
        control = {
            "harness": "production",
            "harness_explicit": True,
            "aws_region": "us-west-2",
            "aws_region_explicit": False,
            "aws_instance_type": "c7a.xlarge",
            "aws_instance_type_explicit": False,
            "vm_concurrency": 50,
            "vm_concurrency_explicit": False,
            "aws_bucket": None,
            "aws_bucket_explicit": False,
            "aws_secret_env": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            state_path = data / "remote_campaigns" / "old-prefix.json"
            state_path.parent.mkdir()
            state_path.write_text(json.dumps(old))
            with (
                patch("exp_aws_trajectory.aws_clients", return_value={}),
                patch(
                    "exp_aws_trajectory.run_campaign", return_value={"cells": []}
                ) as run,
            ):
                retry_failed(
                    control, ENVIRONMENTS, data, "old-prefix", dry_run=True
                )

        retry_cfg = run.call_args.args[0]
        self.assertIs(retry_cfg["prefix_only"], True)
        self.assertEqual(retry_cfg["name"], "fraud-control")
        self.assertEqual(
            retry_cfg["_cell_selections"],
            [("qwen3-32b", "fraud_detection", 7)],
        )

    def test_ordinary_pipeline_failures_remain_nonretryable(self):
        old = {
            "campaign_id": "old-audit",
            "bucket": "bucket",
            "region": "us-west-2",
            "pipeline_config": {
                "targets": ["qwen3-32b"],
                "target_models": ["openrouter/qwen/qwen3-32b"],
                "seeds": ["fraud_detection"],
                "seeds_path": "/repo/seeds/ml_training_data_misuse",
                "family": "ml_training_data_misuse",
                "epochs": 1,
                "reasoning": True,
                "condition": "allow",
                "harness": "simple",
                "judge_resolved": "openai/gpt-5.6-luna",
                "gate_model": "openai/gpt-5.6-luna",
                "time_limit": AGENT_TIME_LIMIT_SECONDS,
                "aws_secret_env": [],
            },
            "cells": [{
                "status": "completed",
                "terminal": {"pipeline_exit_code": 1},
                "target": "qwen3-32b",
                "seed": "fraud_detection",
                "original_epoch": 1,
            }],
        }
        control = {
            "harness": "simple",
            "harness_explicit": True,
            "aws_region": "us-west-2",
            "aws_region_explicit": False,
            "aws_instance_type": "c7a.xlarge",
            "aws_instance_type_explicit": False,
            "vm_concurrency": 50,
            "vm_concurrency_explicit": False,
            "aws_bucket": None,
            "aws_bucket_explicit": False,
            "aws_secret_env": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            state_path = data / "remote_campaigns" / "old-audit.json"
            state_path.parent.mkdir()
            state_path.write_text(json.dumps(old))
            with (
                patch("exp_aws_trajectory.aws_clients", return_value={}),
                self.assertRaisesRegex(
                    AwsTrajectoryError, "no retryable infrastructure failures"
                ),
            ):
                retry_failed(
                    control, ENVIRONMENTS, data, "old-audit", dry_run=True
                )

    def test_explicit_retry_includes_completed_pipeline_failures(self):
        old = {
            "campaign_id": "old-continuation",
            "bucket": "bucket",
            "region": "us-west-2",
            "instance_type": "c7a.xlarge",
            "vm_concurrency": 4,
            "pipeline_config": {
                "targets": ["qwen3-32b"],
                "target_models": ["openrouter/qwen/qwen3-32b"],
                "seeds": ["checkout_redesign"],
                "seeds_path": "/repo/seeds/p_hacking",
                "family": "p_hacking",
                "epochs": 2,
                "reasoning": None,
                "condition": "allow",
                "harness": "production",
                "judge_resolved": "openai/gpt-5.6-luna",
                "gate_model": "openai/gpt-5.6-luna",
                "time_limit": 1800,
                "aws_secret_env": [],
                "pressure": "low",
                "continuation": {
                    "treatment": "no-hack",
                    "payloads": [],
                },
            },
            "cells": [
                {
                    "status": "completed",
                    "terminal": {"pipeline_exit_code": 1},
                    "target": "qwen3-32b",
                    "prefix_name": "prefix",
                    "seed": "checkout_redesign",
                    "original_epoch": 1,
                },
                {
                    "status": "completed",
                    "terminal": {"pipeline_exit_code": 0},
                    "target": "qwen3-32b",
                    "prefix_name": "prefix",
                    "seed": "checkout_redesign",
                    "original_epoch": 2,
                },
            ],
        }
        control = {
            "harness": "production",
            "harness_explicit": True,
            "aws_region": "us-west-2",
            "aws_region_explicit": False,
            "aws_instance_type": "c7a.xlarge",
            "aws_instance_type_explicit": False,
            "vm_concurrency": 50,
            "vm_concurrency_explicit": False,
            "aws_bucket": None,
            "aws_bucket_explicit": False,
            "aws_secret_env": [],
            "retry_pipeline_failures": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            state_path = data / "remote_campaigns" / "old-continuation.json"
            state_path.parent.mkdir()
            state_path.write_text(json.dumps(old))
            with (
                patch("exp_aws_trajectory.aws_clients", return_value={}),
                patch(
                    "exp_aws_trajectory.run_campaign", return_value={"cells": []}
                ) as run,
            ):
                retry_failed(
                    control, ENVIRONMENTS, data, "old-continuation", dry_run=True
                )

        self.assertEqual(
            run.call_args.args[0]["_cell_selections"],
            [("prefix", "checkout_redesign", 1)],
        )

    def test_retry_preserves_incomplete_continuation_override(self):
        payload = {
            "name": "incomplete-prefix",
            "sha256": "a" * 64,
            "file_sha256": "b" * 64,
            "local_path": "/tmp/incomplete-prefix.json",
            "target": "qwen3-32b",
            "target_model": "openrouter/qwen/qwen3-32b",
            "reasoning": True,
        }
        old = {
            "campaign_id": "old-continuation",
            "bucket": "bucket",
            "region": "us-west-2",
            "instance_type": "c7a.xlarge",
            "vm_concurrency": 4,
            "pipeline_config": {
                "targets": ["qwen3-32b"],
                "target_models": ["openrouter/qwen/qwen3-32b"],
                "seeds": ["checkout_redesign"],
                "seeds_path": "/repo/seeds/p_hacking",
                "seed_dir": "p_hacking",
                "family": "p_hacking",
                "epochs": 10,
                "reasoning": None,
                "condition": "allow",
                "harness": "simple",
                "judge_resolved": "openai/gpt-5.6-luna",
                "gate_model": "openai/gpt-5.6-luna",
                "time_limit": 1800,
                "aws_secret_env": [],
                "pressure": "low",
                "continuation": {
                    "treatment": "no-honeypot",
                    "payloads": [payload],
                },
                "allow_incomplete_prefixes": True,
            },
            "cells": [{
                "status": "infrastructure_failure",
                "target": "qwen3-32b",
                "prefix_name": "incomplete-prefix",
                "seed": "checkout_redesign",
                "original_epoch": 4,
            }],
        }
        control = {
            "harness": "simple",
            "harness_explicit": True,
            "aws_region": "us-west-2",
            "aws_region_explicit": False,
            "aws_instance_type": "c7a.xlarge",
            "aws_instance_type_explicit": False,
            "vm_concurrency": 50,
            "vm_concurrency_explicit": False,
            "aws_bucket": None,
            "aws_bucket_explicit": False,
            "aws_secret_env": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            state_path = data / "remote_campaigns" / "old-continuation.json"
            state_path.parent.mkdir()
            state_path.write_text(json.dumps(old))
            with (
                patch("exp_aws_trajectory.aws_clients", return_value={}),
                patch(
                    "exp_aws_trajectory.run_campaign", return_value={"cells": []}
                ) as run,
            ):
                retry_failed(
                    control, ENVIRONMENTS, data, "old-continuation", dry_run=True
                )

        retry_cfg = run.call_args.args[0]
        self.assertIs(retry_cfg["allow_incomplete_prefixes"], True)
        self.assertEqual(
            retry_cfg["_cell_selections"],
            [("incomplete-prefix", "checkout_redesign", 4)],
        )


if __name__ == "__main__":
    unittest.main()
