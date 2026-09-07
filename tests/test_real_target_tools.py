"""The real tools must present exactly the environments-owned wire schema."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anyio

ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from target_tool_protocol import (  # noqa: E402
    target_tool_fingerprint,
    target_tool_metadata,
)
from real_target_tools import (  # noqa: E402
    _bash,
    _check_time,
    real_target_tools,
    real_target_tools_infos,
    reset_sample_time_remaining,
)


class RealToolSchemaTests(unittest.TestCase):
    def test_tool_infos_byte_equal_fixed_definitions(self):
        real = [
            info.model_dump(mode="json", exclude_none=False)
            for info in real_target_tools_infos()
        ]
        self.assertEqual(real, target_tool_metadata())

    def test_fingerprint_matches_simulated_interface(self):
        import hashlib
        import json

        payload = json.dumps(
            [i.model_dump(mode="json", exclude_none=False) for i in real_target_tools_infos()],
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertEqual(
            hashlib.sha256(payload.encode()).hexdigest(),
            target_tool_fingerprint(),
        )

    def test_tools_are_registered_inspect_tools(self):
        tools = real_target_tools()
        self.assertEqual(len(tools), 3)
        from inspect_ai.tool import ToolDef

        names = [ToolDef(t).name for t in tools]
        self.assertEqual(names, ["bash", "read_file", "write_file"])

    def test_ml_time_tool_is_explicitly_additive(self):
        from inspect_ai.tool import ToolDef

        names = [ToolDef(t).name for t in real_target_tools(include_time=True)]
        self.assertEqual(names, ["bash", "read_file", "write_file", "check_time"])
        self.assertEqual(real_target_tools_infos(), real_target_tools_infos(include_time=False))


class RealBashTimeoutTests(unittest.IsolatedAsyncioTestCase):
    @patch("real_target_tools.sandbox")
    async def test_bash_has_no_individual_timeout(self, mock_sandbox):
        mock_sandbox.return_value.exec = AsyncMock(return_value=SimpleNamespace(
            stdout="done\n", stderr="", success=True, returncode=0,
        ))

        result = await _bash("python train.py")

        self.assertEqual(result, "done\n")
        self.assertIsNone(mock_sandbox.return_value.exec.await_args.kwargs["timeout"])


class CheckTimeTests(unittest.IsolatedAsyncioTestCase):
    @patch("real_target_tools.sample_limits")
    async def test_reports_inspect_sample_clock(self, mock_limits):
        mock_limits.return_value.time = SimpleNamespace(usage=125.25, limit=4200)

        result = await _check_time()

        self.assertEqual(
            result,
            '{"elapsed_seconds":125.2,"remaining_seconds":4074.8,"total_seconds":4200.0}',
        )

    @patch("real_target_tools.sample_limits")
    async def test_reset_changes_both_reported_and_enforced_deadline(self, mock_limits):
        from inspect_ai.util import time_limit

        active_limit = time_limit(0.02)
        mock_limits.return_value.time = active_limit
        with active_limit:
            reset = reset_sample_time_remaining(0.05)
            await anyio.sleep(0.03)
            result = await _check_time()

        self.assertAlmostEqual(reset["remaining_seconds_after_reset"], 0.05)
        self.assertGreater(reset["deadline_seconds_from_start"], 0.05)
        self.assertIn('"remaining_seconds":', result)
        self.assertGreaterEqual(active_limit.limit, 0.05)

    @patch("real_target_tools.sample_limits")
    async def test_reset_can_shorten_the_original_deadline(self, mock_limits):
        from inspect_ai.util import LimitExceededError, time_limit

        active_limit = time_limit(1.0)
        mock_limits.return_value.time = active_limit
        with self.assertRaises(LimitExceededError):
            with active_limit:
                reset_sample_time_remaining(0.02)
                await anyio.sleep(0.04)

    @patch("real_target_tools.sample_limits")
    async def test_reset_fails_loudly_if_inspect_shape_changes(self, mock_limits):
        mock_limits.return_value.time = SimpleNamespace(usage=10, limit=4200)

        with self.assertRaisesRegex(RuntimeError, "implementation changed"):
            reset_sample_time_remaining(2700)


if __name__ == "__main__":
    unittest.main()
