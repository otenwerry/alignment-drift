"""Free regression tests for long OpenCode prompt transport."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from exp_opencode_transport import (  # noqa: E402
    _OpenCodeStdinSandbox,
    _opencode_stdin_message,
    install_inspect_swe_opencode_stdin_transport,
    rewrite_opencode_exec_remote,
)


def _inspect_swe_command(prompt: str, *, resume: bool = True) -> list[str]:
    command = [
        "bash",
        "-c",
        'exec 0</dev/null; "$@"',
        "bash",
        "/opt/inspect-swe/opencode/node_modules/.bin/opencode",
        "run",
        "--model",
        "openrouter/moonshotai/kimi-k2.6",
        "--format",
        "json",
        "--dangerously-skip-permissions",
    ]
    if resume:
        command.append("--continue")
    command.append(prompt)
    return command


@pytest.mark.parametrize("resume", [False, True])
def test_large_opencode_prompt_uses_stdin_not_argv(resume: bool) -> None:
    prompt = ('activity log says "quoted value"\n' * 20_000) + "finish task"
    options = ExecRemoteAwaitableOptions(
        cwd="/workspace",
        env={"ONE": "two"},
        user="root",
        concurrency=False,
        timeout=123,
    )

    command, rewritten_options, rewritten = rewrite_opencode_exec_remote(
        _inspect_swe_command(prompt, resume=resume), options
    )

    assert rewritten is True
    assert command[:4] == ["bash", "-c", 'exec "$@"', "bash"]
    assert prompt not in command
    assert ("--continue" in command) is resume
    assert sum(len(value.encode()) + 1 for value in command) < 1024
    assert rewritten_options.input == prompt
    assert rewritten_options.cwd == "/workspace"
    assert rewritten_options.env == {"ONE": "two"}
    assert rewritten_options.user == "root"
    assert rewritten_options.concurrency is False
    assert rewritten_options.timeout == 123


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("oneword", "oneword"),
        ("two words", "two words"),
        ('say "hello world"', 'say "hello world"'),
    ],
)
def test_stdin_payload_is_raw_model_visible_text(
    prompt: str, expected: str
) -> None:
    assert _opencode_stdin_message(prompt) == expected


def test_non_opencode_exec_is_untouched() -> None:
    options = ExecRemoteAwaitableOptions(cwd="/workspace")
    command = ["bash", "-c", 'exec "$@"', "bash", "python", "task.py"]

    rewritten_command, rewritten_options, rewritten = rewrite_opencode_exec_remote(
        command, options
    )

    assert rewritten is False
    assert rewritten_command == command
    assert rewritten_options is options


def test_sandbox_proxy_forwards_large_prompt_through_input() -> None:
    captured = SimpleNamespace()

    class FakeSandbox:
        async def exec_remote(self, cmd, options, *, stream):
            captured.cmd = cmd
            captured.options = options
            captured.stream = stream
            return "result"

    prompt = "large prompt " * 30_000
    proxy = _OpenCodeStdinSandbox(FakeSandbox())
    result = asyncio.run(
        proxy.exec_remote(
            _inspect_swe_command(prompt),
            ExecRemoteAwaitableOptions(cwd="/workspace"),
            stream=False,
        )
    )

    assert result == "result"
    assert prompt not in captured.cmd
    assert captured.options.input == prompt
    assert captured.stream is False


def test_install_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("inspect_swe._opencode.opencode")

    def original_lookup(name=None):
        return ("sandbox", name)

    monkeypatch.setattr(module, "sandbox_env", original_lookup)

    install_inspect_swe_opencode_stdin_transport()
    installed = module.sandbox_env
    install_inspect_swe_opencode_stdin_transport()

    assert module.sandbox_env is installed
    assert module.sandbox_env is not original_lookup
