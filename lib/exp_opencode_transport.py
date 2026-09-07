"""Long-prompt stdin transport for the pinned Inspect-SWE OpenCode adapter.

Inspect-SWE 0.2.63 closes stdin and appends the complete prompt to the argv used
to start ``opencode run``.  Large inline continuation logs can therefore fail in
``execve`` with ``E2BIG`` before OpenCode starts.  OpenCode 1.18.14 reads a
non-interactive message from stdin, so this module narrowly rewrites that one
adapter call while leaving every other sandbox operation untouched.

This module can participate in paid agent execution and therefore uses the
repository's ``exp_`` prefix.  Importing or testing it performs no paid work.
"""

from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import PurePath
from typing import Any, Callable, Sequence

from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions


_INSPECT_SWE_SHELL_PREFIX = (
    "bash",
    "-c",
    'exec 0</dev/null; "$@"',
    "bash",
)
_STDIN_SHELL = 'exec "$@"'
_PATCH_MARKER = "__mats_opencode_prompt_stdin_v1__"


def _opencode_stdin_message(prompt: str) -> str:
    """Return the raw prompt expected by OpenCode 1.18.14 on stdin.

    Shell quoting belongs to argv parsing and is removed before a process receives a
    positional argument. OpenCode reads stdin as message content directly, so adding
    argv-style wrapper quotes here would make them and any backslash escapes visible
    to the model.
    """

    return prompt


def rewrite_opencode_exec_remote(
    command: Sequence[str],
    options: ExecRemoteAwaitableOptions,
) -> tuple[list[str], ExecRemoteAwaitableOptions, bool]:
    """Move the pinned adapter's final prompt argument to stdin.

    Returns the command, options, and whether the exact pinned call was
    rewritten.  Calls that do not match the Inspect-SWE OpenCode invocation are
    returned unchanged.
    """

    original = list(command)
    if (
        len(original) < len(_INSPECT_SWE_SHELL_PREFIX) + 4
        or tuple(original[:4]) != _INSPECT_SWE_SHELL_PREFIX
        or PurePath(original[4]).name != "opencode"
        or original[5] != "run"
    ):
        return original, options, False
    if options.input is not None:
        raise RuntimeError(
            "Inspect-SWE OpenCode invocation unexpectedly already has stdin input"
        )

    prompt = original.pop()
    original[2] = _STDIN_SHELL
    return (
        original,
        replace(options, input=_opencode_stdin_message(prompt)),
        True,
    )


class _OpenCodeStdinSandbox:
    """Delegate a sandbox while rewriting only OpenCode's agent invocation."""

    def __init__(self, delegate: Any):
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    async def exec_remote(
        self,
        cmd: list[str],
        options: Any = None,
        *,
        stream: bool = True,
    ) -> Any:
        if isinstance(options, ExecRemoteAwaitableOptions):
            cmd, options, _ = rewrite_opencode_exec_remote(cmd, options)
        return await self._delegate.exec_remote(cmd, options, stream=stream)


def install_inspect_swe_opencode_stdin_transport() -> None:
    """Install the idempotent adapter shim in Inspect-SWE's OpenCode module."""

    module = importlib.import_module("inspect_swe._opencode.opencode")
    sandbox_lookup: Callable[[str | None], Any] = module.sandbox_env
    if getattr(sandbox_lookup, _PATCH_MARKER, False):
        return

    def stdin_sandbox_lookup(name: str | None = None) -> _OpenCodeStdinSandbox:
        return _OpenCodeStdinSandbox(sandbox_lookup(name))

    setattr(stdin_sandbox_lookup, _PATCH_MARKER, True)
    module.sandbox_env = stdin_sandbox_lookup
