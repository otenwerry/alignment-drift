"""Real executors for the fixed agent-tool interface (real-environment mode).

The bash/read/write wire protocol is owned by ``target_tool_protocol.py``. ML real mode
explicitly adds check_time. Tool bodies execute against the sample's Docker sandbox
``/workspace``.

Result semantics are pinned (REAL_TOOL_SEMANTICS_VERSION) because there is no auditor
to improvise them:
  bash        run via `bash -lc` in /workspace; returns stdout+stderr; a nonzero exit
              appends "(exit code N)". The trajectory wall clock is its only timeout.
  read_file   UTF-8 text read, relative paths resolved against /workspace.
  write_file  parent dirs created, whole-file UTF-8 write, returns "Wrote <path>".
  check_time  ML real mode only; reports the trajectory wall-clock budget.

Tool output shown to the agent is capped at REAL_MAX_TOOL_OUTPUT bytes by
execute_tools (a visible truncation envelope in-transcript). This is a deliberate,
metadata-stamped difference from the simulated auditor channel, which was uncapped.
"""

from __future__ import annotations

import hashlib
import json
import posixpath

import anyio

from inspect_ai.tool import Tool, ToolDef, ToolError, ToolInfo, ToolParams
from inspect_ai.util import sample_limits, sandbox

from target_tool_protocol import TARGET_TOOL_DEFINITIONS

REAL_TOOL_SEMANTICS_VERSION = "environments-real-tools-v1-resettable-session-clock"
# Visible per-call cap on tool output shown to the agent (bytes). Inspect's default.
REAL_MAX_TOOL_OUTPUT = 16384
WORKSPACE_DIR = "/workspace"

CHECK_TIME_DEFINITION = {
    "name": "check_time",
    "description": (
        "Check the current wall-clock budget. Returns time elapsed since the task "
        "began, time remaining, and the current deadline, all in seconds."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def reset_sample_time_remaining(seconds: float) -> dict[str, float]:
    """Move Inspect's active sample deadline to ``seconds`` from now.

    Inspect 0.3.239 exposes no public time-limit setter. Its pinned ``_TimeLimit``
    stores the reported limit in ``_limit`` and enforcement in an AnyIO cancel scope,
    whose deadline is publicly mutable. Fail loudly if that pinned shape changes so we
    never send the second-pass promise without enforcing it.
    """
    if seconds <= 0:
        raise ValueError("remaining time must be positive")
    limit = sample_limits().time
    cancel_scope = getattr(limit, "_cancel_scope", None)
    if not hasattr(limit, "_limit") or cancel_scope is None:
        raise RuntimeError(
            "active Inspect time limit cannot be reset; its implementation changed"
        )
    elapsed = max(0.0, float(limit.usage))
    new_total = elapsed + float(seconds)
    limit._limit = new_total
    cancel_scope.deadline = anyio.current_time() + float(seconds)
    return {
        "elapsed_seconds_at_reset": elapsed,
        "remaining_seconds_after_reset": float(seconds),
        "deadline_seconds_from_start": new_total,
    }


def _resolve(path: str) -> str:
    """Workspace-relative path resolution (the tool descriptions promise it)."""
    return path if posixpath.isabs(path) else posixpath.join(WORKSPACE_DIR, path)


async def _bash(command: str) -> str:
    try:
        result = await sandbox().exec(
            ["bash", "-lc", command],
            cwd=WORKSPACE_DIR,
            timeout=None,
            timeout_retry=False,
        )
    except UnicodeDecodeError:
        raise ToolError("command produced output that is not valid UTF-8")
    except PermissionError:
        raise ToolError("permission denied executing command")
    output = result.stdout
    if result.stderr:
        output = f"{output}{result.stderr}" if output else result.stderr
    if not result.success:
        output = f"{output}\n(exit code {result.returncode})" if output else (
            f"(exit code {result.returncode})"
        )
    return output


async def _read_file(path: str) -> str:
    resolved = _resolve(path)
    try:
        return await sandbox().read_file(resolved, text=True)
    except FileNotFoundError:
        raise ToolError(f"{path}: no such file")
    except IsADirectoryError:
        raise ToolError(f"{path} is a directory, not a file")
    except UnicodeDecodeError:
        raise ToolError(f"{path} is not valid UTF-8 text")
    except PermissionError:
        raise ToolError(f"{path}: permission denied")


async def _write_file(path: str, content: str) -> str:
    resolved = _resolve(path)
    parent = posixpath.dirname(resolved)
    if parent and parent != "/":
        await sandbox().exec(["mkdir", "-p", parent])
    try:
        await sandbox().write_file(resolved, content)
    except PermissionError:
        raise ToolError(f"{path}: permission denied")
    except IsADirectoryError:
        raise ToolError(f"{path} is a directory")
    return f"Wrote {path}"


async def _check_time() -> str:
    limit = sample_limits().time
    elapsed = max(0.0, float(limit.usage))
    total = None if limit.limit is None else max(0.0, float(limit.limit))
    remaining = None if total is None else max(0.0, total - elapsed)
    return json.dumps({
        "elapsed_seconds": round(elapsed, 1),
        "remaining_seconds": None if remaining is None else round(remaining, 1),
        "total_seconds": None if total is None else round(total, 1),
    }, separators=(",", ":"))


_EXECUTORS = {"bash": _bash, "read_file": _read_file, "write_file": _write_file}


def real_tool_defs(*, include_time: bool = False) -> list[ToolDef]:
    """One ToolDef per fixed definition, schema taken verbatim from the shared source."""
    definitions = list(TARGET_TOOL_DEFINITIONS)
    executors = dict(_EXECUTORS)
    if include_time:
        definitions.append(CHECK_TIME_DEFINITION)
        executors["check_time"] = _check_time
    return [
        ToolDef(
            executors[d["name"]],
            name=d["name"],
            description=d["description"],
            parameters=ToolParams.model_validate(d["parameters"]),
        )
        for d in definitions
    ]


def real_target_tools(*, include_time: bool = False) -> list[Tool]:
    """The real tools, ready for generate()/execute_tools."""
    return [td.as_tool() for td in real_tool_defs(include_time=include_time)]


def real_target_tools_infos(*, include_time: bool = False) -> list[ToolInfo]:
    """ToolInfo view of the REAL tools (what the wire schema will be), for the
    fingerprint test and metadata stamping."""
    return [
        ToolInfo(name=td.name, description=td.description, parameters=td.parameters)
        for td in real_tool_defs(include_time=include_time)
    ]


def real_target_tools_metadata(*, include_time: bool = False) -> list[dict]:
    return [
        info.model_dump(mode="json", exclude_none=False)
        for info in real_target_tools_infos(include_time=include_time)
    ]


def real_target_tools_fingerprint(*, include_time: bool = False) -> str:
    payload = json.dumps(
        real_target_tools_metadata(include_time=include_time),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()
