"""Canonical agent-facing tool schemas for the real-environment project.

The implementations live in :mod:`real_target_tools`.  Keeping the wire protocol in a
small dependency-free module makes it easy to review and fingerprint without importing
an experiment or the retired Petri runtime.
"""

from __future__ import annotations

import hashlib
import json

from inspect_ai.tool import ToolInfo


TARGET_TOOLS_VERSION = "environments-target-tools-v1"

TARGET_TOOL_DEFINITIONS = (
    {
        "name": "bash",
        "description": "Run a shell command in the task workspace and return its combined output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the task workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to the task workspace.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": "Write the complete UTF-8 contents of a file in the task workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to the task workspace.",
                },
                "content": {
                    "type": "string",
                    "description": "The complete file contents to write.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
)


def target_tool_metadata() -> list[dict]:
    return [
        ToolInfo.model_validate(definition).model_dump(mode="json", exclude_none=False)
        for definition in TARGET_TOOL_DEFINITIONS
    ]


def target_tool_fingerprint() -> str:
    payload = json.dumps(target_tool_metadata(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
