#!/usr/bin/env python3
"""Verify the persisted Codex ChatGPT credential, optionally with a live call.

Usage (from mats/environments):

    uv run tools/check_codex_subscription_auth.py
    uv run tools/check_codex_subscription_auth.py --live

The structural check is free and compares parsed JSON without printing credentials.
The live check runs a minimal GPT-5.5 subscription call from an isolated temporary
CODEX_HOME populated only from the persisted project .env value.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import dotenv_values


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
HOST_AUTH_FILE = Path.home() / ".codex" / "auth.json"
AUTH_KEY = "CODEX_SUBSCRIPTION_AUTH_JSON_GZIP_B64"


def _json_object(data: bytes, *, label: str) -> dict:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def load_persisted_auth() -> tuple[bytes, dict]:
    if not ENV_FILE.is_file():
        raise RuntimeError(f"project environment file is missing: {ENV_FILE}")
    encoded = dotenv_values(ENV_FILE).get(AUTH_KEY)
    if not encoded:
        raise RuntimeError(f"{ENV_FILE} does not define {AUTH_KEY}")
    try:
        packed = base64.b64decode(str(encoded), validate=True)
        data = gzip.decompress(packed)
    except (ValueError, TypeError, OSError, EOFError) as error:
        raise RuntimeError(f"{AUTH_KEY} is not valid gzip-base64") from error
    return data, _json_object(data, label=AUTH_KEY)


def verify_matches_host() -> bytes:
    persisted_bytes, persisted = load_persisted_auth()
    if not HOST_AUTH_FILE.is_file():
        raise RuntimeError(f"current Codex login is missing: {HOST_AUTH_FILE}")
    current = _json_object(HOST_AUTH_FILE.read_bytes(), label=str(HOST_AUTH_FILE))
    if persisted != current:
        raise RuntimeError(
            "persisted Codex credential is stale; refresh the .env snapshot from "
            f"{HOST_AUTH_FILE}"
        )
    print("OK: persisted Codex credential matches the current login")
    return persisted_bytes


def live_check(auth_bytes: bytes, *, model: str) -> None:
    binary = shutil.which("codex")
    if binary is None:
        raise RuntimeError("codex is not on PATH")
    with tempfile.TemporaryDirectory(prefix="mats-codex-auth-check-") as tmp:
        codex_home = Path(tmp)
        auth_path = codex_home / "auth.json"
        auth_path.write_bytes(auth_bytes)
        auth_path.chmod(0o600)
        environment = dict(os.environ)
        environment["CODEX_HOME"] = str(codex_home)
        process = subprocess.run(
            [
                binary,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--model",
                model,
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "-",
            ],
            input="Reply with exactly AUTH_OK. Do not use tools.\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            cwd=Path(__file__).resolve().parents[1],
            timeout=180,
            check=False,
        )
        if process.returncode != 0:
            detail = (process.stderr or process.stdout).strip()[-1500:]
            raise RuntimeError(
                f"live {model} credential check failed with exit "
                f"{process.returncode}: {detail}"
            )
        if "AUTH_OK" not in process.stdout:
            raise RuntimeError(
                f"live {model} call succeeded but did not return AUTH_OK; "
                f"stdout tail: {process.stdout.strip()[-1000:]}"
            )
    print(f"OK: persisted credential completed a live {model} subscription call")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="make one minimal subscription model call using only the persisted blob",
    )
    parser.add_argument("--model", default="gpt-5.5")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        auth_bytes = verify_matches_host()
        if args.live:
            live_check(auth_bytes, model=args.model)
    except RuntimeError as error:
        raise SystemExit(f"ERROR: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
