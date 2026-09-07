from __future__ import annotations

import base64
import gzip
import json
import sys
from pathlib import Path

import pytest


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "tools"))

import check_codex_subscription_auth as check_auth


def test_persisted_auth_matches_host_without_printing_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    auth = {"auth_mode": "chatgpt", "tokens": {"access_token": "secret"}}
    raw = json.dumps(auth).encode()
    encoded = base64.b64encode(gzip.compress(raw)).decode()
    env_file = tmp_path / ".env"
    env_file.write_text(f"{check_auth.AUTH_KEY}={encoded}\n")
    host_file = tmp_path / "auth.json"
    host_file.write_bytes(raw)
    monkeypatch.setattr(check_auth, "ENV_FILE", env_file)
    monkeypatch.setattr(check_auth, "HOST_AUTH_FILE", host_file)

    assert check_auth.verify_matches_host() == raw
    output = capsys.readouterr().out
    assert "matches the current login" in output
    assert "secret" not in output


def test_stale_persisted_auth_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = json.dumps({"tokens": {"access_token": "old"}}).encode()
    new = json.dumps({"tokens": {"access_token": "new"}}).encode()
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"{check_auth.AUTH_KEY}="
        f"{base64.b64encode(gzip.compress(old)).decode()}\n"
    )
    host_file = tmp_path / "auth.json"
    host_file.write_bytes(new)
    monkeypatch.setattr(check_auth, "ENV_FILE", env_file)
    monkeypatch.setattr(check_auth, "HOST_AUTH_FILE", host_file)

    with pytest.raises(RuntimeError, match="stale"):
        check_auth.verify_matches_host()
