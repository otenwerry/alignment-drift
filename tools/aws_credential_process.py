"""Cross-process AWS CLI credential bridge for long-running controllers.

Botocore's ``credential_process`` contract invokes this program whenever its cached
credentials approach expiration.  A file lock and a short-lived, mode-0600 cache make
that refresh a single-flight operation across concurrent campaign controllers.  This
prevents every controller from exchanging the same ``aws login`` session at once.

The credential JSON is intentionally written only to the process-provider stdout and
the protected cache.  Provider stdout/stderr are never included in this program's
errors because either stream may contain temporary credentials.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


CACHE_REFRESH_WINDOW = timedelta(minutes=2)
MINIMUM_EXPORT_LIFETIME = timedelta(seconds=15)
STATIC_CACHE_TTL = timedelta(minutes=5)
REQUIRED_FIELDS = ("AccessKeyId", "SecretAccessKey")


def _expiration(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _valid_credentials(
    value: object, *, now: datetime, minimum_remaining: timedelta
) -> dict | None:
    if not isinstance(value, dict) or value.get("Version") != 1:
        return None
    if any(not value.get(field) for field in REQUIRED_FIELDS):
        return None
    if value.get("Expiration") is None:
        return value
    expires = _expiration(value.get("Expiration"))
    if expires is None or expires <= now + minimum_remaining:
        return None
    return value


def _load_cache(path: Path, *, now: datetime) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(value, dict):
        return None
    credentials = _valid_credentials(
        value.get("credentials"),
        now=now,
        minimum_remaining=CACHE_REFRESH_WINDOW,
    )
    if credentials is None:
        return None
    if credentials.get("Expiration") is not None:
        return credentials
    cached_at = _expiration(value.get("cached_at"))
    if cached_at is None or cached_at <= now - STATIC_CACHE_TTL:
        return None
    return credentials


def _write_cache(path: Path, credentials: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                {
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                    "credentials": credentials,
                },
                stream,
                separators=(",", ":"),
            )
            stream.write("\n")
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def credentials(
    *, profile: str, aws_path: str, cache_path: Path, lock_path: Path
) -> dict:
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(lock_path.parent, 0o700)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        cached = _load_cache(cache_path, now=datetime.now(timezone.utc))
        if cached is not None:
            return cached
        result = subprocess.run(
            [
                aws_path,
                "configure",
                "export-credentials",
                "--profile",
                profile,
                "--format",
                "process",
                "--no-cli-pager",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "AWS CLI credential export failed; refresh the interactive "
                f"login for profile {profile!r}"
            )
        try:
            exported = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "AWS CLI credential export returned malformed process JSON"
            ) from error
        current = _valid_credentials(
            exported,
            now=datetime.now(timezone.utc),
            minimum_remaining=MINIMUM_EXPORT_LIFETIME,
        )
        if current is None:
            raise RuntimeError(
                "AWS CLI credential export returned missing, expired, or "
                "near-expiry credentials"
            )
        _write_cache(cache_path, current)
        return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--aws-path", required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    args = parser.parse_args()
    try:
        value = credentials(
            profile=args.profile,
            aws_path=args.aws_path,
            cache_path=args.cache_path,
            lock_path=args.lock_path,
        )
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(value, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
