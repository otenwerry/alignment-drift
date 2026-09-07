"""Small, behavior-stable CLI primitives shared by experiment endpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def validate_cli_args(
    argv: Sequence[str],
    *,
    value_flags: set[str],
    switch_flags: set[str],
    rejected_flags: Mapping[str, str] | None = None,
) -> None:
    valid = sorted(value_flags | switch_flags)
    rejected_flags = rejected_flags or {}
    for argument in argv[1:]:
        flag, separator, _ = argument.partition("=")
        if flag in rejected_flags:
            raise SystemExit(rejected_flags[flag])
        if flag in value_flags:
            if not separator:
                raise SystemExit(
                    f"{flag} requires a value in the form {flag}=<value>"
                )
            continue
        if flag in switch_flags:
            if separator:
                raise SystemExit(f"{flag} is a switch and does not take a value")
            continue
        raise SystemExit(f"unknown argument {argument!r}; valid flags: {valid}")


def value_arg(
    argv: Sequence[str], flag: str, default: str | None = None
) -> str | None:
    prefix = flag + "="
    return next(
        (argument.split("=", 1)[1] for argument in argv if argument.startswith(prefix)),
        default,
    )


def positive_int_arg(
    argv: Sequence[str], flag: str, default: int | None
) -> int | None:
    raw = value_arg(argv, flag)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{flag} must be an integer, got {raw!r}")
    if value < 1:
        raise SystemExit(f"{flag} must be >= 1, got {value}")
    return value
