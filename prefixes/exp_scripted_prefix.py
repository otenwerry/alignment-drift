"""Shared paid runner for short, fixed-question continuation prefixes.

Sibling ``exp_*_prefix.py`` endpoints provide the actual question scripts. This
module deliberately reuses the NQ conversation engine so simple, production, and
subscription prefixes have the same system-prompt, native-resume, usage, cost, and
provider-cache behavior.
"""

from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv

from prefixes import exp_nq_prefix as nq
from cost_tracking import install_cost_tracking
from exp_real_continuation import (
    PREFIX_FORMAT,
    build_prefix_spec,
    store_prefix_payload,
    validate_prefix_payload,
)
from exp_target_harness import NATIVE_HARNESS_MODES, resolve_harness
from model_catalog import TARGET_CHOICES
from project_paths import ENV_FILE
from prompt_caching import install_inspect_warmup
from protocol_sources import resolve_reasoning


_NO_TOKEN_STOP = sys.maxsize
_VALUE_FLAGS = {"--model", "--harness", "--reasoning", "--name"}
_SWITCH_FLAGS = {"--dry-run"}


@dataclass(frozen=True)
class ScriptedPrefix:
    slug: str
    label: str
    entrypoint: str
    description: str
    questions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.slug):
            raise ValueError(f"invalid scripted-prefix slug: {self.slug!r}")
        if not self.questions or any(not question.strip() for question in self.questions):
            raise ValueError(f"{self.slug}: questions must be non-empty strings")

    @property
    def prefix_type(self) -> str:
        return self.slug.replace("-", "_")


def _validate_cli_args() -> None:
    valid = sorted(_VALUE_FLAGS | _SWITCH_FLAGS)
    for argument in sys.argv[1:]:
        flag, separator, _value = argument.partition("=")
        if flag in _VALUE_FLAGS:
            if not separator:
                raise SystemExit(
                    f"{flag} requires a value in the form {flag}=<value>"
                )
            continue
        if flag in _SWITCH_FLAGS:
            if separator:
                raise SystemExit(f"{flag} is a switch and does not take a value")
            continue
        raise SystemExit(f"unknown argument {argument!r}; valid flags: {valid}")


def _arg(flag: str, default: str | None = None) -> str | None:
    return next(
        (
            argument.split("=", 1)[1]
            for argument in sys.argv
            if argument.startswith(flag + "=")
        ),
        default,
    )


def _model_slug_fragment(model_name: str) -> str:
    fragment = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    return fragment or "model"


def default_name(profile: ScriptedPrefix, model_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{profile.slug}-{_model_slug_fragment(model_name)}-{stamp}"


def parse_args(profile: ScriptedPrefix) -> dict:
    _validate_cli_args()
    model = _arg("--model")
    if model is None:
        raise SystemExit(f"--model is required; choices: {sorted(TARGET_CHOICES)}")
    if model not in TARGET_CHOICES:
        raise SystemExit(
            f"unknown --model {model!r}; choices: {sorted(TARGET_CHOICES)}"
        )
    return {
        "model": model,
        "harness": resolve_harness(_arg("--harness")),
        "reasoning": resolve_reasoning(_arg("--reasoning")),
        "name": _arg("--name") or default_name(profile, model),
        "dry_run": "--dry-run" in sys.argv,
        # The reused NQ engine stops only after every fixed question because this
        # prefix type is defined by a complete script, not a target token count.
        "tokens": _NO_TOKEN_STOP,
        "generator": profile.entrypoint,
        "prefix_type": profile.prefix_type,
    }


def selected_questions(profile: ScriptedPrefix) -> list[tuple[int, str]]:
    return list(enumerate(profile.questions))


def build_payload(profile: ScriptedPrefix, cfg: dict, result: dict) -> dict:
    asked = result["asked"]
    asked_questions = [item.get("question") for item in asked]
    completed = asked_questions == list(profile.questions)
    payload = {
        "format": PREFIX_FORMAT,
        "name": cfg["name"],
        "target": cfg["model"],
        "reasoning": cfg["reasoning"],
        "source": {
            "kind": "external",
            "description": profile.description,
            "generator": profile.entrypoint,
            "prefix_type": profile.prefix_type,
            "prefix_type_label": profile.label,
            "source_label": profile.label,
            "harness": cfg["harness"],
            "script_version": 1,
            "fixed_question_order": True,
            "completed_script": completed,
            "questions": [
                {"position": position, "question": question}
                for position, question in enumerate(profile.questions, start=1)
            ],
            "first_question_preamble": nq.FIRST_QUESTION_PREAMBLE,
            "generated_at": datetime.now().astimezone().isoformat(),
            "measured_context_tokens": result["context_tokens"],
            "token_measurement": result["measurement"],
            "generation_usage": result["usage"],
            "generation_cost": result["cost"],
        },
        "messages": [
            message.model_dump(mode="json", exclude_none=False)
            for message in result["messages"]
        ],
    }
    if cfg["harness"] in NATIVE_HARNESS_MODES:
        payload["source"]["native_harness"] = result["native_harness"]
        payload["native_resume"] = result["native_resume"]
    return validate_prefix_payload(payload, origin=profile.entrypoint)


def _cost_note(cost: dict) -> str:
    amount = cost.get("cost_usd")
    if isinstance(amount, (int, float)):
        return (
            f"${amount:.4f} "
            f"({'EXACT' if cost.get('exact') else 'estimate'}, "
            f"{cost.get('source') or 'unknown source'})"
        )
    if cost.get("source") == "subscription_not_metered":
        return "included subscription usage (tokens recorded; no per-run dollar cost)"
    return "unpriced model"


def main(profile: ScriptedPrefix) -> None:
    cfg = parse_args(profile)
    questions = selected_questions(profile)
    print("=" * 72)
    print(f"{profile.label.upper()} PREFIX  (fixed conversation -> prefix payload)")
    print("=" * 72)
    print(
        f"  model={cfg['model']} [reasoning:"
        f"{'on' if cfg['reasoning'] else 'off'}]  harness={cfg['harness']}"
    )
    print(f"  questions={len(questions)}  name={cfg['name']}")
    for position, (_index, question) in enumerate(questions, start=1):
        rendered = nq.question_prompt(question, position - 1)
        print(f"\n  USER {position}\n  {rendered}")

    if cfg["dry_run"]:
        print("\n[dry-run] no model calls and no files written.")
        return

    load_dotenv(ENV_FILE)
    install_cost_tracking()
    if not install_inspect_warmup():
        raise SystemExit(
            "provider prompt-cache routing could not be installed; refusing to build "
            "a paid growing-conversation prefix without it"
        )
    print(f"\n[generate] asking all {len(questions)} fixed questions ...")
    conversation = (
        nq.run_conversation
        if cfg["harness"] == "simple"
        else nq.run_production_conversation
    )
    result = asyncio.run(conversation(cfg, questions))
    asked_questions = [
        item.get("question") for item in (result.get("asked") or [])
    ]
    if asked_questions != list(profile.questions):
        raise SystemExit(
            "conversation did not answer the complete fixed script in order; "
            "refusing to save it"
        )

    payload = build_payload(profile, cfg, result)
    spec = build_prefix_spec(payload, harness=cfg["harness"])
    path = store_prefix_payload(payload)
    print(
        f"\n[done] {len(questions)} questions, "
        f"~{result['context_tokens']:,} context tokens "
        f"({result['measurement']}), {len(spec.messages)} messages"
    )
    print(f"       generation cost: {_cost_note(result['cost'])}")
    print(f"       payload: {path}")
    print("\nUse it with:")
    print(
        "  uv run exp_continuation_pipeline.py --treatment=<slug> "
        f"--prefix-files={path} --harness={cfg['harness']} \\"
    )
    print("      --seed-dir=<family> --seeds=<members> --epochs=<n>")
