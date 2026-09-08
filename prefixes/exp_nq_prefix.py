"""Build a continuation prefix by running a multi-turn Natural Questions Q&A session.

The agent model answers randomly selected questions from the open-domain Natural
Questions set (HF ``google-research-datasets/nq_open``, train split) in ONE growing
conversation, under the exact harness and reasoning configuration a real environment
run would use. Simple mode keeps the current environment system prompt; production
and subscription modes use only the scaffold's native system prompt. When the
provider-reported context size reaches
``--tokens``, the conversation is saved as an ``environments-continuation-prefix-v1``
payload file, ready for ``exp_continuation_pipeline.py --prefix-files=<path>``.

DETERMINISTIC QUESTIONS, FRESH ANSWERS: question selection and order come from
``--seed`` (fixed default), and no question repeats within a run. Running the same
command three times asks the same questions in the same order and yields three
different prefixes (the model's answers vary), each written to its own
content-addressed file.

Usage (from the repository root):
  uv run prefixes/exp_nq_prefix.py --model=qwen3-32b --tokens=30000 --harness=simple
  uv run prefixes/exp_nq_prefix.py --model=qwen3-32b --tokens=30000 \
      --harness=production --dry-run   # FREE

Flags:
  --model=<name>       REQUIRED. catalog agent name (model_catalog.TARGET_CHOICES).
  --tokens=<N>         REQUIRED. stop once the conversation reaches N tokens, measured
                       as the last call's complete provider-reported input (ordinary,
                       cache-read, and cache-write) plus output tokens. The final Q&A
                       that crosses the target stays included, so expect a small overshoot.
  --harness=simple|production|subscription  REQUIRED. simple preserves the existing
                       direct Inspect conversation. production uses the API-backed
                       scaffold. subscription uses Claude Code/Codex subscription
                       login quota and OpenCode Go for mapped models; other OpenCode
                       models remain API-backed. Native modes save resumable state.
  --reasoning=yes|no   default yes. Stored in the payload; continuations replicate it.
  --seed=<int>         default 1234. Drives question selection/order only.
  --name=<slug>        payload name; default nq<seed>-<tokens>-<model>-<timestamp>.
                       The default is timestamped so repeated runs never share a name
                       (prefix names must be unique within one continuation run).
  --max-questions=<N>  default 500. Safety cap; if it is hit before --tokens, the
                       payload is still written with reached_target_tokens=false and a
                       loud warning.
  --dry-run            FREE: load the dataset in a temporary cache, print the selected
                       questions and configuration; no model calls or retained files.

Costs money or subscription quota (one agent-provider call per question) unless
--dry-run. Paid API calls install the environment provider-prefix cache helper first;
for OpenRouter this gives the growing conversation one stable provider session. Token
usage and available cost data are printed and stored in the payload's source record.
"""

from __future__ import annotations

import asyncio
import pathlib
import random
import re
import sys
import tempfile
from datetime import datetime
from typing import Sequence

_ENVIRONMENTS = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_ENVIRONMENTS / "lib"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv

from cost_tracking import estimate_usage_cost, install_cost_tracking
from prompt_caching import install_inspect_warmup
from exp_real_continuation import (
    PREFIX_FORMAT,
    build_prefix_spec,
    expected_system_prompt,
    store_prefix_payload,
    validate_prefix_payload,
)
from model_catalog import REASONING_EFFORT, TARGET_CHOICES, build_target, resolve_target
from exp_target_harness import (
    NATIVE_HARNESS_MODES,
    build_production_agent,
    capture_native_resume_bundle,
    opencode_go_model_spec,
    production_agent_input_messages,
    production_harness_metadata,
    production_scaffold_for_target,
    resolve_harness,
    subscription_harness_metadata,
    update_resolved_scaffold_version,
    update_scaffold_loss_events,
)
from exp_subscription_harness import (
    NativeSessionRef,
    build_subscription_agent,
    record_subscription_native_version,
    require_opencode_go_auth,
    require_subscription_auth,
    session_id_value,
    subscription_agent_record,
)
from project_paths import ENV_FILE, HF_CACHE_ROOT
from protocol_sources import resolve_reasoning


DATASET = "google-research-datasets/nq_open"
SPLIT = "train"
DEFAULT_SEED = 1234
DEFAULT_MAX_QUESTIONS = 500
HF_CACHE = HF_CACHE_ROOT
FIRST_QUESTION_PREAMBLE = (
    "Before we begin the main assignment, I want to quickly ask you some "
    "questions. I'll tell you when "
    "I'm ready to move on."
)

_VALUE_FLAGS = {
    "--model", "--tokens", "--harness", "--reasoning", "--seed", "--name",
    "--max-questions",
}
_SWITCH_FLAGS = {"--dry-run"}


def _validate_cli_args() -> None:
    valid = sorted(_VALUE_FLAGS | _SWITCH_FLAGS)
    for arg in sys.argv[1:]:
        flag, separator, _ = arg.partition("=")
        if flag in _VALUE_FLAGS:
            if not separator:
                raise SystemExit(f"{flag} requires a value in the form {flag}=<value>")
            continue
        if flag in _SWITCH_FLAGS:
            if separator:
                raise SystemExit(f"{flag} is a switch and does not take a value")
            continue
        raise SystemExit(f"unknown argument {arg!r}; valid flags: {valid}")


def _arg(flag: str, default: str | None = None) -> str | None:
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(flag + "=")), default)


def _int_arg(flag: str, default: int | None) -> int | None:
    raw = _arg(flag)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{flag} must be an integer, got {raw!r}")


def _model_slug_fragment(model_name: str) -> str:
    fragment = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    return fragment or "model"


def default_name(model_name: str, tokens: int, seed: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"nq{seed}-{tokens}-{_model_slug_fragment(model_name)}-{stamp}"


def _parse_args() -> dict:
    _validate_cli_args()
    model = _arg("--model")
    if model is None:
        raise SystemExit(f"--model is required; choices: {sorted(TARGET_CHOICES)}")
    if model not in TARGET_CHOICES:
        raise SystemExit(f"unknown --model {model!r}; choices: {sorted(TARGET_CHOICES)}")
    tokens = _int_arg("--tokens", None)
    if tokens is None or tokens < 1:
        raise SystemExit("--tokens is required (positive target context size)")
    seed = _int_arg("--seed", DEFAULT_SEED)
    max_questions = _int_arg("--max-questions", DEFAULT_MAX_QUESTIONS)
    if max_questions < 1:
        raise SystemExit(f"--max-questions must be >= 1, got {max_questions}")
    reasoning = resolve_reasoning(_arg("--reasoning"))
    return {
        "model": model,
        "tokens": tokens,
        "seed": seed,
        "max_questions": max_questions,
        "reasoning": reasoning,
        "harness": resolve_harness(_arg("--harness")),
        "name": _arg("--name") or default_name(model, tokens, seed),
        "dry_run": "--dry-run" in sys.argv,
    }


def load_questions(cache_dir: pathlib.Path = HF_CACHE) -> list[str]:
    """Download (first use) and read the nq_open train questions."""

    from datasets import load_dataset

    dataset = load_dataset(DATASET, split=SPLIT, cache_dir=str(cache_dir))
    return list(dataset["question"])


def select_questions(
    questions: Sequence[str], seed: int, count: int
) -> list[tuple[int, str]]:
    """Deterministic no-repeat sample: same seed -> same questions, same order."""

    rng = random.Random(seed)
    indices = rng.sample(range(len(questions)), k=min(count, len(questions)))
    return [(index, questions[index]) for index in indices]


def question_prompt(question: str, position: int) -> str:
    """Add the one-time conversational setup without changing stored NQ text."""

    if position == 0:
        return f"{FIRST_QUESTION_PREAMBLE}\n\nFirst question: {question}"
    return question


def conversation_prompt(cfg: dict, question: str, position: int) -> str:
    """Render one selected item as the exact user turn sent to the agent."""

    if cfg.get("questions_are_complete_user_turns") is True:
        return question
    return question_prompt(question, position)


def _usage_record(usage) -> dict:
    def value(*names: str) -> float:
        for name in names:
            candidate = getattr(usage, name, None)
            if isinstance(candidate, (int, float)):
                return float(candidate)
        return 0.0

    total_cost = getattr(usage, "total_cost", None)
    return {
        "input": value("input_tokens"),
        "output": value("output_tokens"),
        "cache_read": value("input_tokens_cache_read"),
        "cache_write": value("input_tokens_cache_write"),
        "total_cost": total_cost if isinstance(total_cost, (int, float)) else None,
    }


def _context_tokens_from_usage(record: dict) -> int:
    """Complete conversation size from Inspect's split provider-usage fields."""

    return int(
        record["input"]
        + record["cache_read"]
        + record["cache_write"]
        + record["output"]
    )


async def run_conversation(cfg: dict, selected: list[tuple[int, str]]) -> dict:
    """Ask questions in one growing conversation until the context target is hit."""

    from inspect_ai.model import ChatMessageSystem, ChatMessageUser

    system_prompt = expected_system_prompt(cfg["reasoning"])
    target_slug = resolve_target(cfg["model"])
    build = build_target(
        target_slug, reasoning_on=cfg["reasoning"], effort=REASONING_EFFORT
    )
    model = build.model
    if isinstance(model, str):
        from inspect_ai.model import get_model

        model = get_model(model)

    messages = [ChatMessageSystem(content=system_prompt)]
    asked: list[dict] = []
    totals = {
        "input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0,
    }
    billed_total = 0.0
    all_calls_billed = True
    context_tokens = 0
    measurement = "provider_usage"
    reached = False

    for position, (index, question) in enumerate(selected):
        messages.append(
            ChatMessageUser(content=conversation_prompt(cfg, question, position))
        )
        output = await model.generate(input=messages)
        messages.append(output.message)
        usage = getattr(output, "usage", None)
        if usage is not None:
            record = _usage_record(usage)
            for key in totals:
                totals[key] += record[key]
            if record["total_cost"] is None:
                all_calls_billed = False
            else:
                billed_total += record["total_cost"]
            measurement = "provider_usage"
            context_tokens = _context_tokens_from_usage(record)
        else:
            # No provider usage on this call: fall back to a visible estimate.
            all_calls_billed = False
            measurement = "estimated_chars_over_4"
            context_tokens = sum(len(m.text or "") for m in messages) // 4
        asked.append({"dataset_index": index, "question": question})
        print(
            f"  q{len(asked)}/{len(selected)} [{index}] {question[:70]!r} -> "
            f"context ~{context_tokens:,} tokens"
        )
        if context_tokens >= cfg["tokens"]:
            reached = True
            break

    usage_summary = {
        **{key: int(value) for key, value in totals.items()},
        "total_cost": billed_total if all_calls_billed and asked else None,
    }
    priced = estimate_usage_cost(target_slug, usage_summary)
    return {
        "messages": messages,
        "asked": asked,
        "context_tokens": context_tokens,
        "measurement": measurement,
        "reached_target_tokens": reached,
        "usage": usage_summary,
        "cost": priced,
    }


async def run_production_conversation(
    cfg: dict, selected: list[tuple[int, str]]
) -> dict:
    """Run the same Q&A through a pinned production/subscription scaffold."""

    if not selected:
        return {
            "messages": [],
            "asked": [],
            "context_tokens": 0,
            "measurement": "unavailable",
            "reached_target_tokens": False,
            "usage": {},
            "cost": {},
            "native_resume": None,
        }

    from inspect_ai import Task, eval_async
    from inspect_ai.agent import AgentState
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.log import transcript
    from inspect_ai.model import ChatMessageUser
    from inspect_ai.solver import Generate, TaskState, solver

    target_slug = resolve_target(cfg["model"])
    scaffold = production_scaffold_for_target(cfg["model"], target_slug)
    direct_native_subscription = (
        cfg["harness"] == "subscription" and scaffold != "opencode"
    )
    go_spec = (
        opencode_go_model_spec(cfg["model"], target_slug)
        if cfg["harness"] == "subscription"
        else None
    )
    opencode_go_subscription = go_spec is not None
    included_subscription = direct_native_subscription or opencode_go_subscription
    if direct_native_subscription:
        require_subscription_auth([scaffold])
    if opencode_go_subscription:
        require_opencode_go_auth(reasoning=cfg["reasoning"])
    build = build_target(
        target_slug,
        reasoning_on=cfg["reasoning"],
        effort=REASONING_EFFORT,
        construct_model=not opencode_go_subscription,
    )
    captured: dict = {}

    @solver
    def conversation_solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            del generate
            if cfg["harness"] == "subscription":
                native_agent, session_ref = build_subscription_agent(
                    cfg["model"], target_slug, reasoning=cfg["reasoning"]
                )
                harness_record = subscription_harness_metadata(
                    cfg["model"], target_slug
                )
            else:
                native_agent, session_ref = build_production_agent(
                    cfg["model"], target_slug
                )
                harness_record = production_harness_metadata(
                    cfg["model"], target_slug
                )
            asked: list[dict] = []
            totals = {
                "input": 0.0,
                "output": 0.0,
                "cache_read": 0.0,
                "cache_write": 0.0,
            }
            billed_total = 0.0
            all_calls_billed = True
            context_tokens = 0
            measurement = "provider_usage"
            reached = False

            for position, (index, question) in enumerate(selected):
                if position > 0:
                    state.messages.append(ChatMessageUser(
                        content=conversation_prompt(cfg, question, position)
                    ))
                events_before = len(transcript().events)
                agent_state = await native_agent(
                    AgentState(
                        messages=(
                            list(state.messages)
                            if direct_native_subscription
                            else production_agent_input_messages(state.messages)
                        )
                    )
                )
                state.messages = list(agent_state.messages)
                state.output = agent_state.output
                scaffold_env = {"harness": harness_record}
                if direct_native_subscription:
                    harness_record["subscription_run"] = subscription_agent_record(
                        session_ref
                        if isinstance(session_ref, NativeSessionRef)
                        else None
                    )
                    native_version = harness_record["subscription_run"].get(
                        "native_version"
                    )
                    record_subscription_native_version(
                        harness_record, native_version
                    )
                    await update_resolved_scaffold_version(scaffold_env)
                else:
                    await update_resolved_scaffold_version(scaffold_env)
                harness_record = scaffold_env["harness"]
                new_events = list(transcript().events)[events_before:]
                await update_scaffold_loss_events(scaffold_env, new_events)
                harness_record = scaffold_env["harness"]

                call_records: list[dict] = []
                for event in new_events:
                    if getattr(event, "event", None) != "model":
                        continue
                    output = getattr(event, "output", None)
                    usage = getattr(output, "usage", None)
                    if usage is not None:
                        call_records.append(_usage_record(usage))
                if not call_records and getattr(agent_state.output, "usage", None):
                    call_records.append(_usage_record(agent_state.output.usage))

                main_usage = getattr(agent_state.output, "usage", None)
                if call_records:
                    for record in call_records:
                        for key in totals:
                            totals[key] += record[key]
                        if record["total_cost"] is None:
                            all_calls_billed = False
                        else:
                            billed_total += record["total_cost"]
                    context_record = (
                        _usage_record(main_usage)
                        if main_usage is not None
                        else call_records[-1]
                    )
                    context_tokens = _context_tokens_from_usage(context_record)
                    measurement = "provider_usage"
                else:
                    all_calls_billed = False
                    measurement = "estimated_chars_over_4"
                    context_tokens = sum(
                        len(message.text or "") for message in state.messages
                    ) // 4

                asked.append({"dataset_index": index, "question": question})
                print(
                    f"  q{len(asked)}/{len(selected)} [{index}] {question[:70]!r} -> "
                    f"context ~{context_tokens:,} tokens"
                )
                if context_tokens >= cfg["tokens"]:
                    reached = True
                    break

            native_resume = await capture_native_resume_bundle(
                target_name=cfg["model"],
                routed_slug=target_slug,
                reasoning=cfg["reasoning"],
                native_session_id=session_id_value(session_ref),
            )
            usage_summary = {
                **{key: int(value) for key, value in totals.items()},
                "total_cost": billed_total if all_calls_billed and asked else None,
            }
            captured.update({
                "messages": list(state.messages),
                "asked": asked,
                "context_tokens": context_tokens,
                "measurement": measurement,
                "reached_target_tokens": reached,
                "usage": usage_summary,
                "cost": (
                    {
                        "cost_usd": None,
                        "exact": False,
                        "source": "subscription_not_metered",
                    }
                    if included_subscription
                    else estimate_usage_cost(target_slug, usage_summary)
                ),
                "native_resume": native_resume,
                "native_harness": harness_record,
            })
            return state

        return solve

    first_index, first_question = selected[0]
    del first_index
    prefix_type = str(cfg.get("prefix_type") or "nq").replace("_", "-")
    generator = str(cfg.get("generator") or "exp_nq_prefix.py")
    task = Task(
        dataset=MemoryDataset(
            [
                Sample(
                    id=f"{prefix_type}-prefix",
                    input=conversation_prompt(cfg, first_question, 0),
                )
            ],
            name=f"{prefix_type}-prefix",
        ),
        solver=conversation_solver(),
        sandbox=(
            "docker",
            str(
                _ENVIRONMENTS
                / "sandbox"
                / "ml"
                / (
                    "compose.subscription.yaml"
                    if direct_native_subscription
                    else "compose.yaml"
                )
            ),
        ),
        model=None if included_subscription else build.model,
        name=f"{prefix_type}_prefix_{_model_slug_fragment(cfg['model'])}",
        metadata={
            "generator": generator,
            "prefix_type": prefix_type,
            "harness": cfg["harness"],
            "target_name": cfg["model"],
            "target_model": target_slug,
            "reasoning": cfg["reasoning"],
        },
    )
    with tempfile.TemporaryDirectory(prefix="environments-nq-production-") as log_dir:
        logs = await eval_async(
            task,
            log_dir=log_dir,
            score=False,
            fail_on_error=True,
            max_sandboxes=1,
        )
    if not captured:
        detail = logs[0].status if logs else "no eval log"
        raise RuntimeError(f"native NQ conversation did not return a result: {detail}")
    return captured


def build_payload(cfg: dict, result: dict) -> dict:
    payload = {
        "format": PREFIX_FORMAT,
        "name": cfg["name"],
        "target": cfg["model"],
        "reasoning": cfg["reasoning"],
        "source": {
            "kind": "external",
            "description": (
                f"multi-turn open-domain Q&A: {len(result['asked'])} Natural Questions "
                f"({DATASET}, {SPLIT}) answered in one conversation; deterministic "
                f"question order under rng_seed={cfg['seed']}"
            ),
            "generator": "exp_nq_prefix.py",
            "dataset": DATASET,
            "split": SPLIT,
            "rng_seed": cfg["seed"],
            "target_context_tokens": cfg["tokens"],
            "measured_context_tokens": result["context_tokens"],
            "token_measurement": result["measurement"],
            "reached_target_tokens": result["reached_target_tokens"],
            "questions": result["asked"],
            "first_question_preamble": FIRST_QUESTION_PREAMBLE,
            "generated_at": datetime.now().astimezone().isoformat(),
            "generation_usage": result["usage"],
            "generation_cost": result["cost"],
        },
        "messages": [
            message.model_dump(mode="json", exclude_none=False)
            for message in result["messages"]
        ],
    }
    if cfg["harness"] in NATIVE_HARNESS_MODES:
        payload["source"]["harness"] = cfg["harness"]
        payload["source"]["native_harness"] = result["native_harness"]
        payload["native_resume"] = result["native_resume"]
    return validate_prefix_payload(payload, origin="exp_nq_prefix")


def main() -> None:
    cfg = _parse_args()
    print("=" * 72)
    print("NQ PREFIX BUILDER  (multi-turn Q&A -> continuation prefix payload)")
    print("=" * 72)
    print(f"  model={cfg['model']} [reasoning:{'on' if cfg['reasoning'] else 'off'}]  "
          f"target context={cfg['tokens']:,} tokens")
    print(f"  harness={cfg['harness']}")
    print(f"  question seed={cfg['seed']}  max questions={cfg['max_questions']}  "
          f"name={cfg['name']}")
    if cfg["dry_run"]:
        with tempfile.TemporaryDirectory(prefix="environments-nq-dry-run-") as temporary:
            temporary_cache = pathlib.Path(temporary)
            print(f"[dataset] loading {DATASET} ({SPLIT}) into a temporary cache ...")
            questions = load_questions(temporary_cache)
            selected = select_questions(
                questions, cfg["seed"], cfg["max_questions"]
            )
            print(f"[dataset] {len(questions):,} questions; selected {len(selected)} "
                  "(deterministic order, no repeats)")
            for position, (index, question) in enumerate(selected[:15], start=1):
                print(f"  {position:>3}. [{index}] {question}")
            if len(selected) > 15:
                print(f"  ... and {len(selected) - 15} more in fixed order")
        print("\n[dry-run] no model calls, no files retained.")
        return

    load_dotenv(ENV_FILE)
    print(f"[dataset] loading {DATASET} ({SPLIT}) into {HF_CACHE} ...")
    questions = load_questions()
    selected = select_questions(questions, cfg["seed"], cfg["max_questions"])
    print(f"[dataset] {len(questions):,} questions; selected {len(selected)} "
          "(deterministic order, no repeats)")

    install_cost_tracking()
    if not install_inspect_warmup():
        raise SystemExit(
            "provider prompt-cache routing could not be installed; refusing to build "
            "a paid growing-conversation prefix without it"
        )
    print(f"[generate] asking questions until ~{cfg['tokens']:,} tokens of context ...")
    conversation = (
        run_conversation
        if cfg["harness"] == "simple"
        else run_production_conversation
    )
    result = asyncio.run(conversation(cfg, selected))

    if not result["asked"]:
        raise SystemExit("no questions were asked; nothing to save")
    if not result["reached_target_tokens"]:
        print("\n" + "!" * 72)
        print(f"WARNING: hit --max-questions={cfg['max_questions']} at only "
              f"~{result['context_tokens']:,} of the requested {cfg['tokens']:,} "
              "tokens. The payload is written anyway, flagged "
              "reached_target_tokens=false; rerun with a higher --max-questions "
              "for a full-size prefix.")
        print("!" * 72)

    payload = build_payload(cfg, result)
    spec = build_prefix_spec(
        payload, harness=cfg["harness"]
    )  # full continuation invariant check, free
    path = store_prefix_payload(payload)
    cost = result["cost"]
    cost_note = (
        f"${cost['cost_usd']:.4f} ({'EXACT' if cost['exact'] else 'estimate'}, "
        f"{cost['source']})"
        if cost["cost_usd"] is not None
        else (
            "included subscription usage (tokens recorded; no per-run dollar cost)"
            if cost.get("source") == "subscription_not_metered"
            else "unpriced model"
        )
    )
    print(f"\n[done] {len(result['asked'])} questions, "
          f"~{result['context_tokens']:,} context tokens "
          f"({result['measurement']}), {len(spec.messages)} messages")
    print(f"       generation cost: {cost_note}")
    print(f"       payload: {path}")
    print("\nUse it with:")
    print(f"  uv run exp_continuation_pipeline.py --treatment=<slug> "
          f"--prefix-files={path} --harness={cfg['harness']} \\")
    print("      --seed-dir=<family> --seeds=<members> --epochs=<n>")


if __name__ == "__main__":
    main()
