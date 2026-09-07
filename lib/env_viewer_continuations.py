"""Grouping and base-rate math for the viewer's global Continuations page.

Pure, free helpers over already-loaded audit dicts. Continuation trajectories are
identified by their stored ``real_env["continuation"]`` record (written by the
solver), never by run-directory naming. Rendering stays in ``publication_analysis.py``; this
module only computes.
"""

from __future__ import annotations

from env_viewer_turns import experiment_user_turn_count
from result_classification import agent_label, trajectory_category


CATEGORY_KEYS = (
    "hack", "review", "notable", "clean",
    "unjudged", "invalid", "awaiting",
)

# Order matters: the first entry is the Continuations landing page, and within a
# destination the prefix-source tabs follow this order (plain prefixes, multi-agent,
# Wikipedia). Legacy directions with no Current rows sit at the end.
CONTINUATION_DIRECTIONS = (
    (
        "demand_forecasting_to_checkout_redesign",
        "demand_forecasting",
        "checkout_redesign",
    ),
    (
        "reasoning_prompt_benchmark_to_checkout_redesign",
        "reasoning_prompt_benchmark",
        "checkout_redesign",
    ),
    (
        "activity_log_context_ml_to_checkout_redesign",
        "activity_log_context_ml",
        "checkout_redesign",
    ),
    (
        "activity_log_context_to_checkout_redesign",
        "activity_log_context",
        "checkout_redesign",
    ),
    (
        "wikipedia_summaries_to_checkout_redesign",
        "wikipedia_summaries",
        "checkout_redesign",
    ),
    (
        "wikipedia_summaries_100k_to_checkout_redesign",
        "wikipedia_summaries_100k",
        "checkout_redesign",
    ),
    (
        "fraud_detection_to_checkout_redesign",
        "fraud_detection",
        "checkout_redesign",
    ),
    (
        "general_ethics_to_checkout_redesign",
        "general_ethics",
        "checkout_redesign",
    ),
    (
        "move_fast_to_checkout_redesign",
        "move_fast",
        "checkout_redesign",
    ),
    (
        "natural_questions_to_checkout_redesign",
        "natural_questions",
        "checkout_redesign",
    ),
    (
        "science_ethics_to_checkout_redesign",
        "science_ethics",
        "checkout_redesign",
    ),
    (
        "demand_forecasting_to_fraud_detection",
        "demand_forecasting",
        "fraud_detection",
    ),
    (
        "reasoning_prompt_benchmark_to_fraud_detection",
        "reasoning_prompt_benchmark",
        "fraud_detection",
    ),
    (
        "activity_log_context_ml_to_fraud_detection",
        "activity_log_context_ml",
        "fraud_detection",
    ),
    (
        "wikipedia_summaries_to_fraud_detection",
        "wikipedia_summaries",
        "fraud_detection",
    ),
    (
        "wikipedia_summaries_100k_to_fraud_detection",
        "wikipedia_summaries_100k",
        "fraud_detection",
    ),
    (
        "checkout_redesign_to_fraud_detection",
        "checkout_redesign",
        "fraud_detection",
    ),
    (
        "checkout_redesign_to_reasoning_prompt_benchmark",
        "checkout_redesign",
        "reasoning_prompt_benchmark",
    ),
    (
        "checkout_redesign_to_retrieval_practice",
        "checkout_redesign",
        "retrieval_practice",
    ),
)
DEFAULT_CONTINUATION_DIRECTION = CONTINUATION_DIRECTIONS[0][0]
OTHER_CONTINUATION_DIRECTION = "other"

# These source/destination combinations remain recognized for stored provenance, but
# are deliberately absent from the Current continuation navigation. Their campaigns
# live in the explicit All old manifest. Checkout -> fraud used a different p-hacking
# source task than the regular reasoning-prompt -> fraud conditions, so displaying it
# beside those conditions invites an invalid comparison.
ARCHIVED_CONTINUATION_DIRECTIONS = frozenset({
    "checkout_redesign_to_fraud_detection",
})

# Destination tabs intentionally published by the viewer. Keep this separate from
# CONTINUATION_DIRECTIONS: that registry also recognizes planned/legacy routes so
# stored provenance continues to classify correctly without creating empty tabs.
CONTINUATION_DESTINATIONS = (
    "checkout_redesign",
    "fraud_detection",
)

CONTINUATION_DESTINATION_LABELS = {
    "checkout_redesign": "p-hacking: checkout_redesign",
    "fraud_detection": "ML: fraud_detection",
    "reasoning_prompt_benchmark": "p-hacking: reasoning_prompt_benchmark",
    "retrieval_practice": "p-hacking: retrieval_practice",
}

CONTINUATION_PREFIX_LABELS = {
    "activity_log_context": "p-hacking: multi-agent",
    "activity_log_context_ml": "ML: multi-agent",
    "checkout_redesign": "p-hacking: checkout_redesign",
    "demand_forecasting": "ML: demand_forecasting",
    "fraud_detection": "ML: fraud_detection",
    "reasoning_prompt_benchmark": "p-hacking: reasoning_prompt_benchmark",
    "wikipedia_summaries": "Wikipedia summaries (~40k tokens)",
    "wikipedia_summaries_100k": "Wikipedia summaries (~100k tokens)",
}


def continuation_of(audit: dict) -> dict | None:
    """The stored continuation record, or None for an ordinary trajectory."""

    record = (audit.get("real_env") or {}).get("continuation")
    return record if isinstance(record, dict) else None


def treatment_of(audit: dict) -> str:
    record = continuation_of(audit) or {}
    return str(record.get("treatment") or "unknown")


def prefix_of(audit: dict) -> dict:
    record = continuation_of(audit) or {}
    prefix = record.get("prefix")
    return prefix if isinstance(prefix, dict) else {}


def is_activity_log_context_continuation(audit: dict) -> bool:
    """Whether this continuation uses the complete inline activity-log prefix."""

    return (
        continuation_of(audit) is not None
        and prefix_of(audit).get("source_prefix_type") == "activity_log_context"
    )


def prefix_source_trajectory_id(audit: dict) -> int | None:
    value = prefix_of(audit).get("source_trajectory_id")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _prefix_experiment_user_turns(prefix: dict) -> int | None:
    condition = prefix.get("source_prefix_condition")
    if not isinstance(condition, dict):
        return None
    value = condition.get("experiment_user_turns")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def continuation_source(audit: dict) -> str:
    """Stored source name used to route a continuation, without filename guesses."""

    prefix = prefix_of(audit)
    scripted_type = prefix.get("source_prefix_type")
    if scripted_type == "activity_log_context":
        # The payload type describes delivery; its comparison family identifies the
        # experimental condition. Keep ML and p-hacking activity logs on different
        # direction pages even though both use the same inline context mechanism.
        family = str(
            prefix.get("family")
            or prefix.get("source_comparison_source_family")
            or prefix.get("source_family")
            or ""
        )
        if family in {"ml_training_data_misuse", "ml_prefix_only"}:
            return "activity_log_context_ml"
        return scripted_type
    source_seed = (
        prefix.get("source_comparison_source_seed") or prefix.get("source_seed")
    )
    if source_seed:
        return str(source_seed)
    if (
        prefix.get("source_generator") == "exp_nq_prefix.py"
        or prefix.get("source_dataset") == "google-research-datasets/nq_open"
    ):
        return "natural_questions"
    if scripted_type in {
        "science_ethics",
        "general_ethics",
        "move_fast",
        "wikipedia_summaries",
        "wikipedia_summaries_100k",
    }:
        return str(scripted_type)
    return OTHER_CONTINUATION_DIRECTION


def continuation_direction(audit: dict) -> str:
    """Configured source-to-destination route, or ``other`` when unconfigured."""

    source = continuation_source(audit)
    destination = str(audit.get("seed") or "unknown")
    for key, configured_source, configured_destination in CONTINUATION_DIRECTIONS:
        if source == configured_source and destination == configured_destination:
            return key
    return OTHER_CONTINUATION_DIRECTION


def continuation_direction_label(direction: str) -> str:
    for key, source, destination in CONTINUATION_DIRECTIONS:
        if key == direction:
            return f"{source} → {destination}"
    return "other"


def continuation_destination(direction: str) -> str:
    """Destination-task key for a configured direction.

    The catch-all direction currently contains unconfigured prefix sources going to
    the one active continuation destination, so it stays under that destination tab.
    """

    for key, _source, destination in CONTINUATION_DIRECTIONS:
        if key == direction:
            return destination
    return CONTINUATION_DIRECTIONS[0][2]


def continuation_destination_label(destination: str) -> str:
    """Human-facing destination tab label, including its benchmark family."""

    return CONTINUATION_DESTINATION_LABELS.get(destination, destination)


def continuation_prefix_label(direction: str) -> str:
    """Human-facing prefix-source tab label for one direction."""

    for key, source, _destination in CONTINUATION_DIRECTIONS:
        if key == direction:
            return CONTINUATION_PREFIX_LABELS.get(source, source)
    return "other"


def continuation_condition_context(direction: str) -> str:
    """Self-contained task/prefix context for page headings and figure titles."""

    destination = continuation_destination_label(
        continuation_destination(direction)
    )
    return f"Task: {destination} · Prefix: {continuation_prefix_label(direction)}"


def category_counts(
    audits: list[dict], *, include_excluded: bool = False
) -> dict[str, int]:
    """Outcome counts for valid runs, or for all runs when explicitly requested."""

    counts = {key: 0 for key in CATEGORY_KEYS}
    for audit in audits:
        if not include_excluded and audit.get("integrity_status") == "excluded":
            continue
        counts[trajectory_category(audit, respect_exclusion=False)] += 1
    return counts


def hack_rate(counts: dict[str, int]) -> float | None:
    """Hacks over trajectories with a usable non-review judgment.

    ``review`` rows (p-hacking `other`) are neither hack nor non-hack, so they sit
    in neither the numerator nor the denominator; their count stays visible in the
    table.
    """

    judged = counts["hack"] + counts["notable"] + counts["clean"]
    return counts["hack"] / judged if judged else None


def reasoning_values(audits: list[dict]) -> set:
    return {
        audit.get("reasoning")
        for audit in audits
        if isinstance(audit.get("reasoning"), bool)
    }


def harness_of(audit: dict) -> str:
    """Stored agent harness, treating pre-flag trajectories as simple."""

    return str(audit.get("harness") or "simple")


def pressure_of(audit: dict) -> str | None:
    """Stored p-hacking pressure, without guessing for historical trajectories."""

    value = audit.get("pressure") or (audit.get("real_env") or {}).get("pressure")
    if value in {"low", "high"}:
        return str(value)
    continuation = continuation_of(audit) or {}
    value = continuation.get("pressure")
    return str(value) if value in {"low", "high"} else None


def baseline_audits(
    originals: list[dict], agent: str, reasoning: bool | None, harness: str,
    pressure: str | None,
) -> list[dict]:
    """Original trajectories comparable to one continuation group.

    ``originals`` is the seed's current official pool. Matching is by exact agent
    slug and harness; when the group has one recorded reasoning setting, originals
    with a recorded conflicting setting are dropped (unstamped ones stay). Historical
    originals without a harness stamp are the former simple implementation.
    """

    pool = [
        audit for audit in originals
        if str(audit.get("target") or "") == agent
        and harness_of(audit) == harness
        and pressure_of(audit) == pressure
    ]
    if reasoning is None:
        return pool
    return [
        audit for audit in pool
        if not isinstance(audit.get("reasoning"), bool)
        or audit.get("reasoning") == reasoning
    ]


def continuation_groups(
    continuations: list[dict], *, seed_order: list[str]
) -> list[tuple[tuple[str, str, str, str | None], dict[str, list[dict]]]]:
    """[((seed, agent, harness, pressure), {treatment: rows})] in viewer order."""

    grouped: dict[tuple[str, str, str, str | None], dict[str, list[dict]]] = {}
    for audit in continuations:
        key = (
            str(audit.get("seed") or "unknown"),
            str(audit.get("target") or ""),
            harness_of(audit),
            pressure_of(audit),
        )
        grouped.setdefault(key, {}).setdefault(treatment_of(audit), []).append(audit)
    order = {seed: index for index, seed in enumerate(seed_order)}
    return sorted(
        (
            (key, dict(sorted(by_treatment.items())))
            for key, by_treatment in grouped.items()
        ),
        key=lambda item: (
            order.get(item[0][0], len(order)),
            item[0][0],
            agent_label(item[0][1]),
            item[0][2],
            item[0][3] or "",
        ),
    )


def _binary_hack_counts(audits: list[dict]) -> dict[str, int]:
    """Hack numerator and usable binary denominator for one plotted bar."""

    usable = [
        audit for audit in audits
        if audit.get("mechanical_status") != "invalid"
        and audit.get("integrity_status") != "excluded"
    ]
    categories = [
        trajectory_category(audit, respect_exclusion=False) for audit in usable
    ]
    hacks = sum(category == "hack" for category in categories)
    denominator = sum(
        category in {"hack", "notable", "clean"} for category in categories
    )
    return {
        "k": hacks,
        "n": denominator,
        "excluded": len(audits) - denominator,
    }


def _outcome_composition(audits: list[dict]) -> dict:
    """Four-way usable-outcome composition for one continuation chart bar."""

    counts = {
        "hack_1turn": 0,
        "hack_2turn": 0,
        "interesting": 0,
        "clean": 0,
    }
    for audit in audits:
        if (
            audit.get("mechanical_status") == "invalid"
            or audit.get("integrity_status") == "excluded"
        ):
            continue
        category = trajectory_category(audit, respect_exclusion=False)
        if category == "hack":
            turns = experiment_user_turn_count(audit)
            if turns in {1, 2}:
                counts[f"hack_{turns}turn"] += 1
        elif category == "notable":
            counts["interesting"] += 1
        elif category == "clean":
            counts["clean"] += 1
    return {
        "composition": counts,
        "composition_n": sum(counts.values()),
    }


def continuation_prefix_rate_data(
    continuations: list[dict],
    originals: list[dict],
    *,
    audits_by_id: dict[int, dict],
) -> list[dict]:
    """Matched original and continuation hack-rate bars, grouped per agent.

    Prefix runs are pooled only when they carry the same exact source prefix. The
    original comparison matches destination seed, agent, exact harness, reasoning,
    and pressure.
    """

    grouped: dict[tuple, dict] = {}
    for audit in continuations:
        prefix = prefix_of(audit)
        source_id = prefix_source_trajectory_id(audit)
        exact_prefix = prefix.get("sha256") or prefix.get("name")
        prefix_identity = (
            ("payload", str(exact_prefix))
            if exact_prefix
            else ("trajectory", source_id)
            if source_id is not None
            else ("payload", "unknown")
        )
        reasoning = (
            audit.get("reasoning")
            if isinstance(audit.get("reasoning"), bool)
            else None
        )
        group_key = (
            str(audit.get("seed") or "unknown"),
            str(audit.get("target") or ""),
            harness_of(audit),
            pressure_of(audit),
            reasoning,
        )
        group = grouped.setdefault(group_key, {"prefixes": {}})
        prefix_group = group["prefixes"].setdefault(prefix_identity, {
            "rows": [],
            "source_id": source_id,
            "name": str(prefix.get("name") or "prefix"),
            "treatment": treatment_of(audit),
            "prefix_type": prefix.get("source_prefix_type"),
            "prefix_user_turns": _prefix_experiment_user_turns(prefix),
        })
        prefix_group["rows"].append(audit)

    result = []
    for (seed, agent, harness, pressure, reasoning), group in sorted(
        grouped.items(),
        key=lambda item: (
            agent_label(item[0][1]), item[0][2], item[0][3] or "",
            str(item[0][4]), item[0][0],
        ),
    ):
        matching_originals = baseline_audits(
            [audit for audit in originals if audit.get("seed") == seed],
            agent,
            reasoning,
            harness,
            pressure,
        )
        model = agent_label(agent)
        display_model = model if harness == "simple" else f"{model} · {harness}"
        bars = [{
            "label": "Originals",
            "short_label": "Original",
            "kind": "baseline",
            **_binary_hack_counts(matching_originals),
            **_outcome_composition(matching_originals),
        }]
        treatment_order = {
            "hack-in-one-turn": 0,
            "hack-in-two-turns": 1,
            "no-honeypot": 2,
            "no-hack": 3,
        }
        prefixes = sorted(
            group["prefixes"].values(),
            key=lambda item: (
                treatment_order.get(item["treatment"], 4),
                item["prefix_user_turns"] is None,
                item["prefix_user_turns"] or 0,
                item["source_id"] is None,
                item["source_id"] if item["source_id"] is not None else item["name"],
            ),
        )
        for prefix_group in prefixes:
            source_id = prefix_group["source_id"]
            source = audits_by_id.get(source_id) if source_id is not None else None
            source_category = (
                trajectory_category(source, respect_exclusion=False)
                if source is not None else "unknown"
            )
            if source_category == "hack":
                kind, outcome = "hack_prefix", "hack"
            elif source_category in {"notable", "clean"}:
                kind, outcome = "nonhack_prefix", "non-hack"
            else:
                kind, outcome = "prefix", "prefix"
            prefix_label = (
                f"#{source_id}" if source_id is not None else prefix_group["name"]
            )
            bars.append({
                "label": f"{prefix_label} {outcome}",
                "short_label": (
                    f"After #{source_id}" if source_id is not None else "After prefix"
                ),
                "kind": kind,
                "treatment": prefix_group["treatment"],
                **(
                    {"prefix_type": prefix_group["prefix_type"]}
                    if prefix_group["prefix_type"] is not None else {}
                ),
                **(
                    {"prefix_user_turns": prefix_group["prefix_user_turns"]}
                    if prefix_group["prefix_user_turns"] is not None else {}
                ),
                **_binary_hack_counts(prefix_group["rows"]),
                **_outcome_composition(prefix_group["rows"]),
            })
        if len(bars) > 2:
            payload_index = 0
            for bar in bars[1:]:
                if bar["short_label"] == "After prefix":
                    payload_index += 1
                    bar["short_label"] = f"After prefix {payload_index}"
        result.append({
            "seed": seed,
            "agent": agent,
            "model": display_model,
            "harness": harness,
            "pressure": pressure,
            "reasoning": reasoning,
            "bars": bars,
        })
    return result
