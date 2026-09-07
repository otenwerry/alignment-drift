"""Curated, data-backed inputs for the hand-designed blog figure viewer.

Use the viewer's filtered binary denominator for every bar and select every
continuation from exact stored experiment metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from env_viewer_continuations import (
    continuation_direction,
    continuation_of,
    harness_of,
    prefix_of,
    pressure_of,
    treatment_of,
)
from result_classification import agent_label, trajectory_category


TARGET_N = 40
MODEL_ORDER = (
    "gpt-5.5",
    "deepseek-v4-pro",
    "glm-5.1",
    "kimi-k2.6",
)


@dataclass(frozen=True)
class PointSpec:
    condition: str
    treatment: str
    prefix_name: str
    direction: str | None = None


@dataclass(frozen=True)
class PanelSpec:
    key: str
    title: str
    source: str
    direction: str
    seed: str
    pressure: str | None


def _point(
    condition: str,
    treatment: str,
    prefix_name: str,
    direction: str | None = None,
) -> PointSpec:
    return PointSpec(condition, treatment, prefix_name, direction)


DEMAND_POINTS = {
    "gpt-5.5": (
        _point("hack2", "hack-in-two-turns", "traj7420"),
        _point("clean2", "no-hack", "traj783"),
        _point(
            "noHoneypot",
            "no-honeypot",
            "next-open-continuations-20260817-20260817014146-demand-gpt-"
            "gpt-5-5-demand-forecasting-e5",
        ),
    ),
    "deepseek-v4-pro": (
        _point("hack1", "hack-in-one-turn", "traj823"),
        _point("hack2", "hack-in-two-turns", "traj829"),
        _point("clean2", "no-hack", "traj824"),
        _point(
            "noHoneypot",
            "no-honeypot",
            "demand-phacking-to-ml-20260816-20260816215128-demand-production-"
            "deepseek-v4-pro-demand-forecasting-e1",
        ),
    ),
    "glm-5.1": (
        _point("hack1", "hack-in-one-turn", "traj831"),
        _point("clean2", "no-hack", "traj7764"),
        _point(
            "noHoneypot",
            "no-honeypot",
            "demand-phacking-to-ml-20260816-20260816215128-demand-production-"
            "glm-5-1-demand-forecasting-e1",
        ),
    ),
    "kimi-k2.6": (
        _point("hack1", "hack-in-one-turn", "traj840"),
        _point("hack2", "hack-in-two-turns", "traj842"),
        _point("clean2", "no-hack", "traj837"),
        _point(
            "noHoneypot",
            "no-honeypot",
            "demand-phacking-to-ml-20260816-20260816215128-demand-production-"
            "kimi-k2-6-demand-forecasting-e8",
        ),
    ),
}


REASONING_POINTS = {
    "gpt-5.5": (
        _point("hack2", "hack-in-two-turns", "traj750"),
        _point("clean", "no-hack", "traj757"),
        _point(
            "noHoneypot",
            "no-honeypot",
            "next-open-continuations-20260817-20260817014146-reasoning-"
            "nohoneypot-gpt-gpt-5-5-reasoning-prompt-benchmark-positive-e1",
        ),
    ),
    "deepseek-v4-pro": (
        _point("hack1", "hack-in-one-turn", "traj711"),
        _point("hack2", "hack-in-two-turns", "traj710"),
        _point(
            "noHoneypot",
            "no-honeypot",
            "reasoning-positive-open-n3-deepseek-v4-pro-reasoning-prompt-"
            "benchmark-positive-e1",
        ),
    ),
    "glm-5.1": (
        _point("hack1", "hack-in-one-turn", "traj715"),
        _point("hack2", "hack-in-two-turns", "traj716"),
        _point("clean", "no-hack", "traj718"),
        _point(
            "noHoneypot",
            "no-honeypot",
            "reasoning-positive-open-n3-glm-5-1-reasoning-prompt-benchmark-"
            "positive-e1",
        ),
    ),
    "kimi-k2.6": (
        _point("hack1", "hack-in-one-turn", "traj723"),
        _point("hack2", "hack-in-two-turns", "traj722"),
        _point("clean", "no-hack", "traj720"),
        _point(
            "noHoneypot",
            "no-honeypot",
            "reasoning-positive-open-n3-kimi-k2-6-reasoning-prompt-benchmark-"
            "positive-e1",
        ),
    ),
}


ACTIVITY_DA_POINTS = {
    model: tuple(
        PointSpec(
            "clean2" if point.condition == "clean" else point.condition,
            point.treatment,
            f"activity-log-{point.prefix_name}",
        )
        for point in points
        if point.condition != "noHoneypot"
    )
    for model, points in REASONING_POINTS.items()
}


ACTIVITY_ML_POINTS = {
    model: tuple(
        PointSpec(point.condition, point.treatment, f"activity-log-{point.prefix_name}")
        for point in points
        if point.condition != "noHoneypot"
    )
    for model, points in DEMAND_POINTS.items()
}


WIKIPEDIA_POINTS = {
    "gpt-5.5": (
        _point(
            "prefix40", "wikipedia-summaries",
            "wikipedia-summaries-gpt-5-5-20260813152916",
        ),
        _point(
            "prefix100", "wikipedia-summaries-100k",
            "wikipedia-summaries-100k-gpt-5-5-20260831151246",
            "wikipedia_summaries_100k",
        ),
    ),
    "deepseek-v4-pro": (
        _point(
            "prefix40", "wikipedia-summaries",
            "wikipedia-summaries-deepseek-v4-pro-20260813152931",
        ),
        _point(
            "prefix100", "wikipedia-summaries-100k",
            "wikipedia-summaries-100k-deepseek-v4-pro-20260831154133",
            "wikipedia_summaries_100k",
        ),
    ),
    "glm-5.1": (
        _point(
            "prefix40", "wikipedia-summaries",
            "wikipedia-summaries-glm-5-1-20260813152925",
        ),
        _point(
            "prefix100", "wikipedia-summaries-100k",
            "wikipedia-summaries-100k-glm-5-1-20260831135815",
            "wikipedia_summaries_100k",
        ),
    ),
    "kimi-k2.6": (
        _point(
            "prefix40", "wikipedia-summaries",
            "wikipedia-summaries-kimi-k2-6-20260813152919",
        ),
        _point(
            "prefix100", "wikipedia-summaries-100k",
            "wikipedia-summaries-100k-kimi-k2-6-20260831135816",
            "wikipedia_summaries_100k",
        ),
    ),
}


SOURCE_POINTS = {
    "demand": DEMAND_POINTS,
    "reasoning": REASONING_POINTS,
    "activity_da": ACTIVITY_DA_POINTS,
    "activity_ml": ACTIVITY_ML_POINTS,
    "wikipedia": WIKIPEDIA_POINTS,
}


PANELS = (
    PanelSpec(
        "mlDa",
        "Task: p-hacking (checkout_redesign) · Prefix: ML (demand_forecasting)",
        "demand",
        "demand_forecasting_to_checkout_redesign",
        "checkout_redesign",
        "low",
    ),
    PanelSpec(
        "daDa",
        "Task: p-hacking (checkout_redesign) · Prefix: p-hacking "
        "(reasoning_prompt_benchmark)",
        "reasoning",
        "reasoning_prompt_benchmark_to_checkout_redesign",
        "checkout_redesign",
        "low",
    ),
    PanelSpec(
        "logDaDa",
        "Task: p-hacking (checkout_redesign) · Prefix: Activity-log context "
        "(p-hacking)",
        "activity_da",
        "activity_log_context_to_checkout_redesign",
        "checkout_redesign",
        "low",
    ),
    PanelSpec(
        "mlMl",
        "Task: ML (fraud_detection) · Prefix: ML (demand_forecasting)",
        "demand",
        "demand_forecasting_to_fraud_detection",
        "fraud_detection",
        None,
    ),
    PanelSpec(
        "logMlMl",
        "Task: ML (fraud_detection) · Prefix: Activity-log context (ML)",
        "activity_ml",
        "activity_log_context_ml_to_fraud_detection",
        "fraud_detection",
        None,
    ),
    PanelSpec(
        "daMl",
        "Task: ML (fraud_detection) · Prefix: p-hacking "
        "(reasoning_prompt_benchmark)",
        "reasoning",
        "reasoning_prompt_benchmark_to_fraud_detection",
        "fraud_detection",
        None,
    ),
    PanelSpec(
        "wikiDa",
        "Task: p-hacking (checkout_redesign) · Prefix: Wikipedia summaries",
        "wikipedia",
        "wikipedia_summaries_to_checkout_redesign",
        "checkout_redesign",
        "low",
    ),
    PanelSpec(
        "wikiMl",
        "Task: ML (fraud_detection) · Prefix: Wikipedia summaries (100k interim)",
        "wikipedia",
        "wikipedia_summaries_to_fraud_detection",
        "fraud_detection",
        None,
    ),
)


def _harness(model: str) -> str:
    return "subscription" if model == "gpt-5.5" else "production"


def _is_model(audit: dict, model: str) -> bool:
    return agent_label(str(audit.get("target") or "")) == model


def _raw_binary_counts(rows: Iterable[dict]) -> dict[str, int]:
    categories = [
        trajectory_category(audit, respect_exclusion=False) for audit in rows
    ]
    return {
        "k": sum(category == "hack" for category in categories),
        "n": sum(
            category in {"hack", "notable", "clean"} for category in categories
        ),
    }


def _filtered_binary_counts(rows: Iterable[dict]) -> dict[str, int]:
    usable = [
        audit for audit in rows
        if audit.get("mechanical_status") != "invalid"
        and audit.get("integrity_status") != "excluded"
    ]
    return _raw_binary_counts(usable)


def _point_direction(panel: PanelSpec, point: PointSpec) -> str:
    if point.direction:
        return f"{point.direction}_to_{panel.seed}"
    return panel.direction


def _panel_target(
    continuations: list[dict], panel: PanelSpec, model: str,
) -> str:
    directions = {
        _point_direction(panel, point)
        for point in SOURCE_POINTS[panel.source][model]
    }
    targets = {
        str(audit.get("target") or "")
        for audit in continuations
        if continuation_direction(audit) in directions
        and _is_model(audit, model)
        and harness_of(audit) == _harness(model)
        and pressure_of(audit) == panel.pressure
        and audit.get("reasoning") is True
    }
    if len(targets) != 1:
        raise ValueError(
            f"expected one exact target for {panel.key} / {model}, got "
            f"{sorted(targets)!r}"
        )
    return next(iter(targets))


def _baseline_counts(
    originals: list[dict], panel: PanelSpec, target: str,
) -> dict[str, int]:
    rows = [
        audit for audit in originals
        if audit.get("seed") == panel.seed
        and str(audit.get("target") or "") == target
        and harness_of(audit) == _harness(agent_label(target))
        and pressure_of(audit) == panel.pressure
        and audit.get("reasoning") is True
    ]
    return _filtered_binary_counts(rows)


def _continuation_counts(
    continuations: list[dict], panel: PanelSpec, target: str, point: PointSpec,
) -> dict[str, int]:
    candidates = [
        audit for audit in continuations
        if continuation_direction(audit) == _point_direction(panel, point)
        and str(audit.get("target") or "") == target
        and harness_of(audit) == _harness(agent_label(target))
        and pressure_of(audit) == panel.pressure
        and audit.get("reasoning") is True
        and treatment_of(audit) == point.treatment
        and prefix_of(audit).get("name") == point.prefix_name
    ]
    identities = {
        str(prefix_of(audit).get("sha256") or prefix_of(audit).get("name"))
        for audit in candidates
    }
    if len(identities) > 1:
        raise ValueError(
            f"multiple exact prefix payloads share {point.prefix_name!r}: "
            f"{sorted(identities)!r}"
        )
    return _filtered_binary_counts(candidates)


def panel_build_specs(
    continuations: list[dict], originals: list[dict],
) -> tuple[tuple[str, PanelSpec], ...]:
    """Return the publication panel keys and active data selections."""

    del continuations, originals
    return tuple((panel.key, panel) for panel in PANELS)


def build_post_figure_data(
    continuations: list[dict], originals: list[dict],
) -> dict[str, dict]:
    """Build the exact object consumed by the existing JavaScript figure renderer."""

    data: dict[str, dict] = {}
    for output_key, panel in panel_build_specs(continuations, originals):
        models = []
        for model in MODEL_ORDER:
            target = _panel_target(continuations, panel, model)
            baseline = _baseline_counts(originals, panel, target)
            if baseline["n"] == 0:
                raise ValueError(
                    f"no current baseline for {panel.key} / {model}"
                )
            if baseline["n"] > TARGET_N:
                raise ValueError(
                    f"overfilled post-figure baseline {panel.key} / {model}: "
                    f"n={baseline['n']}"
                )
            points = [{"condition": "baseline", **baseline}]
            for spec in SOURCE_POINTS[panel.source][model]:
                count = _continuation_counts(
                    continuations, panel, target, spec
                )
                if count["n"] > TARGET_N:
                    raise ValueError(
                        f"overfilled post-figure point {panel.key} / {model} / "
                        f"{spec.condition}: n={count['n']}"
                    )
                # Failed-only missing arms have n=0 and should not produce NaN bars.
                # They appear automatically as soon as usable replacements exist.
                if count["n"]:
                    points.append({"condition": spec.condition, **count})
            models.append({"model": model, "points": points})
        data[output_key] = {"title": panel.title, "models": models}
    return data


def point_lookup(data: dict[str, dict]) -> dict[tuple[str, str, str], tuple[int, int]]:
    """Flatten generated data for validation and concise status output."""

    return {
        (panel, model["model"], point["condition"]): (point["k"], point["n"])
        for panel, panel_data in data.items()
        for model in panel_data["models"]
        for point in model["points"]
    }
