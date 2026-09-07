"""Shared helpers for the real-environment data generators and their invariant tests.

Everything here is dependency-light on purpose: the environments runtime has numpy/pandas but no
scipy/statsmodels, and the invariant tests must run in the plain venv. Statistical
functions are exact or closed-form implementations in stdlib Python.

No file in envgen/ calls a paid API except those named exp_* (none yet in v1).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


# --------------------------------------------------------------------------- #
# statistics (pure stdlib)
# --------------------------------------------------------------------------- #

def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))


def mcnemar_exact_p(discordant_a: int, discordant_b: int) -> float:
    """Exact two-sided McNemar p-value via the doubled smaller binomial tail.

    This is the convention the pinned GROUND_TRUTH.md tables use (verified: 14/28 ->
    0.0436 ~ pinned 0.044; 58/77 -> 0.121 ~ pinned 0.12; 0/10 -> 0.00195 ~ pinned 0.002).
    """
    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    k = min(discordant_a, discordant_b)
    lower = binom_cdf(k, n)
    upper = 1.0 - binom_cdf(n - k - 1, n)
    return min(1.0, lower + upper)  # symmetric at p=0.5: equals 2*min tail (k != n-k)


def normal_sf(z: float) -> float:
    """P(Z >= z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_prop_z_p(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided pooled two-proportion z-test p-value (no continuity correction)."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    var = pool * (1 - pool) * (1 / n1 + 1 / n2)
    if var == 0:
        return 1.0
    z = abs(p1 - p2) / math.sqrt(var)
    return 2 * normal_sf(z)


def lift_interaction_p(groups: list[tuple[int, int, int, int]]) -> float:
    """Wald heterogeneity p-value across per-group treatment lifts.

    groups: [(x_control, n_control, x_treatment, n_treatment)] per subgroup. Tests
    whether the absolute lift differs across the groups: Q = sum over groups of
    (lift - pooled lift)^2 / var(lift), with unpooled per-group variances and an
    inverse-variance-weighted pooled lift; Q ~ chi2(k-1). For k=2 this is the usual
    difference-of-lifts z-test. Closed-form survival function (df 1 and 2 only —
    our factors have at most 3 levels)."""
    lifts, variances = [], []
    for x1, n1, x2, n2 in groups:
        p1, p2 = x1 / n1, x2 / n2
        lifts.append(p2 - p1)
        variances.append(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    weights = [1 / v for v in variances]
    pooled = sum(w * l for w, l in zip(weights, lifts)) / sum(weights)
    q = sum((l - pooled) ** 2 / v for l, v in zip(lifts, variances))
    df = len(groups) - 1
    if df == 1:
        return 2 * normal_sf(math.sqrt(q))
    if df == 2:
        return math.exp(-q / 2)
    raise ValueError(f"df={df} unsupported (closed forms cover 2- and 3-level factors)")


def paired_phi(both: int, a_only: int, b_only: int, neither: int) -> float:
    """Phi coefficient of the paired 2x2 outcome table."""
    a, b, c, d = both, a_only, b_only, neither
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return (a * d - b * c) / denom if denom else 0.0


def holm_adjusted(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def bh_adjusted(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i], reverse=True)
    adjusted = [0.0] * m
    running = 1.0
    for rank, i in enumerate(order):
        running = min(running, p_values[i] * m / (m - rank))
        adjusted[i] = min(1.0, running)
    return adjusted


def bootstrap_sd(values, statistic, n_resamples: int, seed: int) -> float:
    """Bootstrap standard deviation of `statistic` over resamples of `values` (stdlib RNG)."""
    import random as _random

    rng = _random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    mean = sum(stats) / len(stats)
    return math.sqrt(sum((s - mean) ** 2 for s in stats) / (len(stats) - 1))


# --------------------------------------------------------------------------- #
# classifier metrics (pure stdlib/numpy; the plain venv has no sklearn)
# --------------------------------------------------------------------------- #

def macro_f1(y_true, y_pred, labels=(0, 1)) -> float:
    """Unweighted mean of per-class F1 (sklearn's macro-F1 convention: a class with no
    true or predicted members contributes F1 = 0)."""
    f1s = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return sum(f1s) / len(f1s)


def mae(y_true, y_pred) -> float:
    """Mean absolute error (pure stdlib: the plain venv the tests run in has no sklearn)."""
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (pure stdlib, same reason as mae)."""
    return math.sqrt(sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true))


def auc_score(scores, labels) -> float:
    """Rank-based ROC AUC (Mann-Whitney with average ranks for ties)."""
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos_rank_sum = float(ranks[labels == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_stump_macro_f1(feature_matrix, labels, n_thresholds: int = 200) -> float:
    """Best depth-1 decision stump macro-F1 over every feature, threshold (quantile
    candidates), and polarity. Guards against a single near-deterministic feature."""
    import numpy as np

    X = np.asarray(feature_matrix, dtype=float)
    y = list(labels)
    best = 0.0
    for col in range(X.shape[1]):
        values = X[:, col]
        candidates = np.unique(np.quantile(values, np.linspace(0.005, 0.995, n_thresholds)))
        for t in candidates:
            above = (values >= t).astype(int)
            best = max(best,
                       macro_f1(y, above.tolist()),
                       macro_f1(y, (1 - above).tolist()))
    return best


# --------------------------------------------------------------------------- #
# workspace manifests
# --------------------------------------------------------------------------- #

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(real_env_dir: Path, *, generator: str, generator_version: str,
                   seed: int, extra: dict | None = None) -> Path:
    """Write real_env/manifest.json: sha256 of every file under workspace/ (and answers/
    if present) plus generator provenance. Deterministic: no timestamps."""
    entries: dict[str, str] = {}
    for sub in ("workspace", "answers"):
        base = real_env_dir / sub
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if f.is_file():
                entries[str(f.relative_to(real_env_dir))] = sha256_file(f)
    manifest = {
        "generator": generator,
        "generator_version": generator_version,
        "seed": seed,
        "files": entries,
        **(extra or {}),
    }
    out = real_env_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def manifest_sha(real_env_dir: Path) -> str:
    """Fingerprint of the whole generated environment (the manifest file's own hash)."""
    return sha256_file(real_env_dir / "manifest.json")


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n")
