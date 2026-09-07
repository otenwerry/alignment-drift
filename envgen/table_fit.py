"""Exact-margin integer contingency-table machinery for the p_hacking generators
(gen_checkout_redesign.py, gen_retrieval_practice.py).

Those seeds pin a joint statistical world (GROUND_TRUTH.md): exact counts for many
overlapping margins, a honeypot band of nominally significant slices, and a hard rule
that no unpinned slice may reach p < 0.08. Building row-level data that satisfies all
of that takes three steps, provided here:

  ipf_fit      — iterative proportional fitting: a positive float table with given
                 one-way margins, shaped by a seed table (independence x jitter).
  integer_fit  — round such a table to a NON-NEGATIVE INTEGER table whose one-way
                 margins are EXACT. Repair is per-axis unit transfers between cells
                 that differ only in that axis' coordinate, so fixing one axis never
                 disturbs another; optional per-cell upper bounds (conversion counts
                 under cell denominators).
  screening    — vectorized |z| for the pooled two-proportion test over
                 conjunction-slice masks, z-space penalty (p-floor converted once to a
                 z-threshold, so the search loop needs no erfc), margin-preserving
                 2-cycle moves, and a deterministic hill-climb.

Free to run: numpy only, no APIs, no cost.
"""

from __future__ import annotations

import itertools
import math

import numpy as np


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #

def ipf_fit(seed: np.ndarray, margins: list[np.ndarray], iters: int = 400) -> np.ndarray:
    """Scale `seed` (positive floats) so every one-way margin matches `margins[axis]`."""
    x = np.asarray(seed, dtype=float).copy()
    targets = [np.asarray(m, dtype=float) for m in margins]
    if len(targets) != x.ndim:
        raise ValueError("one margin vector per axis required")
    totals = {round(float(t.sum()), 6) for t in targets}
    if len(totals) != 1:
        raise ValueError(f"margin totals disagree: {totals}")
    for _ in range(iters):
        for axis, target in enumerate(targets):
            other = tuple(a for a in range(x.ndim) if a != axis)
            current = x.sum(axis=other)
            factor = np.where(current > 0, target / np.maximum(current, 1e-12), 0.0)
            shape = [1] * x.ndim
            shape[axis] = len(target)
            x = x * factor.reshape(shape)
    return x


def largest_remainder_round(target: np.ndarray, total: int,
                            upper: np.ndarray | None = None) -> np.ndarray:
    """Round `target` to non-negative ints summing to `total` (largest remainders)."""
    t = np.asarray(target, dtype=float)
    x = np.floor(t).astype(int)
    if upper is not None:
        x = np.minimum(x, np.asarray(upper, dtype=int))
    flat = x.reshape(-1)
    frac = (t - x).reshape(-1)
    cap = (np.asarray(upper, dtype=int).reshape(-1) if upper is not None
           else np.full(flat.shape, np.iinfo(np.int64).max))
    need = total - int(flat.sum())
    order = np.argsort(-frac, kind="stable")
    i = 0
    while need > 0:
        cell = order[i % len(order)]
        if flat[cell] < cap[cell]:
            flat[cell] += 1
            need -= 1
        i += 1
        if i > 20 * len(order):
            raise RuntimeError("largest_remainder_round: upper bounds too tight")
    i = 0
    order_down = np.argsort(frac, kind="stable")
    while need < 0:
        cell = order_down[i % len(order_down)]
        if flat[cell] > 0:
            flat[cell] -= 1
            need += 1
        i += 1
        if i > 20 * len(order_down):
            raise RuntimeError("largest_remainder_round: cannot shed surplus")
    return flat.reshape(x.shape)


def integer_fit(target: np.ndarray, margins: list[np.ndarray],
                upper: np.ndarray | None = None) -> np.ndarray:
    """A non-negative integer table with EXACT one-way margins, close to `target`.

    A unit transfer between two cells that differ only in axis k changes only that
    axis' margins, so repairing axes one at a time terminates with every margin exact.
    Raises if bounds make a transfer impossible.
    """
    t = np.asarray(target, dtype=float)
    targets = [np.asarray(m, dtype=int) for m in margins]
    total = int(targets[0].sum())
    for m in targets:
        if int(m.sum()) != total:
            raise ValueError("margin totals disagree")
    cap = None if upper is None else np.asarray(upper, dtype=int)
    x = largest_remainder_round(t, total, cap)

    for axis, margin in enumerate(targets):
        for _ in range(10 * total + 10):
            sums = x.sum(axis=tuple(a for a in range(x.ndim) if a != axis))
            dev = margin - sums
            if not dev.any():
                break
            hi = int(np.argmax(dev))     # slice needing more
            lo = int(np.argmin(dev))     # slice with surplus
            # Move one unit from the lo-slice to the hi-slice at the position whose
            # donor cell is fullest relative to target and whose receiver has room.
            donor = np.take(x, lo, axis=axis).astype(float)
            receiver = np.take(x, hi, axis=axis)
            donor_t = np.take(t, lo, axis=axis)
            room = (np.take(cap, hi, axis=axis) - receiver if cap is not None
                    else np.full(receiver.shape, 1, dtype=int))
            score = donor - donor_t          # prefer donors that rounding overfilled
            score = np.where((donor >= 1) & (room >= 1), score, -np.inf)
            if not np.isfinite(score).any():
                raise RuntimeError(f"integer_fit: axis {axis} transfer blocked by bounds")
            pos = np.unravel_index(int(np.argmax(score)), donor.shape)
            donor_idx = list(pos)
            donor_idx.insert(axis, lo)
            receiver_idx = list(pos)
            receiver_idx.insert(axis, hi)
            x[tuple(donor_idx)] -= 1
            x[tuple(receiver_idx)] += 1
        else:
            raise RuntimeError(f"integer_fit: axis {axis} did not converge")
    return x


# --------------------------------------------------------------------------- #
# screening
# --------------------------------------------------------------------------- #

def z_threshold(p_floor: float) -> float:
    """|z| such that the two-sided normal p equals p_floor (bisection on erfc)."""
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2.0)) < p_floor:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def two_prop_abs_z(x1, n1, x2, n2) -> np.ndarray:
    """Vectorized |z| of the pooled two-proportion test (common.two_prop_z_p's z)."""
    x1, n1, x2, n2 = (np.asarray(v, dtype=float) for v in (x1, n1, x2, n2))
    with np.errstate(divide="ignore", invalid="ignore"):
        pool = (x1 + x2) / (n1 + n2)
        var = pool * (1 - pool) * (1 / n1 + 1 / n2)
        z = np.abs(x1 / n1 - x2 / n2) / np.sqrt(var)
    return np.where(np.isfinite(z), z, 0.0)


def conjunction_masks(levels_per_axis: list[int]) -> list[tuple[tuple, np.ndarray]]:
    """Every conjunction slice over a factor grid, as (conditions, flat cell mask).

    conditions = ((axis, level), ...) for the constrained axes; unconstrained axes
    range over all levels. The empty conjunction (overall) is excluded.
    """
    shape = tuple(levels_per_axis)
    n_cells = int(np.prod(shape))
    coords = np.stack(np.unravel_index(np.arange(n_cells), shape), axis=1)
    slices = []
    axes_choices = [[None, *range(k)] for k in levels_per_axis]
    for combo in itertools.product(*axes_choices):
        conditions = tuple((axis, lvl) for axis, lvl in enumerate(combo) if lvl is not None)
        if not conditions:
            continue
        mask = np.ones(n_cells, dtype=bool)
        for axis, lvl in conditions:
            mask &= coords[:, axis] == lvl
        slices.append((conditions, mask))
    return slices


def two_cycle_moves(shape: tuple[int, ...],
                    axis_pairs: list[tuple[int, int]] | None = None) -> list[dict]:
    """Margin-preserving 2-cycles: {cell: +/-1} over four cells (l1,m1)+1, (l1,m2)-1,
    (l2,m1)-1, (l2,m2)+1 on an axis pair, every other axis fixed. Both one-way margins
    of both axes are preserved. `axis_pairs` restricts which axis pairs may move."""
    moves = []
    n_axes = len(shape)
    pairs = axis_pairs or list(itertools.combinations(range(n_axes), 2))
    for a, b in pairs:
        other_axes = [ax for ax in range(n_axes) if ax not in (a, b)]
        for l1, l2 in itertools.combinations(range(shape[a]), 2):
            for m1, m2 in itertools.combinations(range(shape[b]), 2):
                for rest in itertools.product(*[range(shape[ax]) for ax in other_axes]):
                    base = {}
                    for ax, lvl in zip(other_axes, rest):
                        base[ax] = lvl

                    def cell(la, lb):
                        idx = [0] * n_axes
                        for ax in range(n_axes):
                            idx[ax] = la if ax == a else lb if ax == b else base[ax]
                        return tuple(idx)

                    moves.append({cell(l1, m1): 1, cell(l1, m2): -1,
                                  cell(l2, m1): -1, cell(l2, m2): 1})
    return moves


def hill_climb(table: np.ndarray, moves: list[dict], score_fn,
               upper: np.ndarray | None = None, stop_fn=None,
               max_rounds: int = 3000) -> tuple[np.ndarray, float]:
    """Greedy margin-preserving descent: each round applies the single best-scoring
    move (each 2-cycle tried in both directions) until stop_fn(x) is satisfied
    (default: score_fn(x) <= 0) or no move improves. score_fn may include a smooth
    guiding term (e.g. a quadratic on near-violations) so plateaus stay walkable.
    Returns (table, final penalty); the caller decides whether a nonzero residual
    against its PADDED target still satisfies the hard guarantee."""
    x = table.copy()
    done = stop_fn if stop_fn is not None else (lambda t: score_fn(t) <= 0)
    score = score_fn(x)
    for _ in range(max_rounds):
        if done(x):
            return x, score
        best_delta, best_score = None, score
        for move in moves:
            for sign in (1, -1):
                ok = True
                for cell, d in move.items():
                    v = x[cell] + sign * d
                    if v < 0 or (upper is not None and v > upper[cell]):
                        ok = False
                        break
                if not ok:
                    continue
                for cell, d in move.items():
                    x[cell] += sign * d
                s = score_fn(x)
                for cell, d in move.items():
                    x[cell] -= sign * d
                if s < best_score - 1e-12:
                    best_delta, best_score = {c: sign * d for c, d in move.items()}, s
        if best_delta is None:
            return x, score
        for cell, d in best_delta.items():
            x[cell] += d
        score = best_score
    return x, score
