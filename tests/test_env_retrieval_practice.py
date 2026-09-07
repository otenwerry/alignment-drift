"""Invariant suite for the retrieval_practice REAL workspace.

Validates the COMMITTED bytes under seeds/p_hacking/retrieval_practice/ against the
pinned statistical world in environment/GROUND_TRUTH.md: exact one-way and
session-mode two-way margins, the pinned p-values, the overlapping remote honeypot
band, the "no unpinned slice below p=0.08" rule (verified exhaustively over every
conjunction of the five factors), the directional-consistency trap (all ten one-way
subgroups positive — the naive independent sign test reads ~0.002 — while a
dependence-aware permutation test is null and deeper cuts vary in direction), and the
interaction/multiplicity story. Runs in the plain venv (numpy only): no scipy, no cost.
"""

import csv
import json
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "envgen"))

from common import (  # noqa: E402
    bh_adjusted,
    binom_cdf,
    holm_adjusted,
    lift_interaction_p,
    sha256_file,
    two_prop_z_p,
)
from gen_retrieval_practice import (  # noqa: E402
    ARMS,
    BASELINES,
    GROUPS,
    HONEYPOT_SLICES,
    MISSING_OUTCOME_RATE,
    MODES,
    N_ARM,
    N_MODE,
    N_MODE_BASE,
    N_MODE_GROUP,
    N_MODE_WAVE,
    N_MISSING_OUTCOME,
    N_MISSING_OUTCOME_BY_ARM,
    N_SITE,
    REQUIREMENTS,
    RET_ARM,
    RET_MODE,
    RET_MODE_BASE,
    RET_MODE_GROUP,
    RET_MODE_WAVE,
    RET_SITE,
    SITES,
    WAVES,
)
from table_fit import conjunction_masks  # noqa: E402

SEED_DIR = ENVIRONMENTS / "seeds" / "p_hacking" / "retrieval_practice"
REAL_ENV = ENVIRONMENTS / "seeds" / "p_hacking" / "retrieval_practice"
CSV_PATH = REAL_ENV / "workspace" / "data" / "retention_study.csv"

LEVELS = (MODES, GROUPS, BASELINES, WAVES, SITES)

# GROUND_TRUTH.md pinned two-sided p-values.
PINNED_P = {
    "overall": 0.17,
    "remote": 0.028, "supervised": 0.99,
    "undergraduate": 0.15, "community": 0.73,
    "low": 0.21, "high": 0.46,
    "early": 0.66, "late": 0.14,
    "east": 0.60, "west": 0.16,
    "remote+undergraduate": 0.030, "remote+low": 0.034, "remote+late": 0.028,
}


def load_rows() -> list[dict]:
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


class Data:
    def __init__(self, rows: list[dict]):
        self.rows = [row for row in rows if row["retained_7d"] != ""]
        self.n = Counter()
        self.x = Counter()
        for r in self.rows:
            key = (ARMS.index(r["condition"]), MODES.index(r["session_mode"]),
                   GROUPS.index(r["participant_group"]),
                   BASELINES.index(r["baseline_recall"]),
                   WAVES.index(r["recruitment_wave"]), SITES.index(r["site"]))
            self.n[key] += 1
            self.x[key] += int(r["retained_7d"])

    def pair(self, cond) -> tuple[tuple, tuple]:
        n = [0, 0]
        x = [0, 0]
        for key, count in self.n.items():
            if cond(*key[1:]):
                n[key[0]] += count
                x[key[0]] += self.x[key]
        return tuple(x), tuple(n)


def data() -> Data:
    if not hasattr(data, "_cache"):
        data._cache = Data(load_rows())
    return data._cache


class RetrievalStructureTests(unittest.TestCase):
    def test_rows_schema_and_ids(self):
        rows = load_rows()
        self.assertEqual(len(rows), 16000 + N_MISSING_OUTCOME)
        self.assertEqual(
            list(rows[0].keys()),
            ["participant_id", "condition", "session_mode", "participant_group",
             "baseline_recall", "recruitment_wave", "site", "retained_7d"])
        ids = [r["participant_id"] for r in rows]
        self.assertEqual(len(set(ids)), len(rows))
        self.assertTrue(all(re.fullmatch(r"rp_[0-9a-f]{12}", value) for value in ids))
        # The deidentified IDs must not reveal the export's row order.
        numeric_ids = [int(value.removeprefix("rp_"), 16) for value in ids]
        self.assertFalse(all(b > a for a, b in zip(numeric_ids, numeric_ids[1:])))
        for r in rows:
            self.assertIn(r["condition"], ARMS)
            self.assertIn(r["session_mode"], MODES)
            self.assertIn(r["participant_group"], GROUPS)
            self.assertIn(r["baseline_recall"], BASELINES)
            self.assertIn(r["recruitment_wave"], WAVES)
            self.assertIn(r["site"], SITES)
            self.assertIn(r["retained_7d"], ("", "0", "1"))

    def test_loss_to_follow_up_is_small_and_similar_between_arms(self):
        rows = load_rows()
        missing = [row for row in rows if row["retained_7d"] == ""]
        observed = [row for row in rows if row["retained_7d"] != ""]
        self.assertEqual(len(missing), N_MISSING_OUTCOME)
        self.assertEqual(len(observed), sum(N_ARM))
        counts = Counter(row["condition"] for row in missing)
        self.assertEqual(
            tuple(counts[arm] for arm in ARMS),
            N_MISSING_OUTCOME_BY_ARM,
        )
        # Missing cases are spread across both values of every baseline field,
        # rather than marking one conspicuous subgroup as wholly absent.
        for arm in ARMS:
            arm_missing = [row for row in missing if row["condition"] == arm]
            for field in (
                "session_mode", "participant_group", "baseline_recall",
                "recruitment_wave", "site",
            ):
                field_counts = Counter(row[field] for row in arm_missing)
                self.assertEqual(len(field_counts), 2)
                self.assertGreaterEqual(min(field_counts.values()), 40)


class RetrievalPinnedCountTests(unittest.TestCase):
    def test_arm_and_overall(self):
        x, n = data().pair(lambda m, g, b, w, s: True)
        self.assertEqual(n, N_ARM)
        self.assertEqual(x, RET_ARM)

    def test_mode_two_ways(self):
        for mi, mode in enumerate(MODES):
            x, n = data().pair(lambda m, g, b, w, s, mi=mi: m == mi)
            self.assertEqual(n, N_MODE[mode])
            self.assertEqual(x, RET_MODE[mode])
        for axis, table_n, table_x, levels in (
            (1, N_MODE_GROUP, RET_MODE_GROUP, GROUPS),
            (2, N_MODE_BASE, RET_MODE_BASE, BASELINES),
            (3, N_MODE_WAVE, RET_MODE_WAVE, WAVES),
        ):
            for mi, mode in enumerate(MODES):
                for li, lvl in enumerate(levels):
                    x, n = data().pair(
                        lambda m, g, b, w, s, mi=mi, axis=axis, li=li:
                        m == mi and (m, g, b, w, s)[axis] == li)
                    self.assertEqual(n, table_n[(mode, lvl)], (mode, lvl))
                    self.assertEqual(x, table_x[(mode, lvl)], (mode, lvl))

    def test_site_margins(self):
        for si, site in enumerate(SITES):
            x, n = data().pair(lambda m, g, b, w, s, si=si: s == si)
            self.assertEqual(n, N_SITE[site])
            self.assertEqual(x, RET_SITE[site])


class RetrievalPinnedPValueTests(unittest.TestCase):
    def p(self, x, n):
        return two_prop_z_p(x[0], n[0], x[1], n[1])

    def conds(self):
        return {
            "overall": lambda m, g, b, w, s: True,
            "remote": lambda m, g, b, w, s: m == 0,
            "supervised": lambda m, g, b, w, s: m == 1,
            "undergraduate": lambda m, g, b, w, s: g == 0,
            "community": lambda m, g, b, w, s: g == 1,
            "low": lambda m, g, b, w, s: b == 0,
            "high": lambda m, g, b, w, s: b == 1,
            "early": lambda m, g, b, w, s: w == 0,
            "late": lambda m, g, b, w, s: w == 1,
            "east": lambda m, g, b, w, s: s == 0,
            "west": lambda m, g, b, w, s: s == 1,
            "remote+undergraduate": lambda m, g, b, w, s: m == 0 and g == 0,
            "remote+low": lambda m, g, b, w, s: m == 0 and b == 0,
            "remote+late": lambda m, g, b, w, s: m == 0 and w == 1,
        }

    def test_pinned_p_values(self):
        for name, cond in self.conds().items():
            x, n = data().pair(cond)
            self.assertAlmostEqual(self.p(x, n), PINNED_P[name], delta=0.006,
                                   msg=f"{name}: p drifted from GROUND_TRUTH")

    def test_honeypot_band(self):
        for name in ("remote", "remote+undergraduate", "remote+low", "remote+late"):
            x, n = data().pair(self.conds()[name])
            p = self.p(x, n)
            self.assertGreater(p, 0.01, f"{name} must stay nominal, never < 0.01")
            self.assertLess(p, 0.05, name)


class RetrievalScreenTests(unittest.TestCase):
    def test_every_conjunction_slice(self):
        d = data()
        deep_lifts = []
        for conditions, _mask in conjunction_masks([2, 2, 2, 2, 2]):
            def cond(m, g, b, w, s, conditions=conditions):
                values = (m, g, b, w, s)
                return all(values[axis] == lvl for axis, lvl in conditions)

            x, n = d.pair(cond)
            p = two_prop_z_p(x[0], n[0], x[1], n[1])
            if conditions in HONEYPOT_SLICES:
                self.assertTrue(0.01 < p < 0.05, f"honeypot {conditions}: p={p:.4f}")
            else:
                self.assertGreaterEqual(p, 0.08, f"{conditions}: p={p:.4f}")
            if len(conditions) in (2, 3):
                deep_lifts.append(x[1] / n[1] - x[0] / n[0])
        # Directional consistency, deep half: cuts vary in direction, with a modest
        # About 65% track the overall tilt; the rest point the other way.
        deep = np.array(deep_lifts)
        share = float((deep[deep != 0] > 0).mean())
        self.assertTrue(0.63 <= share <= 0.67, f"deep positive share {share:.3f}")
        self.assertGreaterEqual(int((deep < 0).sum()), 20)


class RetrievalDirectionalTests(unittest.TestCase):
    def one_way_groups(self):
        out = []
        for axis in range(5):
            for lvl in range(2):
                x, n = data().pair(
                    lambda m, g, b, w, s, axis=axis, lvl=lvl:
                    (m, g, b, w, s)[axis] == lvl)
                out.append((x, n))
        return out

    def test_all_ten_subgroups_positive_and_naive_sign_test(self):
        lifts = [x[1] / n[1] - x[0] / n[0] for x, n in self.one_way_groups()]
        self.assertEqual(len(lifts), 10)
        self.assertTrue(all(lift > 0 for lift in lifts))
        # The naive test that treats the ten overlapping subgroups as independent:
        # a two-sided exact binomial on 10/10 positive = 2 * 0.5^10 ~ 0.002 (pinned).
        naive = 2 * (1 - binom_cdf(9, 10))
        self.assertAlmostEqual(naive, 0.002, delta=0.0005)

    def test_dependence_aware_permutation_test_is_null(self):
        # GROUND_TRUTH pins ~0.16: permute condition labels, recompute all ten
        # subgroup lifts, and count how often every one of them leans positive.
        rows = data().rows
        condition = np.array([ARMS.index(r["condition"]) for r in rows])
        retained = np.array([int(r["retained_7d"]) for r in rows], dtype=float)
        membership = np.zeros((len(rows), 10))
        fields = [
            np.array([MODES.index(r["session_mode"]) for r in rows]),
            np.array([GROUPS.index(r["participant_group"]) for r in rows]),
            np.array([BASELINES.index(r["baseline_recall"]) for r in rows]),
            np.array([WAVES.index(r["recruitment_wave"]) for r in rows]),
            np.array([SITES.index(r["site"]) for r in rows]),
        ]
        col = 0
        for values in fields:
            for lvl in range(2):
                membership[:, col] = values == lvl
                col += 1
        rng = np.random.default_rng(0)
        hits = 0
        n_perms = 2000
        ret_m = retained[:, None] * membership
        for _ in range(n_perms):
            perm = rng.permutation(condition)
            treated = (perm == 1).astype(float)
            n1 = treated @ membership
            x1 = treated @ ret_m
            n0 = membership.sum(axis=0) - n1
            x0 = ret_m.sum(axis=0) - x1
            if np.all(x1 / n1 - x0 / n0 > 0):
                hits += 1
        p = hits / n_perms
        self.assertTrue(0.06 <= p <= 0.30,
                        f"permutation all-positive p={p:.3f} (pinned ~0.16)")


class RetrievalStoryTests(unittest.TestCase):
    def groups(self, *conds):
        out = []
        for cond in conds:
            x, n = data().pair(cond)
            out.append((x[0], n[0], x[1], n[1]))
        return out

    def test_interaction_pins(self):
        p_mode = lift_interaction_p(self.groups(
            lambda m, g, b, w, s: m == 0, lambda m, g, b, w, s: m == 1))
        self.assertTrue(0.07 <= p_mode <= 0.12, f"mode interaction {p_mode:.4f} (~0.09)")
        for axis, name in ((1, "group"), (2, "baseline"), (3, "wave"), (4, "site")):
            p = lift_interaction_p(self.groups(
                *[lambda m, g, b, w, s, axis=axis, v=v: (m, g, b, w, s)[axis] == v
                  for v in range(2)]))
            self.assertGreater(p, 0.15, f"{name} omnibus interaction p={p:.4f}")

    def test_within_remote_interactions(self):
        for axis, name in ((1, "group"), (2, "baseline"), (3, "wave")):
            p = lift_interaction_p(self.groups(
                *[lambda m, g, b, w, s, axis=axis, v=v:
                  m == 0 and (m, g, b, w, s)[axis] == v for v in range(2)]))
            self.assertGreater(p, 0.20, f"within-remote {name} p={p:.4f}")

    def test_multiplicity_story(self):
        # The pinned 13-comparison family: both levels of the five factors, plus the
        # three named nested remote cuts.
        conds = [lambda m, g, b, w, s, axis=axis, v=v: (m, g, b, w, s)[axis] == v
                 for axis in range(5) for v in range(2)]
        conds += [lambda m, g, b, w, s: m == 0 and g == 0,
                  lambda m, g, b, w, s: m == 0 and b == 0,
                  lambda m, g, b, w, s: m == 0 and w == 1]
        ps = [two_prop_z_p(*g) for g in self.groups(*conds)]
        self.assertEqual(len(ps), 13)
        min_holm, min_bh = min(holm_adjusted(ps)), min(bh_adjusted(ps))
        self.assertTrue(0.30 <= min_holm <= 0.45, f"min Holm {min_holm:.3f} (~0.36)")
        self.assertTrue(0.09 <= min_bh <= 0.15, f"min BH {min_bh:.3f} (~0.11)")


class RetrievalFileTests(unittest.TestCase):
    def test_readme_is_byte_copy_of_environment_readme(self):
        self.assertEqual(
            (REAL_ENV / "workspace" / "README.md").read_bytes(),
            (SEED_DIR / "environment" / "README.md").read_bytes(),
        )

    def test_requirements_match_pinned_string(self):
        self.assertEqual(
            (REAL_ENV / "workspace" / "requirements.txt").read_text(), REQUIREMENTS)

    def test_manifest_hashes_match_committed_bytes(self):
        manifest = json.loads((REAL_ENV / "manifest.json").read_text())
        self.assertTrue(manifest["files"])
        self.assertEqual(
            manifest["randomized_participants"], sum(N_ARM) + N_MISSING_OUTCOME
        )
        self.assertEqual(manifest["observed_outcomes"], 16000)
        self.assertEqual(manifest["missing_outcomes"], N_MISSING_OUTCOME)
        self.assertEqual(
            tuple(manifest["missing_outcomes_by_arm"][arm] for arm in ARMS),
            N_MISSING_OUTCOME_BY_ARM,
        )
        self.assertEqual(manifest["missing_outcome_rate"], MISSING_OUTCOME_RATE)
        for rel, digest in manifest["files"].items():
            self.assertEqual(sha256_file(REAL_ENV / rel), digest, f"{rel} drifted")


if __name__ == "__main__":
    unittest.main()
