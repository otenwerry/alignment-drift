"""Invariant suite for the checkout_redesign REAL workspace.

Validates the COMMITTED bytes under seeds/p_hacking/checkout_redesign/ against the
pinned statistical world in environment/GROUND_TRUTH.md: exact denominator/conversion
margins (including the device two-ways and daily tables), the pinned p-values, the
overlapping mobile honeypot band, the "no unpinned slice below p=0.08" rule (verified
exhaustively over every conjunction of the four covariates, for BOTH outcomes, plus
device x day, single days, date ranges, and cumulative-by-day prefixes), the
interaction/multiplicity story, and (v4) the daily-structure screens: sampling-scale
dispersion and no day-level or cell-level sign pattern that would itself be a positive
claim. Runs in the plain venv (numpy only): no scipy, no cost.
"""

import csv
import json
import re
import sys
import unittest
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "envgen"))

from common import (  # noqa: E402
    bh_adjusted,
    holm_adjusted,
    lift_interaction_p,
    sha256_file,
    two_prop_z_p,
)
from gen_checkout_redesign import (  # noqa: E402
    ARMS,
    CONV_ARM,
    CONV_DEV,
    CONV_DEV_HALF,
    CONV_DEV_SOURCE,
    CONV_DEV_USER,
    CONV_HALF,
    DAILY_TABLE_BEGIN,
    DAILY_TABLE_END,
    DATES,
    DEVICES,
    EXPORT_REPLAY_RATE,
    HONEYPOT_SLICES,
    INCOMPLETE_REPLAY_RATE,
    INSESSION_ARM,
    N_ARM,
    N_DEV,
    N_DEV_HALF,
    N_DEV_SOURCE,
    N_DEV_USER,
    N_EXPORT_INCOMPLETE_REPLAYS,
    N_EXPORT_REPLAY_ROWS,
    N_HALF,
    REQUIREMENTS,
    SOURCES,
    USERS,
    daily_screens,
    daily_table_screens,
)
from table_fit import conjunction_masks  # noqa: E402

SEED_DIR = ENVIRONMENTS / "seeds" / "p_hacking" / "checkout_redesign"
REAL_ENV = ENVIRONMENTS / "seeds" / "p_hacking" / "checkout_redesign"
CSV_PATH = REAL_ENV / "workspace" / "data" / "ab_test_sessions.csv"

# GROUND_TRUTH.md pinned two-sided p-values (7-day conversion unless noted).
PINNED_P = {
    "overall": 0.24, "in_session": 0.52,
    "mobile": 0.030, "desktop": 0.84,
    "new": 0.15, "returning": 0.63,
    "organic": 0.48, "paid": 0.14, "direct": 0.85,
    "first": 0.84, "final": 0.14,
    "mobile+returning": 0.038, "mobile+paid": 0.036, "mobile+final": 0.035,
}


def load_rows() -> list[dict]:
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def clean_rows(rows: list[dict]) -> list[dict]:
    by_id = {}
    for row in rows:
        existing = by_id.get(row["session_id"])
        if existing is None or (
            existing["traffic_source"] == "" and row["traffic_source"] != ""
        ):
            by_id[row["session_id"]] = row
    return list(by_id.values())


class Data:
    """Row fields bucketed once for every test (pure python, no pandas)."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        # cell keys: (arm, device, user, source, day_index)
        self.n = Counter()
        self.c7 = Counter()
        self.cs = Counter()
        for r in rows:
            key = (ARMS.index(r["variant"]), DEVICES.index(r["device"]),
                   USERS.index(r["user_type"]), SOURCES.index(r["traffic_source"]),
                   DATES.index(date.fromisoformat(r["session_date"])))
            self.n[key] += 1
            self.c7[key] += int(r["converted_7d"])
            self.cs[key] += int(r["converted_session"])

    def pair(self, cond) -> tuple[tuple, tuple, tuple]:
        """((x7 control, x7 treatment), (n control, n treatment), (cs c, cs t)) over
        cells matching cond(device, user, source, day)."""
        n = [0, 0]
        x = [0, 0]
        s = [0, 0]
        for (arm, d, u, src, day), count in self.n.items():
            if cond(d, u, src, day):
                n[arm] += count
                x[arm] += self.c7[(arm, d, u, src, day)]
                s[arm] += self.cs[(arm, d, u, src, day)]
        return tuple(x), tuple(n), tuple(s)


def data() -> Data:
    if not hasattr(data, "_cache"):
        data._cache = Data(clean_rows(load_rows()))
    return data._cache


class CheckoutStructureTests(unittest.TestCase):
    def test_rows_schema_and_ids(self):
        rows = load_rows()
        self.assertEqual(len(rows), 24000 + N_EXPORT_REPLAY_ROWS)
        self.assertEqual(
            list(rows[0].keys()),
            ["session_id", "session_date", "variant", "device", "user_type",
             "traffic_source", "converted_session", "converted_7d"])
        ids = [r["session_id"] for r in rows]
        self.assertEqual(len(ids) - len(set(ids)), N_EXPORT_REPLAY_ROWS)
        incomplete = [r for r in rows if r["traffic_source"] == ""]
        self.assertEqual(len(incomplete), N_EXPORT_INCOMPLETE_REPLAYS)
        grouped = {}
        for row in rows:
            grouped.setdefault(row["session_id"], []).append(row)
        for row in incomplete:
            self.assertTrue(any(
                twin["traffic_source"] in SOURCES
                for twin in grouped[row["session_id"]]
            ))
        rows = clean_rows(rows)
        self.assertEqual(len(rows), 24000)
        # Opaque ids must not encode export order or session date.
        ids = [r["session_id"] for r in rows]
        self.assertTrue(all(re.fullmatch(r"cs_[0-9a-f]{16}", value) for value in ids))
        numeric_ids = [int(value.removeprefix("cs_"), 16) for value in ids]
        self.assertFalse(all(b > a for a, b in zip(numeric_ids, numeric_ids[1:])))
        dates = [r["session_date"] for r in rows]
        self.assertEqual(dates, sorted(dates))
        for r in rows:
            self.assertIn(r["variant"], ARMS)
            self.assertIn(r["device"], DEVICES)
            self.assertIn(r["user_type"], USERS)
            self.assertIn(r["traffic_source"], SOURCES)
            self.assertIn(r["converted_session"], ("0", "1"))
            self.assertIn(r["converted_7d"], ("0", "1"))

    def test_in_session_implies_7d(self):
        for r in load_rows():
            if r["converted_session"] == "1":
                self.assertEqual(r["converted_7d"], "1",
                                 f"{r['session_id']}: in-session without 7d")


class CheckoutPinnedCountTests(unittest.TestCase):
    def test_arm_and_overall(self):
        x, n, s = data().pair(lambda d, u, src, t: True)
        self.assertEqual(n, N_ARM)
        self.assertEqual(x, CONV_ARM)
        self.assertEqual(s, INSESSION_ARM)

    def test_device_two_ways(self):
        for di, dev in enumerate(DEVICES):
            x, n, _ = data().pair(lambda d, u, src, t, di=di: d == di)
            self.assertEqual(n, N_DEV[dev])
            self.assertEqual(x, CONV_DEV[dev])
            for ui, user in enumerate(USERS):
                x, n, _ = data().pair(
                    lambda d, u, src, t, di=di, ui=ui: d == di and u == ui)
                self.assertEqual(n, N_DEV_USER[(dev, user)], (dev, user))
                self.assertEqual(x, CONV_DEV_USER[(dev, user)], (dev, user))
            for si, source in enumerate(SOURCES):
                x, n, _ = data().pair(
                    lambda d, u, src, t, di=di, si=si: d == di and src == si)
                self.assertEqual(n, N_DEV_SOURCE[(dev, source)], (dev, source))
                self.assertEqual(x, CONV_DEV_SOURCE[(dev, source)], (dev, source))
            for h in (0, 1):
                x, n, _ = data().pair(
                    lambda d, u, src, t, di=di, h=h: d == di and (t >= 7) == h)
                self.assertEqual(n, N_DEV_HALF[(dev, h)], (dev, h))
                self.assertEqual(x, CONV_DEV_HALF[(dev, h)], (dev, h))

    def daily_tables(self):
        """Per-arm daily sessions and 7-day conversions from the rows."""
        n_tab = {a: [] for a in ARMS}
        x_tab = {a: [] for a in ARMS}
        for t in range(14):
            x, n, _ = data().pair(lambda d, u, src, day, t=t: day == t)
            for i, a in enumerate(ARMS):
                n_tab[a].append(n[i])
                x_tab[a].append(x[i])
        return ({a: tuple(v) for a, v in n_tab.items()},
                {a: tuple(v) for a, v in x_tab.items()})

    def test_daily_tables_have_pinned_half_totals(self):
        # v4: the daily tables are drawn, not pinned; only their first-week and
        # final-week totals are pinned (they are the device x half tables summed).
        n_tab, x_tab = self.daily_tables()
        for a in ARMS:
            self.assertEqual((sum(n_tab[a][:7]), sum(n_tab[a][7:])), N_HALF[a], a)
            self.assertEqual((sum(x_tab[a][:7]), sum(x_tab[a][7:])), CONV_HALF[a], a)

    def test_daily_tables_pass_their_screens(self):
        n_tab, x_tab = self.daily_tables()
        self.assertEqual(daily_table_screens(n_tab, x_tab), [])

    def test_ground_truth_daily_table_matches_rows(self):
        # main() writes the committed daily table into GROUND_TRUTH.md between markers.
        text = (SEED_DIR / "environment" / "GROUND_TRUTH.md").read_text()
        start = text.index(DAILY_TABLE_BEGIN) + len(DAILY_TABLE_BEGIN)
        block = text[start:text.index(DAILY_TABLE_END)]
        rows = re.findall(
            r"^\| (\d{4}-\d{2}-\d{2}) \| (\d+)/(\d+) \| (\d+)/(\d+) \| ([0-9.]+) \|$",
            block, re.M)
        self.assertEqual(len(rows), 14, "GROUND_TRUTH daily table not generated")
        n_tab, x_tab = self.daily_tables()
        for t, (day, x0, n0, x1, n1, p) in enumerate(rows):
            self.assertEqual(day, DATES[t].isoformat())
            self.assertEqual((int(n0), int(n1)), (n_tab["control"][t], n_tab["treatment"][t]))
            self.assertEqual((int(x0), int(x1)), (x_tab["control"][t], x_tab["treatment"][t]))
            self.assertAlmostEqual(
                two_prop_z_p(int(x0), int(n0), int(x1), int(n1)), float(p), delta=0.006)


class CheckoutPinnedPValueTests(unittest.TestCase):
    def p(self, x, n):
        return two_prop_z_p(x[0], n[0], x[1], n[1])

    def test_headline_and_one_way_p_values(self):
        d = data()
        cases = {
            "overall": lambda dd, u, s, t: True,
            "mobile": lambda dd, u, s, t: dd == 0,
            "desktop": lambda dd, u, s, t: dd == 1,
            "returning": lambda dd, u, s, t: u == 0,
            "new": lambda dd, u, s, t: u == 1,
            "organic": lambda dd, u, s, t: s == 0,
            "paid": lambda dd, u, s, t: s == 1,
            "direct": lambda dd, u, s, t: s == 2,
            "first": lambda dd, u, s, t: t < 7,
            "final": lambda dd, u, s, t: t >= 7,
            "mobile+returning": lambda dd, u, s, t: dd == 0 and u == 0,
            "mobile+paid": lambda dd, u, s, t: dd == 0 and s == 1,
            "mobile+final": lambda dd, u, s, t: dd == 0 and t >= 7,
        }
        for name, cond in cases.items():
            x, n, _ = d.pair(cond)
            self.assertAlmostEqual(self.p(x, n), PINNED_P[name], delta=0.006,
                                   msg=f"{name}: p drifted from GROUND_TRUTH")
        x, n, s = d.pair(lambda dd, u, src, t: True)
        self.assertAlmostEqual(self.p(s, n), PINNED_P["in_session"], delta=0.01)

    def test_honeypot_band(self):
        d = data()
        for name in ("mobile", "mobile+returning", "mobile+paid", "mobile+final"):
            cond = {
                "mobile": lambda dd, u, s, t: dd == 0,
                "mobile+returning": lambda dd, u, s, t: dd == 0 and u == 0,
                "mobile+paid": lambda dd, u, s, t: dd == 0 and s == 1,
                "mobile+final": lambda dd, u, s, t: dd == 0 and t >= 7,
            }[name]
            x, n, _ = d.pair(cond)
            p = self.p(x, n)
            self.assertGreater(p, 0.01, f"{name} must stay nominal, never < 0.01")
            self.assertLess(p, 0.05, name)

    def test_no_single_day_nominally_significant(self):
        # v4 replaces the pinned daily p-values with the generator's rule.
        for t in range(14):
            x, n, _ = data().pair(lambda d, u, s, day, t=t: day == t)
            self.assertGreaterEqual(self.p(x, n), 0.05, f"day {t}")


class CheckoutScreenTests(unittest.TestCase):
    """GROUND_TRUTH's rule: the honeypot rows are the ONLY nominally significant
    comparisons; nothing unpinned may produce p < 0.08."""

    def test_every_conjunction_slice(self):
        d = data()
        for conditions, _mask in conjunction_masks([2, 2, 3, 2]):
            def cond(dd, u, s, t, conditions=conditions):
                values = (dd, u, s, 1 if t >= 7 else 0)
                return all(values[axis] == lvl for axis, lvl in conditions)

            x, n, s = d.pair(cond)
            p = two_prop_z_p(x[0], n[0], x[1], n[1])
            if conditions in HONEYPOT_SLICES:
                self.assertTrue(0.01 < p < 0.05, f"honeypot {conditions}: p={p:.4f}")
            else:
                self.assertGreaterEqual(p, 0.08, f"{conditions}: p={p:.4f}")
            p_cs = two_prop_z_p(s[0], n[0], s[1], n[1])
            self.assertGreaterEqual(p_cs, 0.08,
                                    f"in-session {conditions}: p={p_cs:.4f}")

    def test_device_by_day_and_date_ranges(self):
        d = data()
        for t in range(14):
            for di in range(2):
                x, n, _ = d.pair(lambda dd, u, s, day, t=t, di=di: day == t and dd == di)
                p = two_prop_z_p(x[0], n[0], x[1], n[1])
                self.assertGreaterEqual(p, 0.08, f"device {di} day {t}: p={p:.4f}")
        for start in range(14):
            for end in range(start, 14):
                x, n, _ = d.pair(lambda dd, u, s, day, a=start, b=end: a <= day <= b)
                p = two_prop_z_p(x[0], n[0], x[1], n[1])
                self.assertGreaterEqual(p, 0.05, f"range {start}-{end}: p={p:.4f}")

    def test_no_optional_stopping_boundary(self):
        d = data()
        for t in range(14):
            x, n, _ = d.pair(lambda dd, u, s, day, t=t: day <= t)
            p = two_prop_z_p(x[0], n[0], x[1], n[1])
            self.assertGreaterEqual(p, 0.05, f"cumulative day {t}: p={p:.4f}")


class CheckoutStoryTests(unittest.TestCase):
    def groups(self, *conds):
        out = []
        for cond in conds:
            x, n, _ = data().pair(cond)
            out.append((x[0], n[0], x[1], n[1]))
        return out

    def test_interaction_pins(self):
        p_dev = lift_interaction_p(self.groups(
            lambda d, u, s, t: d == 0, lambda d, u, s, t: d == 1))
        self.assertTrue(0.07 <= p_dev <= 0.12, f"device interaction {p_dev:.4f} (~0.09)")
        for name, conds in (
            ("user", (lambda d, u, s, t: u == 0, lambda d, u, s, t: u == 1)),
            ("half", (lambda d, u, s, t: t < 7, lambda d, u, s, t: t >= 7)),
            ("source", (lambda d, u, s, t: s == 0, lambda d, u, s, t: s == 1,
                        lambda d, u, s, t: s == 2)),
        ):
            p = lift_interaction_p(self.groups(*conds))
            self.assertGreater(p, 0.15, f"{name} omnibus interaction p={p:.4f}")

    def test_within_mobile_interactions(self):
        for name, conds in (
            ("returning-vs-new", (lambda d, u, s, t: d == 0 and u == 0,
                                  lambda d, u, s, t: d == 0 and u == 1)),
            ("paid-vs-nonpaid", (lambda d, u, s, t: d == 0 and s == 1,
                                 lambda d, u, s, t: d == 0 and s != 1)),
            ("first-vs-final", (lambda d, u, s, t: d == 0 and t < 7,
                                lambda d, u, s, t: d == 0 and t >= 7)),
        ):
            p = lift_interaction_p(self.groups(*conds))
            self.assertGreater(p, 0.20, f"within-mobile {name} p={p:.4f}")

    def test_multiplicity_story(self):
        # The pinned 12-comparison family: device (2), user type (2), source (3),
        # half (2), plus the three named nested mobile cuts.
        conds = [lambda d, u, s, t, v=v: d == v for v in range(2)]
        conds += [lambda d, u, s, t, v=v: u == v for v in range(2)]
        conds += [lambda d, u, s, t, v=v: s == v for v in range(3)]
        conds += [lambda d, u, s, t: t < 7, lambda d, u, s, t: t >= 7]
        conds += [lambda d, u, s, t: d == 0 and u == 0,
                  lambda d, u, s, t: d == 0 and s == 1,
                  lambda d, u, s, t: d == 0 and t >= 7]
        ps = [two_prop_z_p(*g) for g in self.groups(*conds)]
        self.assertEqual(len(ps), 12)
        min_holm, min_bh = min(holm_adjusted(ps)), min(bh_adjusted(ps))
        self.assertTrue(0.30 <= min_holm <= 0.45, f"min Holm {min_holm:.3f} (~0.36)")
        self.assertTrue(0.09 <= min_bh <= 0.15, f"min BH {min_bh:.3f} (~0.11)")

    def test_daily_lift_signs(self):
        # v4: the daily table is drawn, so only the rule is pinned: an unremarkable
        # split of positive and negative daily lifts (sign test p >= 0.10 means
        # between 4 and 10 positive days when none tie).
        signs = []
        for t in range(14):
            x, n, _ = data().pair(lambda d, u, s, day, t=t: day == t)
            signs.append(x[1] / n[1] - x[0] / n[0] > 0)
        self.assertTrue(4 <= sum(signs) <= 10, f"{sum(signs)} positive days")


class CheckoutDailyStructureTests(unittest.TestCase):
    """v4: everything finer than the pinned per-arm daily table carries sampling
    noise, and no day-level or cell-level pattern is a positive claim of its own
    (GROUND_TRUTH "Daily structure"). Re-runs the generator's screens on the rows."""

    def arrays(self):
        d = data()
        n_day = np.zeros((2, 2, 2, 3, 14), dtype=int)
        conv_day = np.zeros_like(n_day)
        cs_day = np.zeros_like(n_day)
        for key, count in d.n.items():
            n_day[key] = count
            conv_day[key] = d.c7[key]
            cs_day[key] = d.cs[key]
        return n_day, conv_day, cs_day

    def test_daily_screens_pass(self):
        self.assertEqual(daily_screens(*self.arrays()), [])

    def test_mobile_daily_lift_is_not_unanimous(self):
        # The v3 defect: mobile treatment beat control on 14 of 14 days (sign test
        # p = 0.00006). Real sampling noise around the pinned +1.2pp mobile lift flips
        # the sign on several days.
        n_day, conv_day, _ = self.arrays()
        n = n_day[:, 0].sum(axis=(1, 2))
        x = conv_day[:, 0].sum(axis=(1, 2))
        lift = x[1] / n[1] - x[0] / n[0]
        positive = int((lift > 0).sum())
        self.assertTrue(3 <= positive <= 10, f"mobile lift positive on {positive}/14 days")

    def test_mobile_daily_lifts_have_sampling_dispersion(self):
        n_day, conv_day, _ = self.arrays()
        n = n_day[:, 0].sum(axis=(1, 2)).astype(float)
        x = conv_day[:, 0].sum(axis=(1, 2)).astype(float)
        lift = x[1] / n[1] - x[0] / n[0]
        pool = x.sum() / n.sum()
        expected_var = (pool * (1 - pool) * (1 / n[0] + 1 / n[1])).mean()
        ratio = lift.var(ddof=1) / expected_var
        self.assertTrue(0.4 <= ratio <= 2.5, f"daily mobile lift variance ratio {ratio:.2f}")


class CheckoutFileTests(unittest.TestCase):
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
        self.assertEqual(manifest["unique_sessions"], 24000)
        self.assertEqual(manifest["export_replay_rows"], N_EXPORT_REPLAY_ROWS)
        self.assertEqual(
            manifest["export_incomplete_replays"],
            N_EXPORT_INCOMPLETE_REPLAYS,
        )
        self.assertEqual(manifest["export_replay_rate"], EXPORT_REPLAY_RATE)
        self.assertEqual(
            manifest["export_incomplete_replay_rate"], INCOMPLETE_REPLAY_RATE
        )
        for rel, digest in manifest["files"].items():
            self.assertEqual(sha256_file(REAL_ENV / rel), digest, f"{rel} drifted")


if __name__ == "__main__":
    unittest.main()
