"""Guards for the two evaluation definitions that previously misreported results."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filtering"))
from probe_models import (head_ranking, oracle_recall_at_precision, precision_at_threshold,
                          recall_at_threshold, recall_ci_at_threshold, risk_coverage,
                          threshold_at_precision, wilson_interval)


class RiskCoverageTests(unittest.TestCase):
    """Risk-coverage must describe a filtering gate: accept highest P(correct)
    first, and report the WRONG rate among accepted items."""

    def test_full_coverage_equals_wrong_rate(self):
        rng = np.random.default_rng(0)
        y = (rng.random(500) > 0.2).astype(int)
        score = rng.random(500)
        cov, risk = risk_coverage(score, y)
        self.assertAlmostEqual(cov[-1], 1.0)
        # The old |score-0.5| version returned the probe's classification error here,
        # which does not equal the wrong rate the baseline line draws.
        self.assertAlmostEqual(risk[-1], float((y == 0).mean()))

    def test_perfect_score_gives_zero_risk_until_wrongs_are_reached(self):
        y = np.array([1, 1, 1, 0, 0])
        score = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        cov, risk = risk_coverage(score, y)
        np.testing.assert_allclose(risk[:3], 0.0)
        self.assertGreater(risk[-1], 0.0)

    def test_risk_is_monotone_for_a_perfect_ranker(self):
        y = np.array([1, 1, 0, 0])
        score = np.array([0.9, 0.8, 0.2, 0.1])
        _, risk = risk_coverage(score, y)
        self.assertTrue(np.all(np.diff(risk) >= 0))


class OperatingPointTests(unittest.TestCase):
    """The probe's threshold must be chosen on train and applied unchanged to test."""

    def test_threshold_is_reusable_across_splits(self):
        y_wrong = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.2, 0.8, 0.9])
        thr = threshold_at_precision(y_wrong, score, 1.0)
        self.assertIsNotNone(thr)
        self.assertEqual(recall_at_threshold(y_wrong, score, thr), 1.0)
        self.assertEqual(precision_at_threshold(y_wrong, score, thr), 1.0)

    def test_unreachable_precision_reports_na_not_zero(self):
        """No operating point must be distinguishable from "caught nothing".

        Reporting 0.0 puts a literal 0 in the results table, which reads as a claim
        about the detector rather than about the experiment's setup.
        """
        y_wrong = np.array([0, 0, 0, 1])
        score = np.array([0.9, 0.9, 0.9, 0.1])
        self.assertIsNone(threshold_at_precision(y_wrong, score, 0.99))
        self.assertIsNone(recall_at_threshold(y_wrong, score, None))
        self.assertIsNone(precision_at_threshold(y_wrong, score, None))
        self.assertEqual(recall_ci_at_threshold(y_wrong, score, None), (None, None, None))
        # A detector that genuinely catches nothing still reports a real 0.0.
        thr_none_caught = 2.0
        self.assertEqual(recall_at_threshold(y_wrong, score, thr_none_caught), 0.0)

    def test_oracle_recall_is_an_upper_bound_on_the_fixed_rule(self):
        rng = np.random.default_rng(1)
        y_wrong = (rng.random(400) < 0.3).astype(int)
        score = rng.random(400) + 0.4 * y_wrong
        thr = threshold_at_precision(y_wrong, score, 0.4)
        fixed = recall_at_threshold(y_wrong, score, thr)
        oracle = oracle_recall_at_precision(y_wrong, score, 0.4)
        # Same split, so the test-set maximum can only be >= the fixed rule. This is
        # exactly why the oracle number cannot be compared to a fixed verdict.
        self.assertGreaterEqual(oracle + 1e-12, fixed)

    def test_oracle_recall_is_none_when_unreachable(self):
        """sklearn's PR curve appends a (precision=1, recall=0) endpoint with no
        threshold behind it; matching that sentinel would return a spurious 0.0."""
        y_wrong = np.array([0, 0, 0, 1])
        score = np.array([0.9, 0.9, 0.9, 0.1])
        self.assertIsNone(oracle_recall_at_precision(y_wrong, score, 0.99))


class RecallIntervalTests(unittest.TestCase):
    """Recall of a fixed rule is a binomial proportion, so its interval must always
    bracket the point estimate — that is why it is Wilson and not percentile
    bootstrap, whose bounds can straddle the estimate and force a silent clamp."""

    def test_wilson_always_brackets_the_estimate(self):
        """Exactly, with no tolerance: the plotting code subtracts these to form error
        bar arms, and matplotlib rejects an arm that is negative by even one ulp."""
        for n in (1, 5, 20, 137, 1000):
            for k in (0, 1, n // 2, n - 1, n):
                lo, hi = wilson_interval(k, n)
                p = k / n
                self.assertLessEqual(lo, p, f"k={k} n={n}")
                self.assertGreaterEqual(hi, p, f"k={k} n={n}")
                self.assertGreaterEqual(lo, 0.0)
                self.assertLessEqual(hi, 1.0)

    def test_error_bar_arms_are_never_negative(self):
        """The failure this guards against is a render-time crash, not a wrong number:
        `hi - p` was -2e-14 at k == n."""
        for n in (1, 3, 17, 73, 498, 1000):
            for k in (0, 1, n - 1, n):
                lo, hi = wilson_interval(k, n)
                p = k / n
                self.assertGreaterEqual(p - lo, 0.0, f"lower arm negative at k={k} n={n}")
                self.assertGreaterEqual(hi - p, 0.0, f"upper arm negative at k={k} n={n}")

    def test_wilson_is_defined_at_the_boundaries(self):
        # A Wald interval collapses to zero width at p=0 or p=1; Wilson does not.
        lo, hi = wilson_interval(0, 50)
        self.assertAlmostEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        lo, hi = wilson_interval(50, 50)
        self.assertLess(lo, 1.0)
        self.assertAlmostEqual(hi, 1.0)

    def test_empty_denominator_is_na(self):
        self.assertEqual(wilson_interval(0, 0), (None, None))

    def test_recall_ci_brackets_and_narrows_with_n(self):
        rng = np.random.default_rng(3)
        widths = []
        for n in (50, 500, 5000):
            y_wrong = np.ones(n, int)
            score = rng.random(n)
            r, lo, hi = recall_ci_at_threshold(y_wrong, score, 0.5)
            self.assertLessEqual(lo, r)
            self.assertGreaterEqual(hi, r)
            widths.append(hi - lo)
        self.assertTrue(all(a > b for a, b in zip(widths, widths[1:])), widths)


class HeadRankingTests(unittest.TestCase):
    def test_full_ranking_is_persisted_not_just_the_selected_slice(self):
        aucs = np.linspace(0.5, 0.9, 64)
        top = np.argsort(-aucs)[:8]
        r = head_ranking(top, aucs, n_layers=4, topk=8)
        self.assertEqual(len(r["selected"]), 8)
        # every candidate head survives, so the top-k cut can be re-examined later
        self.assertEqual(len(r["ranking"]), 64)
        self.assertEqual(r["n_candidate_heads"], 64)
        self.assertEqual(r["heads_per_layer"], 16)
        ranked = [h["train_lda_auc"] for h in r["ranking"]]
        self.assertEqual(ranked, sorted(ranked, reverse=True))
        self.assertAlmostEqual(r["auc_summary"]["at_topk_cut"], ranked[7])
        self.assertAlmostEqual(r["auc_summary"]["max"], float(aucs.max()))

    def test_unknown_layer_count_reports_none_not_layer_zero(self):
        """Without dump geometry, `max(n_layers, 1)` used to put every head in layer 0
        with head_in_layer equal to the flat index — a confident wrong attribution."""
        aucs = np.linspace(0.5, 0.9, 32)
        r = head_ranking(np.argsort(-aucs)[:4], aucs, n_layers=0, topk=4)
        self.assertIsNone(r["n_layers"])
        self.assertIsNone(r["heads_per_layer"])
        self.assertTrue(all(h["layer"] is None for h in r["ranking"]))
        self.assertTrue(all(h["head_in_layer"] is None for h in r["selected"]))


if __name__ == "__main__":
    unittest.main()
