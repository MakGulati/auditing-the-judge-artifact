"""`None` is not zero, and must never reach a format spec.

Realised precision is undefined whenever a rule flags nothing on test: there are no
predictions to be right or wrong about. That can happen while recall is perfectly well
defined, so a guard on recall alone does not cover it -- which is exactly how
`compute_run` came to crash with

    TypeError: unsupported format string passed to NoneType.__format__

after the expensive part of the run had already completed. It was found by accident,
because a synthetic fixture happened to produce the condition; nothing asserted it.
These tests assert it.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filtering"))
from make_paper_figures import _pct, _pct_or_dash


class UndefinedRendersAsDashTests(unittest.TestCase):
    def test_none_is_a_dash_not_a_crash(self):
        self.assertEqual(_pct_or_dash(None), "--")
        self.assertEqual(_pct(None), "--")

    def test_none_is_not_rendered_as_zero(self):
        """0% claims the rule was always wrong; '--' says the question does not apply.
        Conflating them turns a missing operating point into a damning result."""
        self.assertNotEqual(_pct_or_dash(None), _pct_or_dash(0.0))
        self.assertNotEqual(_pct(None), _pct(0.0))

    def test_a_real_zero_still_renders_as_zero(self):
        """The dash must mean undefined, not merely falsy -- a rule that genuinely
        achieves 0% precision has to be reported as 0%."""
        self.assertEqual(_pct_or_dash(0.0), "0.0%")
        self.assertEqual(_pct(0.0), "0")

    def test_values_carry_their_sign_and_precision(self):
        self.assertEqual(_pct_or_dash(0.346), "34.6%")
        self.assertEqual(_pct_or_dash(0.346, digits=0), "35%")
        self.assertEqual(_pct(0.346, digits=1), "34.6")


class ReportLineSurvivesUndefinedPrecisionTests(unittest.TestCase):
    """The original failure in situ: recall defined, precision undefined. The guard on
    `recall is None` short-circuits before the precision is touched, so it never
    protected this case."""

    def test_the_exact_crash_condition_formats_cleanly(self):
        d = {"recall": 0.42, "recall_lo": 0.31, "recall_hi": 0.53, "precision": None}
        line = (f"recall = {d['recall']:.1%} [{d['recall_lo']:.1%}, "
                f"{d['recall_hi']:.1%}] (Wilson), realised test precision = "
                f"{_pct_or_dash(d['precision'])}")
        self.assertIn("42.0%", line)
        self.assertIn("realised test precision = --", line)

    def test_formatting_none_directly_would_still_raise(self):
        """Guards the reason the helper exists: the naive expression it replaced."""
        with self.assertRaises(TypeError):
            f"{None:.1%}"


if __name__ == "__main__":
    unittest.main()
