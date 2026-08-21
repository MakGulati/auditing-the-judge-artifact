"""The interval behind the perturbation figure.

`two_sample_delta_boot` is what turns "AUC moved by 0.02" into "AUC did not move".
Three things can silently break it: losing the sign (the figure would report the
perturbed set as the reference), resampling rows instead of templates (intervals
several times too narrow, since ~13 instances share one piece of reasoning), and
returning a degenerate interval when a resample happens to contain one class.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filtering"))
from symbolic_compare import two_sample_delta_boot


def sample(n_templates, per_template, sep, rng):
    """A labelled set whose score separates the classes by `sep`."""
    groups = np.repeat(np.arange(n_templates), per_template)
    y = rng.integers(0, 2, len(groups))
    s = rng.normal(0, 1, len(groups)) + sep * y
    return y, s, groups


class TwoSampleDeltaBootTests(unittest.TestCase):
    def test_the_same_separation_on_both_sides_brackets_zero(self):
        rng = np.random.default_rng(0)
        y_s, s_s, g_s = sample(100, 13, 1.5, rng)
        y_g, s_g, _ = sample(1300, 1, 1.5, rng)
        lo, hi = two_sample_delta_boot(y_s, s_s, g_s, y_g, s_g, n=400, seed=0)
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_a_degraded_perturbed_side_gives_a_negative_interval(self):
        """Signs the delta: perturbed minus reference, not the other way round."""
        rng = np.random.default_rng(1)
        y_s, s_s, g_s = sample(100, 13, 0.0, rng)     # no signal after perturbation
        y_g, s_g, _ = sample(1300, 1, 3.0, rng)       # strong signal on the reference
        lo, hi = two_sample_delta_boot(y_s, s_s, g_s, y_g, s_g, n=400, seed=0)
        self.assertLess(hi, 0)

    def test_clustering_widens_the_interval(self):
        """The same rows, resampled by template instead of individually. Instances of
        one template are perfectly correlated here, so ignoring the clustering would
        claim far more information than 100 templates carry."""
        rng = np.random.default_rng(2)
        y1, s1, _ = sample(100, 1, 1.5, rng)
        y = np.repeat(y1, 13)
        s = np.repeat(s1, 13)
        g_clustered = np.repeat(np.arange(100), 13)
        g_rows = np.arange(len(y))
        y_g, s_g, _ = sample(1300, 1, 1.5, rng)
        lo_c, hi_c = two_sample_delta_boot(y, s, g_clustered, y_g, s_g, n=400, seed=0)
        lo_r, hi_r = two_sample_delta_boot(y, s, g_rows, y_g, s_g, n=400, seed=0)
        self.assertGreater(hi_c - lo_c, 1.5 * (hi_r - lo_r))

    def test_a_single_class_anywhere_returns_no_interval(self):
        rng = np.random.default_rng(3)
        y_s, s_s, g_s = sample(20, 5, 1.0, rng)
        y_g = np.ones(50, dtype=int)                  # reference has no errors at all
        s_g = rng.normal(0, 1, 50)
        self.assertEqual(two_sample_delta_boot(y_s, s_s, g_s, y_g, s_g, n=50, seed=0),
                         (None, None))

    def test_a_detector_compared_against_itself_shifts_by_exactly_zero(self):
        """Difference-in-differences mode: the margin of a score against itself is 0
        in every resample, so the interval must be degenerate. A CI that wandered off
        zero here would mean the two sides are not being paired on the same rows."""
        rng = np.random.default_rng(5)
        y_s, s_s, g_s = sample(40, 8, 1.0, rng)
        y_g, s_g, _ = sample(320, 1, 1.0, rng)
        lo, hi = two_sample_delta_boot(y_s, s_s, g_s, y_g, s_g, n=200, seed=0,
                                       ref_s=s_s, ref_g=s_g)
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0)

    def test_the_shift_resolves_what_the_separate_deltas_cannot(self):
        """A stable detector and a reference that degrades on the perturbed side. The
        detector's own delta is noise; the difference in differences is the effect."""
        rng = np.random.default_rng(6)
        groups = np.repeat(np.arange(100), 13)
        y_s = rng.integers(0, 2, len(groups))
        noise_s = rng.normal(0, 1, len(y_s))
        s_s = noise_s + 1.5 * y_s                     # detector: unchanged separation
        ref_s = noise_s + 0.2 * y_s                   # reference: collapsed
        y_g = rng.integers(0, 2, 1300)
        noise_g = rng.normal(0, 1, 1300)
        s_g = noise_g + 1.5 * y_g
        ref_g = noise_g + 1.5 * y_g                   # reference: fine on GSM8K
        plain = two_sample_delta_boot(y_s, s_s, groups, y_g, s_g, n=600, seed=0)
        shift = two_sample_delta_boot(y_s, s_s, groups, y_g, s_g, n=600, seed=0,
                                      ref_s=ref_s, ref_g=ref_g)
        self.assertLess(plain[0], 0)                  # detector's own delta spans zero
        self.assertGreater(plain[1], 0)
        self.assertGreater(shift[0], 0)               # the shift does not

    def test_it_is_deterministic_for_a_given_seed(self):
        rng = np.random.default_rng(4)
        y_s, s_s, g_s = sample(30, 5, 1.0, rng)
        y_g, s_g, _ = sample(150, 1, 1.0, rng)
        a = two_sample_delta_boot(y_s, s_s, g_s, y_g, s_g, n=200, seed=7)
        b = two_sample_delta_boot(y_s, s_s, g_s, y_g, s_g, n=200, seed=7)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
