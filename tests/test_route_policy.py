"""The routing policy's arithmetic.

`routed_accuracy` is the whole experiment in four lines: take a ranking, send the
first r fraction to the k-sample answer, keep the greedy answer for the rest. The
things that can silently go wrong are an off-by-one in the fraction, using the wrong
end of the ranking (which would report the anti-policy and still produce a plausible
rising curve), and the endpoints not pinning to the two baselines every curve must
join.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filtering"))
from route_compare import gain_draws, gain_over_random, routed_accuracy


class RoutedAccuracyTests(unittest.TestCase):
    def setUp(self):
        # 10 items: greedy right on 0-5, wrong on 6-9. Majority repairs 6 and 7,
        # breaks 0, and leaves the rest alone.
        self.greedy = np.array([1., 1., 1., 1., 1., 1., 0., 0., 0., 0.])
        self.major = np.array([0., 1., 1., 1., 1., 1., 1., 1., 0., 0.])

    def test_routing_nothing_is_the_greedy_baseline(self):
        self.assertEqual(routed_accuracy(np.arange(10), self.greedy, self.major, [0.0]),
                         [self.greedy.mean()])

    def test_routing_everything_is_the_majority_baseline(self):
        self.assertEqual(routed_accuracy(np.arange(10), self.greedy, self.major, [1.0]),
                         [self.major.mean()])

    def test_the_oracle_order_repairs_before_it_breaks(self):
        """Routing the two items majority fixes must gain exactly those two."""
        order = np.array([6, 7, 1, 2, 3, 4, 5, 8, 9, 0])   # fixes first, breaker last
        got = routed_accuracy(order, self.greedy, self.major, [0.2])[0]
        self.assertAlmostEqual(got, (self.greedy.sum() + 2) / 10)

    def test_the_reversed_order_is_the_anti_policy(self):
        """Guards the sign: routing the breaker first must LOSE accuracy."""
        order = np.array([0, 8, 9, 5, 4, 3, 2, 1, 7, 6])   # breaker first
        got = routed_accuracy(order, self.greedy, self.major, [0.1])[0]
        self.assertAlmostEqual(got, (self.greedy.sum() - 1) / 10)

    def test_the_routed_fraction_rounds_to_a_whole_number_of_items(self):
        order = np.arange(10)
        # 0.14 * 10 = 1.4 -> 1 item; 0.16 * 10 = 1.6 -> 2 items.
        a, b = routed_accuracy(order, self.greedy, self.major, [0.14, 0.16])
        self.assertAlmostEqual(a, (self.greedy.sum() - 1) / 10)   # item 0 breaks
        self.assertAlmostEqual(b, (self.greedy.sum() - 1) / 10)   # item 1 unchanged

    def test_curves_are_monotone_in_nothing_but_still_join_the_endpoints(self):
        """Any ranking, any fractions: r=0 and r=1 are fixed points."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            order = rng.permutation(10)
            lo, hi = routed_accuracy(order, self.greedy, self.major, [0.0, 1.0])
            self.assertAlmostEqual(lo, self.greedy.mean())
            self.assertAlmostEqual(hi, self.major.mean())


class GainOverRandomTests(unittest.TestCase):
    """The quantity the figure plots, which is now a closed form rather than a
    simulation. If the algebra is wrong the figure is wrong everywhere at once, and
    nothing downstream would notice: the curves would still start and end at zero."""

    def setUp(self):
        self.greedy = np.array([1., 1., 1., 1., 1., 1., 0., 0., 0., 0.])
        self.major = np.array([0., 1., 1., 1., 1., 1., 1., 1., 0., 0.])
        self.gain = self.major - self.greedy

    def test_it_equals_routed_accuracy_minus_the_random_expectation(self):
        rng = np.random.default_rng(0)
        fracs = [0.0, 0.1, 0.3, 0.5, 1.0]
        for _ in range(10):
            order = rng.permutation(10)
            acc = routed_accuracy(order, self.greedy, self.major, fracs)
            want = [a - (self.greedy.mean() + r * self.gain.mean())
                    for a, r in zip(acc, fracs)]
            got = gain_over_random(order, self.gain, fracs)
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w)

    def test_the_endpoints_are_exactly_zero_for_every_ranking(self):
        """r=0 routes nothing and r=1 routes everything, so no ranking can matter."""
        rng = np.random.default_rng(1)
        for _ in range(10):
            lo, hi = gain_over_random(rng.permutation(10), self.gain, [0.0, 1.0])
            self.assertAlmostEqual(lo, 0.0)
            self.assertAlmostEqual(hi, 0.0)

    def test_a_random_ranking_averages_to_zero(self):
        """The closed form must agree with what it replaced: the mean over many
        permutations is the baseline itself."""
        rng = np.random.default_rng(2)
        draws = [gain_over_random(rng.permutation(10), self.gain, [0.3])[0]
                 for _ in range(4000)]
        self.assertAlmostEqual(float(np.mean(draws)), 0.0, places=2)

    def test_the_oracle_order_is_positive_and_the_reverse_is_its_mirror(self):
        best = np.argsort(-self.gain)
        worst = np.argsort(self.gain)
        b = gain_over_random(best, self.gain, [0.2])[0]
        w = gain_over_random(worst, self.gain, [0.2])[0]
        self.assertGreater(b, 0)
        self.assertLess(w, 0)

    def test_every_detector_sees_the_same_resample(self):
        """Paired differences are only meaningful on shared draws, so a detector
        duplicated under two names must have a difference of exactly zero."""
        s = {"a": np.arange(10.0), "copy_of_a": np.arange(10.0),
             "b": np.arange(10.0)[::-1].copy()}
        draws = gain_draws(s, self.gain, [0.2, 0.5], n_boot=50, seed=3)
        self.assertTrue(np.allclose(draws["a"], draws["copy_of_a"]))
        self.assertFalse(np.allclose(draws["a"], draws["b"]))
        self.assertEqual(draws["a"].shape, (50, 2))


if __name__ == "__main__":
    unittest.main()
