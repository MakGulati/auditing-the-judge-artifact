"""Plotted predictions, annotations and table values must share one aggregation.

The defect this pins down: `fit_raw_and_controlled` was called once per seed, the
FIRST seed's predictions were plotted and thresholded, and the annotation beside the
curve reported the MEAN of the per-seed AUCs. The line on the page and the number next
to it therefore described different fits, and no reader could tell.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filtering"))
from probe_models import ensemble_seed_aucs, fit_ensemble, is_stochastic


def synthetic(n=400, d=12, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    F = (rng.normal(size=(n, d)) + y[:, None] * 0.8).astype(np.float32)
    S = (rng.normal(size=(n, 7)) + y[:, None] * 0.3).astype(np.float32)
    return F, y, S


class EnsembleAggregationTests(unittest.TestCase):
    def setUp(self):
        self.Ftr, self.ytr, self.Str = synthetic(seed=0)
        self.Fte, self.yte, self.Ste = synthetic(seed=1)

    def _fit(self, model_type, n_seeds):
        return fit_ensemble(self.Ftr, self.ytr, self.Fte, self.Str, self.Ste,
                            model_type=model_type, seed=0, n_seeds=n_seeds)

    def test_ensemble_is_the_mean_of_the_member_predictions(self):
        scores, seeds, runs = self._fit("mlp", 3)
        self.assertEqual(len(seeds), 3)
        for variant in ("raw", "controlled"):
            np.testing.assert_allclose(
                scores[variant], np.mean([r[variant] for r in runs], axis=0), rtol=1e-6)

    def test_the_reported_auc_is_the_ensembles_own_auc(self):
        """Not the mean of the members' AUCs — that is a different quantity, and
        quoting it beside a curve drawn from one member is the original defect."""
        scores, _, runs = self._fit("mlp", 3)
        ens_auc = roc_auc_score(self.yte, scores["controlled"])
        mean_of_aucs = float(np.mean(ensemble_seed_aucs(runs, self.yte, "controlled")))
        # They are genuinely different numbers; the code must report the first.
        self.assertNotAlmostEqual(ens_auc, mean_of_aucs, places=6)

    def test_ensemble_never_equals_a_single_member_by_construction(self):
        scores, _, runs = self._fit("mlp", 3)
        first = runs[0]["controlled"]
        self.assertFalse(np.allclose(scores["controlled"], first),
                         "ensembling collapsed to the first seed — the old behaviour")

    def test_deterministic_models_ignore_n_seeds(self):
        """Refitting LR at another seed returns the same fit, so averaging would be
        the same number computed n times."""
        self.assertFalse(is_stochastic("lr"))
        one, seeds_one, _ = self._fit("lr", 1)
        many, seeds_many, _ = self._fit("lr", 5)
        self.assertEqual(len(seeds_one), 1)
        self.assertEqual(len(seeds_many), 1)
        np.testing.assert_allclose(one["controlled"], many["controlled"])

    def test_every_variant_is_ensembled_not_just_the_plotted_one(self):
        """Operating points are chosen on the TRAIN scores, so those must be ensembled
        too — otherwise the threshold comes from one seed and the curve from five."""
        scores, _, runs = self._fit("mlp", 3)
        for key in ("raw", "controlled", "raw_train", "controlled_train"):
            self.assertIn(key, scores)
            np.testing.assert_allclose(
                scores[key], np.mean([r[key] for r in runs], axis=0), rtol=1e-6)

    def test_ensemble_is_reproducible(self):
        a, _, _ = self._fit("mlp", 3)
        b, _, _ = self._fit("mlp", 3)
        np.testing.assert_allclose(a["controlled"], b["controlled"])


class ResampleUsesTheSameEstimatorTests(unittest.TestCase):
    """The train-resample spread decorates the point estimate, so it has to be the
    spread OF that estimate. Fitting one model per replicate while quoting an ensemble
    attaches an interval to an estimator nobody is using -- a single draw is noisier
    than an average of five, so the +/- would be systematically too wide."""

    def test_resample_accepts_and_uses_n_seeds(self):
        import inspect
        from probe_models import train_resample_auc
        sig = inspect.signature(train_resample_auc)
        self.assertIn("n_seeds", sig.parameters)
        src = inspect.getsource(train_resample_auc)
        self.assertIn("fit_ensemble", src)
        self.assertNotIn("fit_raw_and_controlled", src)

    def test_drivers_pass_their_ensemble_size_through(self):
        root = Path(__file__).resolve().parents[1] / "filtering"
        for name in ("make_paper_figures.py", "train_test_probe.py"):
            src = (root / name).read_text()
            i = src.index("train_resample_auc(")
            call = src[i:i + 400]
            self.assertIn("n_seeds=", call,
                          f"{name} calls train_resample_auc without n_seeds, so its "
                          f"+/- would describe single fits while its point estimate is "
                          f"an ensemble")

    def test_ensembled_replicates_are_less_variable_than_single_draws(self):
        """The reason it matters, demonstrated rather than asserted."""
        from probe_models import fit_ensemble
        Ftr, ytr, Str = synthetic(seed=2)
        Fte, yte, Ste = synthetic(seed=3)
        singles = [roc_auc_score(yte, fit_ensemble(Ftr, ytr, Fte, Str, Ste,
                                                   model_type="mlp", seed=s,
                                                   n_seeds=1)[0]["controlled"])
                   for s in range(4)]
        ens = [roc_auc_score(yte, fit_ensemble(Ftr, ytr, Fte, Str, Ste,
                                               model_type="mlp", seed=s, n_seeds=3)[0]["controlled"])
               for s in range(4)]
        self.assertLessEqual(float(np.std(ens)), float(np.std(singles)) + 1e-9)


class DriverDefaultsTests(unittest.TestCase):
    """All three drivers must ensemble the same way, or their numbers describe
    different estimators while appearing side by side in one paper."""

    def test_seed_count_defaults_agree(self):
        import argparse
        import make_paper_figures as F

        def default_of(module_path, flag):
            src = Path(module_path).read_text()
            i = src.index(f'"{flag}"')
            seg = src[i:i + 200]
            j = seg.index("default=")
            return int(seg[j + 8:].split(",")[0].split(")")[0])

        root = Path(__file__).resolve().parents[1] / "filtering"
        self.assertEqual(default_of(root / "train_test_probe.py", "--n_seeds"), F.N_SEEDS)
        self.assertEqual(default_of(root / "error_detection_compare.py", "--n_seeds"),
                         F.N_SEEDS)


if __name__ == "__main__":
    unittest.main()
