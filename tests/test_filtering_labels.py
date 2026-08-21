import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filtering"))
from probe_models import load_split


def write_dump(path, *, z, y, idx, meta=None, p_yes=None):
    arrays = dict(Z_head=z, y=np.asarray(y), idx=np.asarray(idx))
    if p_yes is not None:
        arrays["p_yes"] = np.asarray(p_yes, dtype=np.float32)
    if meta is not None:
        arrays["meta"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)


class FilteringLabelTests(unittest.TestCase):
    def test_numeric_policy_repairs_stored_labels_by_idx(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden = root / "hidden.npz"
            raw = root / "raw.jsonl"
            write_dump(hidden, z=np.zeros((2, 1, 2), np.float32), y=[0, 1], idx=[7, 3],
                       p_yes=[0.9, 0.1])
            rows = [
                {"idx": 3, "phase1_answer": "5", "gold_answer": "6",
                 "phase1_solution": "answer 5", "judge_verdict": "NO"},
                {"idx": 7, "phase1_answer": "64.00", "gold_answer": "64",
                 "phase1_solution": "answer 64.00", "judge_verdict": "YES"},
            ]
            raw.write_text("".join(json.dumps(row) + "\n" for row in rows))

            numeric = load_split(hidden, raw, head_dim=2)
            stored = load_split(hidden, raw, head_dim=2, label_policy="stored")
            np.testing.assert_array_equal(numeric.y, [1, 0])
            np.testing.assert_array_equal(stored.y, [0, 1])
            np.testing.assert_array_equal(numeric.yes, [1, 0])
            np.testing.assert_allclose(numeric.p_yes, [0.9, 0.1])
            self.assertEqual(numeric.info["stored_disagreements"], 2)
            self.assertEqual(numeric.info["policy"], "numeric_equivalence_v2")


class UnlabelableExclusionTests(unittest.TestCase):
    def test_error_and_missing_gold_records_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden, raw = root / "hidden.npz", root / "raw.jsonl"
            write_dump(hidden, z=np.zeros((3, 1, 2), np.float32), y=[1, 0, 0],
                       idx=[0, 1, 2], p_yes=[0.5, 0.5, 0.5])
            rows = [
                {"idx": 0, "phase1_answer": "5", "gold_answer": "5",
                 "phase1_solution": "s", "judge_verdict": "YES"},
                # gold never parsed (GSM8K '#### -12' under the old regex)
                {"idx": 1, "phase1_answer": "-12", "gold_answer": None,
                 "phase1_solution": "s", "judge_verdict": "YES"},
                # generation failed; its placeholder fields are not observations
                {"idx": 2, "phase1_answer": None, "gold_answer": "9",
                 "phase1_solution": "", "judge_verdict": None, "error": "CUDA OOM"},
            ]
            raw.write_text("".join(json.dumps(r) + "\n" for r in rows))

            split = load_split(hidden, raw, head_dim=2)
            self.assertEqual(len(split), 1)
            self.assertEqual(split.idx, [0])
            self.assertEqual(split.info["n_dropped_unlabelable"], 2)


class DatasetAwareLabelTests(unittest.TestCase):
    """The probe's labels are recomputed here, so this is where a dataset-blind
    comparison would do its damage: MATH records survive the labelability filter and
    are then labelled by numeric comparison of LaTeX, which the probe trains on
    without complaint."""

    def _write(self, root, rows, meta=None):
        hidden, raw = root / "hidden.npz", root / "raw.jsonl"
        n = len(rows)
        write_dump(hidden, z=np.zeros((n, 1, 2), np.float32), y=[0] * n,
                   idx=[r["idx"] for r in rows], meta=meta,
                   p_yes=[0.5] * n)
        raw.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return hidden, raw

    def test_math_records_are_labelled_under_the_latex_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {"idx": 0, "dataset": "math", "phase1_answer": r"\dfrac{1}{2}",
                 "gold_answer": r"\frac{1}{2}", "phase1_solution": "s",
                 "judge_verdict": "YES"},
                {"idx": 1, "dataset": "math", "phase1_answer": r"\frac{1}{3}",
                 "gold_answer": r"\frac{1}{2}", "phase1_solution": "s",
                 "judge_verdict": "NO"},
            ]
            hidden, raw = self._write(Path(tmp), rows)
            split = load_split(hidden, raw, head_dim=2)
            self.assertEqual(len(split), 2, "LaTeX golds must not be dropped")
            np.testing.assert_array_equal(split.y, [1, 0])
            self.assertEqual(split.info["policy"], "latex_numeric_equivalence_v1")
            self.assertEqual(split.info["dataset"], "math")

    def test_gsm8k_records_are_unaffected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"idx": 0, "phase1_answer": "64.00", "gold_answer": "64",
                     "phase1_solution": "s", "judge_verdict": "YES"}]
            hidden, raw = self._write(Path(tmp), rows)
            split = load_split(hidden, raw, head_dim=2)
            np.testing.assert_array_equal(split.y, [1])
            self.assertEqual(split.info["policy"], "numeric_equivalence_v2")
            self.assertEqual(split.info["dataset"], "gsm8k")

    def test_mixed_datasets_in_one_raw_file_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {"idx": 0, "dataset": "math", "phase1_answer": "2",
                 "gold_answer": "2", "phase1_solution": "s", "judge_verdict": "YES"},
                {"idx": 1, "dataset": "gsm8k", "phase1_answer": "2",
                 "gold_answer": "2", "phase1_solution": "s", "judge_verdict": "YES"},
            ]
            hidden, raw = self._write(Path(tmp), rows)
            with self.assertRaises(ValueError) as cm:
                load_split(hidden, raw, head_dim=2)
            self.assertIn("different equivalence policies", str(cm.exception))

    def test_dump_and_raw_from_different_datasets_raise(self):
        """Activations extracted with one judge prompt, labels from another dataset."""
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"idx": 0, "dataset": "math", "phase1_answer": "2",
                     "gold_answer": "2", "phase1_solution": "s",
                     "judge_verdict": "YES"}]
            meta = {"model": "m/A", "n_layers": 1, "hidden_size": 2, "n_heads": 1,
                    "o_proj_in": 2, "head_dim": 2, "dataset": "gsm8k"}
            hidden, raw = self._write(Path(tmp), rows, meta=meta)
            with self.assertRaises(ValueError) as cm:
                load_split(hidden, raw, head_dim=2)
            self.assertIn("different runs", str(cm.exception))


class HeadDimValidationTests(unittest.TestCase):
    def test_mismatched_head_dim_raises_instead_of_fragmenting_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden, raw = root / "hidden.npz", root / "raw.jsonl"
            meta = {"model": "google/gemma-3-12b-it", "n_layers": 1, "hidden_size": 8,
                    "n_heads": 2, "o_proj_in": 8, "head_dim": 4}
            write_dump(hidden, z=np.zeros((1, 1, 8), np.float32), y=[1], idx=[0],
                       meta=meta, p_yes=[0.5])
            raw.write_text(json.dumps(
                {"idx": 0, "phase1_answer": "1", "gold_answer": "1",
                 "phase1_solution": "s", "judge_verdict": "YES"}) + "\n")

            # correct head_dim loads
            split = load_split(hidden, raw, head_dim=4)
            self.assertEqual(split.Z.shape, (1, 2, 4))
            # a divisor that would silently reshape must be rejected
            with self.assertRaises(ValueError) as cm:
                load_split(hidden, raw, head_dim=2)
            self.assertIn("head_dim", str(cm.exception))


if __name__ == "__main__":
    unittest.main()


class BalancedTrainingTests(unittest.TestCase):
    """LogisticRegression takes class_weight='balanced'; sklearn's MLPClassifier has no
    equivalent, so without an explicit rebalance the two probes are not compared on
    equal terms. Measured on GSM8K (5.7% wrong) that was worth 0.760 -> 0.800 AUC and
    a six-fold drop in seed variance."""

    def setUp(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filtering"))
        from probe_models import balance_train
        self.balance = balance_train

    def test_minority_is_oversampled_to_parity(self):
        y = np.array([1] * 90 + [0] * 10)
        X = np.arange(100).reshape(100, 1).astype(np.float32)
        Xb, yb = self.balance(X, y, seed=0)
        self.assertEqual((yb == 0).sum(), (yb == 1).sum())
        self.assertEqual((yb == 1).sum(), 90)

    def test_every_majority_row_is_kept(self):
        """Oversampling, not undersampling: discarding majority rows would throw away
        real data to fix a weighting problem."""
        y = np.array([1] * 90 + [0] * 10)
        X = np.arange(100).reshape(100, 1).astype(np.float32)
        Xb, yb = self.balance(X, y, seed=0)
        kept = set(Xb[yb == 1].ravel().tolist())
        self.assertEqual(kept, set(range(90)))

    def test_already_balanced_data_is_untouched(self):
        y = np.array([0, 1] * 50)
        X = np.arange(100).reshape(100, 1).astype(np.float32)
        Xb, yb = self.balance(X, y, seed=0)
        self.assertEqual(len(yb), 100)

    def test_single_class_input_is_returned_unchanged(self):
        """A degenerate split must not crash here; check_split_usable reports it."""
        y = np.ones(10, dtype=int)
        X = np.arange(10).reshape(10, 1).astype(np.float32)
        Xb, yb = self.balance(X, y, seed=0)
        self.assertEqual(len(yb), 10)

    def test_seeded_and_reproducible(self):
        y = np.array([1] * 90 + [0] * 10)
        X = np.arange(100).reshape(100, 1).astype(np.float32)
        a, _ = self.balance(X, y, seed=3)
        b, _ = self.balance(X, y, seed=3)
        c, _ = self.balance(X, y, seed=4)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_lr_is_not_rebalanced(self):
        """LR already carries class_weight='balanced'; rebalancing it too would apply
        the correction twice."""
        from probe_models import fit_scores
        rng = np.random.default_rng(0)
        y = np.array([1] * 180 + [0] * 20)
        X = rng.normal(size=(200, 4)).astype(np.float32) + (1 - y)[:, None]
        # Deterministic for LR: the same call twice must give identical scores, which
        # a seeded resample of a skewed set would not.
        s1 = fit_scores(X, y, X, model_type="lr", seed=0)[1]
        s2 = fit_scores(X, y, X, model_type="lr", seed=7)[1]
        np.testing.assert_allclose(s1, s2)
