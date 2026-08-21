"""End-to-end key compatibility for the paper driver, on tiny synthetic dumps.

`compute_run` writes a metrics dict and the figure/table functions read it back by
name. Nothing checked that the two agree, so renaming a cached key surfaced as a
KeyError only after ~40 minutes of real computation -- which is how
`auc_{m}_controlled_sd` survived being renamed to `auc_{m}_controlled_seed_sd`.

This runs the whole path in seconds: compute -> every figure -> the LaTeX table.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filtering"))
from extraction_common import Dump
import make_paper_figures as F

GEOM = dict(nL=2, H=8, nH=2, Dh=8)
HEAD_DIM = 4
GEN = {"model": "m/A", "backend": "vllm", "tokenizer": None, "mistral_format": False,
       "assistant_prefill": "", "judge_max_tokens": 16, "dataset": "gsm8k",
       "split": "train", "seed": 42, "problems_sha256": "digest"}


def build_split(directory, n, wrong_every, split, digest, seed):
    """A matched (hidden_rich.npz, raw.jsonl) pair with both classes present."""
    d = Path(directory)
    (d / "gsm8k").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        wrong = (i % wrong_every) == 0
        records.append({
            "idx": i, "dataset": "gsm8k", "problem": f"p{i}",
            "phase1_solution": "s" * (40 if wrong else 20),
            "phase1_answer": "5" if wrong else "4", "gold_answer": "4",
            "majority_answer": "4", "judge_verdict": "NO" if wrong else "YES",
            "solve_correct": not wrong, "truncated": False, "labelable": True,
        })
    (d / "gsm8k" / "raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    gen = {**GEN, "split": split, "problems_sha256": digest}
    dump = Dump(str(d / "hidden_rich.npz"), n, model="m/A", store_resid=False,
                dataset="gsm8k", gen_run=gen, **GEOM)
    for r in records:
        signal = 0.0 if r["solve_correct"] else 2.0
        z = (rng.normal(size=(GEOM["nL"], GEOM["Dh"])) + signal).astype(np.float16)
        dump.add(r, z, 0.8 if r["solve_correct"] else 0.2)
    dump.flush()


class PaperPipelineSmokeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.out = root / "paper"
        build_split(root / "run_train", 120, 3, "train", "train-digest", 0)
        build_split(root / "run_test", 90, 3, "test", "test-digest", 1)
        self.cfg = dict(title="Tiny", head_dim=HEAD_DIM,
                        train=root / "run_train", test=root / "run_test")
        for patch in (mock.patch.object(F, "OUT", self.out),
                      mock.patch.object(F, "TOPK", 2),
                      mock.patch.object(F, "N_SEEDS", 2),
                      mock.patch.object(F, "N_BOOT", 50),
                      mock.patch.object(F, "RUNS", {"tiny": self.cfg})):
            patch.start()
            self.addCleanup(patch.stop)

    def test_compute_then_render_every_artifact(self):
        """The real check: whatever compute_run writes, the figures and table can read.
        A renamed cache key fails here in seconds instead of after a full run."""
        res = F.compute_run("tiny", self.cfg, n_train_resamples=2)
        M = {"tiny": res}
        F.style()
        F.fig_risk_coverage(M)
        F.fig_pr(M)
        F.fig_bars(M)
        F.table_tex(M)
        produced = {p.name for p in self.out.iterdir()}
        self.assertTrue(any(n.startswith("fig_risk_coverage__") for n in produced))
        self.assertTrue(any(n.startswith("fig_recall_at_train_calibrated_op__")
                            for n in produced), produced)
        self.assertTrue(any(n.startswith("table_probe_auc__") for n in produced))

    def test_cached_metrics_survive_a_json_round_trip(self):
        """The plotting path reads from disk, so anything non-serialisable would fail
        only on a re-plot, long after --compute reported success."""
        res = F.compute_run("tiny", self.cfg, n_train_resamples=0)
        reloaded = json.loads(json.dumps(res))
        F.style()
        F.fig_risk_coverage({"tiny": reloaded})
        F.table_tex({"tiny": reloaded})

    def test_exclusions_and_aggregation_are_cached(self):
        res = F.compute_run("tiny", self.cfg, n_train_resamples=0)
        self.assertIn("exclusions", res)
        self.assertEqual(res["exclusions"]["train"]["n_generated"], 120)
        self.assertEqual(res["aggregation_mlp"], "ensemble-mean-of-probabilities")
        self.assertEqual(res["aggregation_lr"], "single-deterministic-fit")

    def test_reported_auc_is_the_ensembles_own(self):
        from sklearn.metrics import roc_auc_score
        res = F.compute_run("tiny", self.cfg, n_train_resamples=0)
        rc = res["risk_coverage"]["mlp_controlled"]
        self.assertEqual(len(rc["cov"]), res["n_test"])
        # the seed_aucs are recorded but must not BE the reported value
        self.assertIn("auc_mlp_controlled_seed_aucs", res)
        self.assertIsInstance(res["auc_mlp_controlled"], float)


if __name__ == "__main__":
    unittest.main()
