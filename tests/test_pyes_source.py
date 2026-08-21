"""Cross-checking `p_yes` between the activation dump and the top-K dump.

Both files are written by the same forward pass at the same position, so their p_yes
columns must be equal. They are produced by different scripts on different days, and
nothing else in the pipeline can notice that one of them ran under a different
environment -- `rec_sha` pins the RECORD, not the run, so a stale dump passes every
other check. A stale gemma3 top-K dump did exactly that.

The value always comes from the activation dump (it is the column aligned with the
activations). These tests pin the reporting: agreement, disagreement, the two
inconclusive cases, and the one condition still fatal -- fingerprints that show the
dumps describe different records.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filtering"))
from extraction_common import Dump
from probe_models import load_split

GEOM = dict(nL=2, H=8, nH=2, Dh=8)
HEAD_DIM = 4
STORED_P_YES = 0.5

GEN = {"model": "m/A", "backend": "vllm", "tokenizer": None, "mistral_format": False,
       "assistant_prefill": "", "judge_max_tokens": 16, "dataset": "gsm8k",
       "split": "train", "seed": 42, "problems_sha256": "train-digest"}


def record(idx, correct=True):
    answer = "4" if correct else "5"
    return {"idx": idx, "dataset": "gsm8k", "problem": f"problem {idx}",
            "phase1_solution": f"solution {idx}", "phase1_answer": answer,
            "gold_answer": "4", "majority_answer": answer,
            "judge_verdict": "YES" if correct else "NO", "labelable": True}


class PyesCrossCheckTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.records = [record(i, correct=i % 2 == 0) for i in range(6)]
        self.raw = self.dir / "raw.jsonl"
        self.raw.write_text("".join(json.dumps(r) + "\n" for r in self.records))
        self.hidden = self.dir / "hidden_rich.npz"
        dump = Dump(str(self.hidden), len(self.records), model="m/A", store_resid=False,
                    dataset="gsm8k", gen_run=dict(GEN), **GEOM)
        rng = np.random.default_rng(0)
        for rec in self.records:
            dump.add(rec, rng.normal(size=(GEOM["nL"], GEOM["Dh"])).astype(np.float16),
                     STORED_P_YES)
        dump.flush()
        self.topk = self.dir / "verdict_topk.npz"

    def _shas(self):
        return np.load(self.hidden)["rec_sha"]

    def _write_topk(self, *, idx=None, p_yes=None, rec_sha=None):
        d = np.load(self.hidden)
        idx = d["idx"] if idx is None else np.asarray(idx)
        if p_yes is None:
            p_yes = np.linspace(0.1, 0.9, len(idx)).astype(np.float32)
        rec_sha = self._shas() if rec_sha is None else np.asarray(rec_sha)
        np.savez(self.topk, idx=np.asarray(idx, dtype=np.int32),
                 p_yes_shipped=np.asarray(p_yes, dtype=np.float32), rec_sha=rec_sha)

    def load(self):
        return load_split(str(self.hidden), str(self.raw), HEAD_DIM)

    def test_without_a_topk_dump_there_is_nothing_to_check(self):
        split = self.load()
        self.assertEqual(split.info["p_yes_check"]["status"], "no_topk_dump")
        np.testing.assert_allclose(split.p_yes, STORED_P_YES)

    def test_an_agreeing_topk_dump_is_reported_as_agreeing(self):
        self._write_topk(p_yes=np.full(len(self.records), STORED_P_YES))
        split = self.load()
        chk = split.info["p_yes_check"]
        self.assertEqual(chk["status"], "agrees")
        self.assertEqual(chk["n_disagree"], 0)
        np.testing.assert_allclose(split.p_yes, STORED_P_YES)

    def test_a_disagreeing_topk_dump_is_reported_and_does_not_change_p_yes(self):
        """The stale-dump case: report it, keep the activation dump's own column."""
        self._write_topk()                       # 0.1 .. 0.9 vs a stored 0.5
        split = self.load()
        chk = split.info["p_yes_check"]
        self.assertEqual(chk["status"], "disagrees")
        # linspace(0.1, 0.9, 6) puts no point within 0.01 of the stored 0.5.
        self.assertEqual(chk["n_disagree"], len(self.records))
        self.assertGreater(chk["max_delta"], 0.3)
        np.testing.assert_allclose(split.p_yes, STORED_P_YES)

    def test_the_check_is_aligned_by_idx_not_by_row_order(self):
        """Rows listed in another order must not read as a disagreement."""
        d = np.load(self.hidden)
        order = np.argsort(-d["idx"])            # reversed
        self._write_topk(idx=d["idx"][order],
                         p_yes=np.full(len(self.records), STORED_P_YES),
                         rec_sha=self._shas()[order])
        self.assertEqual(self.load().info["p_yes_check"]["status"], "agrees")

    def test_partial_coverage_is_inconclusive_rather_than_a_disagreement(self):
        d = np.load(self.hidden)
        keep = slice(0, len(self.records) - 1)
        self._write_topk(idx=d["idx"][keep],
                         p_yes=np.full(len(self.records) - 1, 0.9),
                         rec_sha=self._shas()[keep])
        chk = self.load().info["p_yes_check"]
        self.assertEqual(chk["status"], "partial_coverage")
        self.assertEqual(chk["n_missing"], 1)

    def test_a_fingerprint_disagreement_is_still_fatal(self):
        """Different RECORDS is a different claim than a different environment."""
        bad = np.array(["deadbeef" + "0" * 8] * len(self.records))
        self._write_topk(rec_sha=bad)
        with self.assertRaises(ValueError) as cm:
            self.load()
        self.assertIn("different records", str(cm.exception))

    def test_a_topk_dump_without_p_yes_is_inconclusive(self):
        d = np.load(self.hidden)
        np.savez(self.topk, idx=d["idx"], top_ids=np.zeros((len(d["idx"]), 4), np.int32))
        self.assertEqual(self.load().info["p_yes_check"]["status"],
                         "topk_dump_lacks_p_yes")


if __name__ == "__main__":
    unittest.main()
