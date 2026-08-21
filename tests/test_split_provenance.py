"""The probe side of the same hole: two splits are loaded independently and never
compared, so a GSM8K train dump and a MATH test dump — or two runs of the same
problems — fit and report a held-out AUC without an error anywhere."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filtering"))
from extraction_common import Dump
from probe_models import check_split_provenance, load_split

GEOM = dict(nL=2, H=8, nH=2, Dh=8)
HEAD_DIM = 4

GEN = {"model": "m/A", "backend": "vllm", "tokenizer": None, "mistral_format": False,
       "assistant_prefill": "", "judge_max_tokens": 16, "dataset": "gsm8k",
       "split": "train", "seed": 42, "problems_sha256": "train-digest"}


def record(idx, correct=True, problem=None):
    answer = "4" if correct else "5"
    return {"idx": idx, "dataset": "gsm8k",
            "problem": problem or f"problem {idx}", "phase1_solution": f"solution {idx}",
            "phase1_answer": answer, "gold_answer": "4", "majority_answer": answer,
            "judge_verdict": "YES" if correct else "NO", "labelable": True}


class SplitFixture:
    """A matched (hidden_rich.npz, raw.jsonl) pair on disk."""

    def __init__(self, directory, records, *, model="m/A", dataset="gsm8k", gen=None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.raw = self.dir / "raw.jsonl"
        self.raw.write_text("".join(json.dumps(r) + "\n" for r in records))
        self.hidden = self.dir / "hidden_rich.npz"
        dump = Dump(str(self.hidden), len(records), model=model, store_resid=False,
                    dataset=dataset, gen_run=gen if gen is not None else dict(GEN), **GEOM)
        rng = np.random.default_rng(0)
        for rec in records:
            dump.add(rec, rng.normal(size=(GEOM["nL"], GEOM["Dh"])).astype(np.float16), 0.5)
        dump.flush()

    def load(self):
        return load_split(str(self.hidden), str(self.raw), HEAD_DIM)


class LoadSplitAlignmentTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.records = [record(i, correct=i % 2 == 0) for i in range(6)]

    def test_a_matched_pair_loads(self):
        split = SplitFixture(self.root / "a", self.records).load()
        self.assertEqual(len(split), 6)
        self.assertEqual(split.Z.shape, (6, GEOM["nL"] * GEOM["Dh"] // HEAD_DIM, HEAD_DIM))

    def test_activations_from_another_run_are_rejected(self):
        """Same indices, same model, same dataset, different problems."""
        fixture = SplitFixture(self.root / "a", self.records)
        other = [record(i, problem=f"unrelated {i}") for i in range(6)]
        fixture.raw.write_text("".join(json.dumps(r) + "\n" for r in other))
        with self.assertRaises(ValueError) as cm:
            fixture.load()
        self.assertIn("different records", str(cm.exception))

    def test_a_legacy_dump_without_fingerprints_still_loads(self):
        fixture = SplitFixture(self.root / "a", self.records)
        z = dict(np.load(fixture.hidden))
        z.pop("rec_sha")
        np.savez(fixture.hidden, **z)
        self.assertEqual(len(fixture.load()), 6)


class CheckSplitProvenanceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.records = [record(i, correct=i % 2 == 0) for i in range(6)]

    def make(self, name, **over):
        gen = {**GEN, **over.pop("gen", {})}
        return SplitFixture(self.root / name, over.pop("records", self.records),
                            gen=gen, **over).load()

    def test_a_matched_pair_passes(self):
        tr = self.make("train")
        te = self.make("test", gen={"split": "test", "problems_sha256": "test-digest"})
        check_split_provenance(tr, te)

    def test_a_different_model_is_fatal(self):
        tr = self.make("train")
        te = self.make("test", model="m/B",
                       gen={"split": "test", "problems_sha256": "test-digest",
                            "model": "m/B"})
        with self.assertRaises(SystemExit) as cm:
            check_split_provenance(tr, te)
        self.assertIn("model", str(cm.exception))

    def test_a_different_judge_budget_is_fatal(self):
        tr = self.make("train")
        te = self.make("test", gen={"split": "test", "problems_sha256": "test-digest",
                                    "judge_max_tokens": 96})
        with self.assertRaises(SystemExit) as cm:
            check_split_provenance(tr, te)
        self.assertIn("judge_max_tokens", str(cm.exception))

    def test_a_prefill_on_only_one_split_is_fatal(self):
        tr = self.make("train")
        te = self.make("test", gen={"split": "test", "problems_sha256": "test-digest",
                                    "assistant_prefill": "\n</think>\n\n"})
        with self.assertRaises(SystemExit):
            check_split_provenance(tr, te)

    def test_the_same_problem_set_on_both_sides_is_fatal(self):
        """Not a mismatch — leakage. It inflates the headline number instead of
        corrupting it, so nothing else would catch it."""
        tr = self.make("train")
        te = self.make("test")
        with self.assertRaises(SystemExit) as cm:
            check_split_provenance(tr, te)
        self.assertIn("SAME problem set", str(cm.exception))

    def test_a_math_test_split_against_a_gsm8k_train_split_is_fatal(self):
        """Their answers obey different equivalence policies and their judge prompts
        differ, so the two feature spaces are not the same space."""
        math_records = [{**record(i), "dataset": "math",
                         "phase1_answer": "\\frac{1}{2}", "gold_answer": "\\frac{1}{2}",
                         "majority_answer": "\\frac{1}{2}"} for i in range(6)]
        tr = self.make("train")
        te = SplitFixture(self.root / "test_math", math_records, dataset="math",
                          gen={**GEN, "dataset": "math", "split": "test",
                               "problems_sha256": "test-digest"}).load()
        with self.assertRaises(SystemExit) as cm:
            check_split_provenance(tr, te)
        self.assertIn("dataset", str(cm.exception))

    def test_the_check_can_be_downgraded_to_a_warning(self):
        tr = self.make("train")
        te = self.make("test")
        check_split_provenance(tr, te, strict=False)

    def test_run_meta_beside_raw_jsonl_wins_over_the_dump_copy(self):
        """It describes the file the labels actually came from."""
        tr = self.make("train")
        te = self.make("test", gen={"split": "test", "problems_sha256": "test-digest"})
        Path(te.info["paths"]["raw"]).parent.joinpath("run_meta.json").write_text(
            json.dumps({**GEN, "split": "test", "problems_sha256": "test-digest",
                        "model": "m/B"}))
        te = load_split(te.info["paths"]["hidden"], te.info["paths"]["raw"], HEAD_DIM)
        with self.assertRaises(SystemExit) as cm:
            check_split_provenance(tr, te)
        self.assertIn("generation model", str(cm.exception))


if __name__ == "__main__":
    unittest.main()


class DigestVersionResumeTests(unittest.TestCase):
    """Adding a field to the record digest invalidates every stored rec_sha at once.
    Reporting that as "these are different records" would send you hunting for a data
    problem that does not exist, so it must be reported as what it is."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.records = [record(i) for i in range(4)]
        self.fixture = SplitFixture(self.root / "run", self.records)

    def _resume(self):
        dump = Dump(str(self.fixture.hidden), len(self.records), model="m/A",
                    store_resid=False, dataset="gsm8k", gen_run=dict(GEN), **GEOM)
        dump.resume(self.records)
        return dump

    def test_a_matching_dump_resumes(self):
        self.assertEqual(self._resume().filled, len(self.records))

    def test_the_dump_records_the_digest_version_it_used(self):
        import metrics.provenance as prov
        z = np.load(self.fixture.hidden)
        self.assertEqual(json.loads(str(z["meta"]))["digest_version"],
                         prov.DIGEST_VERSION)

    def test_a_stale_digest_version_is_reported_as_a_definition_change(self):
        import metrics.provenance as prov

        original = prov.DIGEST_VERSION
        try:
            prov.DIGEST_VERSION = original + 1
            with self.assertRaises(SystemExit) as cm:
                self._resume()
        finally:
            prov.DIGEST_VERSION = original
        msg = str(cm.exception)
        self.assertIn("digest_version", msg)
        self.assertIn("definition change", msg)
        # It must NOT accuse the raw file of holding different records.
        self.assertNotIn("different records", msg)


class MissingProvenanceTests(unittest.TestCase):
    """Dumps predating the provenance work carry none of these fields, and nothing can
    add them after the fact. Such a dump is not wrong, it is unverifiable — a different
    claim, and one the caller should have to make deliberately."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.records = [record(i) for i in range(4)]

    def test_a_current_dump_is_fully_authenticated(self):
        split = SplitFixture(self.root / "new", self.records).load()
        self.assertEqual(split.info["unauthenticated"], [])

    def test_a_dump_without_a_generation_fingerprint_is_flagged(self):
        split = SplitFixture(self.root / "nogen", self.records, gen={}).load()
        self.assertIn("gen_run", split.info["unauthenticated"])
        self.assertIn("prompt_identity", split.info["unauthenticated"])

    def test_a_dump_without_rec_sha_is_flagged(self):
        fixture = SplitFixture(self.root / "nosha", self.records)
        z = dict(np.load(fixture.hidden))
        z.pop("rec_sha")
        np.savez(fixture.hidden, **z)
        self.assertIn("rec_sha", fixture.load().info["unauthenticated"])
