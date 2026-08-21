"""Nothing downstream of generation can see a swapped raw file.

`idx` is a position in a shuffled subset, so every stage joins on a key that two
different runs share. The dump's model/geometry/dataset checks all pass against a raw
file it was never extracted from: the labels are recomputed from the new text, the
activations stay where they are, and the probe reports a number about nothing. These
guard the per-record fingerprint and the generation binding that make that impossible.
"""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extraction_common import Dump, prepare_extraction
from metrics.provenance import read_run_meta, record_digest

GEOM = dict(nL=2, H=8, nH=2, Dh=8)

RUN_META = {
    "dataset": "gsm8k", "split": "train", "seed": 42,
    "problems_sha256": "abc123", "problems_fingerprint_n": 64, "n_problems": 3,
    "model": "m/A", "backend": "vllm", "gguf_file": None, "quantization": None,
    "kv_cache_dtype": None, "tokenizer": None, "mistral_format": False,
    "assistant_prefill": "", "judge_max_tokens": 16, "max_model_len": 4096,
    "k_samples": 1, "label_policy": "numeric_equivalence_v2",
}


def record(idx, problem="what is 2+2?", solution="it is 4", answer="4", gold="4"):
    return {"idx": idx, "dataset": "gsm8k", "problem": problem,
            "phase1_solution": solution, "phase1_answer": answer, "gold_answer": gold,
            "majority_answer": answer, "judge_verdict": "YES", "labelable": True}


class RecordDigestTests(unittest.TestCase):
    def test_the_prompt_fields_are_covered(self):
        """problem and phase1_solution ARE the judge prompt; a change in either means
        the activation was read at a different token sequence."""
        base = record(0)
        for field, value in (("problem", "different"), ("phase1_solution", "different"),
                             ("phase1_answer", "5"), ("gold_answer", "5"),
                             ("majority_answer", "5"), ("idx", 1)):
            self.assertNotEqual(record_digest(base), record_digest({**base, field: value}),
                                f"{field} does not change the digest")

    def test_judge_verdict_is_covered(self):
        """The verdict is an analysis input, not a derived value: probe_models reads it
        as the binary judge baseline and extract_verdict_topk exists to study it. Two
        raw files whose problems match but whose verdicts differ — the same set
        re-judged at another budget — used to digest identically, so a resume spliced
        them together in silence."""
        base = record(0)
        self.assertNotEqual(record_digest(base),
                            record_digest({**base, "judge_verdict": "NO"}))

    def test_digest_is_versioned(self):
        """Adding a field makes every stored digest mismatch at once. The version is
        what lets that be reported as a definition change rather than as a data
        mismatch, so it must actually enter the hash."""
        import metrics.provenance as prov

        self.assertGreaterEqual(prov.DIGEST_VERSION, 2)
        at_current = record_digest(record(0))
        original = prov.DIGEST_VERSION
        try:
            prov.DIGEST_VERSION = original + 1
            self.assertNotEqual(at_current, record_digest(record(0)))
        finally:
            prov.DIGEST_VERSION = original

    def test_irrelevant_fields_do_not_change_it(self):
        """Re-deriving a label must not invalidate an activation that is still correct."""
        base = record(0)
        self.assertEqual(record_digest(base),
                         record_digest({**base, "solve_correct": False, "labelable": 1}))

    def test_legacy_records_without_a_dataset_field_hash_as_gsm8k(self):
        base = record(0)
        legacy = {k: v for k, v in base.items() if k != "dataset"}
        self.assertEqual(record_digest(base), record_digest(legacy))


class DumpResumeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = str(Path(self._tmp.name) / "hidden_rich.npz")
        self.records = [record(i) for i in range(3)]

    def write_dump(self, records):
        dump = Dump(self.out, len(records), model="m/A", store_resid=False,
                    dataset="gsm8k", **GEOM)
        for rec in records:
            dump.add(rec, np.zeros((GEOM["nL"], GEOM["Dh"]), np.float16), 0.5)
        dump.flush()
        return dump

    def fresh(self, n):
        return Dump(self.out, n, model="m/A", store_resid=False, dataset="gsm8k", **GEOM)

    def test_resume_against_the_same_records_succeeds(self):
        self.write_dump(self.records[:2])
        dump = self.fresh(3)
        dump.resume(self.records)
        self.assertEqual(dump.filled, 2)
        self.assertEqual(dump.done_ids, {0, 1})

    def test_resume_against_a_different_raw_file_is_fatal(self):
        """Same model, same dataset, same indices — and completely different problems."""
        self.write_dump(self.records[:2])
        other = [record(i, problem=f"unrelated problem {i}", solution="quite different")
                 for i in range(3)]
        with self.assertRaises(SystemExit) as cm:
            self.fresh(3).resume(other)
        self.assertIn("different records", str(cm.exception))

    def test_a_changed_gold_answer_is_fatal(self):
        self.write_dump(self.records[:2])
        edited = [dict(r) for r in self.records]
        edited[1]["gold_answer"] = "7"
        with self.assertRaises(SystemExit):
            self.fresh(3).resume(edited)

    def test_a_legacy_dump_without_fingerprints_still_resumes(self):
        """Warns instead of failing: an old dump cannot be re-fingerprinted, and
        refusing it would strand every activation extracted before this check."""
        self.write_dump(self.records[:2])
        z = dict(np.load(self.out))
        z.pop("rec_sha")
        np.savez(self.out, **z)
        dump = self.fresh(3)
        dump.resume(self.records)
        self.assertEqual(dump.filled, 2)

    def test_a_changed_judge_prompt_is_fatal(self):
        """Two extraction passes into one dump under different judge wording would mix
        activations read at two different prompts, with nothing recording it."""
        dump = self.write_dump(self.records[:2])
        resumed = self.fresh(3)
        resumed.meta["prompt_identity"] = dict(dump.meta["prompt_identity"],
                                               judge_prompt_sha256="different")
        with self.assertRaises(SystemExit) as cm:
            resumed.resume(self.records)
        self.assertIn("prompt_identity", str(cm.exception))


class PrepareExtractionTests(unittest.TestCase):
    """Generation settings that extraction cannot reproduce must stop it, not be
    silently ignored: each one moves the token the activation is read at."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.input = self.dir / "raw.jsonl"
        self.input.write_text(json.dumps(record(0)) + "\n")
        self.parser = argparse.ArgumentParser()

    def write_meta(self, **over):
        (self.dir / "run_meta.json").write_text(json.dumps({**RUN_META, **over}))

    def args(self, **over):
        base = dict(input=str(self.input), out=str(self.dir / "o.npz"), model="m/A",
                    dataset="gsm8k", tokenizer=None, limit=0, save_every=1500)
        return argparse.Namespace(**{**base, **over})

    def test_matching_run_records_the_generation_identity(self):
        self.write_meta()
        gen = prepare_extraction(self.parser, self.args())
        self.assertEqual(gen["model"], "m/A")
        self.assertEqual(gen["problems_sha256"], "abc123")
        self.assertEqual(len(gen["judge_prompt_sha256"]), 16)

    def test_mistral_format_generation_cannot_be_extracted(self):
        """It templates with mistral_common; every extractor uses AutoTokenizer, which
        tokenizes the same conversation differently."""
        self.write_meta(mistral_format=True)
        with self.assertRaises(SystemExit) as cm:
            prepare_extraction(self.parser, self.args())
        self.assertIn("mistral", str(cm.exception).lower())

    def test_a_custom_generation_tokenizer_is_adopted(self):
        self.write_meta(tokenizer="other/tok")
        args = self.args()
        prepare_extraction(self.parser, args)
        self.assertEqual(args.tokenizer, "other/tok")

    def test_a_conflicting_tokenizer_is_fatal(self):
        self.write_meta(tokenizer="other/tok")
        with self.assertRaises(SystemExit) as cm:
            prepare_extraction(self.parser, self.args(tokenizer="third/tok"))
        self.assertIn("tokenizer", str(cm.exception))

    def test_a_prefill_used_at_generation_but_not_at_extraction_is_fatal(self):
        self.write_meta(assistant_prefill="\n</think>\n\n")
        with self.assertRaises(SystemExit) as cm:
            prepare_extraction(self.parser, self.args())
        self.assertIn("assistant_prefill", str(cm.exception))

    def test_a_prefill_applied_only_at_extraction_is_fatal(self):
        """The mirror case run_full.sh could produce: generation on a backend that
        ignores the prefill, extraction applying it."""
        self.write_meta()
        with self.assertRaises(SystemExit):
            prepare_extraction(self.parser, self.args(assistant_prefill="\n</think>\n\n"))

    def test_a_matching_prefill_passes(self):
        self.write_meta(assistant_prefill="\n</think>\n\n")
        gen = prepare_extraction(self.parser,
                                 self.args(assistant_prefill="\n</think>\n\n"))
        self.assertEqual(gen["extraction_assistant_prefill"], "\n</think>\n\n")

    def test_a_different_model_is_fatal(self):
        self.write_meta(model="other/B")
        with self.assertRaises(SystemExit):
            prepare_extraction(self.parser, self.args())

    def test_a_different_dataset_is_fatal(self):
        self.write_meta(dataset="math")
        with self.assertRaises(SystemExit):
            prepare_extraction(self.parser, self.args())

    def test_a_missing_run_meta_warns_but_proceeds(self):
        """Dumps generated before run_meta.json existed must stay extractable."""
        self.assertIsNone(read_run_meta(str(self.input)))
        gen = prepare_extraction(self.parser, self.args())
        self.assertEqual(gen["extraction_tokenizer"], "m/A")

    def test_invalid_numeric_args_are_rejected(self):
        self.write_meta()
        for over in ({"save_every": 0}, {"limit": -1}):
            with self.assertRaises(SystemExit):
                prepare_extraction(self.parser, self.args(**over))


if __name__ == "__main__":
    unittest.main()
