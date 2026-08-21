"""Resume safety: `idx` is a position in the shuffled subset and raw.jsonl carries no
per-record provenance, so resuming into the wrong directory mixes incompatible records
with no visible symptom. These guard the checks that make that impossible."""
import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_eval import (_ADVISORY_KEYS, _MONOTONIC_KEYS, check_fingerprint,
                      problems_digest, read_raw_records, rewrite_raw, run_fingerprint)

# Every generation setting that changes the weights, the prompt, or the shape of a
# record — and therefore must make a resume fatal.
OUTPUT_CHANGING = {
    "seed": 7, "model": "other/y", "backend": "transformers",
    "gguf_file": "q4.gguf", "quantization": "fp8", "kv_cache_dtype": "fp8",
    "tokenizer": "other/tok", "mistral_format": True,
    "assistant_prefill": "\n</think>\n\n", "judge_max_tokens": 96,
    # A solution cut off by this budget has no final answer and is labelled wrong, so
    # two budgets in one file are two different experiments.
    "solve_max_tokens": 2048,
    "max_model_len": 8192, "k_samples": 5,
}

_ARG_DEFAULTS = dict(dataset="gsm8k", seed=42, model="m/A", backend="vllm",
                     gguf_file="", quantization=None, kv_cache_dtype=None,
                     tokenizer=None, mistral_format=False, assistant_prefill="",
                     judge_max_tokens=16, solve_max_tokens=1024,
                     max_model_len=4096, k_samples=1)

EXAMPLES = [{"question": f"q{i}"} for i in range(80)]


def args(**over):
    return argparse.Namespace(**{**_ARG_DEFAULTS, **over})


def fp(**over):
    os.environ["GSM8K_SPLIT"] = "train"
    base = run_fingerprint(args(), EXAMPLES)
    base.update(over)
    return base


class ProblemDigestTests(unittest.TestCase):
    def test_digest_is_prefix_stable_as_n_grows(self):
        """The loader shuffles then truncates, so a 100-problem pilot and a full run
        share a prefix; hashing all N would spuriously differ whenever N grew."""
        ex = [{"question": f"q{i}"} for i in range(500)]
        self.assertEqual(problems_digest(ex[:100]), problems_digest(ex))

    def test_digest_changes_when_the_problems_change(self):
        a = [{"question": f"q{i}"} for i in range(80)]
        b = [{"question": f"q{i}"} for i in range(80)]
        b[0] = {"question": "different"}
        self.assertNotEqual(problems_digest(a), problems_digest(b))


class FingerprintCoverageTests(unittest.TestCase):
    def test_every_output_changing_setting_is_an_identity_field(self):
        """Two models' generations must not be able to land in one raw.jsonl, and the
        same holds for two quantizations, two prompt templates, or two judge budgets."""
        identity = set(fp()) - _ADVISORY_KEYS
        for field in OUTPUT_CHANGING:
            self.assertIn(field, identity,
                          f"{field} can change a record but is not an identity field")
        for field in ("dataset", "split", "problems_sha256"):
            self.assertIn(field, identity)

    def test_operational_knobs_are_not_fingerprinted(self):
        """They cannot change what was recorded, so they must not block a resume on
        different hardware."""
        got = fp()
        for field in ("gpu_memory_utilization", "max_num_seqs", "max_concurrent",
                      "output_dir"):
            self.assertNotIn(field, got)

    def test_sample_size_is_recorded_but_is_not_an_identity_field(self):
        """It may grow between resumes, so it is compared with >= rather than ==."""
        self.assertEqual(fp()["n_problems"], len(EXAMPLES))
        self.assertIn("n_problems", _MONOTONIC_KEYS)


class CheckFingerprintTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.meta = Path(self._tmp.name) / "run_meta.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_fresh_directory_writes_the_fingerprint(self):
        check_fingerprint(self.meta, fp(), raw_exists=False)
        self.assertEqual(json.loads(self.meta.read_text())["model"], "m/A")

    def test_existing_raw_without_metadata_is_rejected(self):
        """Writing a fingerprint here would stamp pre-existing records with THIS
        invocation's identity regardless of what actually produced them."""
        with self.assertRaises(SystemExit) as cm:
            check_fingerprint(self.meta, fp(), raw_exists=True)
        self.assertIn("no run_meta.json", str(cm.exception))
        self.assertFalse(self.meta.exists())

    def test_existing_raw_without_metadata_can_be_adopted_explicitly(self):
        check_fingerprint(self.meta, fp(), raw_exists=True, adopt_existing=True)
        self.assertTrue(self.meta.exists())

    def test_every_output_changing_setting_makes_a_resume_fatal(self):
        check_fingerprint(self.meta, fp(), raw_exists=False)
        for field, value in OUTPUT_CHANGING.items():
            with self.assertRaises(SystemExit, msg=f"{field} drift not caught") as cm:
                check_fingerprint(self.meta, fp(**{field: value}), raw_exists=True)
            self.assertIn(field, str(cm.exception))

    def test_problem_set_drift_is_fatal(self):
        check_fingerprint(self.meta, fp(), raw_exists=False)
        with self.assertRaises(SystemExit):
            check_fingerprint(self.meta, fp(problems_sha256="zzz"), raw_exists=True)

    def test_label_policy_change_is_advisory_only(self):
        check_fingerprint(self.meta, fp(), raw_exists=False)
        check_fingerprint(self.meta, fp(label_policy="something_v9"), raw_exists=True)
        self.assertEqual(json.loads(self.meta.read_text())["label_policy"],
                         "something_v9")


class SampleSizeTests(unittest.TestCase):
    """The digest only covers a 64-problem prefix, so it cannot by itself tell a
    100-problem run from an 80-problem one. Without the recorded size, resuming a
    100-record directory with --n_problems 80 passes every check and still reports all
    100 records."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.meta = Path(self._tmp.name) / "run_meta.json"
        self.big = [{"question": f"q{i}"} for i in range(100)]

    def _fp(self, examples):
        os.environ["GSM8K_SPLIT"] = "train"
        return run_fingerprint(args(), examples)

    def test_the_prefix_digest_alone_cannot_see_the_shrink(self):
        self.assertEqual(problems_digest(self.big), problems_digest(self.big[:80]))

    def test_shrinking_the_sample_is_fatal(self):
        check_fingerprint(self.meta, self._fp(self.big), raw_exists=False)
        with self.assertRaises(SystemExit) as cm:
            check_fingerprint(self.meta, self._fp(self.big[:80]), raw_exists=True)
        self.assertIn("n_problems", str(cm.exception))

    def test_growing_the_sample_is_allowed_and_recorded(self):
        check_fingerprint(self.meta, self._fp(self.big[:80]), raw_exists=False)
        check_fingerprint(self.meta, self._fp(self.big), raw_exists=True)
        self.assertEqual(json.loads(self.meta.read_text())["n_problems"], 100)

    def test_a_pilot_below_the_digest_width_can_be_grown(self):
        """Its digest covers 40 questions and the resumed run's covers 64; comparing
        them directly would read identical data as a different problem set."""
        small = self.big[:40]
        check_fingerprint(self.meta, self._fp(small), raw_exists=False,
                          digest_at=lambda k: problems_digest(small, k))
        check_fingerprint(self.meta, self._fp(self.big), raw_exists=True,
                          digest_at=lambda k: problems_digest(self.big, k))
        stored = json.loads(self.meta.read_text())
        self.assertEqual(stored["problems_fingerprint_n"], 64)
        self.assertEqual(stored["problems_sha256"], problems_digest(self.big))

    def test_a_grown_pilot_with_different_problems_is_still_fatal(self):
        small = self.big[:40]
        other = [{"question": f"other{i}"} for i in range(100)]
        check_fingerprint(self.meta, self._fp(small), raw_exists=False,
                          digest_at=lambda k: problems_digest(small, k))
        with self.assertRaises(SystemExit):
            check_fingerprint(self.meta, self._fp(other), raw_exists=True,
                              digest_at=lambda k: problems_digest(other, k))


class RawFileRepairTests(unittest.TestCase):
    """A killed run can leave a partial final line; a pre-repair resume can leave
    duplicate idx rows. Appending onto a corrupt tail destroys a second valid record,
    and a duplicated idx is counted twice in the report while extraction keeps one."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.raw = Path(self._tmp.name) / "raw.jsonl"

    def write(self, text):
        self.raw.write_text(text)

    def test_partial_final_line_is_dropped(self):
        self.write(json.dumps({"idx": 0, "problem": "a"}) + "\n" + '{"idx": 1, "prob')
        records, stats = read_raw_records(self.raw)
        self.assertEqual(set(records), {0})
        self.assertEqual(stats["malformed"], 1)

    def test_rewrite_truncates_the_corrupt_tail(self):
        self.write(json.dumps({"idx": 0}) + "\n" + '{"idx": 1, "prob')
        records, _ = read_raw_records(self.raw)
        rewrite_raw(self.raw, records)
        again, stats = read_raw_records(self.raw)
        self.assertEqual(stats["malformed"], 0)
        self.assertEqual(set(again), {0})

    def test_duplicate_idx_keeps_one_row_and_prefers_the_successful_one(self):
        self.write("\n".join([
            json.dumps({"idx": 0, "error": "OOM"}),
            json.dumps({"idx": 0, "phase1_answer": "5"}),
            json.dumps({"idx": 1, "phase1_answer": "7"}),
            json.dumps({"idx": 1, "error": "timeout"}),
        ]) + "\n")
        records, stats = read_raw_records(self.raw)
        self.assertEqual(stats["duplicate"], 2)
        self.assertEqual(records[0]["phase1_answer"], "5")
        self.assertIsNone(records[1].get("error"))

    def test_rewrite_is_ordered_by_idx(self):
        rewrite_raw(self.raw, {2: {"idx": 2}, 0: {"idx": 0}, 1: {"idx": 1}})
        self.assertEqual([json.loads(l)["idx"] for l in self.raw.read_text().splitlines()],
                         [0, 1, 2])


class LegacyMetadataTests(unittest.TestCase):
    """An identity field absent from an older fingerprint is UNKNOWN, not matching.
    Backfilling it silently would stamp this invocation's value onto records that may
    have been produced with a different one — the same unverified assertion the
    raw-without-metadata check already refuses."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.meta = Path(self._tmp.name) / "run_meta.json"
        self.current = fp()
        legacy = {k: self.current[k]
                  for k in ("dataset", "split", "seed", "model", "backend",
                            "assistant_prefill", "problems_sha256", "label_policy")}
        self.meta.write_text(json.dumps(legacy, indent=2, sort_keys=True))

    def test_the_sample_size_fields_are_backfilled_without_adoption(self):
        """They are not identity fields: recording this invocation's value cannot
        mislabel a record, and requiring --adopt_existing for them would make every
        run_meta.json written before they existed need a manual assertion to resume."""
        older = {k: v for k, v in self.current.items() if k not in _MONOTONIC_KEYS}
        self.meta.write_text(json.dumps(older, indent=2, sort_keys=True))
        check_fingerprint(self.meta, self.current, raw_exists=True)
        stored = json.loads(self.meta.read_text())
        for field in _MONOTONIC_KEYS:
            self.assertEqual(stored[field], self.current[field])

    def test_missing_identity_fields_require_explicit_adoption(self):
        with self.assertRaises(SystemExit) as cm:
            check_fingerprint(self.meta, self.current, raw_exists=True)
        msg = str(cm.exception)
        for field in ("judge_max_tokens", "k_samples", "quantization", "mistral_format",
                      "tokenizer"):
            self.assertIn(field, msg)
        self.assertIn("--adopt_existing", msg)

    def test_refusal_leaves_the_file_untouched(self):
        before = self.meta.read_text()
        with self.assertRaises(SystemExit):
            check_fingerprint(self.meta, self.current, raw_exists=True)
        self.assertEqual(self.meta.read_text(), before)

    def test_adoption_records_the_previously_unknown_values(self):
        check_fingerprint(self.meta, self.current, raw_exists=True, adopt_existing=True)
        stored = json.loads(self.meta.read_text())
        for field in OUTPUT_CHANGING:
            self.assertIn(field, stored)

    def test_drift_is_still_fatal_after_adoption(self):
        check_fingerprint(self.meta, self.current, raw_exists=True, adopt_existing=True)
        with self.assertRaises(SystemExit):
            check_fingerprint(self.meta, fp(k_samples=5), raw_exists=True)


if __name__ == "__main__":
    unittest.main()
