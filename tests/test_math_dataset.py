"""MATH is an *additive* dataset: every GSM8K path must behave exactly as before.

The dispatch key is a per-record `dataset` field. A record written before that field
existed can only be GSM8K, so absence must mean GSM8K everywhere — several tests below
exist purely to pin that down, because a regression there silently relabels every
historical raw.jsonl.
"""
import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset_registry
from extraction_common import check_records_dataset, read_records
from metrics.correctness import (DEFAULT_DATASET, LABEL_POLICY, MATH_LABEL_POLICY,
                                 answers_equal, answers_numerically_equal,
                                 canonical_answer, is_labelable, label_correctness,
                                 label_policy_for, record_dataset, relabel_result)
from prompts.gsm8k import GSM8KPrompts
from prompts.math import MATHPrompts
from pipeline.backend import LLMBackend as _LLMBackend
from pipeline.solver import parse_answer as numeric_parse_answer


def record(**over):
    base = {"idx": 0, "gold_answer": "42", "phase1_answer": "42",
            "majority_answer": "42", "error": None}
    base.update(over)
    return base


class DispatchDefaultsToGSM8KTests(unittest.TestCase):
    """A record with no `dataset` field must behave bit-identically to before."""

    def test_record_dataset_defaults(self):
        self.assertEqual(record_dataset({}), DEFAULT_DATASET)
        self.assertEqual(record_dataset({"dataset": None}), DEFAULT_DATASET)
        self.assertEqual(DEFAULT_DATASET, "gsm8k")

    def test_answers_equal_without_dataset_is_the_numeric_comparison(self):
        for answer, gold in [("42", "42"), ("42.0", "42"), ("42", "43"),
                             (None, "42"), ("42", None), (r"\frac{1}{2}", "0.5")]:
            self.assertEqual(answers_equal(answer, gold),
                             answers_numerically_equal(answer, gold),
                             f"{answer!r} vs {gold!r}")

    def test_legacy_record_is_labelable_exactly_as_before(self):
        self.assertTrue(is_labelable(record()))
        self.assertFalse(is_labelable(record(gold_answer="not a number")))
        self.assertFalse(is_labelable(record(gold_answer=None)))
        self.assertFalse(is_labelable(record(error="cuda oom")))

    def test_legacy_record_keeps_the_numeric_label_policy(self):
        out = relabel_result(record())
        self.assertEqual(out["label_policy"], LABEL_POLICY)
        self.assertTrue(out["solve_correct"])
        self.assertTrue(out["labelable"])

    def test_gsm8k_still_rejects_latex_gold(self):
        """The numeric policy must not quietly gain LaTeX tolerance."""
        latex = record(dataset="gsm8k", gold_answer=r"\frac{1}{2}",
                       phase1_answer=r"\frac{1}{2}")
        self.assertFalse(is_labelable(latex))
        self.assertIsNone(label_correctness(r"\frac{1}{2}", r"\frac{1}{2}", "gsm8k"))


class MathDispatchTests(unittest.TestCase):
    def test_latex_gold_is_labelable_under_math(self):
        """The whole point of the add-on: without this, roughly half of MATH is
        dropped as 'unparseable gold' at extraction and probe time."""
        rec = record(dataset="math", gold_answer=r"\frac{2}{3}",
                     phase1_answer=r"\dfrac{2}{3}")
        self.assertTrue(is_labelable(rec))
        self.assertTrue(label_correctness(rec["phase1_answer"], rec["gold_answer"],
                                          "math"))

    def test_math_relabel_uses_the_latex_policy(self):
        rec = relabel_result(record(dataset="math", gold_answer=r"\text{even}",
                                    phase1_answer="even", majority_answer="odd"))
        self.assertEqual(rec["label_policy"], MATH_LABEL_POLICY)
        self.assertTrue(rec["solve_correct"])
        self.assertFalse(rec["majority_correct"])
        self.assertTrue(rec["labelable"])

    def test_math_record_with_unboxed_gold_is_unlabelable(self):
        self.assertFalse(is_labelable(record(dataset="math", gold_answer=None)))
        self.assertIsNone(label_correctness("2", None, "math"))

    def test_math_generation_error_is_still_unlabelable(self):
        """An OOM must never read as 'wrong answer the judge caught'."""
        self.assertFalse(is_labelable(record(dataset="math", error="cuda oom")))

    def test_unparseable_model_answer_is_wrong_not_unlabelable(self):
        rec = record(dataset="math", gold_answer=r"\frac{2}{3}", phase1_answer=None)
        self.assertTrue(is_labelable(rec))
        self.assertFalse(label_correctness(None, r"\frac{2}{3}", "math"))

    def test_canonical_answer_dispatch(self):
        self.assertEqual(canonical_answer(r"\dfrac{2}{3}", "math"), r"\frac{2}{3}")
        self.assertIsNone(canonical_answer(r"\dfrac{2}{3}", "gsm8k"))
        self.assertEqual(canonical_answer("2.00", "math"), "2")
        self.assertEqual(canonical_answer("2.00", "gsm8k"), "2")

    def test_label_policy_strings_are_distinct(self):
        self.assertNotEqual(label_policy_for("gsm8k"), label_policy_for("math"))
        self.assertEqual(label_policy_for(None), LABEL_POLICY)
        self.assertEqual(label_policy_for("unknown"), LABEL_POLICY)


class PromptHookTests(unittest.TestCase):
    def test_gsm8k_parse_answer_is_the_untouched_numeric_parser(self):
        """The base-class hook must not change GSM8K parsing in any way."""
        prompts = GSM8KPrompts()
        for text in ["#### 42", "the answer is -12", "**64**", "no number here",
                     "1,234 total\n#### 1,234", "= 7"]:
            self.assertEqual(prompts.parse_answer(text), numeric_parse_answer(text),
                             text)

    def test_gsm8k_canonical_answer_is_numeric(self):
        self.assertEqual(GSM8KPrompts().canonical_answer("64.00"), "64")
        self.assertIsNone(GSM8KPrompts().canonical_answer(r"\frac{1}{2}"))

    def test_math_parse_answer_prefers_the_boxed_value(self):
        prompts = MATHPrompts()
        self.assertEqual(prompts.parse_answer(r"...so \boxed{\frac{1}{2}}."),
                         r"\frac{1}{2}")
        self.assertEqual(prompts.parse_answer(r"\boxed{3} then \boxed{5}"), "5")

    def test_math_parse_answer_falls_back_to_an_explicit_statement(self):
        self.assertEqual(MATHPrompts().parse_answer("The final answer is 42"), "42")

    def test_math_unboxed_fallback_keeps_the_decimal_point(self):
        """Excluding '.' from the capture stopped at the decimal point, so
        "the final answer is 0.5." parsed as 0 — not a parse failure but a different
        number, which can mark a wrong answer correct as easily as the reverse."""
        self.assertEqual(MATHPrompts().parse_answer("The final answer is 0.5."), "0.5")
        self.assertEqual(MATHPrompts().parse_answer("The answer is -2.75"), "-2.75")

    def test_math_unboxed_fallback_still_stops_at_the_sentence_end(self):
        """A period followed by prose ends the answer; only a digit continues it."""
        self.assertEqual(
            MATHPrompts().parse_answer("The answer is 5. This is because x=5."), "5")

    def test_math_parse_answer_returns_none_when_nothing_is_stated(self):
        for text in ["I could not solve this.", "", r"\boxed{}"]:
            self.assertIsNone(MATHPrompts().parse_answer(text), repr(text))

    def test_math_solve_prompt_states_the_boxed_contract(self):
        self.assertIn(r"\boxed", MATHPrompts().solve_prompt("Compute 1+1."))
        self.assertIn("Compute 1+1.", MATHPrompts().solve_prompt("Compute 1+1."))

    def test_math_judge_prompt_asks_for_one_word(self):
        text = MATHPrompts().judge_prompt("P", "S")
        self.assertIn("YES or NO", text)
        self.assertIn("P", text)
        self.assertIn("S", text)

    def test_math_prompts_offer_the_same_judge_variants_as_gsm8k(self):
        """error_detection_compare and friends pick a variant by name; a missing one
        would only surface as an AttributeError mid-run."""
        for name in ("judge_prompt", "judge_prompt_errorhunt",
                     "judge_prompt_errorhunt_soft", "judge_prompt_stepgrade"):
            self.assertTrue(hasattr(MATHPrompts(), name), name)


class MajorityVotingTests(unittest.TestCase):
    def test_math_votes_group_by_latex_form(self):
        """`canonical_numeric_answer` discards every non-numeric answer, which would
        leave most MATH problems with no votes and therefore no majority."""
        import asyncio
        from pipeline.majority import majority_vote_sample

        answers = [r"\dfrac{1}{2}", r"\frac{1}{2}", r"\frac12", r"\frac{1}{3}"]

        from pipeline.backend import LLMBackend

        class FakeBackend(LLMBackend):
            def __init__(self):
                self.remaining = list(answers)

            async def chat_complete(self, messages, **kwargs):
                return rf"work \boxed{{{self.remaining.pop(0)}}}"

        _, _, majority, truncated = asyncio.run(
            majority_vote_sample("P", MATHPrompts(), FakeBackend(), len(answers)))
        self.assertEqual(majority, r"\frac{1}{2}")
        # This double implements only chat_complete, so truncation is unknown — which
        # must be recorded as None, never silently as "not truncated".
        self.assertEqual(truncated, [None] * len(answers))


class RunProblemIntegrationTests(unittest.TestCase):
    """dataset -> prompts -> parse -> label, end to end against a stubbed backend."""

    class FakeBackend(_LLMBackend):
        """Implements only chat_complete, so the base `complete()` reports
        truncated=None — the honest 'this backend cannot tell us' case."""

        def __init__(self, solve_reply, judge_reply="YES"):
            self.solve_reply, self.judge_reply = solve_reply, judge_reply
            self.prompts_seen = []

        async def chat_complete(self, messages, **kwargs):
            content = messages[0]["content"]
            self.prompts_seen.append(content)
            return self.judge_reply if "YES or NO" in content else self.solve_reply

    def _run_problem(self, dataset, solve_reply, gold):
        import asyncio
        from run_eval import run_problem

        backend = self.FakeBackend(solve_reply)
        result = asyncio.run(run_problem(
            idx=3, example={"question": "Compute it."}, gold_answer=gold,
            prompts=dataset_registry.prompts_for(dataset)(), backend=backend,
            k_samples=1, dataset=dataset))
        return result, backend

    def test_math_record_carries_its_dataset_and_policy(self):
        result, backend = self._run_problem(
            "math", r"Working... \boxed{\dfrac{1}{2}}", r"\frac{1}{2}")
        self.assertEqual(result["dataset"], "math")
        self.assertEqual(result["label_policy"], MATH_LABEL_POLICY)
        self.assertEqual(result["phase1_answer"], r"\dfrac{1}{2}")
        self.assertTrue(result["solve_correct"])
        self.assertTrue(result["majority_correct"])
        self.assertTrue(is_labelable(result))
        self.assertTrue(any(r"\boxed" in p for p in backend.prompts_seen))

    def test_math_wrong_answer_is_labelled_wrong(self):
        result, _ = self._run_problem("math", r"\boxed{\frac{1}{3}}", r"\frac{1}{2}")
        self.assertFalse(result["solve_correct"])
        self.assertTrue(is_labelable(result))

    def test_gsm8k_record_is_unchanged(self):
        result, backend = self._run_problem("gsm8k", "reasoning\n#### 42", "42")
        self.assertEqual(result["dataset"], "gsm8k")
        self.assertEqual(result["label_policy"], LABEL_POLICY)
        self.assertEqual(result["phase1_answer"], "42")
        self.assertTrue(result["solve_correct"])
        self.assertTrue(all(r"\boxed" not in p for p in backend.prompts_seen))

    def test_relabel_round_trips_a_math_record(self):
        """The report is rebuilt from raw.jsonl, so relabel must reach the same
        verdict as generation did."""
        result, _ = self._run_problem("math", r"\boxed{\text{even}}", "even")
        serialized = json.loads(json.dumps(result))
        self.assertEqual(relabel_result(serialized)["solve_correct"],
                         result["solve_correct"])
        self.assertTrue(serialized["solve_correct"])


class DatasetRegistryTests(unittest.TestCase):
    def test_every_dataset_has_a_loader_and_prompts(self):
        for name in dataset_registry.DATASETS:
            self.assertIn(name, dataset_registry._LOADER_PATHS, name)
            self.assertTrue(callable(dataset_registry.prompts_for(name)), name)

    def test_unknown_dataset_raises(self):
        with self.assertRaises(KeyError):
            dataset_registry.prompts_for("mmlu")

    def test_split_env_is_per_dataset(self):
        self.assertEqual(dataset_registry.split_env("gsm8k"), "GSM8K_SPLIT")
        self.assertEqual(dataset_registry.split_env("math"), "MATH_SPLIT")

    def test_dataset_split_reads_the_right_env_var(self):
        env = {"GSM8K_SPLIT": "train", "MATH_SPLIT": "test"}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(dataset_registry.dataset_split("gsm8k"), "train")
            self.assertEqual(dataset_registry.dataset_split("math"), "test")

    def test_dataset_split_defaults_to_test(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(dataset_registry.dataset_split("gsm8k"), "test")
            self.assertEqual(dataset_registry.dataset_split("math"), "test")

    def test_math_split_does_not_leak_into_gsm8k(self):
        """Reading $GSM8K_SPLIT for a MATH run would fingerprint the wrong split."""
        with mock.patch.dict(os.environ, {"GSM8K_SPLIT": "train"}, clear=True):
            self.assertEqual(dataset_registry.dataset_split("math"), "test")


class FingerprintSplitTests(unittest.TestCase):
    def test_fingerprint_records_the_math_split(self):
        from run_eval import run_fingerprint

        args = argparse.Namespace(
            dataset="math", seed=42, model="m/A", backend="vllm", gguf_file="",
            quantization=None, kv_cache_dtype=None, tokenizer=None,
            mistral_format=False, assistant_prefill="", judge_max_tokens=16,
            solve_max_tokens=1024, max_model_len=4096, k_samples=1)
        with mock.patch.dict(os.environ, {"MATH_SPLIT": "train", "GSM8K_SPLIT": "test"},
                             clear=False):
            fingerprint = run_fingerprint(args, [{"question": "q"}])
        self.assertEqual(fingerprint["split"], "train")
        self.assertEqual(fingerprint["dataset"], "math")
        self.assertEqual(fingerprint["label_policy"], MATH_LABEL_POLICY)


class ExtractionGuardTests(unittest.TestCase):
    """Extraction re-templates the judge prompt, so the wrong prompt class reads
    activations at a different token sequence than produced the recorded verdict —
    a mismatch with no visible symptom downstream."""

    def test_matching_dataset_passes(self):
        check_records_dataset([{"dataset": "math"}], "math", "f.jsonl")

    def test_mismatched_dataset_aborts(self):
        with self.assertRaises(SystemExit) as cm:
            check_records_dataset([{"dataset": "math"}], "gsm8k", "f.jsonl")
        self.assertIn("math", str(cm.exception))
        self.assertIn("gsm8k", str(cm.exception))

    def test_legacy_records_count_as_gsm8k(self):
        check_records_dataset([{"idx": 0}], "gsm8k", "f.jsonl")
        with self.assertRaises(SystemExit):
            check_records_dataset([{"idx": 0}], "math", "f.jsonl")

    def test_mixed_records_abort(self):
        with self.assertRaises(SystemExit):
            check_records_dataset([{"dataset": "math"}, {"dataset": "gsm8k"}],
                                  "math", "f.jsonl")

    def test_empty_file_is_not_an_error(self):
        check_records_dataset([], "math", "f.jsonl")

    def test_read_records_checks_before_dropping_unlabelable(self):
        """A MATH file read as GSM8K drops every record as 'unparseable gold'; the
        mismatch must be reported instead of a silent '0 labelable records'."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            path.write_text(json.dumps(
                record(dataset="math", gold_answer=r"\frac{1}{2}")) + "\n")
            with self.assertRaises(SystemExit) as cm:
                read_records(str(path), 0, "gsm8k")
            self.assertIn("math", str(cm.exception))
            kept, stats = read_records(str(path), 0, "math")
            self.assertEqual(stats["kept"], 1)
            self.assertEqual(len(kept), 1)

    def test_read_records_without_a_dataset_argument_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            path.write_text(json.dumps(record()) + "\n")
            kept, stats = read_records(str(path))
            self.assertEqual(stats, {"loaded": 1, "kept": 1, "errors": 0,
                                     "truncated": 0, "no_gold": 0})
            self.assertEqual(len(kept), 1)


class DumpMetadataTests(unittest.TestCase):
    def test_dump_records_the_dataset_and_its_policy(self):
        from extraction_common import Dump

        dump = Dump("/dev/null", 1, nL=2, H=8, nH=2, Dh=8, model="m/A",
                    store_resid=False, dataset="math")
        self.assertEqual(dump.meta["dataset"], "math")
        self.assertEqual(dump.meta["label_policy"], MATH_LABEL_POLICY)

    def test_dump_defaults_to_gsm8k(self):
        from extraction_common import Dump

        dump = Dump("/dev/null", 1, nL=2, H=8, nH=2, Dh=8, model="m/A",
                    store_resid=False)
        self.assertEqual(dump.meta["dataset"], DEFAULT_DATASET)
        self.assertEqual(dump.meta["label_policy"], LABEL_POLICY)

    def test_dataset_is_an_identity_field_not_an_advisory_one(self):
        """Two datasets' activations in one dump would be labelled under two policies
        and judged by two prompts."""
        from extraction_common import Dump

        self.assertNotIn("dataset", Dump._META_ADVISORY)


class MATHLoaderTests(unittest.TestCase):
    """Offline: the HF download is exercised by smoke_test.sh, not by the suite."""

    def setUp(self):
        from loaders.math import MATHLoader
        self.loader = MATHLoader()

    def test_gold_is_the_boxed_answer_of_the_reference_solution(self):
        example = {"solution": r"Factoring gives $(x-2)(x+3)$, so there are "
                               r"$\boxed{2}$ asymptotes."}
        self.assertEqual(self.loader.parse_gold(example), "2")

    def test_gold_keeps_latex_structure(self):
        example = {"solution": r"Thus the answer is $\boxed{\frac{\sqrt{3}}{2}}$."}
        self.assertEqual(self.loader.parse_gold(example), r"\frac{\sqrt{3}}{2}")

    def test_solution_without_a_box_is_unlabelable(self):
        self.assertIsNone(self.loader.parse_gold({"solution": "The answer is 2."}))
        self.assertIsNone(self.loader.parse_gold({}))

    def test_empty_box_is_unlabelable(self):
        self.assertIsNone(self.loader.parse_gold({"solution": r"$\boxed{}$"}))

    def test_split_env_name(self):
        self.assertEqual(self.loader.SPLIT_ENV, "MATH_SPLIT")

    def _fake_load(self):
        """Stand-in for datasets.load_dataset returning one row per subject."""
        rows = {}

        class FakeDS(list):
            def filter(self, fn):
                return FakeDS([r for r in self if fn(r)])

            def shuffle(self, seed=0):
                return self

            def __getitem__(self, i):
                return list.__getitem__(self, i)

        def fake_load_dataset(name, subject, split=None):
            rows.setdefault("calls", []).append((subject, split))
            return FakeDS([
                {"problem": f"{subject}-p{level}", "solution": rf"$\boxed{{{level}}}$",
                 "level": f"Level {level}", "type": subject}
                for level in (1, 3, 5)
            ])

        def fake_concat(parts):
            out = FakeDS()
            for part in parts:
                out.extend(part)
            return out

        return fake_load_dataset, fake_concat, rows

    def test_load_injects_the_question_key_run_eval_reads(self):
        """run_problem and problems_digest both read `question`; MATH calls it
        `problem`, so a missing copy would evaluate empty prompts."""
        fake_load, fake_concat, _ = self._fake_load()
        with mock.patch("loaders.math.load_dataset", fake_load), \
             mock.patch("loaders.math.concatenate_datasets", fake_concat), \
             mock.patch.dict(os.environ, {}, clear=True):
            examples = self.loader.load(5, seed=0)
        self.assertEqual(len(examples), 5)
        for example in examples:
            self.assertEqual(example["question"], example["problem"])
            self.assertTrue(example["question"])

    def test_load_reads_all_seven_subjects_by_default(self):
        fake_load, fake_concat, rows = self._fake_load()
        with mock.patch("loaders.math.load_dataset", fake_load), \
             mock.patch("loaders.math.concatenate_datasets", fake_concat), \
             mock.patch.dict(os.environ, {"MATH_SPLIT": "train"}, clear=True):
            self.loader.load(100, seed=0)
        from loaders.math import SUBJECTS
        self.assertEqual([c[0] for c in rows["calls"]], list(SUBJECTS))
        self.assertTrue(all(c[1] == "train" for c in rows["calls"]))

    def test_subject_selection(self):
        fake_load, fake_concat, rows = self._fake_load()
        with mock.patch("loaders.math.load_dataset", fake_load), \
             mock.patch("loaders.math.concatenate_datasets", fake_concat), \
             mock.patch.dict(os.environ, {"MATH_SUBJECTS": "geometry, algebra"},
                             clear=True):
            self.loader.load(100, seed=0)
        # Canonical order, not the order the env var happened to list them in.
        self.assertEqual([c[0] for c in rows["calls"]], ["algebra", "geometry"])

    def test_level_selection_accepts_several_spellings(self):
        for spelling in ("3", "Level 3", "level3"):
            fake_load, fake_concat, _ = self._fake_load()
            with mock.patch("loaders.math.load_dataset", fake_load), \
                 mock.patch("loaders.math.concatenate_datasets", fake_concat), \
                 mock.patch.dict(os.environ, {"MATH_LEVELS": spelling}, clear=True):
                examples = self.loader.load(100, seed=0)
            self.assertTrue(examples, spelling)
            self.assertTrue(all(e["level"] == "Level 3" for e in examples), spelling)

    def test_unknown_subject_aborts(self):
        """A typo must not silently narrow the problem set to something that still
        looks like a normal run in the logs."""
        with mock.patch.dict(os.environ, {"MATH_SUBJECTS": "algebra,calculus"},
                             clear=True):
            with self.assertRaises(SystemExit) as cm:
                self.loader.load(10, seed=0)
        self.assertIn("calculus", str(cm.exception))

    def test_unknown_level_aborts(self):
        with mock.patch.dict(os.environ, {"MATH_LEVELS": "9"}, clear=True):
            with self.assertRaises(SystemExit) as cm:
                self.loader.load(10, seed=0)
        self.assertIn("9", str(cm.exception))

    def test_n_problems_larger_than_the_split_is_clamped(self):
        fake_load, fake_concat, _ = self._fake_load()
        with mock.patch("loaders.math.load_dataset", fake_load), \
             mock.patch("loaders.math.concatenate_datasets", fake_concat), \
             mock.patch.dict(os.environ, {}, clear=True):
            examples = self.loader.load(10**6, seed=0)
        self.assertEqual(len(examples), 21)   # 7 subjects x 3 levels


if __name__ == "__main__":
    unittest.main()
