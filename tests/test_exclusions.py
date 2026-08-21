"""The analysed population is the non-truncated generations, and that has to be stated
rather than left implicit.

Every budget-truncated record is marked wrong -- the parser finds no final answer in a
solution that was cut off before writing one -- so keeping them does not merely add
noise, it biases observed error prevalence upward and hands the judge free true
negatives (text stopping mid-sentence is trivially rejectable).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filtering"))
from metrics.correctness import audit_stored_labelable, is_labelable
from probe_models import exclusion_summary


def rec(idx, **over):
    base = {"idx": idx, "gold_answer": "4", "phase1_answer": "4",
            "majority_answer": "4", "solve_correct": True, "error": None}
    base.update(over)
    return base


class ExclusionSummaryTests(unittest.TestCase):
    def setUp(self):
        self.records = (
            [rec(i) for i in range(90)]
            + [rec(100 + i, truncated=True, phase1_answer=None, solve_correct=False)
               for i in range(8)]
            + [rec(200, error="cuda oom", solve_correct=False)]
            + [rec(300, gold_answer=None, solve_correct=False)]
        )

    def test_counts_each_exclusion_reason_apart(self):
        e = exclusion_summary(self.records)
        self.assertEqual(e["n_generated"], 100)
        self.assertEqual(e["n_analysed"], 90)
        self.assertEqual(e["n_excluded_truncated"], 8)
        self.assertEqual(e["n_excluded_error"], 1)
        self.assertEqual(e["n_excluded_no_gold"], 1)

    def test_reports_that_truncations_were_all_marked_wrong(self):
        """This is the reason exclusion matters: they are not a random subset."""
        e = exclusion_summary(self.records)
        self.assertEqual(e["truncated_marked_wrong"], 8)
        self.assertEqual(e["truncated_marked_wrong"], e["n_excluded_truncated"])

    def test_quantifies_the_prevalence_inflation(self):
        e = exclusion_summary(self.records)
        self.assertAlmostEqual(e["prevalence_analysed"], 0.0)
        self.assertAlmostEqual(e["prevalence_if_truncations_included"], 0.10)
        self.assertGreater(e["prevalence_if_truncations_included"],
                           e["prevalence_analysed"])

    def test_a_clean_split_reports_no_exclusions(self):
        e = exclusion_summary([rec(i) for i in range(50)])
        self.assertEqual(e["n_analysed"], 50)
        self.assertEqual(e["n_excluded_truncated"], 0)
        self.assertEqual(e["prevalence_analysed"],
                         e["prevalence_if_truncations_included"])


class StoredLabelableAuditTests(unittest.TestCase):
    """The stored field is an advisory snapshot; the recomputed value is authoritative.
    Divergence is allowed but must never be silent."""

    def test_detects_a_stale_stored_true(self):
        """The exact historical defect: `labelable` was written as
        `gold_answer is not None`, so a truncated record (which has a gold answer)
        was stored as labelable while the analysis excluded it."""
        records = [rec(0, truncated=True, phase1_answer=None,
                       solve_correct=False, labelable=True)]
        a = audit_stored_labelable(records)
        self.assertEqual(a["n_disagree"], 1)
        self.assertEqual(a["stored_true_recomputed_false"], 1)
        self.assertFalse(is_labelable(records[0]))

    def test_agreeing_records_are_not_flagged(self):
        records = [rec(0, labelable=True), rec(1, error="oom", labelable=False)]
        self.assertEqual(audit_stored_labelable(records)["n_disagree"], 0)

    def test_records_without_the_field_are_not_counted_as_disagreeing(self):
        records = [rec(0)]
        a = audit_stored_labelable(records)
        self.assertEqual(a["n_with_field"], 0)
        self.assertEqual(a["n_disagree"], 0)

    def test_reports_a_sample_of_offending_indices(self):
        records = [rec(i, truncated=True, phase1_answer=None,
                       solve_correct=False, labelable=True) for i in range(20)]
        a = audit_stored_labelable(records)
        self.assertEqual(a["n_disagree"], 20)
        self.assertEqual(len(a["idx"]), 10)   # capped, so the message stays readable


class WriteTimeSnapshotTests(unittest.TestCase):
    """run_eval writes the snapshot with the same rule the analysis applies, so a
    raw.jsonl read by hand does not contradict the figures built from it."""

    def test_truncated_record_is_written_unlabelable(self):
        from run_eval import _labelable_snapshot
        self.assertFalse(_labelable_snapshot("4", "gsm8k", True))

    def test_completed_record_with_usable_gold_is_labelable(self):
        from run_eval import _labelable_snapshot
        self.assertTrue(_labelable_snapshot("4", "gsm8k", False))

    def test_unknown_truncation_does_not_force_exclusion(self):
        """None means the backend could not report it, which is not evidence of
        truncation; treating it as such would drop every record from such a backend."""
        from run_eval import _labelable_snapshot
        self.assertTrue(_labelable_snapshot("4", "gsm8k", None))

    def test_unusable_gold_is_unlabelable_under_the_dataset_policy(self):
        from run_eval import _labelable_snapshot
        self.assertFalse(_labelable_snapshot(r"\frac{1}{2}", "gsm8k", False))
        self.assertTrue(_labelable_snapshot(r"\frac{1}{2}", "math", False))


class ExclusionsCoverTheGeneratedPopulationTests(unittest.TestCase):
    """The summary must describe every GENERATED record, not the dump's subset.

    Extraction drops unlabelable records before writing the dump, so a summary computed
    over the dump's index list sees only survivors and reports zero exclusions --
    silently answering "how many of the kept records were excluded" instead of "how many
    generations were excluded", which is the question a paper's population statement
    asks. Caught on real data: GSM8K train reported 7466/7466 rather than 7466/7473.
    """

    def test_dump_aligned_subset_would_report_no_exclusions(self):
        generated = ([rec(i) for i in range(90)]
                     + [rec(100 + i, truncated=True, phase1_answer=None,
                            solve_correct=False) for i in range(10)])
        survivors = [r for r in generated if is_labelable(r)]

        full = exclusion_summary(generated)
        subset = exclusion_summary(survivors)

        self.assertEqual(full["n_generated"], 100)
        self.assertEqual(full["n_excluded_truncated"], 10)
        # The failure mode: over survivors alone the exclusions vanish entirely.
        self.assertEqual(subset["n_excluded_truncated"], 0)
        self.assertEqual(subset["n_generated"], 90)

    def test_analysed_count_agrees_between_the_two_views(self):
        """Whichever population is summarised, the analysed count is the same -- so a
        reader who only sees `n_analysed` cannot tell the two apart. That is why the
        denominator has to be right."""
        generated = ([rec(i) for i in range(90)]
                     + [rec(100 + i, truncated=True, phase1_answer=None,
                            solve_correct=False) for i in range(10)])
        survivors = [r for r in generated if is_labelable(r)]
        self.assertEqual(exclusion_summary(generated)["n_analysed"],
                         exclusion_summary(survivors)["n_analysed"])


if __name__ == "__main__":
    unittest.main()
