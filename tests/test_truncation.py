"""Truncation must be recorded, not inferred.

A reply that ran out of token budget is indistinguishable from one the model chose to
end, unless generation records it. That gap silently relabelled 24% of MATH as "wrong
answer" — and on GSM8K it was worse, because the answer parser's last-number fallback
turned a cut-off solution into a confident wrong answer rather than a visible failure.
"""
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics.confusion import _truncation_line
from pipeline.backend import Completion, LLMBackend
from pipeline.majority import majority_vote_sample
from pipeline.solver import DEFAULT_SOLVE_MAX_TOKENS, solve_greedy
from prompts.math import MATHPrompts


class ReportingBackend(LLMBackend):
    """Stands in for a backend that CAN report finish_reason."""

    def __init__(self, text, truncated):
        self.text, self.flag = text, truncated
        self.max_tokens_seen = []

    async def chat_complete(self, messages, **kw):
        return (await self.complete(messages, **kw)).text

    async def complete(self, messages, temperature=0.0, max_tokens=1024, top_p=1.0):
        self.max_tokens_seen.append(max_tokens)
        return Completion(text=self.text, truncated=self.flag)


class SilentBackend(LLMBackend):
    """Implements only chat_complete — cannot report truncation."""

    async def chat_complete(self, messages, temperature=0.0, max_tokens=1024, top_p=1.0):
        return r"work \boxed{7}"


class SolveTruncationTests(unittest.TestCase):
    def test_truncation_flag_is_propagated(self):
        backend = ReportingBackend(r"cut off mid-sent", True)
        _, _, truncated = asyncio.run(solve_greedy("P", MATHPrompts(), backend))
        self.assertIs(truncated, True)

    def test_a_completed_solve_reports_false(self):
        backend = ReportingBackend(r"done \boxed{7}", False)
        text, answer, truncated = asyncio.run(solve_greedy("P", MATHPrompts(), backend))
        self.assertIs(truncated, False)
        self.assertEqual(answer, "7")

    def test_a_backend_that_cannot_tell_reports_none_not_false(self):
        """'Unknown' and 'not truncated' are different claims. Recording the first as
        the second is exactly how this stayed invisible."""
        _, _, truncated = asyncio.run(solve_greedy("P", MATHPrompts(), SilentBackend()))
        self.assertIsNone(truncated)

    def test_the_budget_is_passed_through(self):
        backend = ReportingBackend(r"\boxed{7}", False)
        asyncio.run(solve_greedy("P", MATHPrompts(), backend, 4096))
        self.assertEqual(backend.max_tokens_seen, [4096])

    def test_the_default_budget_is_unchanged(self):
        """1024 is what every existing run used; changing the default silently would
        make new runs incomparable to them."""
        self.assertEqual(DEFAULT_SOLVE_MAX_TOKENS, 1024)
        backend = ReportingBackend(r"\boxed{7}", False)
        asyncio.run(solve_greedy("P", MATHPrompts(), backend))
        self.assertEqual(backend.max_tokens_seen, [1024])

    def test_a_truncated_solution_yields_no_answer(self):
        """The failure this whole flag exists to explain: the final answer is the last
        thing written, so cutting the reply off removes it."""
        backend = ReportingBackend("We have $c = a", True)
        _, answer, truncated = asyncio.run(solve_greedy("P", MATHPrompts(), backend))
        self.assertIsNone(answer)
        self.assertIs(truncated, True)


class MajorityTruncationTests(unittest.TestCase):
    def test_flags_are_returned_per_sample(self):
        backend = ReportingBackend(r"\boxed{7}", True)
        _, _, _, truncated = asyncio.run(
            majority_vote_sample("P", MATHPrompts(), backend, 3))
        self.assertEqual(truncated, [True, True, True])

    def test_the_budget_reaches_every_sample(self):
        backend = ReportingBackend(r"\boxed{7}", False)
        asyncio.run(majority_vote_sample("P", MATHPrompts(), backend, 3, 2048))
        self.assertEqual(backend.max_tokens_seen, [2048] * 3)


class ReportLineTests(unittest.TestCase):
    def test_rate_is_reported(self):
        line = _truncation_line([{"truncated": True}] * 24 + [{"truncated": False}] * 76, 100)
        self.assertIn("24/100", line)
        self.assertIn("24.0%", line)

    def test_a_high_rate_says_what_to_do(self):
        line = _truncation_line([{"truncated": True}] * 24 + [{"truncated": False}] * 76, 100)
        self.assertIn("--solve_max_tokens", line)

    def test_a_clean_run_does_not_nag(self):
        line = _truncation_line([{"truncated": False}] * 100, 100)
        self.assertNotIn("--solve_max_tokens", line)

    def test_unknown_is_reported_as_unknown_not_as_zero(self):
        line = _truncation_line([{} for _ in range(100)], 100)
        self.assertIn("unknown", line)
        self.assertNotIn("0.0%", line)

    def test_partial_reporting_states_the_coverage(self):
        line = _truncation_line([{"truncated": True}] * 2 + [{} for _ in range(8)], 10)
        self.assertIn("[2/10 records report it]", line)


if __name__ == "__main__":
    unittest.main()


class CombinedQuantileTests(unittest.TestCase):
    """The truncated records are the upper tail, not a random sample. Reading the
    budget straight off their own p99 overshoots badly; the strata have to be
    recombined."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from calibrate_solve_budget import combined_quantile
        self.q = combined_quantile
        # 24% truncated, and their true lengths run 1000..2999.
        self.lengths = list(range(1000, 3000))

    def test_quantiles_inside_the_completed_part_need_no_new_budget(self):
        # 76% finished under the old budget, so covering 50% or 70% is already done.
        self.assertIsNone(self.q(self.lengths, 0.24, 0.50))
        self.assertIsNone(self.q(self.lengths, 0.24, 0.70))

    def test_covering_the_whole_dataset_takes_the_stratum_max(self):
        self.assertEqual(self.q(self.lengths, 0.24, 1.0), 2999)

    def test_dataset_p99_is_far_below_the_stratum_p99(self):
        """The point of the correction: 99% of the DATASET is only the ~96th
        percentile of the truncated group, not its 99th."""
        dataset_p99 = self.q(self.lengths, 0.24, 0.99)
        stratum_p99 = self.lengths[int(0.99 * (len(self.lengths) - 1))]
        self.assertLess(dataset_p99, stratum_p99)
        # (0.99 - 0.76) / 0.24 = 0.958 -> the 95.8th percentile of the stratum
        self.assertAlmostEqual(dataset_p99, self.lengths[int(round(0.958 * 1999))],
                               delta=3)

    def test_more_truncation_means_a_lower_stratum_quantile_suffices(self):
        # If HALF the set was truncated, 99% coverage sits at the 98th percentile of
        # the stratum; if only 5% was, it needs the 80th.
        self.assertGreater(self.q(self.lengths, 0.50, 0.99),
                           self.q(self.lengths, 0.05, 0.99))

    def test_quantiles_are_monotonic_in_coverage(self):
        vals = [self.q(self.lengths, 0.24, q) for q in (0.80, 0.90, 0.95, 0.99, 1.0)]
        self.assertEqual(vals, sorted(vals))


class TruncatedRecordsAreUnlabelableTests(unittest.TestCase):
    """A solution the budget cut off never stated its final answer, so its
    correctness is unknown — not wrong.

    Labelling it wrong asserts something about the model when the cause was our
    budget, and it is the EASY kind of wrong: the judge sees text stopping
    mid-sentence and rejects it, so the record becomes a free true negative for the
    judge and a trivially separable example for the probe. That is precisely why
    generation errors were already excluded.
    """

    def setUp(self):
        from metrics.correctness import is_labelable, relabel_result
        self.is_labelable = is_labelable
        self.relabel = relabel_result

    def record(self, **over):
        base = {"idx": 0, "gold_answer": "42", "phase1_answer": "15",
                "majority_answer": "15", "error": None}
        base.update(over)
        return base

    def test_a_truncated_record_is_not_labelable(self):
        self.assertFalse(self.is_labelable(self.record(truncated=True)))

    def test_a_completed_record_is_labelable(self):
        self.assertTrue(self.is_labelable(self.record(truncated=False)))

    def test_unknown_truncation_stays_labelable(self):
        """None means the backend could not report it. Treating that as truncated
        would drop every record from a backend that cannot tell us."""
        self.assertTrue(self.is_labelable(self.record(truncated=None)))

    def test_records_predating_the_field_are_unaffected(self):
        """No historical raw.jsonl carries `truncated`; their labels must not move."""
        legacy = self.record()
        self.assertNotIn("truncated", legacy)
        self.assertTrue(self.is_labelable(legacy))

    def test_relabel_marks_it_unlabelable(self):
        out = self.relabel(self.record(truncated=True))
        self.assertFalse(out["labelable"])

    def test_the_fabricated_answer_is_still_not_counted_as_wrong(self):
        """The GSM8K failure this prevents: the last-number fallback pulls a
        leftover number out of cut-off text, so the record looks like a confident
        wrong answer rather than a visible failure."""
        rec = self.relabel(self.record(truncated=True, phase1_answer="15",
                                       gold_answer="50"))
        self.assertFalse(rec["labelable"])

    def test_a_generation_error_still_wins_over_truncation(self):
        self.assertFalse(self.is_labelable(
            self.record(error="cuda oom", truncated=False)))


class ExclusionReportingTests(unittest.TestCase):
    """The three exclusion reasons mean different things and must be counted apart."""

    def test_report_names_truncation_separately(self):
        from metrics.confusion import generate_report

        base = {"gold_answer": "4", "phase1_answer": "4", "majority_answer": "4",
                "phase1_solution": "s", "judge_verdict": "YES", "k_answers": ["4"],
                "solve_correct": True, "majority_correct": True}
        results = [
            {**base, "idx": 0},
            {**base, "idx": 1, "truncated": True},
            {**base, "idx": 2, "error": "cuda oom"},
            {**base, "idx": 3, "gold_answer": "n/a"},
        ]
        report = generate_report(results)
        self.assertIn("1 generation error(s)", report)
        self.assertIn("1 budget-truncated", report)
        self.assertIn("1 unparseable gold answer(s)", report)
