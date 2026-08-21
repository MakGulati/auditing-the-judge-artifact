import unittest

from metrics.correctness import (LABEL_POLICY, answers_numerically_equal,
                                 canonical_numeric_answer, is_labelable,
                                 label_correctness, relabel_result)


class NumericCorrectnessTests(unittest.TestCase):
    def test_equivalent_decimal_spellings(self):
        for answer in ("64", "64.0", "64.00", "064.000", "64,000.00"):
            gold = "64000" if "," in answer else "64"
            self.assertTrue(answers_numerically_equal(answer, gold))

    def test_non_equivalent_and_invalid_values(self):
        self.assertFalse(answers_numerically_equal("64.01", "64"))
        self.assertFalse(answers_numerically_equal(None, "64"))
        self.assertFalse(answers_numerically_equal("NaN", "NaN"))

    def test_canonicalization_groups_votes(self):
        self.assertEqual(canonical_numeric_answer("-0.00"), "0")
        self.assertEqual(canonical_numeric_answer("1,200.5000"), "1200.5")

    def test_relabel_result_overwrites_legacy_boole(self):
        record = {
            "gold_answer": "64",
            "phase1_answer": "64.00",
            "majority_answer": "63.0",
            "solve_correct": False,
            "majority_correct": True,
        }
        relabel_result(record)
        self.assertTrue(record["solve_correct"])
        self.assertFalse(record["majority_correct"])
        self.assertTrue(record["labelable"])
        self.assertEqual(record["label_policy"], LABEL_POLICY)


class UnlabelableRecordTests(unittest.TestCase):
    def test_unparseable_gold_is_none_not_false(self):
        # A correct answer against an unparseable gold must not read as "wrong".
        self.assertIsNone(label_correctness("-12", None))
        self.assertIsNone(label_correctness("-12", "n/a"))
        self.assertIs(label_correctness("-12", "-12"), True)
        self.assertIs(label_correctness("-11", "-12"), False)

    def test_is_labelable_rejects_missing_gold_and_errors(self):
        self.assertTrue(is_labelable({"gold_answer": "5"}))
        self.assertFalse(is_labelable({"gold_answer": None}))
        # A generation failure would otherwise donate a free true negative to the judge.
        self.assertFalse(is_labelable({"gold_answer": "5", "error": "CUDA OOM"}))

    def test_relabel_marks_error_records_unlabelable(self):
        record = {"gold_answer": "5", "phase1_answer": None, "error": "boom"}
        relabel_result(record)
        self.assertFalse(record["labelable"])


if __name__ == "__main__":
    unittest.main()
