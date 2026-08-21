"""MATH answer extraction and equivalence.

The stakes here are label quality: `extract_boxed` produces the gold answer and
`latex_answers_equal` decides correctness, so a bug in either one is silent label
noise that the probe then learns from.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics.latex_answer import (canonical_latex_answer, extract_boxed,
                                  gold_is_parsable, latex_answers_equal,
                                  normalize_latex)


class ExtractBoxedTests(unittest.TestCase):
    def test_nested_braces_are_not_truncated(self):
        """`\\boxed{([^}]*)}` would return '\\frac{1' — a gold that matches nothing."""
        self.assertEqual(extract_boxed(r"so \boxed{\frac{1}{2}}."), r"\frac{1}{2}")

    def test_deeply_nested_braces(self):
        self.assertEqual(extract_boxed(r"\boxed{\frac{\sqrt{3}}{2}}"),
                         r"\frac{\sqrt{3}}{2}")

    def test_last_box_wins(self):
        """Solutions box an intermediate result before boxing the final answer."""
        self.assertEqual(extract_boxed(r"first \boxed{3}, corrected to \boxed{7}"), "7")

    def test_fbox_is_accepted(self):
        self.assertEqual(extract_boxed(r"answer is \fbox{12}"), "12")

    def test_last_box_wins_across_commands(self):
        self.assertEqual(extract_boxed(r"\fbox{3} then \boxed{9}"), "9")
        self.assertEqual(extract_boxed(r"\boxed{9} then \fbox{3}"), "3")

    def test_braceless_form(self):
        self.assertEqual(extract_boxed(r"\boxed 5"), "5")
        self.assertEqual(extract_boxed(r"\boxed{-3}"), "-3")

    def test_no_box_returns_none(self):
        self.assertIsNone(extract_boxed("the answer is 42"))

    def test_empty_and_none_input(self):
        self.assertIsNone(extract_boxed(""))
        self.assertIsNone(extract_boxed(None))

    def test_unbalanced_braces_do_not_hang_or_raise(self):
        self.assertIsNone(extract_boxed(r"\boxed{\frac{1}{2}"))

    def test_text_answers_survive(self):
        self.assertEqual(extract_boxed(r"\boxed{\text{even}}"), r"\text{even}")


class NormalizationTests(unittest.TestCase):
    def assert_equiv(self, a, b):
        self.assertTrue(latex_answers_equal(a, b), f"{a!r} should equal {b!r}")

    def test_fraction_spellings(self):
        self.assert_equiv(r"\dfrac{1}{2}", r"\frac{1}{2}")
        self.assert_equiv(r"\tfrac{1}{2}", r"\frac{1}{2}")
        self.assert_equiv(r"\frac12", r"\frac{1}{2}")
        self.assert_equiv(r"\frac{1}2", r"\frac{1}{2}")

    def test_delimiters_and_spacing(self):
        self.assert_equiv(r"\left(1,2\right)", "(1,2)")
        self.assert_equiv(r"$2$", "2")
        self.assert_equiv(r"\(x+1\)", "x+1")
        self.assert_equiv(r"1 + 2", "1+2")
        self.assert_equiv(r"\!2\,", "2")

    def test_units_and_decorations(self):
        self.assert_equiv(r"50\%", "50")
        self.assert_equiv(r"90^\circ", "90")
        self.assert_equiv(r"\text{even}", "even")
        self.assert_equiv(r"\textbf{7}", "7")

    def test_trailing_units_are_stripped(self):
        """`12\\text{ cm}` vs a model's `12` is a formatting difference, not a wrong
        answer. The leading space inside the braces is what marks a unit."""
        self.assert_equiv(r"12\text{ cm}", "12")
        self.assert_equiv(r"5\text{ inches}", "5")
        self.assertEqual(normalize_latex(r"12\text{ cm}"), "12")

    def test_word_answers_are_not_mistaken_for_units(self):
        """`\\text{even}` has no leading space, so it is the answer itself."""
        self.assertEqual(normalize_latex(r"\text{even}"), "even")
        self.assertEqual(normalize_latex(r"\text{ odd}"), "odd")
        self.assertTrue(latex_answers_equal(r"\text{even}", "even"))
        self.assertFalse(latex_answers_equal(r"\text{even}", ""))

    def test_currency(self):
        """`\\$` must come off before `$`, or `\\$5` normalizes to a stray backslash."""
        self.assert_equiv(r"\$5", "5")
        self.assertEqual(normalize_latex(r"\$5"), "5")

    def test_operators_are_not_stripped(self):
        """Removing `\\cdot` would silently turn `2\\cdot3` into the number 23."""
        self.assertFalse(latex_answers_equal(r"2\cdot3", "23"))
        self.assertFalse(latex_answers_equal(r"2\times10", "210"))

    def test_thousands_separator(self):
        self.assert_equiv(r"1{,}000", "1000")
        self.assert_equiv("1,000", "1000")

    def test_trailing_period_and_leading_zero(self):
        self.assert_equiv("42.", "42")
        self.assert_equiv(".5", "0.5")
        self.assert_equiv("-.5", "-0.5")

    def test_radical_spelling(self):
        self.assert_equiv(r"\sqrt3", r"\sqrt{3}")
        self.assert_equiv(r"2\sqrt{3}", r"2\sqrt3")

    def test_single_variable_assignment_is_stripped(self):
        self.assert_equiv("x=5", "5")
        self.assert_equiv("y = -3", "-3")

    def test_tuple_commas_are_not_stripped(self):
        """`(1,2)` must not collapse to `(12)` when thousands separators come off."""
        self.assertEqual(normalize_latex("(1,2)"), "(1,2)")
        self.assertFalse(latex_answers_equal("(1,2)", "(12)"))

    def test_inequality_is_not_treated_as_an_assignment(self):
        self.assertFalse(latex_answers_equal("x<5", "5"))


class NumericFallbackTests(unittest.TestCase):
    def test_plain_numeric_literals_compare_by_value(self):
        self.assertTrue(latex_answers_equal("2", "2.00"))
        self.assertTrue(latex_answers_equal("-3", "-3.0"))
        self.assertTrue(latex_answers_equal("0", "-0"))

    def test_comma_separated_answers_are_not_read_as_one_number(self):
        """The GSM8K parser strips every comma before parsing, which is right there and
        catastrophic here: `1,2` (two roots, or a coordinate list) would become the
        single number 12 and compare equal to it. The numeric path is therefore entered
        only for a bare decimal literal."""
        self.assertEqual(canonical_latex_answer("1,2"), "1,2")
        self.assertEqual(canonical_latex_answer("-1,2"), "-1,2")
        self.assertFalse(latex_answers_equal("1,2", "12"))
        self.assertFalse(latex_answers_equal("-1,2", "-12"))
        self.assertTrue(latex_answers_equal("1,2", "1, 2"))

    def test_real_thousands_separators_still_compare_numerically(self):
        """normalize_latex removes them before this point, so `1,000` is a number."""
        self.assertEqual(canonical_latex_answer("1,000"), "1000")
        self.assertTrue(latex_answers_equal("1,000", "1000.0"))

    def test_numeric_canonical_form_matches_the_gsm8k_canonicaliser(self):
        from metrics.correctness import canonical_numeric_answer
        for value in ("2.00", "-3.0", "0", "1000", "0.500"):
            self.assertEqual(canonical_latex_answer(value),
                             canonical_numeric_answer(value), value)


class NonEquivalenceTests(unittest.TestCase):
    def test_different_values_are_not_equal(self):
        self.assertFalse(latex_answers_equal(r"\frac{1}{2}", r"\frac{1}{3}"))
        self.assertFalse(latex_answers_equal("even", "odd"))
        self.assertFalse(latex_answers_equal("(1,2)", "(2,1)"))

    def test_fraction_does_not_equal_its_decimal_value(self):
        """Documented policy: strict, Hendrycks-comparable. Only plain numeric
        literals are compared by value, so `\\frac{1}{2}` != `0.5`. Changing this
        would make MATH accuracy incomparable to published numbers."""
        self.assertFalse(latex_answers_equal(r"\frac{1}{2}", "0.5"))

    def test_missing_answer_is_never_equal(self):
        self.assertFalse(latex_answers_equal(None, "2"))
        self.assertFalse(latex_answers_equal("2", None))
        self.assertFalse(latex_answers_equal(None, None))
        self.assertFalse(latex_answers_equal("", ""))


class GoldParsabilityTests(unittest.TestCase):
    def test_latex_gold_is_parsable(self):
        for gold in (r"\frac{2}{3}", r"\text{even}", "(3,4)", "17", r"2\sqrt{3}"):
            self.assertTrue(gold_is_parsable(gold), gold)

    def test_empty_gold_is_not_parsable(self):
        for gold in (None, "", "   ", "$$", r"\text{}"):
            self.assertFalse(gold_is_parsable(gold), repr(gold))

    def test_canonical_of_nothing_is_none(self):
        self.assertIsNone(canonical_latex_answer(None))
        self.assertIsNone(canonical_latex_answer(""))


if __name__ == "__main__":
    unittest.main()
