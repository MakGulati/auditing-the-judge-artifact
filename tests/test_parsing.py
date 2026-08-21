import unittest

from loaders.gsm8k import GSM8KLoader
from pipeline.judge import parse_verdict
from pipeline.solver import parse_answer


class GoldParsingTests(unittest.TestCase):
    """`#### -12` used to yield None, which made every comparison False and
    silently relabelled a correct answer as wrong."""

    def setUp(self):
        self.loader = GSM8KLoader()

    def gold(self, tail):
        return self.loader.parse_gold({"answer": f"some work\n{tail}"})

    def test_negative_gold(self):
        self.assertEqual(self.gold("#### -12"), "-12")

    def test_plain_and_comma_grouped(self):
        self.assertEqual(self.gold("#### 42"), "42")
        self.assertEqual(self.gold("#### 1,200"), "1200")

    def test_decimal_is_not_truncated(self):
        self.assertEqual(self.gold("#### 3.5"), "3.5")

    def test_unparseable_returns_none(self):
        self.assertIsNone(self.gold("#### n/a"))
        self.assertIsNone(self.loader.parse_gold({"answer": "no marker here"}))


class AnswerParsingTests(unittest.TestCase):
    def test_hash_marker(self):
        self.assertEqual(parse_answer("work\n#### 64"), "64")

    def test_negative_answer(self):
        self.assertEqual(parse_answer("the average is -12.\n#### -12"), "-12")

    def test_fallback_skips_bare_punctuation(self):
        # `[-\d,]+` matches a lone "-", which parsed to None and threw away a
        # perfectly good number earlier in the text.
        self.assertEqual(parse_answer("we get 17 apples - "), "17")

    def test_unparseable(self):
        self.assertIsNone(parse_answer("no numbers at all"))


class VerdictParsingTests(unittest.TestCase):
    def test_leading_verdict(self):
        self.assertEqual(parse_verdict("YES"), "YES")
        self.assertEqual(parse_verdict("NO, it is wrong"), "NO")

    def test_trailing_verdict_wins(self):
        self.assertEqual(parse_verdict("I found no errors.\nVERDICT: YES"), "YES")

    def test_ambiguous_defaults_to_no(self):
        self.assertEqual(parse_verdict("maybe?"), "NO")


if __name__ == "__main__":
    unittest.main()
