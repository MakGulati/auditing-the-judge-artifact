"""The YES/NO binning rules behind the p_yes black-box baseline.

The point of these tests is that PREFIX must stay bit-identical to the shipped
`extraction_common.yes_no_mass` — it is the rule the published figures used, so a
"cleanup" of it would silently rewrite history — while EXACT must differ from it
only by rejecting prefix matches.
"""
from __future__ import annotations

import unittest

from metrics.verdict_tokens import (EXACT, NO, PREFIX, YES, classify, mass, normalize,
                                    p_yes, vocab_classes)


class TestNormalize(unittest.TestCase):
    def test_strips_whitespace_quotes_colons_stars(self):
        for raw in (' yes', 'yes ', '"yes"', '*yes*', ':yes:', '"*yes*"'):
            self.assertEqual(normalize(raw), "YES", raw)

    def test_does_not_strip_other_punctuation(self):
        # Both rules share this, so a token like 'Yes,' is a YES under PREFIX and
        # nothing at all under EXACT. That is a real difference, not an oversight:
        # documenting it is why the test exists.
        self.assertEqual(normalize("Yes,"), "YES,")
        self.assertEqual(classify("Yes,", PREFIX), YES)
        self.assertIsNone(classify("Yes,", EXACT))

    def test_whitespace_only_token_is_unclassified(self):
        for raw in ("", " ", "\n", '":*'):
            self.assertIsNone(classify(raw, PREFIX), raw)
            self.assertIsNone(classify(raw, EXACT), raw)


class TestClassify(unittest.TestCase):
    def test_real_verdict_tokens_agree_under_both_rules(self):
        for raw, want in ((" Yes", YES), ("YES", YES), ("yes", YES), ("Y", YES),
                          ("ye", YES), (" No", NO), ("NO", NO), ("no", NO), ("N", NO)):
            self.assertEqual(classify(raw, PREFIX), want, raw)
            self.assertEqual(classify(raw, EXACT), want, raw)

    def test_prefix_rule_invents_no_votes(self):
        # The whole reason the comparison exists: a judge about to write prose puts
        # mass on these, and the shipped rule scores every one of them as "wrong".
        for raw in (" not", " now", " note", " nothing", " normal", " North",
                    " November", "Node", " noise", " none", " nor"):
            self.assertEqual(classify(raw, PREFIX), NO, raw)
            self.assertIsNone(classify(raw, EXACT), raw)

    def test_prefix_rule_invents_yes_votes_too(self):
        # Asymmetric but not one-sided; the YES side has to be checked as well or the
        # comparison would only ever be able to find a bias in one direction.
        for raw in (" yesterday", "yester", " Yeshu"):
            self.assertEqual(classify(raw, PREFIX), YES, raw)
            self.assertIsNone(classify(raw, EXACT), raw)

    def test_lone_n_is_no_but_ne_is_not(self):
        # 'N' is matched by membership, not by prefix, so the near-miss must not leak.
        self.assertEqual(classify("N", PREFIX), NO)
        self.assertIsNone(classify("NE", PREFIX))
        self.assertIsNone(classify("Nx", PREFIX))

    def test_unknown_rule_rejected(self):
        with self.assertRaises(ValueError):
            classify("yes", "startswith")


class TestMass(unittest.TestCase):
    def test_other_mass_is_reported_not_silently_dropped(self):
        my, mn, other = mass([(" Yes", 0.5), (" No", 0.2), (" The", 0.3)], EXACT)
        self.assertAlmostEqual(my, 0.5)
        self.assertAlmostEqual(mn, 0.2)
        self.assertAlmostEqual(other, 0.3)

    def test_p_yes_normalises_over_yes_no_only(self):
        # 0.3 on an unrelated token must not dilute the verdict.
        self.assertAlmostEqual(p_yes([(" Yes", 0.5), (" No", 0.2), (" The", 0.3)], EXACT),
                               0.5 / 0.7)

    def test_rules_disagree_when_not_carries_mass(self):
        pairs = [(" Yes", 0.6), (" not", 0.4)]
        self.assertAlmostEqual(p_yes(pairs, PREFIX), 0.6)   # ' not' counted against
        self.assertAlmostEqual(p_yes(pairs, EXACT), 1.0)    # ' not' ignored

    def test_fallback_is_a_half_when_no_verdict_token_appears(self):
        # 0.5 here means "the rule found nothing", not "the judge was undecided";
        # pyes_rule_compare counts these separately for exactly that reason.
        self.assertEqual(p_yes([(" The", 0.9), (" answer", 0.1)], PREFIX), 0.5)
        self.assertEqual(p_yes([], EXACT), 0.5)

    def test_empty_yes_side_gives_zero_not_fallback(self):
        self.assertEqual(p_yes([(" No", 0.8)], EXACT), 0.0)


class FakeTokenizer:
    """Minimal tokenizer stand-in: id -> decoded string."""

    def __init__(self, vocab):
        self.vocab = list(vocab)

    def __len__(self):
        return len(self.vocab)

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


class TestVocabClasses(unittest.TestCase):
    def test_partitions_by_rule(self):
        tok = FakeTokenizer([" Yes", " No", " not", " yesterday", " the"])
        yes_p, no_p = vocab_classes(tok, PREFIX)
        yes_e, no_e = vocab_classes(tok, EXACT)
        self.assertEqual((yes_p, no_p), ([0, 3], [1, 2]))
        self.assertEqual((yes_e, no_e), ([0], [1]))


class TestMatchesShippedRule(unittest.TestCase):
    """PREFIX must reproduce extraction_common.yes_no_mass exactly."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_agrees_with_yes_no_mass_on_random_distributions(self):
        import torch
        from extraction_common import yes_no_mass

        # More than 40 entries, so the shipped function's top-40 truncation is
        # exercised rather than assumed away.
        vocab = [" Yes", " No", " not", " now", " yesterday", "Y", "N", " the", " a",
                 " solution", " correct", " wrong", "NO", "YES", " none", " Node"]
        vocab += [f" tok{i}" for i in range(60)]
        tok = FakeTokenizer(vocab)
        g = torch.Generator().manual_seed(0)
        for _ in range(25):
            probs = torch.softmax(torch.randn(len(vocab), generator=g) * 3, -1)
            topv, topi = torch.topk(probs, 40)
            pairs = [(tok.decode([int(i)]), float(p)) for p, i in zip(topv, topi)]
            self.assertAlmostEqual(p_yes(pairs, PREFIX), yes_no_mass(probs, tok),
                                   places=6)

    def test_agrees_when_no_verdict_token_is_in_the_window(self):
        import torch
        from extraction_common import yes_no_mass

        tok = FakeTokenizer([f" tok{i}" for i in range(50)])
        probs = torch.softmax(torch.arange(50, dtype=torch.float), -1)
        topv, topi = torch.topk(probs, 40)
        pairs = [(tok.decode([int(i)]), float(p)) for p, i in zip(topv, topi)]
        self.assertEqual(p_yes(pairs, PREFIX), 0.5)
        self.assertEqual(yes_no_mass(probs, tok), 0.5)


if __name__ == "__main__":
    unittest.main()
