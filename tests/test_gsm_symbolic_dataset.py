"""GSM-Symbolic: GSM8K's task with every instance's surface form resampled.

It exists here as a contamination probe, so the things worth pinning are the ones
that would quietly make it stop being one: loading a variant that changes difficulty
rather than only surface form, a gold answer that silently fails to parse (which
would relabel a correct answer as wrong), and the prefix stability the resume
fingerprint depends on.
"""
import os
import sys
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset_registry
from metrics.correctness import label_policy_for
from prompts.gsm8k import GSM8KPrompts

ROWS = [
    {"question": f"q{i}", "answer": f"reasoning {i}\n#### {i}", "original_id": i % 3,
     "instance": i, "id": i}
    for i in range(10)
]


class FakeDS(list):
    def shuffle(self, seed=0):
        # Deterministic and seed-dependent, like the real thing; the loader takes a
        # PREFIX of this, so the order must not depend on how many rows are requested.
        return FakeDS(sorted(self, key=lambda r: (seed * 7 + r["id"] * 13) % 10))


class GSMSymbolicRegistrationTests(unittest.TestCase):
    def test_it_is_registered_with_a_loader_and_prompts(self):
        self.assertIn("gsm_symbolic", dataset_registry.DATASETS)
        self.assertIn("gsm_symbolic", dataset_registry._LOADER_PATHS)

    def test_it_shares_gsm8k_prompts_and_policy(self):
        """Same task and same answer contract, so a divergence here is a bug."""
        self.assertIs(dataset_registry.prompts_for("gsm_symbolic"), GSM8KPrompts)
        self.assertEqual(label_policy_for("gsm_symbolic"), label_policy_for("gsm8k"))


class GSMSymbolicLoaderTests(unittest.TestCase):
    def setUp(self):
        from loaders.gsm_symbolic import GSMSymbolicLoader
        self.loader = GSMSymbolicLoader()

    def _load(self, n, seed=0, env=None):
        with mock.patch.dict(os.environ, env or {}, clear=False), \
             mock.patch("loaders.gsm_symbolic.load_dataset",
                        return_value=FakeDS(ROWS)) as fake:
            out = self.loader.load(n, seed)
        return out, fake

    def test_it_loads_the_main_variant_by_default(self):
        """p1/p2 ADD clauses, so they change difficulty as well as surface form."""
        _, fake = self._load(4)
        self.assertEqual(fake.call_args.args[1], "main")

    def test_an_unknown_variant_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            self._load(4, env={"GSM_SYMBOLIC_VARIANT": "p3"})
        self.assertIn("main/p1/p2", str(cm.exception))

    def test_a_requested_variant_is_honoured(self):
        _, fake = self._load(4, env={"GSM_SYMBOLIC_VARIANT": "p2"})
        self.assertEqual(fake.call_args.args[1], "p2")

    def test_a_smaller_request_is_a_prefix_of_a_larger_one(self):
        """run_eval may grow n_problems between resumes but never reshuffle."""
        small, _ = self._load(3, seed=42)
        large, _ = self._load(8, seed=42)
        self.assertEqual([r["question"] for r in small],
                         [r["question"] for r in large][:3])

    def test_requesting_more_than_exists_is_clamped(self):
        out, _ = self._load(999)
        self.assertEqual(len(out), len(ROWS))

    def test_gold_answers_parse_from_the_gsm8k_contract(self):
        golds = [self.loader.parse_gold(r) for r in ROWS]
        self.assertEqual(golds, [str(i) for i in range(10)])

    def test_an_answer_without_the_marker_yields_no_gold(self):
        """None gold means unlabelable, which must not be confused with 'wrong'."""
        self.assertIsNone(self.loader.parse_gold({"answer": "no marker here"}))


if __name__ == "__main__":
    unittest.main()
