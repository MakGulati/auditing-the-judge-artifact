"""Extraction must template the judge prompt exactly the way generation did.

A tokenizer rebuilt from tokenizer.json alone carries no special tokens, so a chat
template interpolating `{{ bos_token }}` renders it as the empty string: the prompt
loses a leading token and every activation is read one position off from the sequence
that produced the recorded verdict. That failure is silent, so there is no degraded
fallback — extraction either gets a fully configured tokenizer or stops.
"""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extraction_common import _validate_tokenizer, load_tokenizer


class StubTokenizer:
    def __init__(self, chat_template=None, **special):
        self.chat_template = chat_template
        self._special = special

    def __getattr__(self, name):
        # unset special tokens read as None, as they do on a bare tokenizer
        if name.endswith("_token"):
            return self._special.get(name)
        raise AttributeError(name)


class ValidationTests(unittest.TestCase):
    def test_missing_chat_template_is_rejected(self):
        with self.assertRaises(ValueError) as cm:
            _validate_tokenizer(StubTokenizer(None), "stub")
        self.assertIn("chat_template", str(cm.exception))

    def test_template_referencing_an_absent_special_token_is_rejected(self):
        for token in ("bos_token", "eos_token", "pad_token", "unk_token"):
            with self.assertRaises(ValueError, msg=token) as cm:
                _validate_tokenizer(
                    StubTokenizer("{{ %s }}{{ messages[0].content }}" % token), "stub")
            self.assertIn(token, str(cm.exception))

    def test_template_is_accepted_when_the_token_exists(self):
        tok = StubTokenizer("{{ bos_token }}{{ messages[0].content }}", bos_token="<s>")
        self.assertIs(_validate_tokenizer(tok, "stub"), tok)

    def test_model_without_bos_is_accepted_if_its_template_never_asks(self):
        """Some checkpoints legitimately have no BOS; only referenced tokens matter."""
        tok = StubTokenizer("<|im_start|>{{ messages[0].content }}")
        self.assertIs(_validate_tokenizer(tok, "stub"), tok)


class NoDegradedFallbackTests(unittest.TestCase):
    def test_no_module_constructs_a_bare_tokenizer(self):
        """`PreTrainedTokenizerFast(tokenizer_file=...)` is exactly the object that
        drops the special-token configuration; nothing may build one."""
        offenders = []
        for path in list(ROOT.glob("*.py")) + list((ROOT / "pipeline").glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and getattr(node.func, "id", "") == "PreTrainedTokenizerFast"):
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_unloadable_tokenizer_raises_rather_than_degrading(self):
        with self.assertRaises(SystemExit) as cm:
            load_tokenizer("definitely-not/a-real-model-xyzzy")
        msg = str(cm.exception)
        self.assertIn("could not load a fully configured tokenizer", msg)
        self.assertIn("PreTrainedTokenizerFast", msg)   # explains why there is no fallback


if __name__ == "__main__":
    unittest.main()
