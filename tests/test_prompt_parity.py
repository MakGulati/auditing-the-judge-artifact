"""Activations must be read at the same token sequence that produced the verdict.

`VLLMBackend.chat_complete` builds the prompt as
``apply_chat_template(..., tokenize=False) + prefill`` and hands vLLM one string, so
the tokenizer sees the join. Concatenating two independently tokenized ID sequences
instead can differ wherever a merge spans the boundary.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extraction_common import encode_judge_prompt


class FakeTokenizer:
    """Minimal stand-in with a merge rule that spans the template/prefill join.

    Mirrors real BPE behaviour: "\\n\\n" is one token when the tokenizer sees it as a
    unit, but templating and prefill tokenized separately each yield a lone "\\n".
    """

    def __init__(self):
        self.vocab = {"<s>": 0, "<user>": 1, "<assistant>": 2, "\n": 3, "\n\n": 4,
                      "</think>": 5, "P": 6}

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=True, return_tensors=None, return_dict=False):
        text = "<s><user>" + messages[0]["content"]
        if add_generation_prompt:
            text += "<assistant>\n"
        if not tokenize:
            return text
        ids = [self._encode(text)]
        return {"input_ids": ids} if return_dict else ids

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        assert not add_special_tokens, "would double the template's BOS"
        return {"input_ids": [self._encode(text)]}

    def _encode(self, text):
        out, i = [], 0
        toks = sorted(self.vocab, key=len, reverse=True)   # longest-match first
        while i < len(text):
            for t in toks:
                if text.startswith(t, i):
                    out.append(self.vocab[t])
                    i += len(t)
                    break
            else:
                out.append(6)
                i += 1
        return out


class PrefillTokenizationTests(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()

    def test_prefill_is_tokenized_with_the_template_not_appended_as_ids(self):
        got = encode_judge_prompt(self.tok, "P", "\n</think>\n\n")[0]
        # what generation does: template to text, concatenate, tokenize once
        want = self.tok._encode("<s><user>P<assistant>\n" + "\n</think>\n\n")
        self.assertEqual(got, want)

    def test_id_concatenation_would_have_differed(self):
        """Guards the actual failure mode: the '\\n' ending the generation prompt and
        the '\\n' starting the prefill merge into one '\\n\\n' token in the string path
        but stay two tokens when the ID sequences are concatenated."""
        joined = encode_judge_prompt(self.tok, "P", "\n</think>\n\n")[0]
        template_ids = self.tok.apply_chat_template(
            [{"role": "user", "content": "P"}], add_generation_prompt=True)[0]
        naive = template_ids + self.tok._encode("\n</think>\n\n")
        self.assertNotEqual(joined, naive)
        self.assertLess(len(joined), len(naive))

    def test_no_prefill_uses_the_plain_template_path(self):
        got = encode_judge_prompt(self.tok, "P")[0]
        want = self.tok.apply_chat_template(
            [{"role": "user", "content": "P"}], add_generation_prompt=True)[0]
        self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
