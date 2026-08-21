from __future__ import annotations

import re

from metrics.correctness import parse_numeric_answer
from pipeline.backend import LLMBackend
from prompts.base import PromptTemplate

# The reply budget for a solve. 1024 was sized for GSM8K, whose solutions are short;
# it truncated 24% of MATH. Overridable via `run_eval.py --solve_max_tokens`, and
# recorded in the run fingerprint because it changes the generated text.
DEFAULT_SOLVE_MAX_TOKENS = 1024

# Ordered from most specific to least — first match wins
_ANSWER_PATTERNS = [
    r"####\s*([-\d,]+(?:\.\d+)?)",
    r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([-\d,]+(?:\.\d+)?)",
    r"(?:therefore|thus|so|hence)[,.]?\s+(?:the\s+)?(?:answer|result|total)\s+is\s*:?\s*([-\d,]+(?:\.\d+)?)",
    r"answer\s*:\s*([-\d,]+(?:\.\d+)?)",
    r"\*\*([-\d,]+(?:\.\d+)?)\*\*",
    r"=\s*([-\d,]+(?:\.\d+)?)\s*$",
]


def parse_answer(text: str) -> str | None:
    """Extract the final numerical answer from model output. Returns None if unparseable."""
    for pattern in _ANSWER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            candidate = match.group(1).replace(",", "").strip()
            if parse_numeric_answer(candidate) is not None:
                return candidate
    # Fall back to the last standalone number in the response. `[-\d,]+` can match
    # a lone "-" or ",", so walk backwards to the last token that actually parses
    # instead of failing outright on punctuation the model happened to end with.
    numbers = re.findall(r"(?<!\w)([-\d,]+(?:\.\d+)?)(?!\w)", text)
    for candidate in reversed(numbers):
        candidate = candidate.replace(",", "").strip()
        if parse_numeric_answer(candidate) is not None:
            return candidate
    return None


async def solve_greedy(
    problem: str,
    prompts: PromptTemplate,
    backend: LLMBackend,
    max_tokens: int = DEFAULT_SOLVE_MAX_TOKENS,
) -> tuple[str, str | None, bool | None]:
    """Single greedy (temperature=0) solve.

    Returns (solution_text, parsed_answer, truncated). ``truncated`` is None when the
    backend cannot report it — unknown, which is not the same as False.
    """
    messages = [{"role": "user", "content": prompts.solve_prompt(problem)}]
    result = await backend.complete(messages, temperature=0.0, max_tokens=max_tokens)
    # The prompt defines the output contract, so the prompt owns the parser. The base
    # template's default IS `parse_answer` above, so GSM8K is unchanged.
    return result.text, prompts.parse_answer(result.text), result.truncated
