from __future__ import annotations

import re

from pipeline.backend import LLMBackend
from prompts.base import PromptTemplate


def parse_verdict(text: str) -> str:
    """Extract YES or NO from judge response. Defaults to NO if ambiguous."""
    upper = text.strip().upper()
    if upper.startswith("YES"):
        return "YES"
    if upper.startswith("NO"):
        return "NO"
    # justification-first judges (e.g. no-think reasoning models) end with the
    # verdict, and their prose can contain stray "no"s ("no errors") — take the
    # LAST standalone YES/NO
    matches = re.findall(r"\b(YES|NO)\b", upper)
    return matches[-1] if matches else "NO"


async def judge_solution(
    problem: str,
    solution: str,
    prompts: PromptTemplate,
    backend: LLMBackend,
    max_tokens: int = 16,
) -> str:
    """Ask the model to judge correctness of a solution. Returns 'YES' or 'NO'."""
    messages = [{"role": "user", "content": prompts.judge_prompt(problem, solution)}]
    text = await backend.chat_complete(messages, temperature=0.0, max_tokens=max_tokens)
    return parse_verdict(text)
