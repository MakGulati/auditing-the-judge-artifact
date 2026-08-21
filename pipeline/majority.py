from __future__ import annotations

import asyncio
from collections import Counter

from pipeline.backend import LLMBackend
from pipeline.solver import DEFAULT_SOLVE_MAX_TOKENS
from prompts.base import PromptTemplate


async def _sample_once(
    problem: str,
    prompts: PromptTemplate,
    backend: LLMBackend,
    max_tokens: int = DEFAULT_SOLVE_MAX_TOKENS,
) -> tuple[str, str | None, bool | None]:
    messages = [{"role": "user", "content": prompts.solve_prompt(problem)}]
    result = await backend.complete(
        messages, temperature=0.7, top_p=0.95, max_tokens=max_tokens
    )
    return result.text, prompts.parse_answer(result.text), result.truncated


async def majority_vote_sample(
    problem: str,
    prompts: PromptTemplate,
    backend: LLMBackend,
    k: int,
    max_tokens: int = DEFAULT_SOLVE_MAX_TOKENS,
) -> tuple[list[str], list[str | None], str | None, list[bool | None]]:
    """Run k stochastic solves in parallel.

    Returns (solutions, answers, majority_answer, truncated_flags).
    """
    pairs = await asyncio.gather(
        *[_sample_once(problem, prompts, backend, max_tokens) for _ in range(k)]
    )
    solutions = [p[0] for p in pairs]
    answers = [p[1] for p in pairs]
    truncated = [p[2] for p in pairs]
    # Group formatting variants (for example ``64`` and ``64.00``) into one vote.
    # Canonicalisation is the prompt's, because it is dataset-specific: the numeric
    # canonicaliser discards every non-numeric MATH answer, which would leave most
    # MATH problems with no votes at all and therefore no majority.
    valid = [prompts.canonical_answer(a) for a in answers]
    valid = [a for a in valid if a is not None]
    majority = Counter(valid).most_common(1)[0][0] if valid else None
    return solutions, answers, majority, truncated
