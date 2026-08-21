"""Prompts for the Hendrycks MATH dataset.

A mirror of prompts/gsm8k.py with two differences that matter:

  * the output contract is ``\\boxed{...}`` rather than ``#### <number>``, because MATH
    answers are LaTeX expressions and half of them are not numbers;
  * the judge prompts say "final answer" where GSM8K's say "final numerical answer" —
    telling a grader to check a *numerical* answer on ``\\text{even}`` invites it to
    mark a correct solution wrong.

The judge wording is otherwise kept as close to GSM8K's as the subject matter allows,
so verdicts across the two datasets are comparable.
"""
import re
from typing import Any

from metrics.latex_answer import (canonical_latex_answer, extract_boxed,
                                  normalize_latex)

from .base import PromptTemplate

_SOLVE = """\
Solve the following math problem step by step. Show your reasoning clearly.
At the very end, put your final answer inside \\boxed{{}}.

Problem: {problem}"""

_JUDGE = """\
Problem: {problem}

Proposed solution:
{solution}

Does the above solution arrive at the correct final answer?
Reply with exactly one word: YES or NO. Do not re-solve the problem."""

_JUDGE_ERRORHUNT = """\
You are a strict math grader. Your task is to find any mistake in the proposed solution.

Problem: {problem}

Proposed solution:
{solution}

Independently work through the problem and scrutinize every step of the proposed
solution:
- Re-read the problem and note exactly what is being asked.
- Verify each algebraic manipulation, computation, and logical step.
- Confirm the final boxed answer actually answers the question.
Actively look for errors. If you find even one mistake in the reasoning or in the
final answer, the solution is INCORRECT.

After your analysis, end your response with a final line in exactly this format:
VERDICT: YES   (if the final answer is correct)
VERDICT: NO    (if the final answer is incorrect)"""


_JUDGE_ERRORHUNT_SOFT = """\
You are a careful math grader. Your job is to decide whether the proposed solution
reaches the correct final answer.

Problem: {problem}

Proposed solution:
{solution}

Work through the problem yourself and check the proposed solution step by step:
- Identify what the problem is asking under its most natural, intended reading.
- Verify each algebraic manipulation, computation, and logical step.
- Confirm the final boxed answer is correct.

Judge ONLY the mathematics and the reasoning. Important:
- An answer written in a different but mathematically equivalent form (an unsimplified
  fraction, a rearranged expression, a different but equal radical form) is CORRECT.
- Do NOT treat formatting choices, missing units, or LaTeX style as errors.
- Mark the solution INCORRECT only if it contains an actual mathematical or logical
  mistake, or its final answer is wrong under the intended reading.

After your analysis, end your response with a final line in exactly this format:
VERDICT: YES   (if the final answer is correct)
VERDICT: NO    (if the final answer is incorrect)"""


_JUDGE_STEPGRADE = """\
You are a strict math grader. Grade the proposed solution ONE STEP AT A TIME.

Problem: {problem}

Proposed solution:
{solution}

Go through the solution in order. For EACH reasoning step, output exactly one line:
Step <k>: CORRECT
or
Step <k>: INCORRECT - <short reason>
Judge each step on its own algebra, computation and logic, given the steps before it.
Number the steps yourself sequentially (1, 2, 3, ...) in the order they appear. Do not
skip steps and do not merge them. Keep each reason to a few words.

After grading every step, output a final line in EXACTLY this format:
VERDICT: YES   (if the final answer is correct)
VERDICT: NO    (if the final answer is incorrect)"""

# Last-resort patterns for a solution that never used \boxed{}. Kept deliberately
# short: a loose fallback invents an answer out of prose and labels a model wrong for
# the grader's mistake rather than its own.
#
# `_ANSWER_TAIL` stops at the sentence-ending period but keeps a decimal point: a
# period is consumed only when a digit follows it. Excluding '.' outright truncated
# "the final answer is 0.5." to "0", which is not a parse failure but a *different
# number* — it can mark a wrong answer correct as easily as the reverse.
_ANSWER_TAIL = r"((?:[^\n$.]|\.(?=\d))+)"
_UNBOXED_FALLBACKS = (
    r"(?:the\s+)?final\s+answer\s+is\s*:?\s*\$?" + _ANSWER_TAIL,
    r"(?:the\s+)?answer\s+is\s*:?\s*\$?" + _ANSWER_TAIL,
)


class MATHPrompts(PromptTemplate):
    def solve_prompt(self, problem: str) -> str:
        return _SOLVE.format(problem=problem)

    def judge_prompt(self, problem: str, solution: str) -> str:
        return _JUDGE.format(problem=problem, solution=solution)

    def judge_prompt_errorhunt(self, problem: str, solution: str) -> str:
        return _JUDGE_ERRORHUNT.format(problem=problem, solution=solution)

    def judge_prompt_errorhunt_soft(self, problem: str, solution: str) -> str:
        return _JUDGE_ERRORHUNT_SOFT.format(problem=problem, solution=solution)

    def judge_prompt_stepgrade(self, problem: str, solution: str) -> str:
        return _JUDGE_STEPGRADE.format(problem=problem, solution=solution)

    def parse_answer(self, text: str) -> str | None:
        """Boxed answer if present, else an explicit 'the answer is ...' statement.

        Returning None marks the record *wrong*, not unlabelable: the gold answer is
        still known, the model simply failed the output contract.
        """
        boxed = extract_boxed(text)
        if boxed is not None and normalize_latex(boxed):
            return boxed
        for pattern in _UNBOXED_FALLBACKS:
            match = re.search(pattern, text or "", re.IGNORECASE)
            if match and normalize_latex(match.group(1)):
                return match.group(1).strip()
        return None

    def canonical_answer(self, value: Any) -> str | None:
        return canonical_latex_answer(value)
