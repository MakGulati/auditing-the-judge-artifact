from .base import PromptTemplate

_SOLVE = """\
Solve the following math problem step by step. Show your reasoning clearly.
At the very end, write your final numerical answer on a new line starting with "#### ".

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
- Verify each arithmetic operation and every logical step.
- Confirm the final numerical answer actually answers the question.
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
- Verify each arithmetic operation and every logical step.
- Confirm the final numerical answer is correct.

Judge ONLY the mathematics and the reasoning. Important:
- Assume the standard, intended interpretation of any ambiguous or loosely worded
  problem. Grade-school word problems are often informal — do NOT treat reasonable
  assumptions, minor wording issues, or obvious typos in the PROBLEM as errors.
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
Judge each step on its own arithmetic and logic, given the steps before it. Number
the steps yourself sequentially (1, 2, 3, ...) in the order they appear. Do not skip
steps and do not merge them. Keep each reason to a few words.

After grading every step, output a final line in EXACTLY this format:
VERDICT: YES   (if the final numerical answer is correct)
VERDICT: NO    (if the final numerical answer is incorrect)"""


class GSM8KPrompts(PromptTemplate):
    def solve_prompt(self, problem: str) -> str:
        return _SOLVE.format(problem=problem)

    def judge_prompt_stepgrade(self, problem: str, solution: str) -> str:
        return _JUDGE_STEPGRADE.format(problem=problem, solution=solution)

    def judge_prompt(self, problem: str, solution: str) -> str:
        return _JUDGE.format(problem=problem, solution=solution)

    def judge_prompt_errorhunt(self, problem: str, solution: str) -> str:
        return _JUDGE_ERRORHUNT.format(problem=problem, solution=solution)

    def judge_prompt_errorhunt_soft(self, problem: str, solution: str) -> str:
        return _JUDGE_ERRORHUNT_SOFT.format(problem=problem, solution=solution)
