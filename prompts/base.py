from abc import ABC, abstractmethod
from typing import Any


class PromptTemplate(ABC):
    @abstractmethod
    def solve_prompt(self, problem: str) -> str:
        """Return the prompt for solving a problem."""
        ...

    @abstractmethod
    def judge_prompt(self, problem: str, solution: str) -> str:
        """Return the prompt for judging a proposed solution."""
        ...

    # The parser belongs with the prompt because the prompt defines the output
    # contract: GSM8K's asks for `#### <number>`, MATH's asks for `\boxed{...}`.
    # Both default to the numeric pair, so a template that does not override them
    # behaves exactly as before these hooks existed.

    def parse_answer(self, text: str) -> str | None:
        """Extract the final answer from model output; None if unparseable."""
        # Imported here, not at module scope: pipeline.solver imports this module.
        from pipeline.solver import parse_answer as parse_numeric
        return parse_numeric(text)

    def canonical_answer(self, value: Any) -> str | None:
        """Canonical form used to group majority votes; None if there is no answer."""
        from metrics.correctness import canonical_numeric_answer
        return canonical_numeric_answer(value)
