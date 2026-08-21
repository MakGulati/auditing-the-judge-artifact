from abc import ABC, abstractmethod
from typing import Any


class DatasetLoader(ABC):
    @abstractmethod
    def load(self, n_problems: int, seed: int) -> list[dict[str, Any]]:
        """Load and return a shuffled subset of n_problems examples."""
        ...

    @abstractmethod
    def parse_gold(self, example: dict[str, Any]) -> str | None:
        """Extract the canonical gold answer string from a raw example."""
        ...
