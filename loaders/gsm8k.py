from __future__ import annotations

import os
import re
from typing import Any

from datasets import load_dataset

from metrics.correctness import parse_numeric_answer

from .base import DatasetLoader


class GSM8KLoader(DatasetLoader):
    # Name of the env var that selects the split. Read by dataset_registry so the
    # fingerprint records the split this loader actually used.
    SPLIT_ENV = "GSM8K_SPLIT"

    def load(self, n_problems: int, seed: int) -> list[dict[str, Any]]:
        split = os.environ.get(self.SPLIT_ENV, "test")
        ds = load_dataset("gsm8k", "main", split=split)
        ds = ds.shuffle(seed=seed)
        n = min(n_problems, len(ds))
        return [ds[i] for i in range(n)]

    def parse_gold(self, example: dict[str, Any]) -> str | None:
        """Extract the number after '#### ' in a GSM8K answer string.

        Must accept a leading sign and a decimal part: GSM8K contains negative
        golds (e.g. ``#### -12``). A digits-only pattern returns None for those,
        and a None gold makes every comparison False — silently relabelling a
        correct answer as wrong. Decimals must not be truncated for the same reason.
        """
        answer = example.get("answer", "")
        match = re.search(r"####\s*\$?\s*([+-]?[\d,]+(?:\.\d+)?)", answer)
        if not match:
            return None
        text = match.group(1).replace(",", "").strip()
        # Only hand back something the shared numeric parser can actually read.
        return text if parse_numeric_answer(text) is not None else None
