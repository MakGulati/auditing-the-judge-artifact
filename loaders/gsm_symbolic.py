from __future__ import annotations

import os
from typing import Any

from datasets import load_dataset

from .gsm8k import GSM8KLoader


class GSMSymbolicLoader(GSM8KLoader):
    """Apple's GSM-Symbolic: GSM8K test problems re-instantiated from templates.

    Same reasoning, same answer format (``#### N``), different surface form -- the
    names and numbers are resampled, so an instance here was never in any pretraining
    corpus even though the template behind it was. That is what makes it a
    contamination probe: a model that solves GSM8K by recalling solved instances
    should lose accuracy here, while one that solves it by reasoning should not.

    ``main`` is the direct re-instantiation and the variant this pipeline wants.
    ``p1``/``p2`` also ADD clauses, so they change the difficulty as well as the
    surface form and cannot separate memorisation from reasoning load.

    Only a test split exists, which suits the intended use: fit the probe on the
    real GSM8K train split and score it here.

    Beware the shape when doing statistics on the output. The 5,000 rows are 100
    templates x 50 instances, so rows are NOT independent -- instances of one
    template share their reasoning and differ only in surface values. The effective
    sample size for anything template-level is nearer 100 than 5,000. The template is
    recoverable by joining ``question`` back to this dataset's ``original_id``.
    """

    SPLIT_ENV = "GSM_SYMBOLIC_SPLIT"
    HF_DATASET = "apple/GSM-Symbolic"

    def load(self, n_problems: int, seed: int) -> list[dict[str, Any]]:
        split = os.environ.get(self.SPLIT_ENV, "test")
        variant = os.environ.get("GSM_SYMBOLIC_VARIANT", "main")
        if variant not in {"main", "p1", "p2"}:
            raise ValueError(
                f"GSM_SYMBOLIC_VARIANT={variant!r} is not one of main/p1/p2. An "
                f"unrecognised variant would otherwise silently fall through to a "
                f"different problem set than the fingerprint records.")
        ds = load_dataset(self.HF_DATASET, variant, split=split)
        ds = ds.shuffle(seed=seed)
        n = min(n_problems, len(ds))
        return [ds[i] for i in range(n)]
