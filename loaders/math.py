from __future__ import annotations

import os
from typing import Any

from datasets import concatenate_datasets, load_dataset

from metrics.latex_answer import extract_boxed, normalize_latex

from .base import DatasetLoader

# The original `hendrycks/competition_math` is disabled on the Hub. This mirror carries
# the same data, one config per subject, each with a train and a test split.
HF_DATASET = "EleutherAI/hendrycks_math"

# Fixed order — the subsets are concatenated before shuffling, so the order is part of
# what `--seed` permutes. Sorting it here rather than taking $MATH_SUBJECTS' order keeps
# a given seed reproducible no matter how the env var is written.
SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

LEVELS = ("1", "2", "3", "4", "5")


def _selected(env_var: str, allowed: tuple[str, ...], normalize) -> tuple[str, ...]:
    """Parse a comma-separated env var into a validated, canonically ordered subset.

    An unrecognised value aborts instead of silently narrowing the problem set: a typo
    in $MATH_SUBJECTS would otherwise produce a run over fewer problems that looks
    completely normal in the logs.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return allowed
    wanted = [normalize(part) for part in raw.split(",") if part.strip()]
    unknown = sorted({w for w in wanted if w not in allowed})
    if unknown:
        raise SystemExit(
            f"[FATAL] {env_var}={raw!r} names unknown value(s) {', '.join(unknown)}.\n"
            f"        Valid values: {', '.join(allowed)}."
        )
    if not wanted:
        raise SystemExit(f"[FATAL] {env_var}={raw!r} selects nothing.")
    return tuple(a for a in allowed if a in set(wanted))


def _normalize_level(part: str) -> str:
    """Accept '3', 'Level 3' and 'level3' alike."""
    return part.strip().lower().removeprefix("level").strip()


class MATHLoader(DatasetLoader):
    """Hendrycks MATH, shaped like the GSM8K loader so run_eval.py stays dataset-blind.

    Selection knobs mirror the GSM8K_SPLIT idiom:
      MATH_SPLIT     train | test        (default test)
      MATH_SUBJECTS  csv of SUBJECTS     (default: all seven)
      MATH_LEVELS    csv of 1..5         (default: all)

    Subject and level selection are NOT separate fingerprint fields; they change which
    problems are sampled, and `run_eval.problems_digest` hashes the sampled questions,
    so a changed selection already makes a resume fatal.
    """

    SPLIT_ENV = "MATH_SPLIT"

    def load(self, n_problems: int, seed: int) -> list[dict[str, Any]]:
        split = os.environ.get(self.SPLIT_ENV, "test")
        subjects = _selected("MATH_SUBJECTS", SUBJECTS, lambda s: s.strip().lower())
        levels = _selected("MATH_LEVELS", LEVELS, _normalize_level)

        parts = [load_dataset(HF_DATASET, subject, split=split) for subject in subjects]
        ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        if set(levels) != set(LEVELS):
            wanted = {f"level {lvl}" for lvl in levels}
            ds = ds.filter(lambda ex: str(ex.get("level", "")).strip().lower() in wanted)
            if len(ds) == 0:
                raise SystemExit(
                    f"[FATAL] MATH_LEVELS={','.join(levels)} leaves no problems in "
                    f"split={split!r} for subjects {','.join(subjects)}."
                )
        ds = ds.shuffle(seed=seed)
        n = min(n_problems, len(ds))
        # `question` is the key run_eval.run_problem and problems_digest read; MATH
        # calls it `problem`. Copy rather than rename so `problem` stays available.
        return [{**ds[i], "question": ds[i]["problem"]} for i in range(n)]

    def parse_gold(self, example: dict[str, Any]) -> str | None:
        """Gold is whatever the reference solution puts in its last ``\\boxed{}``.

        Returns None when the solution boxes nothing (a handful of MATH solutions do
        not), which makes the record unlabelable rather than wrong.
        """
        boxed = extract_boxed(example.get("solution", ""))
        if boxed is None:
            return None
        # Hand back a form the shared comparator can actually read, the same contract
        # GSM8KLoader.parse_gold honours.
        return boxed if normalize_latex(boxed) else None
