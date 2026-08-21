"""Single place that knows which loader, prompts and split env var a dataset uses.

`run_eval.py` and all three `extract_hidden_rich*.py` need the same mapping, and they
need to agree: extraction re-templates the judge prompt from scratch, so if it picked a
different prompt class than generation did, activations would be read at a different
token sequence than produced the recorded verdict.

Loaders are imported lazily because they pull in the `datasets` library, which the
analysis-only environment (`requirements-analysis.txt`) deliberately does not install.
Prompt classes have no dependencies worth deferring.
"""
from __future__ import annotations

import os

from prompts.gsm8k import GSM8KPrompts
from prompts.math import MATHPrompts

PROMPTS_MAP = {
    "gsm8k": GSM8KPrompts,
    "math": MATHPrompts,
    # GSM-Symbolic re-instantiates GSM8K templates: same task, same answer contract,
    # so it shares GSM8K's prompts and its numeric equivalence policy. Only the
    # surface form of each instance differs, which is the point of running it.
    "gsm_symbolic": GSM8KPrompts,
}

DATASETS = tuple(PROMPTS_MAP)

# Kept in sync with PROMPTS_MAP by `test_dataset_registry`.
_LOADER_PATHS = {
    "gsm8k": ("loaders.gsm8k", "GSM8KLoader"),
    "math": ("loaders.math", "MATHLoader"),
    "gsm_symbolic": ("loaders.gsm_symbolic", "GSMSymbolicLoader"),
}


def _check(dataset: str) -> str:
    if dataset not in PROMPTS_MAP:
        raise KeyError(f"unknown dataset {dataset!r}; known: {', '.join(DATASETS)}")
    return dataset


def prompts_for(dataset: str):
    """Prompt class for ``dataset``. Cheap — safe to call from the extractors."""
    return PROMPTS_MAP[_check(dataset)]


def loader_for(dataset: str):
    """Loader class for ``dataset``, imported on demand."""
    import importlib

    module_name, class_name = _LOADER_PATHS[_check(dataset)]
    return getattr(importlib.import_module(module_name), class_name)


def split_env(dataset: str) -> str:
    """Name of the env var selecting ``dataset``'s split (e.g. ``GSM8K_SPLIT``)."""
    return loader_for(dataset).SPLIT_ENV


def dataset_split(dataset: str) -> str:
    """The split this invocation will actually read, for the run fingerprint."""
    return os.environ.get(split_env(dataset), "test")
