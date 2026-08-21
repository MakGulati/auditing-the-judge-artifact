"""Provenance primitives shared by generation, extraction and analysis.

Deliberately free of torch/transformers/datasets: the analysis-only environment
(`requirements-analysis.txt`) installs none of them, and the probe scripts need the
same record fingerprint the extractors write.

The problem these solve: `idx` is a position in a shuffled subset, not a stable
dataset ID, and `raw.jsonl` carries no per-record provenance. Every stage downstream
of generation joins on `idx`, so two runs over *different problems* align perfectly —
model, geometry and dataset all agree, labels are recomputed from whichever text is
present, and the activations that stay in place were read at another file's prompts.
Nothing about the result looks wrong.
"""
from __future__ import annotations

import hashlib
import json
import os

from metrics.correctness import record_dataset

# Bumped whenever _DIGEST_FIELDS changes. Stored beside the digests so a definition
# change is reported as such instead of as "your records changed" — without it, adding
# a field makes every stored digest mismatch at once, and the honest message ("the
# rule moved") is indistinguishable from the alarming one ("this dump is not yours").
DIGEST_VERSION = 2

# Fields of a raw record that decide either the judge prompt (problem, solution) or the
# labels and scores stored next to the activations. `idx` is deliberately included: it
# is what the dump aligns on, so it belongs inside the thing that proves the alignment
# is real.
#
# `judge_verdict` is in here (v2) because it is an *analysis input*, not a derived
# value: probe_models reads it as the binary judge baseline, and extract_verdict_topk
# exists purely to study it. Two raw files that differ only in their verdicts — the
# same problems re-judged at a different budget or temperature — used to produce
# identical digests, so a resume spliced them together without a word.
_DIGEST_FIELDS = ("idx", "problem", "phase1_solution", "phase1_answer",
                  "gold_answer", "majority_answer", "judge_verdict")


def record_digest(rec: dict) -> str:
    """Content fingerprint of one raw record — what binds a dump row to its prompt."""
    h = hashlib.sha256()
    # Version first, so v1 and v2 digests of the same record never collide.
    h.update(f"v{DIGEST_VERSION}".encode())
    h.update(b"\0")
    h.update(record_dataset(rec).encode())
    h.update(b"\0")
    for key in _DIGEST_FIELDS:
        h.update(str(rec.get(key)).encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def read_run_meta(path: str) -> dict | None:
    """The generation fingerprint `run_eval.py` wrote beside ``path``, if any.

    ``path`` may be the raw.jsonl itself or the directory holding it.
    """
    directory = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    meta_path = os.path.join(directory, "run_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)
