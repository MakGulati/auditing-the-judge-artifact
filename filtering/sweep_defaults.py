#!/usr/bin/env python3
"""Is the MLP-over-LR gap a finding, or a consequence of the defaults?

The headline comparison is between two probe families, and each was handed an
arbitrary capacity knob: LR gets C=0.5, the MLP gets one hidden layer of 128 units,
and both read the top TOPK=32 heads selected by train-only LDA. A reader is entitled
to ask whether the gap survives when the loser is given a fairer setting, and nothing
in the repository answered that.

This sweeps one axis at a time around the shipped defaults, because that is the
question being asked: not "what is the best configuration" (which would overfit the
test split by selection) but "does the conclusion move when each knob moves". It also
reports the best-of-grid for EACH probe, which is the strongest form of the fairness
check -- if LR at its best still loses to the MLP at its best, the gap is not a
capacity artefact.

Everything is fit on TRAIN and scored on held-out TEST through the same
`fit_ensemble` the paper figures use, so the numbers are comparable to
`metrics_{tag}.json` at the default cell.

  python filtering/sweep_defaults.py --runs gemma3,gemma3_math --out sweep.json

Costs are dominated by the per-head LDA sweep, which does not depend on any knob
here, so it is computed once per run and reused across the whole grid.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe_models as PM
from probe_models import fit_ensemble, load_split
from metrics.correctness import label_policy_for

ROOT = Path(__file__).resolve().parent.parent

# One axis at a time around the shipped defaults (topk=32, C=0.5, hidden=(128,)).
TOPKS = [8, 16, 32, 64, 128]
CS = [0.05, 0.1, 0.5, 2.0, 10.0]
HIDDENS = [(32,), (64,), (128,), (256,), (512,), (128, 128)]
DEFAULT_TOPK, DEFAULT_C, DEFAULT_HIDDEN = 32, 0.5, (128,)


def patched_make_classifier(C, hidden):
    """`make_classifier` with the two capacity knobs overridden, everything else as shipped."""
    def make(model_type, seed=0, y=None):
        if model_type == "lr":
            return LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
        if model_type == "mlp":
            vf = 0.15
            es = True if y is None else PM._can_use_mlp_early_stopping(y, vf)
            return MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-4,
                                 learning_rate_init=1e-3, max_iter=300,
                                 early_stopping=es, validation_fraction=vf,
                                 random_state=seed)
        raise ValueError(model_type)
    return make


def cell(Ftr, ytr, Fte, yte, Str, Ste, model_type, C, hidden, n_seeds):
    original = PM.make_classifier
    PM.make_classifier = patched_make_classifier(C, hidden)
    try:
        scores, _, _ = fit_ensemble(Ftr, ytr, Fte, Str, Ste, model_type=model_type,
                                    seed=0, n_seeds=n_seeds)
    finally:
        PM.make_classifier = original
    return {v: float(roc_auc_score(yte, scores[v])) for v in ("raw", "controlled")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated tags from make_paper_figures.RUNS")
    ap.add_argument("--n_seeds", type=int, default=3,
                    help="MLP initialisations averaged per cell (paper uses 5; 3 keeps "
                         "the grid affordable and the ensemble still beats a single draw)")
    ap.add_argument("--out", default="filtering/figures/sweep_defaults.json")
    args = ap.parse_args()

    from make_paper_figures import RUNS

    report = {"defaults": {"topk": DEFAULT_TOPK, "C": DEFAULT_C, "hidden": list(DEFAULT_HIDDEN)},
              "n_seeds": args.n_seeds, "runs": {}}

    for tag in [t.strip() for t in args.runs.split(",") if t.strip()]:
        cfg = RUNS[tag]
        ds = cfg.get("dataset", "gsm8k")
        t0 = time.time()
        print(f"=== {tag} ===", flush=True)
        tr = load_split(str(cfg["train"] / "hidden_rich.npz"),
                        str(cfg["train"] / ds / "raw.jsonl"), cfg["head_dim"])
        te = load_split(str(cfg["test"] / "hidden_rich.npz"),
                        str(cfg["test"] / ds / "raw.jsonl"), cfg["head_dim"])
        ytr, yte = tr.y, te.y
        # The expensive part, and it depends on no knob in this sweep: compute the
        # per-head train LDA AUCs once, then slice different top-k from the ranking.
        nH = tr.Z.shape[1]
        aucs = np.array([PM.lda_auc(tr.Z[:, h, :].astype(np.float32), ytr) for h in range(nH)])
        order = np.argsort(-aucs)
        print(f"  {len(ytr)} train / {len(yte)} test, {nH} heads, LDA sweep "
              f"{time.time()-t0:.0f}s", flush=True)

        def feats(topk):
            top = order[:topk]
            return (tr.Z[:, top, :].reshape(len(ytr), -1).astype(np.float32),
                    te.Z[:, top, :].reshape(len(yte), -1).astype(np.float32))

        rows = []

        def run_cell(topk, C, hidden, axis):
            Ftr, Fte = feats(topk)
            r = {"axis": axis, "topk": topk, "C": C, "hidden": list(hidden)}
            for m in ("lr", "mlp"):
                a = cell(Ftr, ytr, Fte, yte, tr.S, te.S, m, C, hidden, args.n_seeds)
                r[f"{m}_raw"], r[f"{m}_controlled"] = a["raw"], a["controlled"]
            r["gap_controlled"] = r["mlp_controlled"] - r["lr_controlled"]
            rows.append(r)
            print(f"    topk={topk:<4} C={C:<5} hidden={str(hidden):<10} "
                  f"LRc={r['lr_controlled']:.4f} MLPc={r['mlp_controlled']:.4f} "
                  f"gap={r['gap_controlled']:+.4f}", flush=True)
            return r

        run_cell(DEFAULT_TOPK, DEFAULT_C, DEFAULT_HIDDEN, "default")
        for k in TOPKS:
            if k != DEFAULT_TOPK:
                run_cell(k, DEFAULT_C, DEFAULT_HIDDEN, "topk")
        for c in CS:
            if c != DEFAULT_C:
                run_cell(DEFAULT_TOPK, c, DEFAULT_HIDDEN, "C")
        for h in HIDDENS:
            if h != DEFAULT_HIDDEN:
                run_cell(DEFAULT_TOPK, DEFAULT_C, h, "hidden")

        best_lr = max(r["lr_controlled"] for r in rows)
        best_mlp = max(r["mlp_controlled"] for r in rows)
        report["runs"][tag] = {
            "dataset": ds, "n_train": len(ytr), "n_test": len(yte),
            "wrong_test": int((yte == 0).sum()),
            "label_policy": label_policy_for(ds),
            "rows": rows,
            # The fairness check: each probe at ITS best cell anywhere in the grid.
            # Both are test-selected, so neither is an honest estimate on its own --
            # the comparison between them is the point.
            "best_of_grid": {"lr_controlled": best_lr, "mlp_controlled": best_mlp,
                             "gap": best_mlp - best_lr},
        }
        print(f"  best-of-grid: LRc={best_lr:.4f} MLPc={best_mlp:.4f} "
              f"gap={best_mlp-best_lr:+.4f}   [{time.time()-t0:.0f}s]", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
