#!/usr/bin/env python3
"""Compare detectors at a matched REVIEW BUDGET instead of a matched precision target.

The operating point this pipeline reports elsewhere is calibrated on train at the
judge's train precision and then applied unchanged to test. That does not equalise
anything on test: precision is a property of the score AND the prevalence, so a
threshold that hits 0.70 on train lands somewhere else on test, and each rule ends up
at a different point on its own PR curve. Recalls read off different points are not
comparable, and the caption warning that says so does not repair the comparison.

Matching the flag rate fixes it outright.

  * The threshold needs no test labels -- "flag the lowest-scoring m" is a property of
    the score distribution, where "flag until precision is 0.70" is not.
  * It is exactly comparable. Every rule flags the same m items out of the same n,
    against the same W wrong ones, so precision = TP/m and recall = TP/W are both
    monotone in TP. There is one quantity to compare, not two that trade off.
  * It is the constraint a practitioner actually has. Review budgets are expressed as
    "we can look at 10% of outputs", never as "we want precision 0.70".

Budgets reported: the judge's own flag rate on that split -- the one the binary
verdict picks for itself, so the verdict's point is exact rather than interpolated --
and fixed 5/10/20% budgets, because the judge's rate is idiosyncratic (Llama flags
2.9% on GSM8K) and a fixed budget is what a reader can act on.

Everything is fit on TRAIN, as everywhere else here.

  python filtering/matched_budget.py --runs gemma3,qwen25_7b,llama31_8b
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probe_models import (fit_ensemble, load_split, prepare_head_features,
                          shallow_only_scores)

ROOT = Path(__file__).resolve().parent.parent
DETECTORS = ["mlp_controlled", "mlp_raw", "lr_controlled", "shallow", "p_yes", "verdict"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def at_budget(scores, wrong, m, rng):
    """Flag the m lowest-scoring items; return TP, recall, precision.

    Ties matter here and are broken at random rather than by index. The binary
    verdict is constant within each class, so an index-order tiebreak would hand it
    whatever ordering the dataset happened to have -- a real effect on its score that
    has nothing to do with the detector.
    """
    s = np.asarray(scores, dtype=float)
    order = np.lexsort((rng.random(len(s)), s))
    flagged = order[:m]
    tp = int(wrong[flagged].sum())
    W = int(wrong.sum())
    return {"tp": tp, "m": int(m),
            "recall": tp / W if W else None,
            "precision": tp / m if m else None,
            "recall_ci": wilson(tp, W) if W else (None, None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="gemma3,qwen25_7b,llama31_8b,gemma3_math,"
                                      "qwen25_7b_math,llama31_8b_math")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--budgets", default="0.05,0.10,0.20")
    ap.add_argument("--out", default="filtering/figures/paper/matched_budget.json")
    args = ap.parse_args()

    from make_paper_figures import RUNS

    fixed = [float(b) for b in args.budgets.split(",")]
    report = {"topk": args.topk, "n_seeds": args.n_seeds, "fixed_budgets": fixed,
              "runs": {}}

    for tag in [t.strip() for t in args.runs.split(",") if t.strip()]:
        cfg = RUNS[tag]
        ds = cfg.get("dataset", "gsm8k")
        t0 = time.time()
        print(f"=== {tag} ===", flush=True)
        tr = load_split(str(cfg["train"] / "hidden_rich.npz"),
                        str(cfg["train"] / ds / "raw.jsonl"), cfg["head_dim"])
        te = load_split(str(cfg["test"] / "hidden_rich.npz"),
                        str(cfg["test"] / ds / "raw.jsonl"), cfg["head_dim"])

        Ftr, Fte, _, _ = prepare_head_features(tr.Z, tr.y, te.Z, args.topk)
        s = {}
        for m in ("lr", "mlp"):
            e, _, _ = fit_ensemble(Ftr, tr.y, Fte, tr.S, te.S, model_type=m, seed=0,
                                   n_seeds=args.n_seeds)
            s[f"{m}_raw"], s[f"{m}_controlled"] = e["raw"], e["controlled"]
        _, s["shallow"] = shallow_only_scores(tr.S, tr.y, te.S, seed=0)
        s["p_yes"] = te.p_yes
        s["verdict"] = te.yes.astype(float)

        wrong = (te.y == 0).astype(int)
        n, W = len(te.y), int(wrong.sum())
        judge_rate = float((te.yes == 0).mean())
        rng = np.random.default_rng(0)

        run = {"dataset": ds, "n": n, "wrong": W, "prevalence": W / n,
               "judge_flag_rate": judge_rate, "budgets": {}}
        for label, rate in [("judge_rate", judge_rate)] + [(f"{b:.0%}", b) for b in fixed]:
            m = int(round(rate * n))
            cell = {"rate": rate, "m": m,
                    "detectors": {d: at_budget(s[d], wrong, m, rng) for d in DETECTORS}}
            run["budgets"][label] = cell
        report["runs"][tag] = run

        print(f"  n={n} wrong={W} ({W/n:.1%})  judge flags {judge_rate:.1%}")
        for label, cell in run["budgets"].items():
            head = f"  budget {label:>10s} (m={cell['m']:4d})"
            best = max(cell["detectors"], key=lambda d: cell["detectors"][d]["tp"])
            parts = []
            for d in DETECTORS:
                c = cell["detectors"][d]
                mark = "*" if d == best else " "
                parts.append(f"{d}={c['recall']:.3f}{mark}")
            print(f"{head}  " + "  ".join(parts), flush=True)

        print(f"  [{time.time()-t0:.0f}s]", flush=True)
        del tr, te, Ftr, Fte
        gc.collect()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
