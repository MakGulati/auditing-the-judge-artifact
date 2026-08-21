#!/usr/bin/env python3
"""Does the probe's advantage survive on the problems least likely to be memorised?

GSM8K and MATH are both from 2021 and predate every checkpoint here, so contamination
cannot be ruled out and the honest question is what it would do to the claims. One
observable proxy: MATH ships a difficulty label, and Level 5 problems are both the
hardest and the least likely to have been absorbed as a solved instance. If the
probe's margin over the judge holds there, the result is not an artefact of the model
recognising problems it has already seen. If the margin lives entirely in Levels 1-2,
that is a finding the paper has to report rather than a detail.

The level is not stored in raw.jsonl -- it is used at load time to select subjects and
levels and then dropped -- so it is recovered here by joining problem text back to the
source dataset. The join is exact and reported: a partial join would silently analyse
a biased subset.

Everything is fit on the MATH TRAIN split as usual; only the scoring of the held-out
test split is broken out by level.

  python filtering/level_breakdown.py --models gemma3,qwen25_7b,llama31_8b
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probe_models import fit_ensemble, load_split, prepare_head_features, shallow_only_scores

ROOT = Path(__file__).resolve().parent.parent
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]
MATH_TAGS = {"gemma3": ("gemma3_math", 256), "qwen25_7b": ("qwen25_7b_math", 128),
             "llama31_8b": ("llama31_8b_math", 128)}


def level_index():
    """problem text -> 'Level N', built from the same mirror the loader reads."""
    from datasets import load_dataset
    idx = {}
    for subject in SUBJECTS:
        for row in load_dataset("EleutherAI/hendrycks_math", subject, split="test"):
            idx[row["problem"].strip()] = row["level"]
    return idx


def auc(y, s):
    """AUC, or None where a level is single-class and the number is undefined."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma3,qwen25_7b,llama31_8b")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--out", default="filtering/figures/paper/level_breakdown.json")
    args = ap.parse_args()

    print("building level index ...", flush=True)
    levels = level_index()
    print(f"  {len(levels)} MATH test problems indexed", flush=True)

    report = {"topk": args.topk, "n_seeds": args.n_seeds, "models": {}}
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        tag, head_dim = MATH_TAGS[key]
        t0 = time.time()
        print(f"=== {key} ({tag}) ===", flush=True)
        tr = load_split(str(ROOT / f"results_{tag}_train" / "hidden_rich.npz"),
                        str(ROOT / f"results_{tag}_train" / "math" / "raw.jsonl"), head_dim)
        te = load_split(str(ROOT / f"results_{tag}_test" / "hidden_rich.npz"),
                        str(ROOT / f"results_{tag}_test" / "math" / "raw.jsonl"), head_dim)

        raw = {}
        for line in open(ROOT / f"results_{tag}_test" / "math" / "raw.jsonl"):
            line = line.strip()
            if line:
                r = json.loads(line)
                raw[r["idx"]] = r
        lv = np.array([levels.get(raw[i]["problem"].strip(), "UNMATCHED") for i in te.idx])
        n_unmatched = int((lv == "UNMATCHED").sum())
        print(f"  joined {len(lv)-n_unmatched}/{len(lv)} test records to a level "
              f"({n_unmatched} unmatched)", flush=True)

        Ftr, Fte, _, _ = prepare_head_features(tr.Z, tr.y, te.Z, args.topk)
        scores = {}
        for m in ("lr", "mlp"):
            s, _, _ = fit_ensemble(Ftr, tr.y, Fte, tr.S, te.S, model_type=m, seed=0,
                                   n_seeds=args.n_seeds)
            scores[f"{m}_controlled"] = s["controlled"]
            scores[f"{m}_raw"] = s["raw"]
        _, scores["shallow"] = shallow_only_scores(tr.S, tr.y, te.S, seed=0)
        scores["judge_p_yes"] = te.p_yes

        rows = {}
        print(f"  {'level':>9s} {'n':>5s} {'%wrong':>7s} {'MLPc':>7s} {'MLPraw':>7s} "
              f"{'shallow':>7s} {'judge':>7s} {'MLPc-judge':>10s}")
        for level in [f"Level {i}" for i in range(1, 6)] + ["UNMATCHED"]:
            sel = lv == level
            if not sel.any():
                continue
            y = te.y[sel]
            row = {"n": int(sel.sum()), "wrong": int((y == 0).sum()),
                   "prevalence": float((y == 0).mean())}
            for name, s in scores.items():
                row[name] = auc(y, np.asarray(s)[sel])
            mj = (None if row["mlp_controlled"] is None or row["judge_p_yes"] is None
                  else row["mlp_controlled"] - row["judge_p_yes"])
            row["mlp_minus_judge"] = mj
            rows[level] = row
            fmt = lambda v: f"{v:7.3f}" if v is not None else f"{'n/a':>7s}"
            mj_s = f"{mj:+10.3f}" if mj is not None else f"{'n/a':>10s}"
            print(f"  {level:>9s} {row['n']:5d} {row['prevalence']:6.1%} "
                  f"{fmt(row['mlp_controlled'])} {fmt(row['mlp_raw'])} "
                  f"{fmt(row['shallow'])} {fmt(row['judge_p_yes'])} {mj_s}")

        report["models"][key] = {"tag": tag, "n_unmatched": n_unmatched, "levels": rows}
        print(f"  [{time.time()-t0:.0f}s]", flush=True)
        del tr, te, Ftr, Fte
        gc.collect()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
