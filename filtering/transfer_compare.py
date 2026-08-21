#!/usr/bin/env python3
"""Does the probe transfer, or does it need in-distribution labels to be any use?

Every other result here fits on a dataset's TRAIN split and scores its own TEST
split. That is the favourable case, and it is also the one a practitioner cannot
have: training the probe needs thousands of gold-labelled answers from the same
model on the same task, which is exactly the supervision that is missing at
deployment. If the probe only works in-distribution, "white-box probes detect
errors" is a much narrower claim than it sounds.

So: fit everything -- head selection, the shallow residualiser, the probe -- on ONE
dataset's train split, and score the OTHER dataset's held-out test split, same model
throughout. Both directions, three models.

The comparison that matters is not transfer-vs-zero. It is:

  in-distribution   the same probe fit on the target's own train split (the ceiling)
  judge p_YES       the black-box baseline on that same target split (the thing to beat)
  shallow           surface text features, fit on source, applied to target

Beating the judge after a dataset shift is the claim worth having. Merely being
above chance is not.

  python filtering/transfer_compare.py --models gemma3,qwen25_7b,llama31_8b
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
from probe_models import (fit_ensemble, load_split, prepare_head_features,
                          shallow_only_scores)

ROOT = Path(__file__).resolve().parent.parent

# model key -> (gsm8k tag, math tag, head_dim)
PAIRS = {
    "gemma3": ("gemma3", "gemma3_math", 256),
    "qwen25_7b": ("qwen25_7b", "qwen25_7b_math", 128),
    "llama31_8b": ("llama31_8b", "llama31_8b_math", 128),
}


def split_path(tag, split, dataset):
    d = ROOT / f"results_{tag}_{split}"
    return str(d / "hidden_rich.npz"), str(d / dataset / "raw.jsonl")


def evaluate(src, tgt, topk, n_seeds):
    """Fit on `src` (a train split), score `tgt` (a test split).

    Returns the point AUCs and the per-item scores that produced them. The
    scores are kept because the interesting uncertainty here is *paired*: the
    transferred probe, the in-distribution probe and the judge all score the
    same target items, so resampling those items jointly cancels the shared
    item-difficulty noise that separate per-detector intervals would keep.
    """
    Ftr, Fte, _, _ = prepare_head_features(src.Z, src.y, tgt.Z, topk)
    out, scores = {}, {}
    for m in ("lr", "mlp"):
        s, _, _ = fit_ensemble(Ftr, src.y, Fte, src.S, tgt.S, model_type=m,
                               seed=0, n_seeds=n_seeds)
        for v in ("raw", "controlled"):
            scores[f"{m}_{v}"] = np.asarray(s[v], dtype=float)
            out[f"{m}_{v}"] = float(roc_auc_score(tgt.y, s[v]))
    _, sh = shallow_only_scores(src.S, src.y, tgt.S, seed=0)
    scores["shallow"] = np.asarray(sh, dtype=float)
    out["shallow"] = float(roc_auc_score(tgt.y, sh))
    return out, scores


def _auc(y, s):
    """AUC via rank sum. sklearn's roc_auc_score is too slow for 2000 draws."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0    # average rank over ties
        i = j + 1
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def paired_target_boot(y, score_sets, judge, n_boot, seed):
    """Class-stratified paired bootstrap over one target split's items.

    Every detector in `score_sets` plus the judge is re-scored on the *same*
    resampled items, so any difference of two entries is a paired difference.
    Conditional on the fitted probes: the draw resamples evaluation items, it
    does not refit.
    """
    rng = np.random.default_rng(seed)
    correct = np.flatnonzero(y == 1)
    wrong = np.flatnonzero(y == 0)
    names = list(score_sets)
    draws = {n: np.empty(n_boot) for n in names}
    draws["judge_p_yes"] = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(correct, len(correct), replace=True),
                              rng.choice(wrong, len(wrong), replace=True)])
        yy = y[idx]
        for n in names:
            draws[n][b] = _auc(yy, score_sets[n][idx])
        draws["judge_p_yes"][b] = _auc(yy, judge[idx])
    return draws


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"mean": float(np.mean(v)), "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma3,qwen25_7b,llama31_8b")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000,
                    help="paired class-stratified target-item resamples; 0 disables")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--out", default="filtering/figures/paper/transfer_compare.json")
    args = ap.parse_args()

    report = {"topk": args.topk, "n_seeds": args.n_seeds,
              "n_boot": args.n_boot, "boot_seed": args.seed,
              "boot_scheme": "paired, class-stratified over target test items, "
                             "conditional on the fitted probes",
              "models": {}}
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        g_tag, m_tag, head_dim = PAIRS[key]
        t0 = time.time()
        print(f"=== {key} (head_dim={head_dim}) ===", flush=True)
        splits = {}
        for name, tag, split, ds in (("g_train", g_tag, "train", "gsm8k"),
                                     ("g_test", g_tag, "test", "gsm8k"),
                                     ("m_train", m_tag, "train", "math"),
                                     ("m_test", m_tag, "test", "math")):
            h, r = split_path(tag, split, ds)
            splits[name] = load_split(h, r, head_dim)
        # Same checkpoint on both datasets, or the head axis means different things.
        models = {s.info["dump_meta"].get("model") for s in splits.values()}
        if len(models) != 1:
            raise SystemExit(f"[FATAL] {key} mixes checkpoints {models}; transferring a "
                             f"head selection between them is meaningless.")
        heads = {s.Z.shape[1] for s in splits.values()}
        if len(heads) != 1:
            raise SystemExit(f"[FATAL] {key} splits disagree on head count {heads}.")

        res, score_cache = {}, {}
        for label, src, tgt in (("gsm8k->gsm8k", "g_train", "g_test"),
                                ("math->math", "m_train", "m_test"),
                                ("gsm8k->math", "g_train", "m_test"),
                                ("math->gsm8k", "m_train", "g_test")):
            r, sc = evaluate(splits[src], splits[tgt], args.topk, args.n_seeds)
            score_cache[label] = (tgt, sc)
            tgt_split = splits[tgt]
            r["judge_p_yes"] = (float(roc_auc_score(tgt_split.y, tgt_split.p_yes))
                                if not np.isnan(tgt_split.p_yes).all() else None)
            r["n_target"] = int(len(tgt_split.y))
            r["wrong_target"] = int((tgt_split.y == 0).sum())
            res[label] = r
            print(f"  {label:14s} MLPc={r['mlp_controlled']:.4f} LRc={r['lr_controlled']:.4f} "
                  f"shallow={r['shallow']:.4f} judge={r['judge_p_yes']:.4f}", flush=True)

        for direction, transfer, indist in (("gsm8k->math", "gsm8k->math", "math->math"),
                                            ("math->gsm8k", "math->gsm8k", "gsm8k->gsm8k")):
            t, i = res[transfer], res[indist]
            res[transfer]["retained_vs_indist"] = t["mlp_controlled"] - i["mlp_controlled"]
            res[transfer]["margin_vs_judge"] = t["mlp_controlled"] - t["judge_p_yes"]
            print(f"  {direction}: MLPc {t['mlp_controlled']:.4f} vs in-dist "
                  f"{i['mlp_controlled']:.4f} ({t['retained_vs_indist']:+.4f}), "
                  f"vs judge {t['judge_p_yes']:.4f} ({t['margin_vs_judge']:+.4f})",
                  flush=True)

        # Paired intervals, one bootstrap per TARGET split so that the
        # transferred probe, its in-distribution counterpart and the judge are
        # all resampled together on identical items.
        if args.n_boot:
            for direction, transfer, indist in (
                    ("gsm8k->math", "gsm8k->math", "math->math"),
                    ("math->gsm8k", "math->gsm8k", "gsm8k->gsm8k")):
                tgt_name, sc_x = score_cache[transfer]
                _, sc_i = score_cache[indist]
                tgt_split = splits[tgt_name]
                sets = {f"transfer_{k}": v for k, v in sc_x.items()}
                sets.update({f"indist_{k}": v for k, v in sc_i.items()})
                d = paired_target_boot(tgt_split.y, sets, tgt_split.p_yes,
                                       args.n_boot, args.seed)
                block = {"n_boot": args.n_boot, "n_target": int(len(tgt_split.y)),
                         "auroc": {}, "vs_indist": {}, "vs_judge": {}}
                for k in sc_x:
                    block["auroc"][f"transfer_{k}"] = ci(d[f"transfer_{k}"])
                    block["auroc"][f"indist_{k}"] = ci(d[f"indist_{k}"])
                    block["vs_indist"][k] = ci(d[f"transfer_{k}"] - d[f"indist_{k}"])
                    block["vs_judge"][k] = ci(d[f"transfer_{k}"] - d["judge_p_yes"])
                block["auroc"]["judge_p_yes"] = ci(d["judge_p_yes"])
                res[direction]["bootstrap"] = block
                b = block["vs_indist"]["mlp_raw"]
                j = block["vs_judge"]["mlp_raw"]
                print(f"  {direction}: raw MLP vs in-dist {b['mean']:+.4f} "
                      f"[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]  vs judge "
                      f"{j['mean']:+.4f} [{j['ci_lo']:+.4f},{j['ci_hi']:+.4f}]",
                      flush=True)

        report["models"][key] = res
        print(f"  [{time.time()-t0:.0f}s]", flush=True)
        del splits
        gc.collect()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
