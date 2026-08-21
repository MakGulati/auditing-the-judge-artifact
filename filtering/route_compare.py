#!/usr/bin/env python3
"""Is the probe worth anything as a COMPUTE-ALLOCATION policy, not just a detector?

Everything else here measures how well a score ranks wrong answers. This asks the
question a practitioner actually has: given a budget, which items should get k-sample
self-consistency instead of a single greedy solve?

The policy is: score every item, route the lowest-scoring fraction `r` to a k-sample
majority vote, keep the greedy answer for the rest. Accuracy and cost are then both
functions of r, so every detector traces a curve between two fixed endpoints --
r=0 is all-greedy, r=1 is self-consistency everywhere -- and the question is which
detector gets closest to the r=1 accuracy for the least compute.

Two baselines make the curve interpretable, and without them it says nothing:

  random   route a random r fraction. This is the honest null: self-consistency helps
           on average, so ANY routing policy gains accuracy as r grows, and a curve
           that merely rises proves nothing at all.
  oracle   route the items that self-consistency actually fixes. The ceiling.

Cost is reported in solve-calls per item: (1-r)*1 + r*k. The judge pass is unchanged
by routing, so it is excluded rather than diluting the ratio.

The probe is fit on the TRAIN split of a k=1 run and applied to the k=5 test records,
which is the same train-only discipline as everywhere else in this pipeline.

  python filtering/route_compare.py --train_tag llama31_8b --k5_dir results_llama31_8b_k5_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probe_models import (fit_ensemble, load_split, prepare_head_features,
                          shallow_only_scores)

ROOT = Path(__file__).resolve().parent.parent


def gain_over_random(order, gain, fracs):
    """Accuracy gained over RANDOM routing at the same budget, in accuracy points.

    Random routing needs no simulation. Write g_i = (majority correct) - (greedy
    correct); routing a set R gives accuracy mean(greedy) + (1/n) * sum_{i in R} g_i,
    and a random R of size m has expectation mean(greedy) + r * mean(g). The two
    mean(greedy) terms cancel, so the gain is exact:

        (1/n) * sum_{i in R} g_i  -  r * mean(g)

    The permutation average in `curves["random"]` is the same quantity with Monte
    Carlo noise on top, which is tolerable on an absolute-accuracy axis and is not
    once the plot shows differences of a couple of points.
    """
    n = len(order)
    gbar = float(np.mean(gain))
    out = []
    for r in fracs:
        m = int(round(r * n))
        out.append(float(gain[order[:m]].sum() / n - r * gbar))
    return out


def gain_draws(scores, gain, fracs, n_boot=2000, seed=0):
    """Bootstrap draws of the gain for every detector, on SHARED item resamples.

    The routing order is recomputed inside each resample, so the interval covers both
    which items the score ranks lowest and whether self-consistency happens to repair
    them -- the two things that could make this run's gain a fluke.

    Every detector sees the same resampled items, which is what makes the paired
    difference between two detectors meaningful: two separately-drawn intervals can
    overlap while the paired difference is nowhere near zero, because both detectors
    move together when the resample happens to contain easy items.
    """
    rng = np.random.default_rng(seed)
    names = list(scores)
    s = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    n = len(gain)
    draws = {k: np.empty((n_boot, len(fracs))) for k in names}
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        g_i = gain[i]
        for k in names:
            draws[k][b] = gain_over_random(np.argsort(s[k][i]), g_i, fracs)
    return draws


def pct_ci(draws):
    return (np.percentile(draws, 2.5, axis=0).tolist(),
            np.percentile(draws, 97.5, axis=0).tolist())


def routed_accuracy(order, greedy_ok, major_ok, fracs):
    """Accuracy when the first `r` fraction of `order` gets the majority answer."""
    n = len(order)
    out = []
    for r in fracs:
        m = int(round(r * n))
        routed = np.zeros(n, dtype=bool)
        routed[order[:m]] = True
        ok = np.where(routed, major_ok, greedy_ok)
        out.append(float(ok.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tag", required=True, help="tag whose TRAIN split fits the probe")
    ap.add_argument("--k5_dir", required=True, help="results dir of the k>1 run to route over")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default="filtering/figures/paper/route_compare.json")
    args = ap.parse_args()

    tr_dir = ROOT / f"results_{args.train_tag}_train"
    tr = load_split(str(tr_dir / "hidden_rich.npz"),
                    str(tr_dir / args.dataset / "raw.jsonl"), args.head_dim)
    te = load_split(str(Path(args.k5_dir) / "hidden_rich.npz"),
                    str(Path(args.k5_dir) / args.dataset / "raw.jsonl"), args.head_dim)

    raw = {}
    for line in open(Path(args.k5_dir) / args.dataset / "raw.jsonl"):
        line = line.strip()
        if line:
            r = json.loads(line)
            raw[r["idx"]] = r

    greedy_ok = (te.y == 1).astype(float)
    # `majority_correct` is recorded by run_eval against the same gold answer.
    major_ok = np.array([1.0 if raw[i].get("majority_correct") else 0.0 for i in te.idx])
    k = len(raw[te.idx[0]].get("k_answers") or [])

    print(f"train {len(tr.y)} / route-over {len(te.y)} records, k={k}")
    print(f"  greedy accuracy   : {greedy_ok.mean():.4f}")
    print(f"  majority accuracy : {major_ok.mean():.4f}  "
          f"(fixes {int(((major_ok == 1) & (greedy_ok == 0)).sum())}, "
          f"breaks {int(((major_ok == 0) & (greedy_ok == 1)).sum())})")

    Ftr, Fte, _, _ = prepare_head_features(tr.Z, tr.y, te.Z, args.topk)
    scores = {}
    for m in ("lr", "mlp"):
        s, _, _ = fit_ensemble(Ftr, tr.y, Fte, tr.S, te.S, model_type=m, seed=0,
                               n_seeds=args.n_seeds)
        scores[f"{m}_controlled"] = s["controlled"]
        # The paper's primary method is the RAW probe, fixed in advance for every
        # model, dataset and budget; carrying it here keeps the routing arm on the
        # same detector as every other result rather than on the ablation.
        scores[f"{m}_raw"] = s["raw"]
    # Returns (train_scores, test_scores); only the held-out side is a routing signal.
    _, scores["shallow"] = shallow_only_scores(tr.S, tr.y, te.S, seed=0)
    if not np.isnan(te.p_yes).all():
        scores["p_yes"] = te.p_yes
    # The verbalized verdict is binary, so it cannot ORDER items within a class; ties
    # are broken at random, which is what a practitioner using it would actually get.
    rng = np.random.default_rng(0)
    scores["verdict"] = te.yes.astype(float) + rng.normal(0, 1e-6, len(te.yes))

    fracs = [round(x, 3) for x in np.linspace(0, 1, 21)]
    cost = [(1 - r) + r * k for r in fracs]
    report = {"train_tag": args.train_tag, "k5_dir": str(args.k5_dir), "k": k,
              "n": len(te.y), "fracs": fracs, "cost_per_item": cost,
              "greedy_acc": float(greedy_ok.mean()),
              "majority_acc": float(major_ok.mean()), "curves": {}}

    # Ascending score = most likely WRONG first, which is the routing order.
    gain = major_ok - greedy_ok
    report["gain_over_random"] = {}
    draws = gain_draws(scores, gain, fracs, n_boot=args.n_boot)
    for name, s in scores.items():
        order = np.argsort(np.asarray(s, dtype=float))
        report["curves"][name] = routed_accuracy(order, greedy_ok, major_ok, fracs)
        lo, hi = pct_ci(draws[name])
        report["gain_over_random"][name] = {
            "mean": gain_over_random(order, gain, fracs), "ci_lo": lo, "ci_hi": hi}
        if name != "verdict":
            report.setdefault("auc", {})[name] = float(roc_auc_score(te.y, s))

    # Probe minus judge at the same budget, paired on the resample. This is the
    # comparison the paper makes; reading it off two overlapping bands would be wrong.
    if "p_yes" in draws:
        report["gain_vs_p_yes"] = {}
        for name in draws:
            if name == "p_yes":
                continue
            lo, hi = pct_ci(draws[name] - draws["p_yes"])
            mean = [a - b for a, b in zip(report["gain_over_random"][name]["mean"],
                                          report["gain_over_random"]["p_yes"]["mean"])]
            report["gain_vs_p_yes"][name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}

    report["curves"]["random"] = list(np.mean(
        [routed_accuracy(rng.permutation(len(te.y)), greedy_ok, major_ok, fracs)
         for _ in range(200)], axis=0))
    # The same baseline without Monte Carlo noise; see gain_over_random.
    report["curves"]["random_exact"] = [float(greedy_ok.mean() + r * gain.mean())
                                        for r in fracs]
    # Oracle: route exactly the items self-consistency repairs, best first.
    o_order = np.argsort(-gain)
    report["curves"]["oracle"] = routed_accuracy(o_order, greedy_ok, major_ok, fracs)
    report["gain_over_random"]["oracle"] = {
        "mean": gain_over_random(o_order, gain, fracs), "ci_lo": None, "ci_hi": None}

    print(f"\n{'route %':>8s} {'cost/item':>9s} " +
          " ".join(f"{n:>12s}" for n in report["curves"]))
    for j, r in enumerate(fracs):
        if j % 2:
            continue
        print(f"{r*100:7.0f}% {cost[j]:9.2f} " +
              " ".join(f"{report['curves'][n][j]:12.4f}" for n in report["curves"]))

    def gain_table(title, block):
        print(f"\n{title}")
        for name, g in block.items():
            row = []
            for j, r in enumerate(fracs):
                if r not in (0.05, 0.1, 0.2, 0.5):
                    continue
                ci = ("" if g["ci_lo"] is None
                      else f" [{100*g['ci_lo'][j]:+.2f},{100*g['ci_hi'][j]:+.2f}]")
                row.append(f"{r:.0%}: {100*g['mean'][j]:+.2f}{ci}")
            print(f"  {name:16s} " + "   ".join(row))

    gain_table("gain over random routing, accuracy points (95% CI over items)",
               report["gain_over_random"])
    if "gain_vs_p_yes" in report:
        gain_table("gain over the judge's p_YES at the same budget (paired CI)",
                   report["gain_vs_p_yes"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
