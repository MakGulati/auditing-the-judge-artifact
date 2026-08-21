#!/usr/bin/env python3
"""Check the manuscript's propositions against the experimental artifacts.

Every identity asserted in Appendix A is re-derived here from the checked-in
caches and the aligned held-out predictions, so the appendix tables are generated
evidence rather than transcribed claims.

Usage (from the repository root):

    python3 scripts/verify_identities.py [--predictions DIR]

``--predictions`` points at the ``aligned_predictions`` directory of the analysis
pipeline. Proposition 2's numerical check needs per-item scores and is skipped
when that directory is absent; Propositions 1 and the routing comparators are
checked from ``data/`` alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

RUN_LABEL = {
    "gemma3": "Gemma 3/GSM8K",
    "qwen25_7b": "Qwen2.5/GSM8K",
    "llama31_8b": "Llama 3.1/GSM8K",
    "gemma3_math": "Gemma 3/MATH",
    "qwen25_7b_math": "Qwen2.5/MATH",
    "llama31_8b_math": "Llama 3.1/MATH",
}


def check_prop1() -> float:
    """rho_m = pi0 (1 - R_m) / (1 - M_m/N); return max discrepancy."""
    with (DATA / "main_metrics.csv").open() as fh:
        stats = {
            r["tag"]: (float(r["wrong_prevalence"]), int(r["n"]))
            for r in csv.DictReader(fh)
        }
    with (DATA / "fixed_budget_metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))

    print("== Proposition 1: residual-risk identity ==")
    print(f"{'run':16s} {'m':>6s} {'pi0':>7s} {'dRecall':>9s} {'predicted':>10s} "
          f"{'recorded':>10s} {'diff':>10s}")
    worst = 0.0
    for r in rows:
        m = float(r["budget"])
        pi0, n = stats[r["tag"]]
        b = int(r["n_reviewed"]) / n
        predicted = pi0 * float(r["recall_gain"]) / (1.0 - b)
        recorded = float(r["residual_risk_improvement"])
        diff = abs(predicted - recorded)
        worst = max(worst, diff)
        if m == 0.05:
            print(f"{RUN_LABEL[r['tag']]:16s} {m:6.2f} {pi0:7.4f} "
                  f"{float(r['recall_gain']):+9.4f} {predicted:10.5f} "
                  f"{recorded:10.5f} {diff:10.2e}")
    print(f"  max |predicted - recorded| over all {len(rows)} cell x budget rows: {worst:.3e}\n")
    return worst


def check_prop2(pred_dir: pathlib.Path) -> float | None:
    """int_0^1 R_m dm == pi0/2 + (1-pi0) AUROC; returns the max abs discrepancy."""
    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("== Proposition 2: skipped (numpy/scikit-learn unavailable) ==\n")
        return None
    if not pred_dir.is_dir():
        print(f"== Proposition 2: skipped (no predictions at {pred_dir}) ==\n")
        return None

    print("== Proposition 2: AUROC as a uniform prior over budgets ==")
    print(f"{'run':16s} {'score':6s} {'AUROC':>9s} {'LHS':>9s} {'RHS':>9s} {'diff':>10s}")
    worst = 0.0
    worst_adjusted = 0.0
    for tag, label in RUN_LABEL.items():
        path = pred_dir / f"{tag}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        y = d["y"].astype(int)
        for name, disp in (("mlp_raw", "probe"), ("p_yes", "judge")):
            s = d[name].astype(float)
            n, w = len(y), int((y == 0).sum())
            pi0 = w / n
            auroc = roc_auc_score(y, s)
            order = np.argsort(s, kind="mergesort")          # worst-scoring first
            s_sorted = s[order]
            err_sorted = (y[order] == 0)
            # Exact expected left Riemann sum under uniform random ordering
            # within each group of exactly tied scores (Appendix B-A).
            cumulative_sum = 0.0
            errors_before = 0
            lo = 0
            while lo < n:
                hi = lo + 1
                while hi < n and s_sorted[hi] == s_sorted[lo]:
                    hi += 1
                group_size = hi - lo
                group_errors = int(err_sorted[lo:hi].sum())
                cumulative_sum += (
                    group_size * errors_before
                    + group_errors * (group_size + 1) / 2
                )
                errors_before += group_errors
                lo = hi
            lhs = cumulative_sum / (n * w)
            rhs = pi0 / 2 + (1 - pi0) * auroc
            raw_diff = abs(lhs - rhs)
            adjusted_diff = abs((lhs - rhs) - 1 / (2 * n))
            worst = max(worst, raw_diff)
            worst_adjusted = max(worst_adjusted, adjusted_diff)
            print(f"{label:16s} {disp:6s} {auroc:9.5f} {lhs:9.5f} {rhs:9.5f} "
                  f"{raw_diff:10.2e}")
    print(f"  max |LHS - RHS|: {worst:.3e}  (finite-sample offset 1/(2N))")
    print(f"  max |LHS - RHS - 1/(2N)|: {worst_adjusted:.3e}\n")
    return worst_adjusted


def check_routing() -> float:
    worst = 0.0
    for filename, label in (
        ("route_compare.json", "Llama 3.1 8B / GSM8K"),
        ("route_compare_gemma3.json", "Gemma 3 12B / GSM8K"),
    ):
        d = json.loads((DATA / filename).read_text())
        fr, curves = d["fracs"], d["curves"]
        greedy, majority, n, k = d["greedy_acc"], d["majority_acc"], d["n"], d["k"]
        peak = max(curves["oracle"])
        repaired = round((peak - greedy) * n)
        broken = round((peak - majority) * n)
        pi_plus, pi_minus = repaired / n, broken / n

        errs = []
        for m, recorded in zip(fr, curves["oracle"]):
            b = int(m * n) / n
            predicted = greedy + min(b, pi_plus) - max(0.0, b - (1.0 - pi_minus))
            errs.append(abs(predicted - recorded))
        worst = max(worst, max(errs))

        i10 = fr.index(0.10)
        b10 = int(0.10 * n) / n
        rand_pred = greedy + b10 * (majority - greedy)
        rand_rec = curves["random"][i10]
        print(f"== Routing comparators ({label}, k={k}) ==")
        print(f"  random @ nominal m=0.10 predicted {rand_pred:.4f}  cached {rand_rec:.4f}")
        print(f"  implemented cost  1 + k(M/N) = {1 + k * b10:.4f} calls/item")
        print(f"  repairs {repaired}  breaks {broken}  pi_+={pi_plus:.4f}  pi_-={pi_minus:.4f}")
        print(f"  max oracle formula discrepancy across all budgets: {max(errs):.3e}")
        print(f"  route-all {majority:.4f}; peak {peak:.4f}; excess {peak - majority:.4f}\n")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions",
        type=pathlib.Path,
        default=pathlib.Path(
            "filtering/"
            "figures/ieee_tps_2026/aligned_predictions"
        ),
    )
    args = ap.parse_args()

    w1 = check_prop1()
    w2 = check_prop2(args.predictions)
    wr = check_routing()

    ok = w1 < 1e-12 and (w2 is None or w2 < 1e-3) and wr < 1e-12
    print("VERIFIED" if ok else "DISCREPANCY -- inspect above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
