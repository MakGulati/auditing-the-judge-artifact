#!/usr/bin/env python3
"""Recall at a matched review budget: who catches more errors for the same effort.

Every rule flags the same m lowest-scoring items, so precision and recall are both
monotone in true positives and there is a single quantity to compare -- unlike the
train-calibrated operating point, where each rule lands at a different realised
precision and the recalls cannot be read against each other.

Plotted as probe MINUS judge, because the absolute recall at a given budget is
dominated by the run's prevalence (Llama's MATH split is 49% wrong, Gemma3's GSM8K
6%) and putting six such curves on one axis compares datasets rather than detectors.
The difference at equal budget is the quantity the paper claims.

The x axis is log-scaled: the interesting region is 2-10%, which is what a review
budget actually looks like, and a linear axis spends most of its width on budgets
nobody has.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "gemma3": ("Gemma3-12B", "#0072B2"),
    "qwen25_7b": ("Qwen2.5-7B", "#E69F00"),
    "llama31_8b": ("Llama-3.1-8B", "#009E73"),
}
GSM = ["gemma3", "qwen25_7b", "llama31_8b"]


PRIMARY = "mlp_raw"


def probe_recall(det):
    """The probe curve, fixed to ONE variant for every run and every budget.

    Taking max(raw, controlled) per budget -- which this figure used to do -- selects
    a variant on the test set at every point on the x axis, so the resulting curve is
    not a method anyone could deploy. It barely moves the numbers here (one third
    decimal in one run), which is exactly why it is worth removing: the curve looks
    the same either way, and only one of the two is a claim.
    """
    return det[PRIMARY]["recall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?",
                    default="filtering/figures/paper/matched_budget.json")
    ap.add_argument("--out", default="filtering/figures/paper/fig_budget")
    args = ap.parse_args()
    d = json.loads(Path(args.json).read_text())

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "legend.fontsize": 6.8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)

    for ax, suffix, title in ((axes[0], "", "GSM8K"), (axes[1], "_math", "MATH")):
        for base in GSM:
            tag = base + suffix
            if tag not in d["runs"]:
                continue
            run = d["runs"][tag]
            pts = sorted((c["rate"], probe_recall(c["detectors"]) - c["detectors"]["p_yes"]["recall"])
                         for k, c in run["budgets"].items() if k != "judge_rate")
            xs = [100 * p[0] for p in pts]
            ys = [p[1] for p in pts]
            label, colour = STYLE[base]
            ax.plot(xs, ys, color=colour, lw=1.5, marker="o", ms=2.6, label=label,
                    zorder=3)
            # The judge's own flag rate: the only budget where the binary verdict
            # sits exactly on its own operating point rather than interpolated.
            jr = 100 * run["judge_flag_rate"]
            jc = run["budgets"]["judge_rate"]["detectors"]
            ax.plot([jr], [probe_recall(jc) - jc["p_yes"]["recall"]], marker="*",
                    ms=8, color=colour, zorder=5, linestyle="none")
        ax.axhline(0, color="#444444", lw=0.9, ls="--", zorder=2)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50])
        ax.set_xticklabels(["2%", "5%", "10%", "20%", "50%"])
        ax.set_xlabel("Review budget (% of outputs flagged)")
        ax.set_title(title)
    axes[0].set_ylabel("Probe $-$ judge recall\n(same budget)")

    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], marker="*", ms=8, color="#444444", linestyle="none"))
    labels.append("judge's own flag rate")
    fig.legend(handles, labels, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.19), ncols=4, columnspacing=1.4,
               handlelength=1.8)
    fig.text(0.5, -0.33,
             "Above zero: the probe catches more errors than the judge's "
             "$p_{\\mathrm{YES}}$ for the same review effort.\nEvery rule flags the "
             "same number of items, so this needs no test labels to set a threshold "
             "and no precision/recall trade-off.",
             ha="center", fontsize=6.2, color="#5f5f5f")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"saved {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
