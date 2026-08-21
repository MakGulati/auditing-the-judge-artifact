#!/usr/bin/env python3
"""What changes when every instance is re-instantiated from its template?

Both panels plot a CHANGE against zero, because the absolute levels are not the
question and putting them on the axis buries the answer. Accuracy differs by twelve
points across these three models and AUC by nine; a reader comparing bar heights is
reading the models, not the perturbation. What the experiment asks is whether each
quantity MOVED, so each panel is centred on "did not move".

  A  solver accuracy, perturbed minus its matched originals, paired over the 100
     templates -- the unconfounded baseline, since the templates are not a random
     sample of the test split.
  B  error-detection AUC, perturbed minus the full GSM8K test split. The matched
     originals cannot carry an AUC: one model gets 95 of those 100 right, so its
     matched AUC rests on five wrong answers and its interval swamps every effect
     here. The price is that this delta mixes perturbation with template selection;
     the matched deltas are in the JSON.

Main figure: the controlled MLP probe and the judge's p_YES, which is the comparison
the paper makes. --appendix adds the raw MLP, the LR probe and shallow text features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TITLE = {"qwen25_7b": "Qwen2.5-7B", "llama31_8b": "Llama-3.1-8B", "gemma3": "Gemma3-12B"}
ORDER = ["qwen25_7b", "llama31_8b", "gemma3"]
MAIN = [("mlp_controlled", "MLP probe (controlled)", "#E69F00"),
        ("p_yes", "judge $p_{\\mathrm{YES}}$", "#5f5f5f")]
EXTRA = [("mlp_raw", "MLP probe (raw)", "#CC79A7"),
         ("lr_controlled", "LR probe (controlled)", "#0072B2"),
         ("shallow", "Shallow text feats", "#009E73")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?",
                    default="filtering/figures/paper/symbolic_compare.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--appendix", action="store_true",
                    help="also draw the raw MLP, LR and shallow features in panel B")
    args = ap.parse_args()
    d = json.loads(Path(args.json).read_text())
    if "delta_symbolic_minus_full" not in next(iter(d["models"].values())):
        raise SystemExit("[FATAL] this symbolic_compare.json predates the "
                         "delta-vs-full output; re-run filtering/symbolic_compare.py")
    out = args.out or ("filtering/figures/paper/fig_symbolic_appendix" if args.appendix
                       else "filtering/figures/paper/fig_symbolic")
    models = [m for m in ORDER if m in d["models"]]
    dets = MAIN + (EXTRA if args.appendix else [])

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "legend.fontsize": 6.8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.8, 2.5 + 0.25 * len(dets)),
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    y = np.arange(len(models))[::-1]  # first model at the top

    # Alternating bands, not separator lines: in the appendix panel five markers per
    # model fill the row, and without a background it is not obvious which model a
    # marker near the boundary belongs to.
    for ax in (ax0, ax1):
        for i, yi in enumerate(y):
            if i % 2 == 0:
                ax.axhspan(yi - 0.5, yi + 0.5, color="#f2f2f2", lw=0, zorder=0)

    # --- A: solver accuracy, paired over templates -----------------------------
    for yi, m in zip(y, models):
        a = d["models"][m]["accuracy"]
        c = 100 * a["paired_delta"]
        lo, hi = 100 * a["ci_lo"], 100 * a["ci_hi"]
        ax0.errorbar(c, yi, xerr=[[c - lo], [hi - c]], fmt="o", ms=4.5,
                     color="#0072B2", ecolor="#0072B2", elinewidth=1.2, capsize=2.5,
                     zorder=3)
        ax0.text(hi + 0.4, yi, f"{c:+.1f} pp".replace("-", "−"), va="center",
                 ha="left", fontsize=6.5, color="#333333")
    ax0.axvline(0, color="#444444", lw=1.0, ls="--", zorder=2)
    ax0.set_yticks(y); ax0.set_yticklabels([TITLE[m] for m in models], fontsize=7.5)
    ax0.set_ylim(-0.6, len(models) - 0.4)
    ax0.set_xlabel("Change in accuracy (percentage points)")
    ax0.set_title("A. Does perturbation change accuracy?")

    # --- B: detection AUC ------------------------------------------------------
    span = 0.30 if len(dets) <= 2 else 0.62
    step = span / max(len(dets) - 1, 1)
    for j, (key, label, colour) in enumerate(dets):
        off = span / 2 - j * step
        for yi, m in zip(y, models):
            e = d["models"][m]["delta_symbolic_minus_full"][key]
            v, lo, hi = e["delta"], e["ci_lo"], e["ci_hi"]
            if v is None:
                continue
            ax1.errorbar(v, yi + off, xerr=[[v - lo], [hi - v]], fmt="o", ms=4.2,
                         color=colour, ecolor=colour, elinewidth=1.1, capsize=2.2,
                         zorder=3, label=label if yi == y[0] else None)
    ax1.axvline(0, color="#444444", lw=1.0, ls="--", zorder=2)
    ax1.set_yticks(y); ax1.set_yticklabels([TITLE[m] for m in models], fontsize=7.5)
    ax1.set_ylim(-0.6, len(models) - 0.4)
    ax1.set_xlabel("Change in error-detection AUC")
    ax1.set_title("B. Does perturbation change error detection?")
    for yi in y[:-1]:
        ax1.axhline(yi - 0.5, color="#dddddd", lw=0.6, zorder=1)
        ax0.axhline(yi - 0.5, color="#dddddd", lw=0.6, zorder=1)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.14 if len(dets) <= 2 else -0.19),
               ncols=min(len(dets), 3), handlelength=1.4, columnspacing=1.6)
    fig.text(0.5, -0.26 if len(dets) <= 2 else -0.36,
             "Both panels are centred on no change. A: perturbed minus matched "
             "originals, paired over the 100 templates.\nB: perturbed minus the full "
             "GSM8K test split; every detector is fit on the REAL GSM8K train split "
             "only.\nIntervals are 95%, bootstrapped over TEMPLATES on the perturbed "
             "side. Every interval here covers zero: with 100 templates the design\n"
             "resolves AUC changes of roughly 0.1, so this bounds degradation rather "
             "than excluding it.",
             ha="center", fontsize=6.2, color="#5f5f5f")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}")
    print(f"saved {out}.png/.pdf")


if __name__ == "__main__":
    main()
