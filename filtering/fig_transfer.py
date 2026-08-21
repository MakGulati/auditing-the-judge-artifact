#!/usr/bin/env python3
"""The transfer figure: what survives a dataset shift, and what the shift breaks.

One panel per direction. Each detector is drawn twice -- fit in-distribution (the
ceiling, hollow) and fit on the other dataset (the transfer, filled) -- against the
judge baseline on that same target split, which is the line the probe has to clear to
be worth its supervision.

Drawing raw and controlled side by side is the point rather than a detail: the
residualiser is fit on the SOURCE dataset, so under shift it subtracts a
miscalibrated prediction and does active damage. Showing only `controlled`, the
variant the rest of the paper reports, would attribute that damage to the probe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TITLE = {"gemma3": "Gemma3-12B", "qwen25_7b": "Qwen2.5-7B", "llama31_8b": "Llama-3.1-8B"}
SERIES = [("mlp_raw", "MLP raw", "#E69F00"),
          ("mlp_controlled", "MLP controlled", "#B37700"),
          ("shallow", "Shallow text feats", "#009E73")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?", default="filtering/figures/paper/transfer_compare.json")
    ap.add_argument("--out", default="filtering/figures/paper/fig_transfer")
    args = ap.parse_args()
    d = json.loads(Path(args.json).read_text())
    models = list(d["models"])

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "legend.fontsize": 6.8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    dirs = [("gsm8k->math", "math->math", "Fit GSM8K $\\rightarrow$ score MATH"),
            ("math->gsm8k", "gsm8k->gsm8k", "Fit MATH $\\rightarrow$ score GSM8K")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)

    width = 0.24
    for ax, (transfer, indist, title) in zip(axes, dirs):
        x = np.arange(len(models))
        for j, (key, label, colour) in enumerate(SERIES):
            off = (j - 1) * width
            tv = [d["models"][m][transfer][key] for m in models]
            iv = [d["models"][m][indist][key] for m in models]
            ax.bar(x + off, tv, width * 0.92, color=colour,
                   label=label if ax is axes[0] else None, zorder=3)
            # In-distribution ceiling for the same detector, as a cap the bar falls short of.
            ax.plot(np.repeat(x + off, 3).reshape(-1, 3).T[0:1].ravel()[:0], [])  # no-op
            for xi, v in zip(x + off, iv):
                ax.hlines(v, xi - width * 0.46, xi + width * 0.46, color=colour,
                          lw=1.1, linestyle=(0, (2, 1.5)), zorder=4)
        jv = [d["models"][m][transfer]["judge_p_yes"] for m in models]
        for xi, v in zip(x, jv):
            ax.hlines(v, xi - 0.42, xi + 0.42, color="#D55E00", lw=1.6, zorder=5)
        ax.axhline(0.5, color="#999999", lw=0.8, ls=":", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([TITLE[m] for m in models], fontsize=7)
        ax.set_title(title)
        ax.set_ylim(0.35, 1.0)
    axes[0].set_ylabel("AUC on the target test split")
    axes[0].text(0.02, 0.52, "chance", fontsize=6.2, color="#777777",
                 transform=axes[0].get_yaxis_transform())

    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([0], [0], color="#555555", lw=1.1, linestyle=(0, (2, 1.5))),
                Line2D([0], [0], color="#D55E00", lw=1.6)]
    labels += ["same detector, fit in-distribution", "judge $p_{\\mathrm{YES}}$ (target split)"]
    fig.legend(handles, labels, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.17), ncols=3, columnspacing=1.4, handlelength=1.8)
    fig.text(0.5, -0.30,
             "Bars: detector fit on the OTHER dataset. Dashes: the same detector fit on the "
             "target's own train split.\nThe residualiser is fit on the source, so under "
             "shift `controlled` subtracts a miscalibrated prediction and loses more than "
             "`raw`.",
             ha="center", fontsize=6.2, color="#5f5f5f")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"saved {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
