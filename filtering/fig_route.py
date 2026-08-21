#!/usr/bin/env python3
"""Does the probe pick the right problems to spend extra compute on?

Plotted as the gain over RANDOM routing at the same budget, not as absolute accuracy.
Every routing curve joins the same two endpoints -- all-greedy at r=0 and
self-consistency everywhere at r=1 -- and self-consistency helps on average, so an
absolute-accuracy plot rises for any policy whatsoever, including a coin flip. The
only thing worth reading off it is the gap to random, so that gap is what the axis
shows: above zero means the score allocated compute usefully, and how far above is
how much the ranking was worth.

Random routing is computed exactly rather than simulated. With g_i = (majority
correct) - (greedy correct), routing a set R gives mean(greedy) + (1/n)*sum_R g_i and
a random R of size m has expectation mean(greedy) + r*mean(g); the difference drops
the common term. Bands are a 95% percentile interval over items, with the routing
order recomputed inside each resample.

The main figure carries the three scores a reader would actually consider deploying.
--appendix adds the LR probe, shallow text features and the oracle ceiling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"mlp_controlled": "#E69F00", "lr_controlled": "#0072B2", "shallow": "#009E73",
     "p_yes": "#5f5f5f", "verdict": "#D55E00", "oracle": "#000000"}
LABEL = {"mlp_controlled": "MLP probe", "lr_controlled": "LR probe",
         "shallow": "Shallow text feats", "p_yes": "judge $p_{\\mathrm{YES}}$",
         "verdict": "verbalized verdict", "oracle": "oracle (routes what SC fixes)"}
SHORT = {"mlp_controlled": "MLP", "p_yes": "$p_{\\mathrm{YES}}$",
         "verdict": "verdict"}
MAIN = ["mlp_controlled", "p_yes", "verdict"]
EXTRA = ["lr_controlled", "shallow", "oracle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", default="filtering/figures/paper/route_compare.json", nargs="?")
    ap.add_argument("--out", default=None)
    ap.add_argument("--appendix", action="store_true",
                    help="also draw LR, shallow features and the oracle ceiling")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    d = json.loads(Path(args.json).read_text())
    if "gain_over_random" not in d:
        raise SystemExit("[FATAL] this route_compare.json predates the gain-over-random "
                         "output; re-run filtering/route_compare.py")
    out = args.out or ("filtering/figures/paper/fig_route_appendix" if args.appendix
                       else "filtering/figures/paper/fig_route")
    k = d["k"]
    xs = [100 * r for r in d["fracs"]]
    names = MAIN + (EXTRA if args.appendix else [])

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "legend.fontsize": 6.8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.axhline(0, color="#444444", lw=1.0, ls="--", zorder=2)

    for name in names:
        g = d["gain_over_random"].get(name)
        if g is None:
            continue
        ys = [100 * v for v in g["mean"]]
        ls = ":" if name == "oracle" else "-"
        ax.plot(xs, ys, color=C[name], lw=1.6 if name == "mlp_controlled" else 1.3,
                ls=ls, label=LABEL[name],
                zorder=4 if name == "mlp_controlled" else 3)
        if g.get("ci_lo") is not None:
            ax.fill_between(xs, [100 * v for v in g["ci_lo"]],
                            [100 * v for v in g["ci_hi"]], color=C[name], alpha=0.13,
                            lw=0, zorder=1)

    # One budget read out in words, so the figure states a result rather than leaving
    # the reader to measure one off the axis. The link is a full-height guide at 10%
    # and a marker on the curve: a leader line to a text box crosses the curves it is
    # pointing at, whichever empty corner the box goes in.
    j = d["fracs"].index(0.1)
    mg = 100 * d["gain_over_random"]["mlp_controlled"]["mean"][j]
    cost = d["cost_per_item"][j]
    # Reserve a clear strip under the lowest band for the readout line, rather than
    # letting it land on top of a curve or under the axis spine.
    vals = [100 * v for n in names if n in d["gain_over_random"]
            for key in ("mean", "ci_lo", "ci_hi")
            for v in (d["gain_over_random"][n][key] or [])]
    ylo, yhi = min(vals), max(vals)
    ax.set_ylim(ylo - 0.10 * (yhi - ylo) - 0.55, yhi + 0.05 * (yhi - ylo) + 0.2)
    y_read = ylo - 0.05 * (yhi - ylo) - 0.30

    ax.axvline(10, color="#999999", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    ax.plot([10], [mg], marker="o", ms=4, color=C["mlp_controlled"],
            markeredgecolor="white", markeredgewidth=0.6, zorder=6, linestyle="none")
    # round() before adding 0.0 so a gain of -0.02 prints as +0.0, not -0.0.
    read = ",  ".join(
        f"{SHORT[n]} {round(100 * d['gain_over_random'][n]['mean'][j], 1) + 0.0:+.1f}"
        for n in MAIN)
    ax.text(12.5, y_read, f"At 10% routed ({cost:.1f} calls/item):  {read} pts",
            fontsize=6.5, color="#333333", ha="left", va="center")

    ax.set_xlabel("Problems sent to self-consistency (%)")
    ax.set_ylabel("Accuracy gain over random routing\n(percentage points)")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 10, 25, 50, 75, 100])
    sec = ax.secondary_xaxis("top", functions=(lambda r: 1 + (k - 1) * r / 100,
                                               lambda c: 100 * (c - 1) / (k - 1)))
    sec.set_xlabel(f"Solve calls per problem  (1 = greedy, {k} = self-consistency everywhere)",
                   fontsize=7.2)
    sec.set_xticks([1, 1.4, 2, 3, 4, 5])
    sec.tick_params(labelsize=7)
    ax.set_title(args.title or "Selective self-consistency: is the ranking worth its compute?")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.16 if not args.appendix else -0.22),
               ncols=3, handlelength=1.8, columnspacing=1.4)
    fig.text(0.5, -0.26 if not args.appendix else -0.34,
             "Score every problem, send the lowest-scoring fraction to a "
             f"{k}-sample majority vote, keep the greedy answer for the rest.\n"
             "Above zero, the score chose better problems than chance at the same "
             "compute. Bands are 95% CIs over items.",
             ha="center", fontsize=6.2, color="#5f5f5f")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}")
    print(f"saved {out}.png/.pdf")


if __name__ == "__main__":
    main()
