#!/usr/bin/env python3
"""Two-arm selective-routing figure (Fig. 3).

Reads the two routing caches under ``data/`` and plots accuracy gain over random
routing at equal solve cost, one panel per model. Nothing is transcribed: every
plotted value comes from the JSON written by ``filtering/route_compare.py``.

    python3 scripts/fig_routing.py

Writes ``figures/fig3_routing_two_arms.pdf`` (and .png at 600 dpi).

Design notes:
  * Full-width small multiples with a SHARED y-axis. The two arms have very
    different headroom (Llama's k=5 vote repairs 120 items, Gemma 3's repairs
    19), and a per-panel y-scale would hide that difference.
  * Routing fraction is the primary x-axis because it exposes the evaluated
    operating point directly; solve cost is 1 + 5m calls per problem.
  * The 10% operating point is annotated with the exact probe--judge comparison.
  * Okabe--Ito hues in fixed order, paired with distinct dashes and markers, so
    identity survives greyscale printing and CVD. Validated: worst adjacent CVD
    dE 11.0 (deutan), normal-vision dE 24.2.
  * Only the primary probe, judge, and oracle are shown. The shallow-feature
    ablation remains in the text and appendix rather than competing for space.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"

# IEEE two-column width.
WIDTH_IN, HEIGHT_IN = 7.16, 2.55

ARMS = [
    ("route_compare.json", "Llama 3.1 8B"),
    ("route_compare_gemma3.json", "Gemma 3 12B"),
]

# Okabe--Ito, fixed order; each hue carries a dash + marker as redundant encoding.
STYLE = {
    "mlp_raw":  ("#0072B2", "-",  "o", "probe"),
    "p_yes":    ("#E69F00", "--", "s", r"judge $p(\mathrm{YES})$"),
}
BANDED = ("mlp_raw", "p_yes")
ORACLE = "#4d4d4d"


def _style_rc() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.5,
        "figure.dpi": 600,
    })


def main() -> None:
    _style_rc()
    arms = [(json.loads((DATA / f).read_text()), label) for f, label in ARMS]

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, HEIGHT_IN), sharey=True)

    for ax, (d, label) in zip(axes, arms):
        routed = [100.0 * f for f in d["fracs"]]
        gor = d["gain_over_random"]

        # Oracle ceiling first, so the data lines sit on top of it.
        if "oracle" in gor:
            ax.plot(routed, [100 * v for v in gor["oracle"]["mean"]],
                    color=ORACLE, lw=1.2, ls=(0, (4, 2)), zorder=1)

        for key, (color, dash, marker, _) in STYLE.items():
            if key not in gor:
                continue
            b = gor[key]
            y = [100 * v for v in b["mean"]]
            if key in BANDED and b.get("ci_lo") is not None:
                ax.fill_between(routed,
                                [100 * v for v in b["ci_lo"]],
                                [100 * v for v in b["ci_hi"]],
                                color=color, alpha=0.16, lw=0, zorder=2)
            ax.plot(routed, y, color=color, ls=dash, marker=marker,
                    markersize=3.6, markevery=2, markeredgewidth=0, zorder=3)

        ax.axhline(0, color="black", lw=0.8, alpha=0.65, zorder=1)
        ax.axvline(10, color="#777777", lw=0.8, ls=":", zorder=1)
        ax.set_xlim(0.0, 100.0)
        ax.set_xticks([0, 10, 20, 40, 60, 80, 100])
        ax.set_xlabel("problems routed to five-sample vote (%)")
        judge_auc = d["auc"]["p_yes"]
        strength = "weaker judge" if judge_auc < 0.75 else "stronger judge"
        ax.set_title(f"{label} — {strength} (judge AUROC {judge_auc:.3f})", pad=5)
        ax.grid(axis="y", lw=0.45, alpha=0.25)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        # Direct labels explain the baselines without requiring caption lookup.
        ax.text(72, 0.14, "equal-cost random baseline", color="black",
                fontsize=7.5, ha="center", va="bottom")
        j = 7
        ax.annotate("ex-post oracle ceiling",
                    xy=(routed[j], 100 * gor["oracle"]["mean"][j]),
                    xytext=(4, 4), textcoords="offset points",
                    color=ORACLE, fontsize=7.5, va="bottom")

        # Exact values at the operating point discussed in the paper.
        op = 2  # m = 0.10
        probe = 100 * gor["mlp_raw"]["mean"][op]
        judge = 100 * gor["p_yes"]["mean"][op]
        comp = d["gain_vs_p_yes"]["mlp_raw"]
        delta = 100 * comp["mean"][op]
        lo = 100 * comp["ci_lo"][op]
        hi = 100 * comp["ci_hi"][op]
        text = (f"at 10%: probe {probe:+.2f}, judge {judge:+.2f} pts\n"
                f"probe − judge {delta:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        ax.annotate(text, xy=(10, probe), xytext=(16, 5.25 if label.startswith("Llama") else 2.55),
                    textcoords="data", fontsize=7.7, ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999",
                              lw=0.6, alpha=0.94),
                    arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7))

    axes[0].set_ylabel("accuracy gain over equal-cost\nrandom routing (percentage points)")

    handles = [Line2D([], [], color=c, ls=ls, marker=m, markersize=2.6,
                      markeredgewidth=0, label=lab)
               for c, ls, m, lab in STYLE.values()]
    handles.append(Line2D([], [], color=ORACLE, ls=(0, (4, 2)), lw=1.2,
                          label="ex-post oracle"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.01), handlelength=2.5, columnspacing=1.8)

    fig.tight_layout(rect=(0, 0, 1, 0.91), pad=0.4, w_pad=1.1)
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "fig3_routing_two_arms.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig3_routing_two_arms.png", dpi=600, bbox_inches="tight")
    print("wrote figures/fig3_routing_two_arms.{pdf,png}")


if __name__ == "__main__":
    main()
