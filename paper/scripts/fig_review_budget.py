#!/usr/bin/env python3
"""Plot matched-budget residual-risk effects from the checked-in metrics."""

from __future__ import annotations

import csv
import pathlib

import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "fixed_budget_metrics.csv"
OUT = ROOT / "figures" / "fig1_selective_risk"

BUDGETS = (0.05, 0.10, 0.15, 0.20)
MODELS = (
    ("gemma3", "gemma3_math", "Gemma 3 12B"),
    ("qwen25_7b", "qwen25_7b_math", "Qwen2.5 7B"),
    ("llama31_8b", "llama31_8b_math", "Llama 3.1 8B"),
)
COLORS = ("#0072B2", "#E69F00", "#009E73", "#7A5195")
MARKERS = ("o", "s", "^", "D")
OFFSETS = (0.24, 0.08, -0.08, -0.24)


def load_rows() -> dict[tuple[str, float], dict[str, float]]:
    rows: dict[tuple[str, float], dict[str, float]] = {}
    with DATA.open(newline="") as handle:
        for row in csv.DictReader(handle):
            budget = float(row["budget"])
            if budget not in BUDGETS:
                continue
            rows[(row["tag"], budget)] = {
                "effect": 100.0 * float(row["residual_risk_improvement"]),
                "lo": 100.0 * float(row["risk_improvement_ci_lo"]),
                "hi": 100.0 * float(row["risk_improvement_ci_hi"]),
            }
    return rows


def signed(value: float) -> str:
    return f"{value:+.2f}".replace("-", "\N{MINUS SIGN}")


def main() -> None:
    rows = load_rows()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15), sharex=True, sharey=True)
    datasets = (("GSM8K", 0), ("MATH", 1))

    for ax, (dataset, tag_index) in zip(axes, datasets):
        for model_index, tags in enumerate(MODELS):
            base_y = 2 - model_index
            tag = tags[tag_index]
            for budget_index, budget in enumerate(BUDGETS):
                row = rows[(tag, budget)]
                y = base_y + OFFSETS[budget_index]
                effect, lo, hi = row["effect"], row["lo"], row["hi"]
                ax.errorbar(
                    effect,
                    y,
                    xerr=[[effect - lo], [hi - effect]],
                    fmt=MARKERS[budget_index],
                    color=COLORS[budget_index],
                    ecolor=COLORS[budget_index],
                    markersize=4.2,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    elinewidth=1.15,
                    capsize=2.0,
                    zorder=3,
                )
                ax.text(
                    hi + 0.10,
                    y,
                    signed(effect),
                    va="center",
                    ha="left",
                    fontsize=6.5,
                    color=COLORS[budget_index],
                )

        ax.axvline(0, color="#555555", linestyle="--", linewidth=0.9, zorder=1)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, zorder=0)
        ax.set_title(dataset, fontweight="bold", pad=5)
        ax.set_xlim(-2.0, 7.0)
        ax.set_ylim(-0.55, 2.55)
        ax.set_yticks((2, 1, 0), [model[2] for model in MODELS])
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle="none",
            markersize=4.5,
            label=f"{int(100 * budget)}% review",
        )
        for budget, color, marker in zip(BUDGETS, COLORS, MARKERS)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        handletextpad=0.35,
        columnspacing=1.4,
    )
    fig.supxlabel(
        "Reduction in residual wrong-answer rate vs. judge (percentage points; positive favours probe)",
        y=0.015,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.93), w_pad=1.2)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
