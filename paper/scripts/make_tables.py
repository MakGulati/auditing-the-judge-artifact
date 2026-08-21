#!/usr/bin/env python3
"""Generate the manuscript's LaTeX tables from the checked-in metric caches.

Reads only files under ``data/`` -- no numbers are transcribed by hand. Run from
the repository root:

    python3 scripts/make_tables.py

Writes ``tables/table1_main.tex`` and ``tables/table2_budget.tex``.
"""

from __future__ import annotations

import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TABLES = ROOT / "tables"

# Display order: GSM8K block then MATH block, models in a fixed order within each.
ORDER = [
    "gemma3",
    "qwen25_7b",
    "llama31_8b",
    "gemma3_math",
    "qwen25_7b_math",
    "llama31_8b_math",
]
MODEL_TEX = {
    "Gemma 3 12B": "Gemma~3 12B",
    "Qwen2.5 7B": "Qwen2.5 7B",
    "Llama 3.1 8B": "Llama~3.1 8B",
}


def _rows() -> dict[str, dict[str, str]]:
    with (DATA / "main_metrics.csv").open() as fh:
        return {r["tag"]: r for r in csv.DictReader(fh)}


def _budget_rows() -> list[dict[str, str]]:
    with (DATA / "fixed_budget_metrics.csv").open() as fh:
        return list(csv.DictReader(fh))


def _ci(lo: str, hi: str) -> str:
    return f"[{float(lo):+.3f}, {float(hi):+.3f}]"


def _higher_pair(probe: str, judge: str) -> tuple[str, str]:
    """Format a probe--judge pair, bolding the higher point estimate."""
    probe_cell = f"{float(probe):.3f}"
    judge_cell = f"{float(judge):.3f}"
    if float(probe) > float(judge):
        probe_cell = rf"\textbf{{{probe_cell}}}"
    elif float(judge) > float(probe):
        judge_cell = rf"\textbf{{{judge_cell}}}"
    return probe_cell, judge_cell


def table_main() -> str:
    rows = _rows()
    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Held-out discrimination, six model--dataset cells. The probe is the",
        r"primary method fixed in advance (raw top-32-head five-seed MLP ensemble); the",
        r"baseline is the judge's continuous $\pyes$. $\prev$ is the observed error",
        r"prevalence in the analysed population. Deltas are probe minus judge with paired",
        r"class-stratified 95\% bootstrap intervals over test items (2000 resamples);",
        r"the six cells are not family-wise adjusted. AP treats errors as the positive",
        r"class and is therefore sensitive to $\prev$. Intervals are conditional on the",
        r"generated answer population and fitted ensemble (\cref{sec:limitations}).",
        r"Within each probe--judge pair, bold marks the higher point estimate; the",
        r"paired interval indicates whether the difference is statistically resolved.}",
        r"\label{tab:main}",
        r"\small",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"& & & & \multicolumn{3}{c}{AUROC (correctness)} & \multicolumn{3}{c}{AP (error class)} \\",
        r"\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Model & Data & $N$ & $\prev$ & probe & judge & $\Delta$ [95\% CI] & probe & judge & $\Delta$ [95\% CI] \\",
        r"\midrule",
    ]
    for i, tag in enumerate(ORDER):
        r = rows[tag]
        if i == 3:
            out.append(r"\midrule")
        auroc_probe, auroc_judge = _higher_pair(
            r["raw_auroc"], r["judge_auroc"]
        )
        ap_probe, ap_judge = _higher_pair(
            r["raw_ap_error"], r["judge_ap_error"]
        )
        out.append(
            "{model} & {data} & {n} & {prev:.3f} & "
            "{ra} & {ja} & {da:+.3f}~{dci} & "
            "{rp} & {jp} & {dp:+.3f}~{pci} \\\\".format(
                model=MODEL_TEX[r["model"]],
                data="GSM8K" if r["dataset"] == "gsm8k" else "MATH",
                n=r["n"],
                prev=float(r["wrong_prevalence"]),
                ra=auroc_probe,
                ja=auroc_judge,
                da=float(r["delta_auroc"]),
                dci=_ci(r["delta_auroc_ci_lo"], r["delta_auroc_ci_hi"]),
                rp=ap_probe,
                jp=ap_judge,
                dp=float(r["delta_ap_error"]),
                pci=_ci(r["delta_ap_ci_lo"], r["delta_ap_ci_hi"]),
            )
        )
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(out)


def table_budget() -> str:
    rows = _budget_rows()
    by = {(r["tag"], float(r["budget"])): r for r in rows}
    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Primary matched 5\% review operating point. Both methods flag the",
        r"same number of answers. Error recall is the fraction of wrong answers",
        r"flagged. $\Delta$ recall is probe minus judge; unreviewed error-rate",
        r"reduction is the corresponding decrease among answers not flagged. The",
        r"final-column brackets are paired class-stratified 95\% bootstrap intervals",
        r"(2000 resamples). Positive values favour the probe. Bold marks the higher",
        r"recall point estimate; the paired interval quantifies its uncertainty.}",
        r"\label{tab:budget}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Error recall at 5\% review} &",
        r"\multicolumn{2}{c}{Probe advantage over judge} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Model & Data & probe & judge & $\Delta$ recall &",
        r"unreviewed error-rate reduction (pp) [95\% CI] \\",
        r"\midrule",
    ]
    names = {
        "gemma3": ("Gemma~3 12B", "GSM8K"),
        "qwen25_7b": ("Qwen2.5 7B", "GSM8K"),
        "llama31_8b": ("Llama~3.1 8B", "GSM8K"),
        "gemma3_math": ("Gemma~3 12B", "MATH"),
        "qwen25_7b_math": ("Qwen2.5 7B", "MATH"),
        "llama31_8b_math": ("Llama~3.1 8B", "MATH"),
    }
    for i, tag in enumerate(ORDER):
        if i == 3:
            out.append(r"\midrule")
        model, data = names[tag]
        r = by[(tag, 0.05)]
        risk = 100.0 * float(r["residual_risk_improvement"])
        risk_lo = 100.0 * float(r["risk_improvement_ci_lo"])
        risk_hi = 100.0 * float(r["risk_improvement_ci_hi"])
        d = float(r["recall_gain"])
        recall_probe, recall_judge = _higher_pair(
            r["raw_recall"], r["judge_recall"]
        )
        out.append(
            "{model} & {data} & {rr} & {jr} & {d:+.3f} & "
            "{risk:+.2f}~[{risk_lo:+.2f}, {risk_hi:+.2f}] \\\\".format(
                model=model,
                data=data,
                rr=recall_probe,
                jr=recall_judge,
                d=d,
                risk=risk,
                risk_lo=risk_lo,
                risk_hi=risk_hi,
            )
        )
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(out)


def table_transfer() -> str:
    """Compact in-domain and cross-dataset AUROC comparison."""
    d = json.loads((DATA / "transfer_compare.json").read_text())
    names = {
        "gemma3": "Gemma~3 12B",
        "qwen25_7b": "Qwen2.5 7B",
        "llama31_8b": "Llama~3.1 8B",
    }
    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Held-out target-split AUROC under the joint dataset and inference-protocol",
        r"change. The transferred probe is fitted on the other dataset; the in-domain probe",
        r"is fitted on the target's training split. The judge is the target-domain $\pyes$",
        r"and requires no probe fitting. Bold marks the higher value in the operational",
        r"comparison between the transferred primary probe and the judge. Deltas are paired",
        r"differences on the same target items. All twelve primary-probe 95\% bootstrap",
        r"intervals exclude zero; \cref{tab:transfer-full} reports the intervals and the full",
        r"ablation matrix. Dataset, precision, and reply budget change together, so their",
        r"effects cannot be separated.}",
        r"\label{tab:transfer}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{deployment comparison} & \multicolumn{2}{c}{primary-probe transfer} & \multicolumn{2}{c}{transferred ablations} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Model & Target & judge & probe & $\Delta$ vs.\ judge & in-domain & "
        r"$\Delta$ vs.\ in-dom. & controlled & shallow \\",
        r"\midrule",
    ]

    for tag, label in names.items():
        m = d["models"][tag]
        for tgt, indom, cross in (
            ("GSM8K", "gsm8k->gsm8k", "math->gsm8k"),
            ("MATH", "math->math", "gsm8k->math"),
        ):
            a, b = m[indom], m[cross]
            boot = b["bootstrap"]
            judge, probe = _higher_pair(
                str(b["judge_p_yes"]), str(b["mlp_raw"])
            )
            out.append(
                "{lab} & {tgt} & {jd} & {x} & {d2:+.3f} & {i:.3f} & {d1:+.3f} & "
                "{c:.3f} & {s:.3f} \\\\".format(
                    lab=label if tgt == "GSM8K" else "",
                    tgt=tgt,
                    jd=judge,
                    x=probe,
                    d2=boot["vs_judge"]["mlp_raw"]["mean"],
                    i=a["mlp_raw"],
                    d1=boot["vs_indist"]["mlp_raw"]["mean"],
                    c=b["mlp_controlled"],
                    s=b["shallow"],
                )
            )
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(out)


def table_transfer_full() -> str:
    """Full transfer matrix with intervals for all three detectors (appendix)."""
    d = json.loads((DATA / "transfer_compare.json").read_text())
    names = {
        "gemma3": "Gemma~3 12B",
        "qwen25_7b": "Qwen2.5 7B",
        "llama31_8b": "Llama~3.1 8B",
    }
    det = [("mlp_raw", "raw MLP"), ("mlp_controlled", "controlled"),
           ("shallow", "shallow")]
    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full transfer matrix under the joint protocol change. Columns as in \cref{tab:transfer};",
        r"$\dagger$ marks a paired 95\% interval excluding zero. The controlled probe loses",
        r"more than the raw probe on the same items in every GSM8K$\to$MATH direction,",
        r"a pattern consistent with \cref{prop:residual-shift} but not a causal test; the shallow detector loses",
        r"least of the three everywhere.}",
        r"\label{tab:transfer-full}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrlrrrr}",
        r"\toprule",
        r"Model & Target & judge & detector & in & $\to$ & "
        r"$\Delta$ vs.\ in-dom.\ [95\% CI] & $\Delta$ vs.\ judge [95\% CI] \\",
        r"\midrule",
    ]

    def cell(b):
        star = r"$^{\dagger}$" if b["excludes_zero"] else r"\phantom{$^{\dagger}$}"
        return f"{b['mean']:+.3f}~[{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}]{star}"

    for k, (tag, label) in enumerate(names.items()):
        if k:
            out.append(r"\midrule")
        m = d["models"][tag]
        for tgt, indom, cross in (
            ("GSM8K", "gsm8k->gsm8k", "math->gsm8k"),
            ("MATH", "math->math", "gsm8k->math"),
        ):
            a, b = m[indom], m[cross]
            boot = b["bootstrap"]
            for j, (key, dname) in enumerate(det):
                first = j == 0
                out.append(
                    "{lab} & {tgt} & {jd} & {det} & {i:.3f} & {x:.3f} & {d1} & {d2} \\\\".format(
                        lab=label if (first and tgt == "GSM8K") else "",
                        tgt=tgt if first else "",
                        jd=f"{b['judge_p_yes']:.3f}" if first else "",
                        det=dname,
                        i=a[key],
                        x=b[key],
                        d1=cell(boot["vs_indist"][key]),
                        d2=cell(boot["vs_judge"][key]),
                    )
                )
            out.append(r"\addlinespace[1.5pt]")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(out)


def table_populations() -> str:
    """Per-run analysed population and exclusion accounting (Appendix D)."""
    names = {
        "gemma3": ("Gemma~3 12B", "GSM8K"),
        "qwen25_7b": ("Qwen2.5 7B", "GSM8K"),
        "llama31_8b": ("Llama~3.1 8B", "GSM8K"),
        "gemma3_math": ("Gemma~3 12B", "MATH"),
        "qwen25_7b_math": ("Qwen2.5 7B", "MATH"),
        "llama31_8b_math": ("Llama~3.1 8B", "MATH"),
    }
    out = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Analysed population and exclusions, per model--dataset cell and split. ``trunc'' is",
        r"generations cut off by the token budget before writing a final answer;",
        r"``no gold'' is reference answers that did not parse. $\prev$ is the error",
        r"prevalence of the analysed population and $\prev^{+}$ what it would be if",
        r"truncations were retained and scored wrong. Excluding them lowers observed",
        r"prevalence in every affected cell.}",
        r"\label{tab:populations}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Run & split & gen. & trunc & no gold & analysed & $\prev$ & $\prev^{+}$ \\",
        r"\midrule",
    ]
    tot = {"gen": 0, "tr": 0, "ng": 0, "an": 0, "trw": 0}
    for i, tag in enumerate(ORDER):
        if i == 3:
            out.append(r"\midrule")
        model, data = names[tag]
        d = json.loads((DATA / f"metrics_{tag}.json").read_text())
        for j, split in enumerate(("train", "test")):
            e = d[f"{split}_label_info"]["exclusions"]
            tot["gen"] += e["n_generated"]
            tot["tr"] += e["n_excluded_truncated"]
            tot["ng"] += e["n_excluded_no_gold"]
            tot["an"] += e["n_analysed"]
            tot["trw"] += e["truncated_marked_wrong"]
            lab = f"{model} / {data}" if j == 0 else ""
            out.append(
                "{lab} & {sp} & {g} & {t} & {ng} & {a} & {p:.3f} & {pp:.3f} \\\\".format(
                    lab=lab,
                    sp=split,
                    g=e["n_generated"],
                    t=e["n_excluded_truncated"],
                    ng=e["n_excluded_no_gold"],
                    a=e["n_analysed"],
                    p=e["prevalence_analysed"],
                    pp=e["prevalence_if_truncations_included"],
                )
            )
    out += [
        r"\midrule",
        "\\textbf{{total}} & & {g} & {t} & {ng} & {a} & & \\\\".format(
            g=tot["gen"], t=tot["tr"], ng=tot["ng"], a=tot["an"]
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        "% truncated_marked_wrong across all runs and splits: "
        f"{tot['trw']} of {tot['tr']}",
    ]
    return "\n".join(out)


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    (TABLES / "table1_main.tex").write_text(table_main())
    (TABLES / "table2_budget.tex").write_text(table_budget())
    (TABLES / "table3_transfer.tex").write_text(table_transfer())
    (TABLES / "table5_transfer_full.tex").write_text(table_transfer_full())
    (TABLES / "table4_populations.tex").write_text(table_populations())
    print("wrote table1_main, table2_budget, table3_transfer, "
          "table4_populations, table5_transfer_full")


if __name__ == "__main__":
    main()
