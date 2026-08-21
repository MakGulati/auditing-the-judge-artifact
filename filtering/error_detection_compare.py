#!/usr/bin/env python3
"""Error-detection comparison at a TRAIN-CALIBRATED operating point.

Each detector's threshold is chosen on TRAIN at the judge's train precision and
applied unchanged to TEST. That calibrates on train; it does NOT equalise
precision on test. Every rule's realised TEST precision is printed beside its
recall precisely because those values differ, and recalls measured at different
realised precisions are not directly comparable -- AUC and risk-coverage are the
primary comparisons. The best-threshold-on-test number is descriptive only and is
labelled a test-oracle upper bound wherever it appears.

"Error detection" = flagging WRONG solutions (label y==0). Detectors compared on the
held-out TEST split:

  * controlled head-probe  (white-box, continuous P(wrong) = 1 - P(correct))
  * raw head-probe         (white-box, no shallow control)
  * judge verdict          (black-box, binary: judge says NO -> flag wrong)
  * judge p_yes            (black-box, continuous: 1 - P(YES) token mass)

The judge verdict is a single operating point. To compare fairly, the probe's
threshold is chosen on TRAIN to hit the judge's train precision, then applied
unchanged to TEST. Reading the best threshold off the test PR curve — which this
script used to do — is a best-of-all-thresholds statistic on the reporting split and
is not comparable to a fixed verdict; it is still reported, explicitly labelled as an
optimistic upper bound.

Probe fit mirrors train_test_probe.py exactly (fit on TRAIN, score TEST).

Example:
  python filtering/error_detection_compare.py \
    --train_hidden results_gemma3_train/hidden_rich.npz --train_raw results_gemma3_train/gsm8k/raw.jsonl \
    --test_hidden  results_gemma3_test/hidden_rich.npz  --test_raw  results_gemma3_test/gsm8k/raw.jsonl \
    --head_dim 256 --title "Gemma3-12b" --out filtering/figures/gemma3_error_detection.png
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from probe_models import (check_split_provenance, check_split_usable,
                          fit_ensemble, load_split, model_label,
                          shallow_only_scores,
                          oracle_recall_at_precision, parse_models, precision_at_threshold,
                          prepare_head_features, recall_ci_at_threshold,
                          threshold_at_precision, wilson_interval)


def log(*a): print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_hidden", required=True)
    ap.add_argument("--train_raw", required=True)
    ap.add_argument("--test_hidden", required=True)
    ap.add_argument("--test_raw", required=True)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--label_policy", choices=("numeric", "stored"), default="numeric")
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--title", default="self-judge")
    ap.add_argument("--out", default="filtering/figures/error_detection_compare.png")
    ap.add_argument("--n_boot", type=int, default=2000,
                    help="bootstrap resamples for 95%% CIs (0 to skip)")
    ap.add_argument("--models", default="lr,mlp",
                    help="comma-separated probe models to compare: lr,mlp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="initialisations to ensemble for stochastic probes (MLP); "
                         "predictions are averaged. Must match the value the other "
                         "drivers use, or this script's operating points describe a "
                         "different estimator than the figures do.")
    ap.add_argument("--allow_split_mismatch", action="store_true",
                    help="downgrade the train/test provenance check to a warning. Only "
                         "for deliberate cross-run comparisons.")
    args = ap.parse_args()
    models = parse_models(args.models)
    t0 = time.time()

    tr = load_split(args.train_hidden, args.train_raw, args.head_dim, args.label_policy)
    te = load_split(args.test_hidden, args.test_raw, args.head_dim, args.label_policy)
    # Each split is validated only against its own raw file; this is what compares them
    # to each other, and what catches two splits built from the same problems.
    check_split_provenance(tr, te, strict=not args.allow_split_mismatch)
    check_split_usable("train", tr)
    check_split_usable("test", te)
    ytr, yte = tr.y, te.y
    log(f"TRAIN n={len(ytr)} wrong={(ytr==0).sum()} ({1-ytr.mean():.1%})  "
        f"TEST n={len(yte)} wrong={(yte==0).sum()} ({1-yte.mean():.1%})  heads={tr.Z.shape[1]}")
    log(f"label policy={tr.info['policy']}  stored disagreements: "
        f"train={tr.info['stored_disagreements']} test={te.info['stored_disagreements']}  "
        f"dropped unlabelable: train={tr.info['n_dropped_unlabelable']} "
        f"test={te.info['n_dropped_unlabelable']}")

    Ftr, Fte, top, aucs = prepare_head_features(tr.Z, ytr, te.Z, args.topk)
    log(f"top head LDA AUC (train)={aucs.max():.3f}; {args.topk} heads ({time.time()-t0:.0f}s)")

    # Ensembled across initialisations, matching train_test_probe.py and
    # make_paper_figures.py: every operating point and bar below is computed from the
    # same averaged predictions rather than from one arbitrary seed.
    scores = {m: fit_ensemble(Ftr, ytr, Fte, tr.S, te.S, model_type=m, seed=args.seed,
                              n_seeds=args.n_seeds)[0]
              for m in models}

    # error-detection target: positive class = WRONG (y==0)
    yw_tr = (ytr == 0).astype(int)
    yw_te = (yte == 0).astype(int)
    wrong_te = {m: {v: 1.0 - scores[m][v] for v in ("raw", "controlled")} for m in models}
    wrong_tr = {m: {v: 1.0 - scores[m][f"{v}_train"] for v in ("raw", "controlled")}
                for m in models}
    base_rate = yw_te.mean()

    # ── judge baselines ─────────────────────────────────────────────────────────
    flag_te = (te.yes == 0).astype(int)
    tp = int(((flag_te == 1) & (yw_te == 1)).sum())
    j_prec = tp / max(int(flag_te.sum()), 1)
    j_rec = tp / max(int(yw_te.sum()), 1)
    # The precision target must come from TRAIN so no test information leaks into
    # the probe's threshold.
    flag_tr = (tr.yes == 0).astype(int)
    j_prec_tr = int(((flag_tr == 1) & (yw_tr == 1)).sum()) / max(int(flag_tr.sum()), 1)
    log(f"judge verdict: test precision={j_prec:.3f} recall={j_rec:.3f} "
        f"(flags {flag_te.mean():.1%} as wrong) | train precision={j_prec_tr:.3f}")

    p_yes_ok = np.isfinite(te.p_yes).all() and np.isfinite(tr.p_yes).all()
    if p_yes_ok:
        log(f"judge p_yes (continuous) AUC(wrong)={roc_auc_score(yw_te, 1.0 - te.p_yes):.4f} "
            f"AP={average_precision_score(yw_te, 1.0 - te.p_yes):.4f}")
    else:
        log("[WARN] dump has no p_yes; continuous black-box baseline unavailable.")

    # ── probe at a TRAIN-selected operating point ───────────────────────────────
    # The shallow features are what `controlled` residualizes away. Scoring them as a
    # detector in their own right is the only way to read the probe's margin over the
    # surface cues: on MATH they reach 0.80 AUC alone, above the controlled LR probe.
    sh_tr, sh_te = shallow_only_scores(tr.S, ytr, te.S, seed=args.seed)
    detectors = {}   # name -> (label, test score, frozen threshold)
    detectors["shallow"] = ("shallow text feats", 1.0 - sh_te,
                            threshold_at_precision(yw_tr, 1.0 - sh_tr, j_prec_tr))
    for m in models:
        for v in ("raw", "controlled"):
            thr = threshold_at_precision(yw_tr, wrong_tr[m][v], j_prec_tr)
            detectors[f"{m}_{v}"] = (f"{model_label(m)} {v}", wrong_te[m][v], thr)
    if p_yes_ok:
        detectors["p_yes"] = ("judge p_yes", 1.0 - te.p_yes,
                              threshold_at_precision(yw_tr, 1.0 - tr.p_yes, j_prec_tr))

    metrics = {}
    for key, (label, score_te, thr) in detectors.items():
        if thr is None:
            log(f"[WARN] {label}: train precision target {j_prec_tr:.3f} unreachable; no "
                f"operating point exists. Reported as N/A — reporting 0 would be "
                f"indistinguishable from a detector that genuinely caught nothing.")
        # Recall of a fixed rule is a binomial proportion over the wrong examples, so
        # the Wilson interval is exact and always brackets the point estimate.
        r, r_lo, r_hi = recall_ci_at_threshold(yw_te, score_te, thr)
        metrics[key] = {
            "label": label, "thr": thr,
            "recall": r, "recall_lo": r_lo, "recall_hi": r_hi,
            "precision": precision_at_threshold(yw_te, score_te, thr),
            "ap": float(average_precision_score(yw_te, score_te)),
            "auc": float(roc_auc_score(yw_te, score_te)),
            "oracle_recall": oracle_recall_at_precision(yw_te, score_te, j_prec),
        }
    j_lo, j_hi = wilson_interval(tp, int(yw_te.sum()))
    metrics["judge"] = {"label": "judge verdict", "thr": None, "recall": j_rec,
                        "recall_lo": j_lo, "recall_hi": j_hi,
                        "precision": j_prec, "ap": float("nan"), "auc": float("nan"),
                        "oracle_recall": j_rec}

    def fmt(v, spec=".1%"):
        return "N/A" if v is None else format(v, spec)

    log(f"--- fixed operating point (threshold chosen on TRAIN at P={j_prec_tr:.0%}) ---")
    log("    recall intervals are Wilson (exact for a fixed rule), not bootstrap")
    for key in ["judge"] + list(detectors):
        d = metrics[key]
        extra = ("" if key == "judge"
                 # DESCRIPTIVE ONLY. Chosen by maximising recall over TEST thresholds,
                 # so it is a best-of-all-thresholds statistic on the reporting split
                 # and is not an unbiased result. Never plotted or tabulated.
                 else f"   [test-oracle, descriptive only: {fmt(d['oracle_recall'])}]")
        ci_s = ("" if d["recall"] is None
                else f" [{fmt(d['recall_lo'])}, {fmt(d['recall_hi'])}]")
        log(f"  {d['label']:>16s}: recall={fmt(d['recall'])}{ci_s}  "
            f"realised precision={fmt(d['precision'])}{extra}")

    # ── bootstrap for the PAIRED probe-minus-judge difference ───────────────────
    # Marginal recalls use the Wilson interval above; the paired difference is not a
    # single proportion, so it still needs resampling. Every rule compared is FIXED (a
    # frozen threshold or a fixed verdict), so a resample re-estimates the same
    # statistic instead of re-maximising over thresholds. Unstratified so prevalence
    # uncertainty propagates — precision depends on prevalence, and holding it fixed
    # understates the interval.
    ci = {}
    live = {k: v for k, v in detectors.items() if v[2] is not None}
    dead = [k for k in detectors if k not in live]
    if dead:
        log(f"  [WARN] no operating point for {', '.join(dead)}; excluded from the "
            f"paired-difference bootstrap.")
    if args.n_boot > 0 and live:
        rng = np.random.default_rng(args.seed)
        n = len(yw_te)
        boot = {"judge": [], **{f"d_{k}": [] for k in live}}
        for _ in range(args.n_boot):
            s = rng.integers(0, n, n)
            yw = yw_te[s]
            nw = int(yw.sum())
            if not nw:
                continue
            rj = ((flag_te[s] == 1) & (yw == 1)).sum() / nw
            boot["judge"].append(rj)
            for key, (_, score_te, thr) in live.items():
                r = float(np.sum((score_te[s] >= thr) & (yw == 1)) / nw)
                boot[f"d_{key}"].append(r - rj)
        ci = {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
              for k, v in boot.items() if v}
        log(f"--- {args.n_boot}x bootstrap 95% CIs, paired difference vs judge ---")
        for k in live:
            m, lo, hi = ci[f"d_{k}"]
            verdict = "CI excludes 0" if lo > 0 else ("CI excludes 0 (negative)" if hi < 0 else "n.s.")
            log(f"  d {metrics[k]['label']:>14s} - judge = {m:+.1%}  [{lo:+.1%}, {hi:+.1%}]  -> {verdict}")

    # ── plot ────────────────────────────────────────────────────────────────────
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {"lr": "#0072B2", "mlp": "#E69F00"}
    for m in models:
        p, r, _ = precision_recall_curve(yw_te, wrong_te[m]["controlled"])
        axL.plot(r, p, color=colors[m], lw=2.4,
                 label=f"{model_label(m)} controlled (AP={metrics[f'{m}_controlled']['ap']:.2f})")
    if p_yes_ok:
        p, r, _ = precision_recall_curve(yw_te, 1.0 - te.p_yes)
        axL.plot(r, p, color="#D55E00", lw=1.8, ls="-.",
                 label=f"judge p_yes (AP={metrics['p_yes']['ap']:.2f})")
    axL.scatter([j_rec], [j_prec], color="#D55E00", s=110, marker="*", zorder=6,
                label=f"judge verdict (P={j_prec:.0%}, R={j_rec:.0%})")
    axL.axhline(base_rate, color="#5f5f5f", ls=":", lw=1.6,
                label=f"random flag (P={base_rate:.0%})")
    axL.set_xlabel("Recall (wrong solutions caught)"); axL.set_ylabel("Precision")
    axL.set_title("Error-detection precision–recall"); axL.set_xlim(0, 1); axL.set_ylim(0, 1)
    axL.legend(fontsize=8.5, loc="upper right"); axL.grid(alpha=0.3)

    bar_keys = (["judge", "shallow"] + [f"{m}_controlled" for m in models]
                + (["p_yes"] if p_yes_ok else []))
    labels = [metrics[k]["label"].replace(" ", "\n") for k in bar_keys]
    # A detector without an operating point gets a zero-height bar annotated "n/a";
    # plotting a real 0 would read as "caught nothing", which is a different claim.
    na = [i for i, k in enumerate(bar_keys) if metrics[k]["recall"] is None]
    recs = [0.0 if metrics[k]["recall"] is None else metrics[k]["recall"] * 100
            for k in bar_keys]
    cols = (["#D55E00", "#009E73"] + [colors[m] for m in models]
            + (["#5f5f5f"] if p_yes_ok else []))
    # Wilson bounds always bracket the point estimate, so no clamping is applied — a
    # negative arm here would indicate a real bug rather than a rendering nuisance.
    lo_err = [0.0 if i in na else recs[i] - metrics[k]["recall_lo"] * 100
              for i, k in enumerate(bar_keys)]
    hi_err = [0.0 if i in na else metrics[k]["recall_hi"] * 100 - recs[i]
              for i, k in enumerate(bar_keys)]
    bars = axR.bar(labels, recs, color=cols, width=0.6, yerr=np.array([lo_err, hi_err]),
                   capsize=6, error_kw=dict(ecolor="#333", lw=1.5))
    for i, (b, v, k) in enumerate(zip(bars, recs, bar_keys)):
        cx = b.get_x() + b.get_width() / 2
        if i in na:
            axR.text(cx, 2, "n/a\n(no operating\npoint)", ha="center", va="bottom",
                     fontsize=8, color="#5f5f5f")
            continue
        pr = metrics[k]["precision"]
        lbl = (f"{v:.0f}%\n[{metrics[k]['recall_lo']*100:.0f}, "
               f"{metrics[k]['recall_hi']*100:.0f}]\nP={pr*100:.0f}%")
        axR.text(cx, metrics[k]["recall_hi"] * 100 + 2, lbl, ha="center", fontsize=8.5,
                 fontweight="bold")
    sub = ""
    if ci and models:
        primary = f"{models[-1]}_controlled"
        if f"d_{primary}" in ci:
            m, lo, hi = ci[f"d_{primary}"]
            sub = (f"\nΔ({metrics[primary]['label']} - judge) = {m:+.0%}  "
                   f"95% CI [{lo:+.0%}, {hi:+.0%}] (paired bootstrap)")
    axR.set_ylabel("Recall — wrong solutions caught (%)")
    axR.set_title(f"Threshold CALIBRATED on TRAIN at the judge's train precision "
                  f"({j_prec_tr:.0%})\nP = precision each rule REALISES on test — "
                  f"calibrating on train does not equalise it{sub}", fontsize=9)
    axR.set_ylim(0, 118); axR.grid(alpha=0.3, axis="y")

    # Derived from the raw records rather than hardcoded, so a MATH run cannot render
    # a figure captioned GSM8K.
    dataset_label = str(tr.info.get("dataset") or "").upper()
    fig.suptitle(f"{args.title}: white-box probe vs black-box judge — error detection\n"
                 f"{dataset_label}, fit on {len(ytr)} train, eval on {len(yte)} held-out test "
                 f"({yw_te.sum()} wrong / {len(yte)})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    log(f"DONE saved {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
