#!/usr/bin/env python3
"""Clean train/test split for the attention-head correctness probe.

Fits the probe on one split and evaluates on a disjoint held-out split — no fold
leakage, and no operating point read off the test set:

  1. rank attention heads by closed-form LDA AUC on TRAIN, take top-32
  2. residualize those head features against shallow text features (LinearRegression
     fit on TRAIN), to control for surface cues
  3. fit a logistic-regression probe on TRAIN residuals; score TEST
  4. report TEST AUC (raw + controlled) + risk-coverage, against BOTH judge baselines:
     the binary YES/NO verdict and the judge's continuous P(YES) token mass

Inputs are the `hidden_rich.npz` dumps (Z_head + y + p_yes + idx) and `raw.jsonl`
(for the shallow features / judge_verdict) of each split.

Example:
  python filtering/train_test_probe.py \
    --train_hidden results_gemma3_train/hidden_rich.npz --train_raw results_gemma3_train/gsm8k/raw.jsonl \
    --test_hidden  results_gemma3_test/hidden_rich.npz \
    --test_raw     results_gemma3_test/gsm8k/raw.jsonl \
    --head_dim 256 --out filtering/figures/gemma3_train_test_probe.png
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from probe_models import (
    check_split_provenance,
    check_split_usable,
    ensemble_seed_aucs,
    fit_ensemble,
    head_ranking,
    is_stochastic,
    load_split,
    model_label,
    parse_models,
    prepare_head_features,
    risk_coverage,
    shallow_only_scores,
    train_resample_auc,
)


def log(*a): print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_hidden", required=True)
    ap.add_argument("--train_raw", required=True)
    ap.add_argument("--test_hidden", required=True)
    ap.add_argument("--test_raw", required=True)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--label_policy", choices=("numeric", "stored"), default="numeric",
                    help="numeric equivalence (default) or legacy labels stored in the NPZ")
    ap.add_argument("--head_dim", type=int, default=128,
                    help="attention head_dim: 128 for Ministral-3-8B, 256 for Gemma3-12b. "
                         "Verified against the dump's recorded geometry.")
    ap.add_argument("--models", default="lr",
                    help="comma-separated probe models to compare: lr,mlp (default: lr)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="seeds for stochastic probes (MLP); reports mean +/- sd so a "
                         "single lucky init is not reported as the result. This spread "
                         "is initialisation only — see --n_train_resamples")
    ap.add_argument("--n_train_resamples", type=int, default=8,
                    help="train bootstrap replicates, each redoing head selection, to "
                         "estimate spread from data sampling + head selection + init. "
                         "Dominates runtime (a full per-head LDA sweep per replicate); "
                         "0 skips it")
    ap.add_argument("--title", default="Ministral-3-8B")
    ap.add_argument("--dataset_label", default=None,
                    help="dataset name shown in the figure title. Derived from the raw "
                         "records by default, so a figure cannot be mislabelled.")
    ap.add_argument("--out", default="filtering/figures/ministral_train_test_probe.png")
    ap.add_argument("--allow_split_mismatch", action="store_true",
                    help="downgrade the train/test provenance check to a warning. Only "
                         "for deliberate cross-run comparisons; the reported AUC then "
                         "describes two different experiments.")
    args = ap.parse_args()
    models = parse_models(args.models)
    # These fail deep inside sklearn (or silently produce a degenerate probe) rather
    # than here, where the cause is still visible.
    for flag, value, minimum in (("--topk", args.topk, 1), ("--head_dim", args.head_dim, 1),
                                 ("--n_seeds", args.n_seeds, 1),
                                 ("--n_train_resamples", args.n_train_resamples, 0)):
        if value < minimum:
            ap.error(f"{flag} must be >= {minimum} (got {value})")
    t0 = time.time()

    tr = load_split(args.train_hidden, args.train_raw, args.head_dim, args.label_policy)
    te = load_split(args.test_hidden, args.test_raw, args.head_dim, args.label_policy)
    # Each split validates only against its own raw file. Nothing so far compares the
    # two, so a mismatched pair (or the same problems on both sides) would train and
    # report a held-out AUC without a single error.
    check_split_provenance(tr, te, strict=not args.allow_split_mismatch)
    check_split_usable("train", tr)
    check_split_usable("test", te)
    ytr, yte = tr.y, te.y
    nH = tr.Z.shape[1]
    log(f"TRAIN n={len(ytr)} wrong={(ytr==0).sum()} ({1-ytr.mean():.1%})  "
        f"TEST n={len(yte)} wrong={(yte==0).sum()} ({1-yte.mean():.1%})  heads={nH}")
    for name, sp in (("TRAIN", tr), ("TEST", te)):
        ex = sp.info.get("exclusions") or {}
        if ex:
            log(f"{name} analysed population: {ex['n_analysed']}/{ex['n_generated']} "
                f"generations  (excluded: {ex['n_excluded_truncated']} budget-truncated, "
                f"{ex['n_excluded_error']} generation error(s), "
                f"{ex['n_excluded_no_gold']} unparseable gold)")
            if ex["n_excluded_truncated"]:
                log(f"  all {ex['truncated_marked_wrong']}/{ex['n_excluded_truncated']} "
                    f"truncated records were marked wrong; keeping them would raise the "
                    f"observed error rate from {ex['prevalence_analysed']:.2%} to "
                    f"{ex['prevalence_if_truncations_included']:.2%}")
    log(f"label policy={tr.info['policy']}  stored disagreements: "
        f"train={tr.info['stored_disagreements']} test={te.info['stored_disagreements']}  "
        f"dropped unlabelable: train={tr.info['n_dropped_unlabelable']} "
        f"test={te.info['n_dropped_unlabelable']}")

    Ftr, Fte, top, aucs = prepare_head_features(tr.Z, ytr, te.Z, args.topk)
    log(f"top head LDA AUC (train) = {aucs.max():.3f}; selected {args.topk} heads ({time.time()-t0:.0f}s)")

    # Persist the FULL candidate ranking, not just the selected slice — for an
    # attention-head probing result this is the interpretability output. Without the
    # whole distribution you cannot tell whether rank topk is meaningfully better than
    # rank 400, i.e. whether the cut reflects structure or is arbitrary.
    n_layers = (tr.info.get("dump_meta") or {}).get("n_layers") or 0
    ranking = head_ranking(top, aucs, n_layers, args.topk)
    ranking["head_dim"] = args.head_dim
    heads_path = Path(args.out).with_suffix("").as_posix() + "_heads.json"
    Path(heads_path).parent.mkdir(parents=True, exist_ok=True)
    Path(heads_path).write_text(json.dumps(ranking, indent=2))
    s = ranking["auc_summary"]
    log(f"head LDA AUC (train): max={s['max']:.3f} cut@top{args.topk}={s['at_topk_cut']:.3f} "
        f"median={s['p50']:.3f}  -> {heads_path}")

    scores = {}
    for model in models:
        # ONE aggregation, used for every number and every curve below: a stochastic
        # probe is ensembled by averaging its predicted probabilities across seeds.
        # Plotting one seed's predictions while quoting the mean of the per-seed AUCs
        # would describe two different fits in the same figure.
        scores[model], seeds, runs = fit_ensemble(
            Ftr, ytr, Fte, tr.S, te.S, model_type=model, seed=args.seed,
            n_seeds=args.n_seeds)
        for variant in ("raw", "controlled"):
            auc = roc_auc_score(yte, scores[model][variant])
            note = ""
            if len(seeds) > 1:
                per_seed = ensemble_seed_aucs(runs, yte, variant)
                # Reported apart from the ensemble, never as its error bar: it says how
                # far a single arbitrary draw would have landed, not how uncertain the
                # ensemble is. The train-resample spread below is the other source.
                note = (f"   [single-draw init spread: sd={np.std(per_seed, ddof=1):.4f} "
                        f"over n={len(seeds)} seeds]")
            log(f"HELD-OUT TEST AUC  {model_label(model):3s} {variant:10s} = "
                f"{auc:.4f} (ensemble of {len(seeds)}){note}")

    # Refitting at several inits holds the data AND the selected heads fixed, so its
    # spread is init noise alone. Resampling train and redoing head selection inside
    # each replicate covers sampling + selection + init — with a few hundred minority
    # examples choosing topk of nH candidates, selection is plausibly the larger term.
    if args.n_train_resamples > 0:
        for model in models:
            vals = train_resample_auc(tr.Z, ytr, tr.S, te.Z, yte, te.S, args.topk,
                                      model_type=model, n_reps=args.n_train_resamples,
                                      seed=args.seed, log=log, n_seeds=args.n_seeds)
            if len(vals) > 1:
                log(f"TRAIN-RESAMPLE TEST AUC  {model_label(model):3s} controlled  = "
                    f"{np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f} "
                    f"(sampling + head selection + init, n={len(vals)})")

    # ── shallow-feature baseline ────────────────────────────────────────────────
    # The features `controlled` residualizes away, scored on their own. Without this
    # the reader cannot tell whether the control removed a surface artifact or the
    # single best legitimate cue: on MATH these reach ~0.80 alone, because solution
    # length tracks problem difficulty.
    _, shallow_te = shallow_only_scores(tr.S, ytr, te.S, seed=args.seed)
    auc_shallow = roc_auc_score(yte, shallow_te)
    log(f"SHALLOW-ONLY baseline (length, words, digits, ops) TEST AUC = {auc_shallow:.4f}"
        f"   <- what `controlled` subtracts")
    for model in models:
        gain = roc_auc_score(yte, scores[model]["raw"]) - auc_shallow
        log(f"  {model_label(model)} raw minus shallow-only = {gain:+.4f} "
            f"(the value the attention heads add over the surface features)")

    # ── judge baselines ─────────────────────────────────────────────────────────
    # The binary verdict is one operating point; its ROC AUC is balanced accuracy and
    # is not comparable to a continuous score. p_yes IS the continuous black-box
    # baseline, so report it whenever the dump carries it.
    yes_te = te.yes
    v_err = (yes_te != yte).mean()
    nwrong = (yte == 0).sum()
    v_rec = ((yes_te == 0) & (yte == 0)).sum() / max(nwrong, 1)
    v_prec = ((yes_te == 0) & (yte == 0)).sum() / max((yes_te == 0).sum(), 1)
    log(f"judge verdict (test): saysYES={yes_te.mean():.1%} err={v_err:.1%} "
        f"recall_wrong={v_rec:.1%} prec={v_prec:.1%}")
    log(f"judge verdict balanced accuracy (NOT an AUC-comparable number): "
        f"{roc_auc_score(yte, yes_te):.4f}")

    p_yes_ok = np.isfinite(te.p_yes).all()
    if p_yes_ok:
        log(f"judge p_yes (continuous black-box baseline) TEST AUC = "
            f"{roc_auc_score(yte, te.p_yes):.4f}")
    else:
        log("[WARN] dump has no p_yes; the continuous black-box baseline is unavailable. "
            "Re-extract to enable it.")

    # ── risk-coverage on test ───────────────────────────────────────────────────
    # Accept highest P(correct) first; risk = wrong rate among ACCEPTED items, so at
    # coverage 100% the curve meets the no-filtering line by construction.
    N = len(yte)
    full_err = (yte == 0).mean()
    primary = "lr" if "lr" in scores else models[0]
    for c in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        vals = []
        for model in models:
            _, risk_m = risk_coverage(scores[model]["controlled"], yte)
            vals.append(f"{model_label(model)}={risk_m[int(c*N)-1]:.3f}")
        _, risk_s = risk_coverage(shallow_te, yte)
        vals.append(f"shallow={risk_s[int(c*N)-1]:.3f}")
        if p_yes_ok:
            _, risk_j = risk_coverage(te.p_yes, yte)
            vals.append(f"p_yes={risk_j[int(c*N)-1]:.3f}")
        log(f"  cov {c:.0%}: risk among accepted  " + "  ".join(vals))

    plt.figure(figsize=(8.2, 5.3))
    colors = {"lr": "#0072B2", "mlp": "#E69F00"}
    for model in models:
        auc_ctrl = roc_auc_score(yte, scores[model]["controlled"])
        covc, riskc = risk_coverage(scores[model]["controlled"], yte)
        plt.plot(covc * 100, riskc * 100, color=colors[model], lw=2.6,
                 label=f"{model_label(model)} controlled gate (test AUC={auc_ctrl:.2f})")
    for model in models:
        auc_raw = roc_auc_score(yte, scores[model]["raw"])
        covr, riskr = risk_coverage(scores[model]["raw"], yte)
        plt.plot(covr * 100, riskr * 100, color=colors[model], lw=1.5, ls="--", alpha=0.6,
                 label=f"{model_label(model)} raw gate (test AUC={auc_raw:.2f})")
    covs, risks = risk_coverage(shallow_te, yte)
    plt.plot(covs * 100, risks * 100, color="#009E73", lw=1.8, ls=":",
             label=f"shallow-features-only gate (test AUC={auc_shallow:.2f})")
    if p_yes_ok:
        covj, riskj = risk_coverage(te.p_yes, yte)
        plt.plot(covj * 100, riskj * 100, color="#D55E00", lw=2.0, ls="-.",
                 label=f"judge p_yes gate (test AUC={roc_auc_score(yte, te.p_yes):.2f})")
    plt.axhline(full_err * 100, color="#5f5f5f", ls=":", lw=1.8,
                label=f"no filtering (accept all)  risk={full_err*100:.1f}%")
    i = int(0.7 * N) - 1
    _, risk_p = risk_coverage(scores[primary]["controlled"], yte)
    plt.scatter([70], [risk_p[i] * 100], color=colors[primary], s=60, zorder=5)
    plt.gca().invert_xaxis()
    plt.xlabel("Coverage (% auto-accepted)"); plt.ylabel("Risk: wrong among accepted (%)")
    dataset_label = args.dataset_label or str(tr.info.get("dataset") or "").upper()
    plt.title(f"Clean train/test split — {args.title} self-judge probe "
              f"({dataset_label})\n"
              f"fit on {len(ytr)} train, eval on {N} held-out test", fontsize=10)
    plt.legend(fontsize=8.5, loc="upper left"); plt.grid(alpha=0.3)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=140)
    log(f"DONE saved {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
