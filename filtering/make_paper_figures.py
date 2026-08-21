#!/usr/bin/env python3
"""Paper figures for the white-box probe-gate filtering experiments.

Recomputes every number from the raw artifacts (results_{tag}_{train,test}/
hidden_rich.npz + gsm8k/raw.jsonl) with the exact pipeline of
train_test_probe.py / error_detection_compare.py (fit on TRAIN only, score
held-out TEST), caches metrics+curves to JSON, then renders:

  figures/paper/fig_risk_coverage__{sel}.{pdf,png}      risk-coverage panels, one per run
  figures/paper/fig_error_detection_pr__{sel}.{pdf,png} PR panels + judge baselines
  figures/paper/fig_recall_at_train_calibrated_op__{sel}.{pdf,png}
                                                       grouped recall bars w/ 95% CIs
  figures/paper/table_probe_auc__{sel}.tex              LaTeX summary table
  figures/paper/metrics_{tag}.json                      all cached numbers
  figures/paper/heads_{tag}.json                        which attention heads were selected

`{sel}` names the runs that went into the file, so two selections cannot overwrite
each other's output. Every figure is therefore traceable to its inputs by filename.

Two things here differ from the earlier version and change the reported numbers:

  * Risk-coverage accepts the highest-P(correct) items first and reports the WRONG
    rate among accepted. The previous version ranked by |score-0.5| and reported the
    probe's 0/1 classification error, which is selective classification, not
    filtering — items the probe confidently called WRONG counted as "accepted", and
    the curve did not meet the no-filtering baseline at 100% coverage.
  * Recall is reported at a TRAIN-CALIBRATED operating point: each detector's
    threshold is chosen on the train split at the judge's train precision, then applied
    unchanged to test. This does NOT equalise precision on test -- the realised test
    precision of each rule is reported beside its recall and the values differ, often
    substantially. Recalls at differing realised precisions are not directly
    comparable, so AUC and risk-coverage remain the primary comparisons.
    Maximising recall over TEST thresholds is a best-of-all-thresholds statistic on the
    reporting split; it is cached as `oracle_recall`, labelled test-oracle/descriptive,
    and kept out of the figures and the main table.

The run selection is always explicit: there is no default, so the same command
produces the same figure on every machine.

Compute once (slow: loads the activation dumps, fits LR+MLP):
  python filtering/make_paper_figures.py --compute --runs gemma3
Re-plot from cache (fast):
  python filtering/make_paper_figures.py --runs gemma3
Everything registered, or just what is on this machine:
  python filtering/make_paper_figures.py --runs all
  python filtering/make_paper_figures.py --runs present
"""
import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_models import (check_split_provenance, ensemble_seed_aucs, fit_ensemble,
                          head_ranking,
                          is_stochastic, load_split,
                          oracle_recall_at_precision, precision_at_threshold,
                          prepare_head_features, recall_ci_at_threshold, risk_coverage,
                          shallow_only_scores,
                          threshold_at_precision, train_resample_auc, wilson_interval)
from metrics.correctness import label_policy_for

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures" / "paper"

RUNS = {
    "gemma3": dict(title="Gemma3-12B", head_dim=256,
                   train=ROOT / "results_gemma3_train", test=ROOT / "results_gemma3_test"),
    "llama31_8b": dict(title="Llama-3.1-8B", head_dim=128,
                       train=ROOT / "results_llama31_8b_train", test=ROOT / "results_llama31_8b_test"),
    "qwen25_7b": dict(title="Qwen2.5-7B", head_dim=128,
                      train=ROOT / "results_qwen25_7b_train", test=ROOT / "results_qwen25_7b_test"),
    # A MATH run is a separate entry, not a flag: it needs its own TAG (so its dumps
    # never share a path with the GSM8K run) and `dataset` picks both the raw.jsonl
    # subdirectory and the answer-equivalence policy the labels are recomputed under.
    "gemma3_math": dict(title="Gemma3-12B", head_dim=256, dataset="math",
                        train=ROOT / "results_gemma3_math_train",
                        test=ROOT / "results_gemma3_math_test"),
    "qwen25_7b_math": dict(title="Qwen2.5-7B", head_dim=128, dataset="math",
                           train=ROOT / "results_qwen25_7b_math_train",
                           test=ROOT / "results_qwen25_7b_math_test"),
    "llama31_8b_math": dict(title="Llama-3.1-8B", head_dim=128, dataset="math",
                            train=ROOT / "results_llama31_8b_math_train",
                            test=ROOT / "results_llama31_8b_math_test"),
}
PROBES = ["lr", "mlp"]
TOPK = 32
SEED = 0
N_SEEDS = 5          # stochastic probes (MLP) are refit this many times
N_BOOT = 2000
# Train resamples with head re-selection inside each replicate. Each one redoes the
# per-head LDA sweep, so this dominates runtime; --n_train_resamples 0 skips it.
N_TRAIN_RESAMPLES = 8
# Bump when the computed quantities change, so cached JSON from an older definition
# is rejected instead of silently re-plotted.
METRICS_VERSION = 7
# Bumped whenever run_config_digest's FORMULA changes. Stored beside the digest so a
# definition change is reported as such: without it, every cache mismatches at once and
# the error prints two identical-looking configurations, blaming a difference that does
# not exist. (v2 dropped `title`, which is display text and cannot change a number.)
CONFIG_DIGEST_VERSION = 2

# Okabe-Ito subset, CVD-validated (dataviz validator, light surface):
C_LR = "#0072B2"
C_MLP = "#E69F00"
C_JUDGE = "#D55E00"
C_PYES = "#5f5f5f"
C_SHALLOW = "#009E73"
C_GRAY = "#5f5f5f"

PROBE_LABEL = {"lr": "LR probe", "mlp": "MLP probe"}
# The features `controlled` residualizes away, scored alone. Without it on the
# figure the reader cannot tell how much of a probe's score is a ruler: on MATH
# these reach 0.80 AUC by themselves, ABOVE the controlled LR probe.
SHALLOW_LABEL = "Shallow text feats"


def log(*a):
    print(*a, flush=True)


def _pct_or_dash(v, digits=1):
    """A percentage with its sign, or an em-dash when the quantity does not exist.

    `None` here is not zero and not an error. Realised precision is undefined whenever
    a rule flags nothing on test -- there are no predictions to be right or wrong
    about -- and that can happen while recall is perfectly well defined, so a guard on
    recall alone does not cover it. Formatting it as 0.0% would claim the rule was
    always wrong; letting it reach a format spec raises TypeError and takes down the
    whole report after the expensive part has already run.
    """
    return "--" if v is None else f"{v * 100:.{digits}f}%"


def panel_label(tag: str, d: dict) -> str:
    """How a run is named on a figure or table row: model AND dataset, always.

    Naming the dataset only in the entries that happen to be non-default produced
    panels reading "Gemma3-12B" beside "Gemma3-12B (MATH)", which invites the reader
    to assume the unlabelled one is the norm rather than a second dataset. Composing
    the label here means a new dataset cannot be added without its name appearing.
    """
    title = RUNS.get(tag, {}).get("title") or d.get("title", tag)
    return f"{title} ({str(d.get('dataset', 'gsm8k')).upper()})"


def compute_run(tag, cfg, n_train_resamples=N_TRAIN_RESAMPLES):
    t0 = time.time()
    log(f"=== {tag} ===")
    # run_eval writes raw.jsonl under a per-dataset subdirectory; a RUNS entry may
    # override `dataset` to point this driver at a MATH run instead.
    dataset = cfg.get("dataset", "gsm8k")
    tr = load_split(cfg["train"] / "hidden_rich.npz", cfg["train"] / dataset / "raw.jsonl",
                    cfg["head_dim"])
    te = load_split(cfg["test"] / "hidden_rich.npz", cfg["test"] / dataset / "raw.jsonl",
                    cfg["head_dim"])
    # Both splits are validated against their own raw file, never against each other.
    check_split_provenance(tr, te)
    ytr, yte = tr.y, te.y
    log(f"TRAIN n={len(ytr)} wrong={(ytr == 0).sum()} ({1 - ytr.mean():.1%})  "
        f"TEST n={len(yte)} wrong={(yte == 0).sum()} ({1 - yte.mean():.1%})  heads={tr.Z.shape[1]}")

    Ftr, Fte, top, aucs = prepare_head_features(tr.Z, ytr, te.Z, TOPK)
    n_layers = (tr.info.get("dump_meta") or {}).get("n_layers") or 0
    # The FULL candidate ranking, not just the selected slice: without it you cannot
    # tell whether rank 32 is meaningfully better than rank 400.
    ranking = head_ranking(top, aucs, n_layers, TOPK)
    ranking["head_dim"] = cfg["head_dim"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"heads_{tag}.json").write_text(json.dumps(ranking, indent=2))
    s = ranking["auc_summary"]
    log(f"head LDA AUC (train): max={s['max']:.3f} cut@top{TOPK}={s['at_topk_cut']:.3f} "
        f"median={s['p50']:.3f} ({time.time() - t0:.0f}s)")

    out = {
        "tag": tag, "title": cfg["title"], "topk": TOPK, "seed": SEED,
        "metrics_version": METRICS_VERSION,
        # The policy the labels were ACTUALLY computed under, which is dataset-specific.
        # Stamping the GSM8K constant here would misdescribe a MATH run's labels.
        "label_policy": tr.info["policy"],
        "dataset": dataset,
        # Binds this cache to the registry entry it was computed from; see
        # run_config_digest. Without it, reusing a tag for a different run silently
        # re-plots the old numbers under the new run's name.
        "config_sha": run_config_digest(tag, cfg),
        "config_digest_version": CONFIG_DIGEST_VERSION,
        # Provenance fields the input artifacts do not carry. Dumps predating the
        # provenance work have none and cannot be given them retroactively, so this
        # records the gap rather than pretending it was checked.
        "unauthenticated": {"train": tr.info.get("unauthenticated") or [],
                            "test": te.info.get("unauthenticated") or []},
        "train_label_info": tr.info, "test_label_info": te.info,
        # Analysed population per split, for the figure captions and the paper text.
        "exclusions": {"train": tr.info.get("exclusions"),
                       "test": te.info.get("exclusions")},
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "wrong_train": int((ytr == 0).sum()), "wrong_test": int((yte == 0).sum()),
        "head_dim": cfg["head_dim"],
        "head_auc_summary": s,
    }

    # ── probes: point scores at SEED, plus seed spread for stochastic probes ─────
    scores = {}
    for m in PROBES:
        # ONE aggregation for everything: the ensemble's averaged probabilities are what
        # gets plotted, thresholded, tabulated and annotated. Previously the curves used
        # runs[0] while the annotations quoted the mean of the per-seed AUCs, so the line
        # on the page and the number beside it came from different fits.
        scores[m], seeds, runs = fit_ensemble(Ftr, ytr, Fte, tr.S, te.S, model_type=m,
                                              seed=SEED, n_seeds=N_SEEDS)
        out[f"n_seeds_{m}"] = len(seeds)
        out[f"aggregation_{m}"] = ("ensemble-mean-of-probabilities" if len(seeds) > 1
                                   else "single-deterministic-fit")
        for v in ("raw", "controlled"):
            # The reported estimate IS the ensemble's AUC, not a mean of AUCs.
            out[f"auc_{m}_{v}"] = float(roc_auc_score(yte, scores[m][v]))
            per_seed = ensemble_seed_aucs(runs, yte, v)
            out[f"auc_{m}_{v}_seed_aucs"] = per_seed
            # INITIALISATION-ONLY spread of the individual draws. This is not an error
            # bar on the ensemble; it says how far one arbitrary seed would have landed.
            out[f"auc_{m}_{v}_seed_sd"] = (float(np.std(per_seed, ddof=1))
                                           if len(per_seed) > 1 else None)
        log(f"AUC {m} (ensemble of {len(seeds)}): raw={out[f'auc_{m}_raw']:.4f} "
            f"ctrl={out[f'auc_{m}_controlled']:.4f}  "
            f"[init-only spread of single draws: raw sd="
            f"{out['auc_%s_raw_seed_sd' % m] or 0:.4f} "
            f"ctrl sd={out['auc_%s_controlled_seed_sd' % m] or 0:.4f}] "
            f"({time.time() - t0:.0f}s)")

    # ── shallow-feature baseline ────────────────────────────────────────────────
    # Exactly what `controlled` subtracts, scored on its own, so the probe's margin
    # over the surface features is readable off the figure instead of inferred.
    sh_tr, sh_te = shallow_only_scores(tr.S, ytr, te.S, seed=SEED)
    out["auc_shallow"] = float(roc_auc_score(yte, sh_te))
    for m in PROBES:
        out[f"gain_{m}_raw_over_shallow"] = out[f"auc_{m}_raw"] - out["auc_shallow"]
    log(f"AUC shallow-only = {out['auc_shallow']:.4f}  "
        + "  ".join(f"({m} raw {out[f'gain_{m}_raw_over_shallow']:+.4f})" for m in PROBES))

    # ── train-resample spread: sampling + head selection + init ─────────────────
    for m in PROBES:
        key = f"auc_{m}_controlled_resample"
        if n_train_resamples <= 0:
            out[f"{key}_sd"] = None
            out[f"{key}_n"] = 0
            continue
        log(f"  train resamples ({m}, head selection redone per replicate):")
        vals = train_resample_auc(tr.Z, ytr, tr.S, te.Z, yte, te.S, TOPK,
                                  model_type=m, n_reps=n_train_resamples, seed=SEED,
                                  log=log, n_seeds=N_SEEDS)
        out[f"{key}_n"] = len(vals)
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{key}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        out[f"{key}_values"] = vals
        if len(vals) > 1:
            # The two spreads are printed side by side but never combined: this one
            # covers sampling + head selection + init, the other is a single draw's
            # distance from the ensemble.
            init_sd = out.get(f"auc_{m}_controlled_seed_sd")
            init_txt = "n/a (deterministic)" if init_sd is None else f"{init_sd:.4f}"
            log(f"  AUC {m} controlled, train-resample: {np.mean(vals):.4f} "
                f"+/- {np.std(vals, ddof=1):.4f} (n={len(vals)}; separately, "
                f"init-only sd of a single draw = {init_txt})")

    tr.Z = te.Z = None          # the activation dumps are large; drop them now

    # ── judge baselines ─────────────────────────────────────────────────────────
    yw_tr = (ytr == 0).astype(int)
    yw_te = (yte == 0).astype(int)
    flag_te = (te.yes == 0).astype(int)
    flag_tr = (tr.yes == 0).astype(int)
    tp = int(((flag_te == 1) & (yw_te == 1)).sum())
    out["judge_says_yes"] = float(te.yes.mean())
    out["judge_prec"] = tp / max(int(flag_te.sum()), 1)
    out["judge_rec"] = tp / max(int(yw_te.sum()), 1)
    out["judge_prec_train"] = (int(((flag_tr == 1) & (yw_tr == 1)).sum())
                               / max(int(flag_tr.sum()), 1))
    # A binary predictor's ROC AUC is its balanced accuracy; it is capped far below
    # what a continuous scorer can reach, so it must not be tabulated as an AUC
    # comparable to the probes'. p_yes is the continuous black-box baseline.
    out["judge_verdict_balanced_acc"] = float(roc_auc_score(yte, te.yes))
    p_yes_ok = bool(np.isfinite(te.p_yes).all() and np.isfinite(tr.p_yes).all())
    out["has_p_yes"] = p_yes_ok
    out["auc_p_yes"] = float(roc_auc_score(yte, te.p_yes)) if p_yes_ok else None
    log(f"judge verdict: P={out['judge_prec']:.3f} R={out['judge_rec']:.3f} "
        f"saysYES={out['judge_says_yes']:.1%} balanced_acc={out['judge_verdict_balanced_acc']:.3f}"
        + (f" | p_yes AUC={out['auc_p_yes']:.3f}" if p_yes_ok else " | p_yes MISSING"))

    # ── risk-coverage (true filtering gate) ─────────────────────────────────────
    out["risk_coverage"] = {}
    out["sel_err"] = {}
    curves = {f"{m}_{v}": scores[m][v] for m in PROBES for v in ("raw", "controlled")}
    curves["shallow"] = sh_te
    if p_yes_ok:
        curves["p_yes"] = te.p_yes
    for name, sc in curves.items():
        cov, risk = risk_coverage(sc, yte)
        out["risk_coverage"][name] = {"cov": cov.tolist(), "risk": risk.tolist()}
        out["sel_err"][name] = {f"{c:.0%}": float(risk[int(c * len(yte)) - 1])
                                for c in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)}

    # ── error detection at a TRAIN-selected operating point ─────────────────────
    wrong_te = {f"{m}_{v}": 1.0 - scores[m][v] for m in PROBES for v in ("raw", "controlled")}
    wrong_tr = {f"{m}_{v}": 1.0 - scores[m][f"{v}_train"]
                for m in PROBES for v in ("raw", "controlled")}
    wrong_te["shallow"] = 1.0 - sh_te
    wrong_tr["shallow"] = 1.0 - sh_tr
    if p_yes_ok:
        wrong_te["p_yes"] = 1.0 - te.p_yes
        wrong_tr["p_yes"] = 1.0 - tr.p_yes

    out["pr_curves"] = {}
    out["error_detection"] = {}
    thresholds = {}
    for name, wr_te in wrong_te.items():
        thr = threshold_at_precision(yw_tr, wrong_tr[name], out["judge_prec_train"])
        thresholds[name] = thr
        prec, rec, _ = precision_recall_curve(yw_te, wr_te)
        out["pr_curves"][name] = {"prec": prec.tolist(), "rec": rec.tolist()}
        # Recall of a fixed rule is a binomial proportion over the wrong examples, so
        # it has an exact interval that always contains the point estimate.
        r_point, r_lo, r_hi = recall_ci_at_threshold(yw_te, wr_te, thr)
        out["error_detection"][name] = {
            "ap": float(average_precision_score(yw_te, wr_te)),
            "auc_wrong": float(roc_auc_score(yw_te, wr_te)),
            "threshold": thr,
            # headline number: fixed rule chosen on train, evaluated on test.
            # None (not 0) when no operating point exists — see recall_at_threshold.
            "recall": r_point,
            "recall_lo": r_lo, "recall_hi": r_hi,
            # the precision the fixed rule actually realises on test. Chosen on train,
            # so it need not equal the judge's test precision; without it the reader
            # cannot check that the recalls being compared are comparable at all.
            "precision": precision_at_threshold(yw_te, wr_te, thr),
            # optimistic upper bound, kept for reference only
            "oracle_recall": oracle_recall_at_precision(yw_te, wr_te, out["judge_prec"]),
        }
        if thr is None:
            log(f"[WARN] {name}: train precision target {out['judge_prec_train']:.3f} "
                f"unreachable; no operating point exists, recall reported as N/A "
                f"(not 0 — that would read as a detector that caught nothing).")

    # judge's own recall interval, same estimator, so the bars are comparable
    j_lo, j_hi = wilson_interval(tp, int(yw_te.sum()))
    out["judge_rec_lo"], out["judge_rec_hi"] = j_lo, j_hi

    # ── bootstrap over the TEST set: the PAIRED probe-minus-judge difference ─────
    # Marginal recalls use the Wilson interval above. The paired difference is not a
    # single proportion, so it still needs resampling; unstratified so uncertainty in
    # prevalence propagates, and with frozen thresholds (no per-resample threshold
    # search) a resample re-estimates the same statistic instead of re-maximising.
    rng = np.random.default_rng(SEED)
    n = len(yw_te)
    keys = [k for k in wrong_te if thresholds[k] is not None]
    skipped = [k for k in wrong_te if thresholds[k] is None]
    if skipped:
        log(f"  [WARN] no operating point for {', '.join(skipped)}; excluded from the "
            f"paired-difference bootstrap.")
    boot = {"judge": [], **{f"d_{k}": [] for k in keys}}
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        yw = yw_te[s]
        nw = int(yw.sum())
        if not nw:
            continue
        rj = ((flag_te[s] == 1) & (yw == 1)).sum() / nw
        boot["judge"].append(rj)
        for k in keys:
            r = float(np.sum((wrong_te[k][s] >= thresholds[k]) & (yw == 1)) / nw)
            boot[f"d_{k}"].append(r - rj)
    out["bootstrap"] = {k: {"mean": float(np.mean(v)),
                            "lo": float(np.percentile(v, 2.5)),
                            "hi": float(np.percentile(v, 97.5))}
                        for k, v in boot.items() if v}
    log(f"  judge recall = {out['judge_rec']:.1%} [{j_lo:.1%}, {j_hi:.1%}] (Wilson)")
    for k in wrong_te:
        d = out["error_detection"][k]
        if d["recall"] is None:
            log(f"  {k}: recall = N/A (no operating point)")
            continue
        log(f"  {k}: recall = {d['recall']:.1%} [{d['recall_lo']:.1%}, {d['recall_hi']:.1%}] "
            f"(Wilson), realised test precision = {_pct_or_dash(d['precision'])}")
    for k in keys:
        b = out["bootstrap"][f"d_{k}"]
        sig = "excludes 0" if b["lo"] > 0 else ("excludes 0 (negative)" if b["hi"] < 0 else "n.s.")
        log(f"  d {k} - judge = {b['mean']:+.1%} [{b['lo']:+.1%}, {b['hi']:+.1%}] -> {sig}")
    log(f"done {tag} ({time.time() - t0:.0f}s)")
    return out


def style():
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def run_config_digest(tag: str, cfg: dict) -> str:
    """Fingerprint of everything about a run that decides its cached numbers.

    `metrics_{tag}.json` is keyed by tag alone, so reusing a tag for a different run —
    repointing it at another checkpoint's directories, correcting a head_dim, renaming
    the title — leaves a cache that validates and plots as if it were current. Only
    label_policy and metrics_version were ever compared, and neither moves when the
    configuration does.

    TOPK and SEED are in here because they change the numbers; METRICS_VERSION is not,
    because it is checked separately with a message of its own.
    """
    payload = json.dumps({
        "digest_version": CONFIG_DIGEST_VERSION,
        "tag": tag,
        # `title` is deliberately absent: it is display text and cannot change any
        # computed value, so including it would force a full recompute for a relabel.
        # The paths, dataset and head_dim below are what prove the entry still points
        # at the same run.
        "head_dim": cfg["head_dim"],
        "dataset": cfg.get("dataset", "gsm8k"),
        "train": str(cfg["train"]),
        "test": str(cfg["test"]),
        "topk": TOPK,
        "seed": SEED,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def selection_slug(tags) -> str:
    """Filename component identifying WHICH runs a figure was built from.

    Without it every selection writes the same `fig_risk_coverage.pdf`, so plotting
    two runs and then re-plotting one silently replaces a three-panel figure with a
    one-panel figure of the same name — and a paper draft referencing it picks up
    whichever came last. Order is preserved because it is the panel order.
    """
    joined = "+".join(tags)
    if len(joined) <= 48:
        return joined
    # Long selections would produce unusable filenames; hash the exact list so two
    # different long selections still differ.
    digest = hashlib.sha256("\0".join(tags).encode()).hexdigest()[:8]
    return f"{len(tags)}runs-{digest}"


def save(fig, name, tags):
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{name}__{selection_slug(tags)}"
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    log(f"saved {OUT / stem}.pdf/.png")


def fig_risk_coverage(M):
    # Width scales with the panel count: five models in a fixed 7in figure squeezed
    # every panel to illegibility.
    fig, axes = plt.subplots(1, len(M), figsize=(max(2.6 * len(M), 3.2), 2.6),
                             sharex=True, squeeze=False)
    axes = axes[0]
    cov_min = 20.0
    for ax, (tag, d) in zip(axes, M.items()):
        ymax = 0.0
        series = [("lr", C_LR), ("mlp", C_MLP)]
        for m, c in series:
            for variant, lw, ls, alpha in (("controlled", 1.4, "-", 1.0), ("raw", 0.9, "--", 0.45)):
                rc = d["risk_coverage"][f"{m}_{variant}"]
                cov = np.array(rc["cov"]) * 100
                risk = np.array(rc["risk"]) * 100
                keep = cov >= cov_min
                lab = PROBE_LABEL[m] if variant == "controlled" else None
                ax.plot(cov[keep], risk[keep], color=c, lw=lw, ls=ls, alpha=alpha, label=lab)
                ymax = max(ymax, risk[keep].max())
        if "shallow" in d.get("risk_coverage", {}):
            rc = d["risk_coverage"]["shallow"]
            cov = np.array(rc["cov"]) * 100
            risk = np.array(rc["risk"]) * 100
            keep = cov >= cov_min
            ax.plot(cov[keep], risk[keep], color=C_SHALLOW, lw=1.2, ls=":",
                    label=SHALLOW_LABEL)
            ymax = max(ymax, risk[keep].max())
        if d.get("has_p_yes") and "p_yes" in d["risk_coverage"]:
            rc = d["risk_coverage"]["p_yes"]
            cov = np.array(rc["cov"]) * 100
            risk = np.array(rc["risk"]) * 100
            keep = cov >= cov_min
            ax.plot(cov[keep], risk[keep], color=C_JUDGE, lw=1.2, ls="-.",
                    label="judge $p_{\\mathrm{YES}}$")
            ymax = max(ymax, risk[keep].max())
        # Accepting everything gives exactly the wrong-answer rate — the same
        # quantity the curves plot, so the curve meets this line at 100% coverage.
        full_err = d["wrong_test"] / d["n_test"]
        ymax = max(ymax, full_err * 100)
        ax.axhline(full_err * 100, color=C_GRAY, ls=":", lw=1.0)
        # inverted x-axis: cov_min is the RIGHT edge, so anchor the label left of it
        ax.text(cov_min + 2, full_err * 100 + 0.02 * ymax,
                f"no filtering {full_err * 100:.1f}%",
                fontsize=6.5, color=C_GRAY, ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))
        auc_note = (f"AUC: LR {d['auc_lr_controlled']:.2f}\n"
                    f"MLP {d['auc_mlp_controlled']:.2f}")
        if d.get("auc_shallow") is not None:
            auc_note += f"\nshallow {d['auc_shallow']:.2f}"
        if d.get("auc_p_yes") is not None:
            auc_note += f"\n$p_{{\\mathrm{{YES}}}}$ {d['auc_p_yes']:.2f}"
        ax.text(0.03, 0.97, auc_note, transform=ax.transAxes, ha="left", va="top",
                fontsize=6.5, linespacing=1.25,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6))
        ax.set_title(f"{panel_label(tag, d)}\n({d['wrong_test']}/{d['n_test']} wrong, "
                     f"{d['wrong_test'] / d['n_test']:.0%})", fontsize=7.5)
        ax.set_xlim(100, cov_min)
        ax.set_ylim(0, ymax * 1.45)
        ax.grid(True)
    axes[0].set_ylabel("Risk: wrong among accepted (%)")
    fig.supxlabel("Coverage (% auto-accepted)", fontsize=8, y=-0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    # One row: at ncols=3 the fourth entry wrapped onto a second line and overran the
    # x-axis label. Anchored below it, not above.
    fig.legend(handles, labels, ncols=len(labels), loc="lower center", frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.19), columnspacing=1.4)
    d0 = next(iter(M.values()))
    # The dataset is per-panel: a MATH run can be selected alone or mixed into the same
    # figure as a GSM8K one, so naming one of them in the caption is a claim the figure
    # cannot support. Name every dataset actually plotted, and drop the sample sizes
    # when the panels do not share them.
    datasets = sorted({str(d.get("dataset", "gsm8k")).upper() for d in M.values()})
    same_n = len({(d["n_train"], d["n_test"]) for d in M.values()}) == 1
    fit_on = (f"Fit on {d0['n_train']:,} {'/'.join(datasets)} train; evaluated on "
              f"{d0['n_test']:,} held-out test." if same_n else
              f"Fit on {'/'.join(datasets)} train; evaluated on held-out test "
              f"(sample sizes differ per panel).")
    fig.text(0.5, -0.23,
             f"Solid: shallow-controlled probe.  Dashed: raw probe.  {fit_on}",
             ha="center", fontsize=6.5, color=C_GRAY)
    fig.tight_layout()
    save(fig, "fig_risk_coverage", list(M))
    plt.close(fig)


def fig_pr(M):
    fig, axes = plt.subplots(1, len(M), figsize=(max(2.4 * len(M), 3.0), 2.3),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (tag, d) in zip(axes, M.items()):
        if "shallow" in d.get("pr_curves", {}):
            pr = d["pr_curves"]["shallow"]
            ax.plot(pr["rec"], pr["prec"], color=C_SHALLOW, lw=1.1, ls=":",
                    label=SHALLOW_LABEL)
        for m, c in (("lr", C_LR), ("mlp", C_MLP)):
            pr = d["pr_curves"][f"{m}_controlled"]
            ax.plot(pr["rec"], pr["prec"], color=c, lw=1.4, label=PROBE_LABEL[m])
        if d.get("has_p_yes") and "p_yes" in d["pr_curves"]:
            pr = d["pr_curves"]["p_yes"]
            ax.plot(pr["rec"], pr["prec"], color=C_PYES, lw=1.2, ls="-.",
                    label="judge $p_{\\mathrm{YES}}$")
        ax.scatter([d["judge_rec"]], [d["judge_prec"]], color=C_JUDGE, marker="*", s=70,
                   zorder=5, label="verbalized verdict")
        ap_lr = d["error_detection"]["lr_controlled"]["ap"]
        ap_mlp = d["error_detection"]["mlp_controlled"]["ap"]
        ax.text(0.97, 0.985, f"AP: LR {ap_lr:.2f}\nMLP {ap_mlp:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
        prev = d["wrong_test"] / d["n_test"]
        ax.axhline(prev, color=C_GRAY, ls=":", lw=1.0)
        ax.text(0.02, prev + 0.025, f"chance {prev:.0%}", fontsize=6.5, color=C_GRAY,
                ha="left", va="bottom")
        ax.set_title(panel_label(tag, d))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True)
    axes[0].set_ylabel("Precision")
    fig.supxlabel("Recall (wrong solutions flagged)", fontsize=8, y=-0.04)
    handles, labels = axes[0].get_legend_handles_labels()
    # One row, same reason as fig_risk_coverage: this figure has FIVE entries, so the
    # hardcoded ncols=4 wrapped the fifth onto a second line that printed straight
    # through the x-axis label. ncols=len(labels) keeps it one row however many
    # detectors a panel happens to draw.
    fig.legend(handles, labels, ncols=len(labels), loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, -0.19), columnspacing=1.4)
    fig.tight_layout()
    save(fig, "fig_error_detection_pr", list(M))
    plt.close(fig)


def _recall_point(d, key):
    """(recall, lo, hi) in %, or None when the detector has no operating point."""
    if key == "judge":
        return d["judge_rec"] * 100, d["judge_rec_lo"] * 100, d["judge_rec_hi"] * 100
    e = d["error_detection"][key]
    if e["recall"] is None:
        return None
    return e["recall"] * 100, e["recall_lo"] * 100, e["recall_hi"] * 100


def fig_bars(M):
    fig, ax = plt.subplots(figsize=(2.3 * len(M) + 1.0, 3.1))
    tags = list(M)
    show_pyes = all(M[t].get("has_p_yes") for t in tags)
    series = [("judge", "Verbalized verdict", C_JUDGE),
              ("shallow", SHALLOW_LABEL, C_SHALLOW),
              ("lr_controlled", "LR probe", C_LR),
              ("mlp_controlled", "MLP probe", C_MLP)]
    if show_pyes:
        series.append(("p_yes", "Judge $p_{\\mathrm{YES}}$", C_PYES))
    width = 0.8 / len(series)
    x = np.arange(len(tags))
    bar_x = {}
    for j, (key, lab, c) in enumerate(series):
        vals, err, missing = [], [[], []], []
        for i, tag in enumerate(tags):
            pt = _recall_point(M[tag], key)
            if pt is None:
                # No operating point exists. A zero-height bar would read as "caught
                # nothing"; draw nothing and annotate instead.
                vals.append(0.0); err[0].append(0.0); err[1].append(0.0)
                missing.append(i)
                continue
            v, lo, hi = pt
            vals.append(v)
            # Wilson bounds always bracket the point estimate, so no clamping is
            # needed and none is applied — a negative arm here would be a real bug.
            err[0].append(v - lo)
            err[1].append(hi - v)
        pos = x + (j - (len(series) - 1) / 2) * width
        bar_x[key] = pos
        ax.bar(pos, vals, width * 0.92, color=c, label=lab,
               yerr=np.array(err), capsize=2, error_kw=dict(ecolor="#333", lw=0.8))
        ink = "#1a1a1a" if key in ("mlp_controlled",) else "white"
        for i, (p, v) in enumerate(zip(pos, vals)):
            if i in missing:
                ax.text(p, 2.5, "n/a", ha="center", fontsize=5.5, color=C_GRAY,
                        rotation=90, va="bottom")
                continue
            ax.text(p, 2.5, f"{v:.0f}", ha="center", fontsize=6.0, color=ink,
                    fontweight="bold")
    # Realised test precision of each rule, so the reader can check the recalls being
    # compared are held at comparable precision. The threshold is picked on train, so
    # the test precisions can drift apart and that drift must be visible.
    # Realised TEST precision, written under the bar it belongs to rather than
    # concatenated into one per-group string: with five series that string ran wider
    # than its group and collided with both its neighbour and the tick labels.
    # Thresholds are calibrated on train, so these values differ across rules and the
    # recalls above are NOT measured at a common precision.
    for i, tag in enumerate(tags):
        d = M[tag]
        for key, _lab, _c in series:
            pr = (d["judge_prec"] if key == "judge"
                  else d["error_detection"][key]["precision"])
            txt = "n/a" if pr is None else f"{pr * 100:.0f}"
            ax.text(bar_x[key][i], -3.5, txt, ha="center", va="top", fontsize=5.4,
                    color=C_GRAY, clip_on=False)
    ax.text(-0.02, -3.5, "realised\ntest P%:", ha="right", va="top", fontsize=5.4,
            color=C_GRAY, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_xticks(x)
    ax.set_xticklabels([panel_label(t, M[t]) for t in tags])
    ax.tick_params(axis="x", pad=14)      # clear the realised-precision row above
    ax.set_ylabel("Recall of wrong solutions (%)\nat a TRAIN-CALIBRATED operating point")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="upper center", ncols=len(series),
              bbox_to_anchor=(0.5, 1.16), columnspacing=0.9, handlelength=1.2)
    fig.text(0.5, -0.10,
             "Thresholds calibrated on TRAIN at the judge's train precision, then "
             "applied unchanged to TEST.\nPrecision is NOT matched on test: the "
             "realised test precision under each group differs across rules, so these "
             "recalls\nare not measured at a common precision. AUC and risk-coverage "
             "are the primary comparisons.",
             ha="center", va="top", fontsize=5.6, color=C_GRAY)
    save(fig, "fig_recall_at_train_calibrated_op", list(M))
    plt.close(fig)


def _pct(v, digits=0):
    """Percent cell, or an em-dash when the quantity does not exist."""
    return "--" if v is None else f"{v * 100:.{digits}f}"


def table_tex(M):
    show_pyes = all(d.get("auc_p_yes") is not None for d in M.values())
    # Prefer the train-resample sd (sampling + head selection + init) over the
    # init-only seed sd; state which one the +/- is in the caption note below.
    resample_sds = [d.get("auc_mlp_controlled_resample_sd") for d in M.values()]
    use_resample = all(s is not None for s in resample_sds)
    rows = []
    for tag, d in M.items():
        # The point estimate is the ENSEMBLE's AUC. The +/- is the train-resample sd
        # (sampling + head selection + init) and is named as such in the caption; it is
        # deliberately never the initialisation-only spread, and the two are never
        # blended into one conventional mean +/- sd.
        sd = d.get("auc_mlp_controlled_resample_sd") if use_resample else None
        mlp_ctrl = (f"{d['auc_mlp_controlled']:.2f}"
                    + (f"\\,$\\pm$\\,{sd:.2f}" if sd else ""))
        ed = d["error_detection"]["mlp_controlled"]
        cells = [
            panel_label(tag, d),
            f"{d['wrong_test'] / d['n_test'] * 100:.1f}",
            f"{d['judge_verdict_balanced_acc']:.2f}",
        ]
        if show_pyes:
            cells.append(f"{d['auc_p_yes']:.2f}")
        # The surface-feature baseline sits immediately before the probes: it is what
        # `controlled` subtracts, so the probe columns are only interpretable next to it.
        cells.append(f"{d['auc_shallow']:.2f}" if d.get("auc_shallow") is not None
                     else "--")
        cells += [
            f"{d['auc_lr_raw']:.2f}", f"{d['auc_lr_controlled']:.2f}",
            f"{d['auc_mlp_raw']:.2f}", mlp_ctrl,
            # Recall columns carry their realised test precision, so a reader can see
            # whether the two recalls are being compared at comparable precision.
            f"{_pct(d['judge_rec'])} ({_pct(d['judge_prec'])})",
            f"{_pct(ed['recall'])} ({_pct(ed['precision'])})",
        ]
        rows.append(" & ".join(cells) + r" \\")
    body = "\n".join(rows)
    # AUC block = verdict bal.acc [+ p_YES] + Shallow + {LR, LR-ctrl, MLP, MLP-ctrl}
    n_auc = 7 if show_pyes else 6
    # + model name, %wrong, and the two recall columns
    ncols = 4 + n_auc
    assert ncols == len(rows[0].split("&")), "tabular spec does not match row width"
    pyes_head = " & $p_{\\mathrm{YES}}$" if show_pyes else ""
    sd_note = ("the TRAIN-RESAMPLE sd (sampling + head selection + init), from "
               f"{next(iter(M.values())).get('auc_mlp_controlled_resample_n')} train "
               "resamples"
               if use_resample else "UNAVAILABLE (run with --n_train_resamples > 0)")
    d0_ = next(iter(M.values()))
    init_note = "; ".join(
        f"{panel_label(t, dd)}: {dd.get('auc_mlp_controlled_seed_sd') or 0:.3f}"
        for t, dd in M.items())
    tex = (
        "% Auto-generated by filtering/make_paper_figures.py -- do not edit by hand.\n"
        "% 'Verdict bal.acc' is the binary verdict's balanced accuracy, NOT an AUC:\n"
        "%   a binary score cannot reach the AUC a continuous score can, so it is not\n"
        "%   comparable to the probe columns. p_YES is the continuous black-box baseline.\n"
        "% Recall columns use a TRAIN-CALIBRATED operating point: the threshold is\n"
        "%   chosen on TRAIN at the judge's train precision and applied unchanged to\n"
        "%   TEST. This does NOT match precision on test -- the parenthesised number is\n"
        "%   the precision each rule ACTUALLY REALISES on test, and these differ across\n"
        "%   rules. Recalls measured at different realised precisions are not directly\n"
        "%   comparable; AUC and risk-coverage are the primary comparisons.\n"
        "%   '--' = the train precision target was unreachable, so no operating point\n"
        "%   exists (NOT zero recall).\n"
        f"% Probe columns are the ENSEMBLE over {d0_.get('n_seeds_mlp', 1)} MLP\n"
        "%   initialisations (probabilities averaged); curves, operating points and\n"
        "%   these values all use that same aggregation.\n"
        f"% The MLP-ctrl +/- is {sd_note}.\n"
        "% SEPARATELY, the initialisation-only spread of a single arbitrary draw is\n"
        f"%   {init_note}. It is NOT an error bar on the ensemble and the two\n"
        "%   uncertainty sources are not combined.\n"
        f"\\begin{{tabular}}{{l{'c' * (ncols - 1)}}}\n\\toprule\n"
        f" & & \\multicolumn{{{n_auc}}}{{c}}{{Correctness AUC (held-out test)}} & "
        "\\multicolumn{2}{c}{Recall \\% (realised test P\\%) @ train-calibrated op.} \\\\\n"
        f"\\cmidrule(lr){{3-{2 + n_auc}}}\\cmidrule(lr){{{3 + n_auc}-{ncols}}}\n"
        "Judge model & \\%wrong & Verdict bal.acc"
        f"{pyes_head} & Shallow & LR & LR-ctrl & MLP & MLP-ctrl & verdict & MLP-ctrl \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"table_probe_auc__{selection_slug(list(M))}.tex"
    (OUT / name).write_text(tex)
    log(f"saved {OUT / name}")


def missing_inputs(tag, cfg):
    """Artifacts `--compute` needs for `tag` that are not on disk."""
    dataset = cfg.get("dataset", "gsm8k")
    return [str(p) for p in (cfg["train"] / "hidden_rich.npz",
                             cfg["train"] / dataset / "raw.jsonl",
                             cfg["test"] / "hidden_rich.npz",
                             cfg["test"] / dataset / "raw.jsonl")
            if not p.exists()]


def parse_tag_list(requested: str) -> list[str]:
    """Split a --runs value into tags: whitespace-tolerant, de-duplicated, ordered.

    `"a, b"` used to fail with `unknown run tag: ' b'`, which reads as a typo in the
    tag rather than in the separator, and `"a,a"` used to plot the same run twice as
    two panels. Order is preserved because it is the panel order on the figure.
    """
    seen, tags = set(), []
    for part in requested.split(","):
        tag = part.strip()
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def resolve_tags(requested, compute):
    """Which runs to act on. The selection is always explicit, never inferred.

    RUNS is a registry of every run the paper has ever included, so on any one machine
    most entries have no data. Selecting "whatever happens to be on disk" made the
    output set a property of the machine: the same command produced a three-panel
    figure on one box and a one-panel figure on another, both named the same thing,
    with nothing in either recording which runs went in.

    So a selection must be named:
      --runs a,b     exactly those, fatal if any lacks data
      --runs all     every registered tag, fatal if any lacks data (machine-independent)
      --runs present only those with data — still machine-dependent, but now a choice
                     someone made rather than a default they never saw
    """
    need = "activation dumps" if compute else f"cached metrics in {OUT}"
    available = {t: (not missing_inputs(t, c) if compute
                     else (OUT / f"metrics_{t}.json").exists())
                 for t, c in RUNS.items()}

    def absent_detail(tags):
        return "\n".join(
            f"  {t}: " + (", ".join(missing_inputs(t, RUNS[t])) if compute
                          else f"{OUT / f'metrics_{t}.json'} (run --compute first)")
            for t in tags)

    if not requested:
        have = [t for t in RUNS if available[t]]
        raise SystemExit(
            f"--runs is required: name the runs to include, so the output is the same "
            f"on every machine.\n"
            f"        --runs all      every registered tag ({len(RUNS)}); fatal if any "
            f"lacks {need}\n"
            f"        --runs present  only the {len(have)} tag(s) with {need} here: "
            f"{', '.join(have) or '(none)'}\n"
            f"        --runs a,b      exactly those\n"
            f"        registered: {', '.join(RUNS)}")

    if requested == "present":
        tags = [t for t in RUNS if available[t]]
        if not tags:
            raise SystemExit(
                f"--runs present selected nothing: no registered run has {need}. Either "
                f"generate a run (see README) or add its entry to RUNS in "
                f"{Path(__file__).name}; known tags: {', '.join(RUNS)}")
        skipped = [t for t in RUNS if not available[t]]
        if skipped:
            log(f"--runs present: skipping {len(skipped)} registered tag(s) with no "
                f"{need}: {', '.join(skipped)}")
        log(f"runs: {', '.join(tags)}")
        return tags

    tags = list(RUNS) if requested == "all" else parse_tag_list(requested)
    unknown = [t for t in tags if t not in RUNS]
    if unknown:
        raise SystemExit(f"unknown run tag(s): {', '.join(unknown)}; "
                         f"known: {', '.join(RUNS)}")
    if not tags:
        raise SystemExit(f"--runs {requested!r} names no tags; "
                         f"known: {', '.join(RUNS)}")
    absent = [t for t in tags if not available[t]]
    if absent:
        raise SystemExit(f"no {need} for requested tag(s) "
                         f"{', '.join(absent)}:\n{absent_detail(absent)}")
    log(f"runs: {', '.join(tags)}")
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", action="store_true", help="recompute metrics from raw artifacts")
    ap.add_argument("--runs", default=None,
                    help="which runs to include: comma-separated tags from RUNS, or "
                         "'all' (every registered tag) or 'present' (only those with "
                         "data on this machine). Required — there is no default, so the "
                         "same command yields the same figure everywhere. Naming a tag "
                         "with no data is fatal rather than a skip.")
    ap.add_argument("--models", default=None,
                    help=argparse.SUPPRESS)   # deprecated alias for --runs
    ap.add_argument("--allow_unauthenticated", action="store_true",
                    help="plot runs whose dumps predate per-record fingerprints "
                         "(rec_sha) or the generation identity (gen_run). Those files "
                         "cannot be verified to come from the raw records beside them; "
                         "without this flag they are refused rather than trusted.")
    ap.add_argument("--n_train_resamples", type=int, default=N_TRAIN_RESAMPLES,
                    help="train bootstrap replicates, each redoing head selection, to "
                         "estimate the probe's real spread (default "
                         f"{N_TRAIN_RESAMPLES}). Dominates runtime; 0 skips it and "
                         "falls back to the init-only seed sd.")
    args = ap.parse_args()
    if args.models is not None:
        if args.runs is not None:
            ap.error("--models is the old name for --runs; pass only one")
        log("[WARN] --models is deprecated: the registry holds runs (a model *and* a "
            "dataset and its paths), not models. Use --runs.")
        args.runs = args.models
    tags = resolve_tags(args.runs, args.compute)

    if args.compute:
        OUT.mkdir(parents=True, exist_ok=True)
        for tag in tags:
            res = compute_run(tag, RUNS[tag], n_train_resamples=args.n_train_resamples)
            (OUT / f"metrics_{tag}.json").write_text(json.dumps(res))
            log(f"cached {OUT / f'metrics_{tag}.json'}")

    M = {}
    for tag in tags:
        p = OUT / f"metrics_{tag}.json"
        if not p.exists():
            raise SystemExit(f"missing {p}; run with --compute first")
        M[tag] = json.loads(p.read_text())
        # Compare against the policy CURRENTLY in force for that run's dataset, not
        # against the GSM8K constant — otherwise every MATH run reads as stale.
        expected_policy = label_policy_for(M[tag].get("dataset"))
        if M[tag].get("label_policy") != expected_policy:
            raise SystemExit(f"stale label policy in {p}; rerun with --compute")
        if M[tag].get("metrics_version") != METRICS_VERSION:
            raise SystemExit(
                f"{p} was written by metrics_version={M[tag].get('metrics_version')}, "
                f"current is {METRICS_VERSION} (risk-coverage and operating-point "
                f"definitions changed). Rerun with --compute.")
        expected_sha = run_config_digest(tag, RUNS[tag])
        stored_sha = M[tag].get("config_sha")
        if stored_sha is None:
            raise SystemExit(
                f"{p} predates config fingerprinting, so nothing verifies it was "
                f"computed from the run {tag!r} now names. Rerun with --compute.")
        stored_dv = M[tag].get("config_digest_version", 1)
        if stored_dv != CONFIG_DIGEST_VERSION:
            raise SystemExit(
                f"{p} stores config_sha under digest_version={stored_dv}; this build "
                f"computes v{CONFIG_DIGEST_VERSION}. The stored fingerprint cannot be "
                f"compared against the current formula.\n"
                f"        This is a definition change, not a configuration mismatch — "
                f"the cached numbers are still valid, but nothing here can prove it. "
                f"Rerun with --compute.")
        if stored_sha != expected_sha:
            raise SystemExit(
                f"{p} was computed from a different configuration than RUNS[{tag!r}] "
                f"now describes (cached title={M[tag].get('title')!r} "
                f"head_dim={M[tag].get('head_dim')!r} dataset={M[tag].get('dataset')!r}; "
                f"registry title={RUNS[tag]['title']!r} "
                f"head_dim={RUNS[tag]['head_dim']!r} "
                f"dataset={RUNS[tag].get('dataset', 'gsm8k')!r}).\n"
                f"        The cache is keyed by tag alone, so a reused tag would plot "
                f"the old run's numbers under the new run's name. Rerun with --compute.")

    unverifiable = {t: m["unauthenticated"] for t, m in M.items()
                    if any((m.get("unauthenticated") or {}).values())}
    if unverifiable and not args.allow_unauthenticated:
        detail = "\n".join(
            f"  {t}: train missing {', '.join(v['train']) or '-'}; "
            f"test missing {', '.join(v['test']) or '-'}"
            for t, v in unverifiable.items())
        raise SystemExit(
            f"these runs' artifacts predate provenance recording, so nothing verifies "
            f"the activations were read from the records beside them:\n{detail}\n"
            f"        Re-extract them (the fields cannot be added after the fact), or "
            f"pass --allow_unauthenticated to plot them anyway.")
    if unverifiable:
        for t, v in unverifiable.items():
            log(f"[WARN] {t}: plotting unverifiable artifacts on your say-so "
                f"(--allow_unauthenticated); train missing "
                f"{', '.join(v['train']) or '-'}, test missing "
                f"{', '.join(v['test']) or '-'}")

    style()
    fig_risk_coverage(M)
    fig_pr(M)
    fig_bars(M)
    table_tex(M)


if __name__ == "__main__":
    main()
