#!/usr/bin/env python3
"""Does the GSM8K probe-vs-judge result survive re-instantiating every instance?

GSM-Symbolic rebuilds 100 GSM8K test questions from their templates with fresh names
and numbers. The reasoning is the same; the exact string was never in a pretraining
corpus. So a detector that only works because the model has seen these instances
should degrade here, and one that reads the model's own error state should not.

Two rules make the comparison mean anything.

Everything is fit on the REAL GSM8K TRAIN split -- head selection, the shallow
residualiser, the probes. No GSM-Symbolic label is used to fit anything. The
perturbed set is only ever scored.

And the baseline is the MATCHED originals, not all of GSM8K. The 100 templates are
not a random sample of the 1,319 test questions -- for one model here they are 3
points easier -- so comparing against the full split confounds perturbation with
template selection. Every original is located by exact `original_question` text, and
a missing, duplicated or ambiguous match is fatal rather than dropped.

That matched baseline carries the ACCURACY comparison, which is paired and needs only
100 templates. It cannot carry the AUC comparison: one model gets 95 of those 100
right, so its matched AUC rests on five wrong answers. Both deltas are reported --
`delta_symbolic_minus_matched` (unconfounded, unusably noisy for AUC) and
`delta_symbolic_minus_full` (against the 1,319-question test split, with a two-sample
interval) -- and the figure plots the latter.

Uncertainty is bootstrapped over TEMPLATES, not rows. The ~1,300 variants are 100
templates x ~13 instances that share their reasoning, so resampling rows would
treat correlated items as independent and report intervals several times too narrow.

  python filtering/symbolic_compare.py --models qwen25_7b,llama31_8b,gemma3
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probe_models import (fit_ensemble, load_split, prepare_head_features,
                          shallow_only_scores)

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"qwen25_7b": ("Qwen/Qwen2.5-7B-Instruct", 128),
          "llama31_8b": ("meta-llama/Llama-3.1-8B-Instruct", 128),
          "gemma3": ("google/gemma-3-12b-it", 256)}
DETECTORS = ["mlp_raw", "mlp_controlled", "lr_raw", "lr_controlled", "shallow",
             "p_yes", "verdict"]


def template_map():
    """perturbed question -> original question, with the mapping validated.

    A silent mismatch here would compare a model against the wrong baseline, so the
    structure GSM-Symbolic promises (each perturbed row names exactly one original)
    is asserted rather than assumed.
    """
    from datasets import load_dataset
    ds = load_dataset("apple/GSM-Symbolic", "main", split="test")
    q2o, o2ids = {}, {}
    for r in ds:
        q, o = r["question"].strip(), r["original_question"].strip()
        if q in q2o and q2o[q] != o:
            raise SystemExit(f"[FATAL] perturbed question maps to two originals: {q[:80]!r}")
        q2o[q] = o
        o2ids.setdefault(o, set()).add(r["original_id"])
    ambiguous = {o: ids for o, ids in o2ids.items() if len(ids) > 1}
    if ambiguous:
        raise SystemExit(f"[FATAL] {len(ambiguous)} original question(s) carry several "
                         f"original_ids; the template identity is not well defined.")
    return q2o


def auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, np.asarray(s, dtype=float)))


def cluster_boot(y, s_a, s_b, groups, n=2000, seed=0):
    """95% CI for AUC(a)-AUC(b), resampling TEMPLATES with replacement."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s_a = np.asarray(s_a, float); s_b = np.asarray(s_b, float)
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx]))
    if not out:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def two_sample_delta_boot(y_s, sc_s, groups_s, y_g, sc_g, n=2000, seed=0,
                          ref_s=None, ref_g=None):
    """95% CI for AUC(perturbed) - AUC(GSM8K test), two independent samples.

    The perturbed side is resampled by TEMPLATE (its rows are ~13 instances of the
    same reasoning); the GSM8K side is resampled by row, where each row is its own
    question. The two draws are independent because the samples are, so the
    difference distribution is just the difference of the two resampled statistics.

    With `ref_s`/`ref_g`, each side's statistic becomes a MARGIN, AUC(score) minus
    AUC(reference) on the same rows, and the result is a difference in differences:
    "did this detector lose more than that one did, going from GSM8K to perturbed?"
    That is a far sharper question than either drop on its own, because the two
    detectors share every resampled row and most of the sampling noise cancels.
    """
    rng = np.random.default_rng(seed)
    y_s = np.asarray(y_s); sc_s = np.asarray(sc_s, float)
    y_g = np.asarray(y_g); sc_g = np.asarray(sc_g, float)
    r_s = None if ref_s is None else np.asarray(ref_s, float)
    r_g = None if ref_g is None else np.asarray(ref_g, float)
    uniq = np.unique(groups_s)
    idx_by_g = {g: np.flatnonzero(groups_s == g) for g in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        i_s = np.concatenate([idx_by_g[g] for g in pick])
        i_g = rng.integers(0, len(y_g), len(y_g))
        if len(np.unique(y_s[i_s])) < 2 or len(np.unique(y_g[i_g])) < 2:
            continue
        a = roc_auc_score(y_s[i_s], sc_s[i_s])
        b = roc_auc_score(y_g[i_g], sc_g[i_g])
        if r_s is not None:
            a -= roc_auc_score(y_s[i_s], r_s[i_s])
            b -= roc_auc_score(y_g[i_g], r_g[i_g])
        out.append(a - b)
    if not out:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def score_all(tr, te, topk, n_seeds):
    """Every detector, all fit on `tr` (GSM8K train) and applied to `te`."""
    Ftr, Fte, _, _ = prepare_head_features(tr.Z, tr.y, te.Z, topk)
    s = {}
    for m in ("lr", "mlp"):
        # One ensemble per probe, used for every downstream number, exactly as
        # make_paper_figures does -- averaged probabilities, not averaged AUCs.
        e, _, _ = fit_ensemble(Ftr, tr.y, Fte, tr.S, te.S, model_type=m, seed=0,
                               n_seeds=n_seeds)
        s[f"{m}_raw"], s[f"{m}_controlled"] = e["raw"], e["controlled"]
    _, s["shallow"] = shallow_only_scores(tr.S, tr.y, te.S, seed=0)
    s["p_yes"] = te.p_yes
    s["verdict"] = te.yes.astype(float)
    return s


def population(split, raw_by_idx):
    trunc = sum(1 for i in split.idx if raw_by_idx[i].get("truncated"))
    return {"n": int(len(split.y)), "wrong": int((split.y == 0).sum()),
            "prevalence": float((split.y == 0).mean()),
            "truncated_in_analysed": int(trunc),
            "n_dropped_unlabelable": split.info["n_dropped_unlabelable"],
            "exclusions": split.info["exclusions"],
            "p_yes_check": split.info.get("p_yes_check"),
            "label_policy": split.info["policy"]}


def read_raw(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["idx"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen25_7b,llama31_8b,gemma3")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default="filtering/figures/paper/symbolic_compare.json")
    args = ap.parse_args()

    print("validating template map ...", flush=True)
    q2o = template_map()
    print(f"  {len(q2o)} perturbed questions -> {len(set(q2o.values()))} originals",
          flush=True)

    report = {"topk": args.topk, "n_seeds": args.n_seeds, "n_boot": args.n_boot,
              "aggregation": "ensemble-mean-of-probabilities", "models": {}}

    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        model_id, head_dim = MODELS[key]
        t0 = time.time()
        print(f"=== {key} ===", flush=True)

        tr = load_split(str(ROOT / f"results_{key}_train" / "hidden_rich.npz"),
                        str(ROOT / f"results_{key}_train" / "gsm8k" / "raw.jsonl"), head_dim)
        g_te = load_split(str(ROOT / f"results_{key}_test" / "hidden_rich.npz"),
                          str(ROOT / f"results_{key}_test" / "gsm8k" / "raw.jsonl"), head_dim)
        s_dir = ROOT / f"results_{key}_symbolic_test"
        s_te = load_split(str(s_dir / "hidden_rich.npz"),
                          str(s_dir / "gsm_symbolic" / "raw.jsonl"), head_dim)

        # --- provenance: same checkpoint and geometry across all three dumps -------
        metas = [s.info["dump_meta"] for s in (tr, g_te, s_te)]
        prov = {"model": {m.get("model") for m in metas},
                "head_dim": {m.get("head_dim") for m in metas},
                "n_heads": {m.get("n_heads") for m in metas},
                "n_layers": {m.get("n_layers") for m in metas}}
        for field, vals in prov.items():
            if len(vals) != 1:
                raise SystemExit(f"[FATAL] {key}: dumps disagree on {field}: {vals}")
        if metas[0].get("model") != model_id:
            raise SystemExit(f"[FATAL] {key}: dumps are from {metas[0].get('model')!r}, "
                             f"expected {model_id!r}")
        heads = {s.Z.shape[1] for s in (tr, g_te, s_te)}
        if len(heads) != 1:
            raise SystemExit(f"[FATAL] {key}: head axis differs across dumps: {heads}")
        print(f"  provenance OK: {model_id}, head_dim={metas[0].get('head_dim')}, "
              f"{heads.pop()} heads", flush=True)

        g_raw, s_raw = (read_raw(ROOT / f"results_{key}_test" / "gsm8k" / "raw.jsonl"),
                        read_raw(s_dir / "gsm_symbolic" / "raw.jsonl"))

        # --- match perturbed rows to their original, and originals to the GSM8K run -
        g_by_text = {}
        for i, r in g_raw.items():
            g_by_text.setdefault(r["problem"].strip(), []).append(i)
        dup = {t: v for t, v in g_by_text.items() if len(v) > 1}
        if dup:
            raise SystemExit(f"[FATAL] {key}: {len(dup)} GSM8K question(s) appear more "
                             f"than once in the test run; the match is ambiguous.")

        tmpl = np.array([q2o.get(s_raw[i]["problem"].strip(), "") for i in s_te.idx])
        unmatched = int((tmpl == "").sum())
        if unmatched:
            raise SystemExit(f"[FATAL] {key}: {unmatched}/{len(tmpl)} perturbed records "
                             f"have no original_question in the template map.")
        wanted = sorted(set(tmpl))
        missing = [o for o in wanted if o not in g_by_text]
        if missing:
            raise SystemExit(f"[FATAL] {key}: {len(missing)}/{len(wanted)} originals are "
                             f"absent from the GSM8K test run; the matched baseline "
                             f"would be a different question set.")
        # Analysed originals: present in the GSM8K dump (i.e. labelable there too).
        g_pos = {ix: j for j, ix in enumerate(g_te.idx)}
        matched_rows = [(o, g_pos[g_by_text[o][0]]) for o in wanted
                        if g_by_text[o][0] in g_pos]
        dropped = len(wanted) - len(matched_rows)
        m_idx = np.array([j for _, j in matched_rows])
        print(f"  templates: {len(wanted)} matched, {dropped} original(s) unlabelable in "
              f"the GSM8K run and excluded from the matched baseline", flush=True)

        # --- score everything, fit on GSM8K train only ----------------------------
        s_g = score_all(tr, g_te, args.topk, args.n_seeds)
        s_s = score_all(tr, s_te, args.topk, args.n_seeds)

        res = {"provenance": {k: list(v)[0] if len(v) == 1 else list(v)
                              for k, v in prov.items()},
               "match": {"n_templates": len(wanted), "n_matched": len(matched_rows),
                         "n_originals_unlabelable": dropped,
                         "n_perturbed_rows": int(len(tmpl)), "n_unmatched_rows": unmatched},
               "population": {"gsm8k_full": population(g_te, g_raw),
                              "gsm8k_matched": {"n": len(m_idx),
                                                "wrong": int((g_te.y[m_idx] == 0).sum()),
                                                "prevalence": float((g_te.y[m_idx] == 0).mean())},
                              "symbolic": population(s_te, s_raw)},
               "auc": {"gsm8k_full": {}, "gsm8k_matched": {}, "symbolic": {}},
               "delta_symbolic_minus_matched": {},
               "delta_symbolic_minus_full": {}, "margin_vs_p_yes": {}}

        for name in DETECTORS:
            res["auc"]["gsm8k_full"][name] = auc(g_te.y, s_g[name])
            res["auc"]["gsm8k_matched"][name] = auc(g_te.y[m_idx],
                                                    np.asarray(s_g[name])[m_idx])
            res["auc"]["symbolic"][name] = auc(s_te.y, s_s[name])
            a, b = res["auc"]["symbolic"][name], res["auc"]["gsm8k_matched"][name]
            res["delta_symbolic_minus_matched"][name] = (
                None if a is None or b is None else a - b)
            # Against the FULL GSM8K test split, which is what the figure plots: the
            # matched originals carry as few as five errors, so their AUC interval is
            # wider than any effect here. The cost is that the 100 templates are not a
            # random sample of the 1,319 questions, so this delta mixes perturbation
            # with template selection -- stated in the caption, and the matched delta
            # above is kept for the reader who wants the unconfounded direction.
            c = res["auc"]["gsm8k_full"][name]
            lo, hi = two_sample_delta_boot(s_te.y, s_s[name], tmpl, g_te.y, s_g[name],
                                           n=args.n_boot)
            res["delta_symbolic_minus_full"][name] = {
                "delta": None if a is None or c is None else a - c,
                "ci_lo": lo, "ci_hi": hi,
                "excludes_zero": lo is not None and (lo > 0 or hi < 0)}

        # Probe-minus-judge margin, the quantity the paper actually claims, with a
        # template-clustered CI on the perturbed set.
        for name in ("mlp_raw", "mlp_controlled", "lr_controlled", "shallow"):
            m_sym = (None if res["auc"]["symbolic"][name] is None
                     or res["auc"]["symbolic"]["p_yes"] is None
                     else res["auc"]["symbolic"][name] - res["auc"]["symbolic"]["p_yes"])
            m_mat = (None if res["auc"]["gsm8k_matched"][name] is None
                     or res["auc"]["gsm8k_matched"]["p_yes"] is None
                     else res["auc"]["gsm8k_matched"][name]
                     - res["auc"]["gsm8k_matched"]["p_yes"])
            lo, hi = cluster_boot(s_te.y, s_s[name], s_s["p_yes"], tmpl,
                                  n=args.n_boot)
            # How much the margin MOVED between the two sets. Each detector's own drop
            # is barely resolvable -- the perturbed side is 100 templates, so its AUC
            # interval is +-0.1 wide -- but the two detectors share every row, so the
            # difference in differences survives what the separate deltas do not. This
            # is the honest form of "the perturbation costs the judge more than the
            # probe": read it, not two overlapping intervals in panel B.
            s_lo, s_hi = two_sample_delta_boot(
                s_te.y, s_s[name], tmpl, g_te.y, s_g[name], n=args.n_boot,
                ref_s=s_s["p_yes"], ref_g=s_g["p_yes"])
            res["margin_vs_p_yes"][name] = {
                "symbolic": m_sym, "gsm8k_matched": m_mat,
                "gsm8k_full": (res["auc"]["gsm8k_full"][name]
                               - res["auc"]["gsm8k_full"]["p_yes"]),
                "symbolic_ci_lo": lo, "symbolic_ci_hi": hi,
                "symbolic_excludes_zero": (lo is not None and (lo > 0 or hi < 0)),
                "shift_vs_gsm8k_full": (None if m_sym is None else
                                        m_sym - (res["auc"]["gsm8k_full"][name]
                                                 - res["auc"]["gsm8k_full"]["p_yes"])),
                "shift_ci_lo": s_lo, "shift_ci_hi": s_hi,
                "shift_excludes_zero": (s_lo is not None and (s_lo > 0 or s_hi < 0))}

        # --- paired accuracy over templates ---------------------------------------
        inst = {}
        for j, ix in enumerate(s_te.idx):
            inst.setdefault(tmpl[j], []).append(float(s_te.y[j] == 1))
        pairs = [(float(g_te.y[j] == 1), float(np.mean(inst[o])))
                 for o, j in matched_rows if o in inst]
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        rng = np.random.default_rng(0)
        boot = np.array([(b[i] - a[i]).mean()
                         for i in (rng.integers(0, len(a), len(a))
                                   for _ in range(args.n_boot))])
        res["accuracy"] = {
            "n_templates_paired": len(pairs),
            "original": float(a.mean()), "perturbed": float(b.mean()),
            "paired_delta": float((b - a).mean()),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5))}

        pop = res["population"]
        print(f"  n: gsm8k {pop['gsm8k_full']['n']} ({pop['gsm8k_full']['prevalence']:.1%} wrong)"
              f" | matched {pop['gsm8k_matched']['n']} ({pop['gsm8k_matched']['prevalence']:.1%})"
              f" | symbolic {pop['symbolic']['n']} ({pop['symbolic']['prevalence']:.1%})",
              flush=True)
        ac = res["accuracy"]
        print(f"  accuracy  original {ac['original']:.1%} -> perturbed {ac['perturbed']:.1%}"
              f"  ({ac['paired_delta']:+.1%}, 95% CI [{ac['ci_lo']:+.1%}, {ac['ci_hi']:+.1%}])",
              flush=True)
        print(f"  {'detector':16s} {'gsm8k_full':>10s} {'matched':>9s} {'symbolic':>9s} "
              f"{'d_match':>8s} {'d_full':>8s} {'d_full 95% CI':>20s}")
        for name in DETECTORS:
            f = lambda v: f"{v:.3f}" if v is not None else "  n/a"
            d = res["delta_symbolic_minus_matched"][name]
            df = res["delta_symbolic_minus_full"][name]
            ci = ("n/a" if df["ci_lo"] is None
                  else f"[{df['ci_lo']:+.3f}, {df['ci_hi']:+.3f}]"
                       + ("*" if df["excludes_zero"] else ""))
            d_s = f"{d:+.3f}" if d is not None else "   n/a"
            df_s = f"{df['delta']:+.3f}" if df["delta"] is not None else "   n/a"
            print(f"  {name:16s} {f(res['auc']['gsm8k_full'][name]):>10s} "
                  f"{f(res['auc']['gsm8k_matched'][name]):>9s} "
                  f"{f(res['auc']['symbolic'][name]):>9s} "
                  f"{d_s:>8s} {df_s:>8s} {ci:>20s}")
        for name, m in res["margin_vs_p_yes"].items():
            ci = ("n/a" if m["symbolic_ci_lo"] is None
                  else f"[{m['symbolic_ci_lo']:+.3f}, {m['symbolic_ci_hi']:+.3f}]")
            sci = ("n/a" if m["shift_ci_lo"] is None
                   else f"[{m['shift_ci_lo']:+.3f}, {m['shift_ci_hi']:+.3f}]"
                        + ("*" if m["shift_excludes_zero"] else ""))
            fm = lambda v: f"{v:+.3f}" if v is not None else " n/a "
            print(f"  margin {name:15s} full {fm(m['gsm8k_full'])}  matched "
                  f"{fm(m['gsm8k_matched'])}  symbolic {fm(m['symbolic'])}  CI {ci}"
                  f"   shift {fm(m['shift_vs_gsm8k_full'])} {sci}")

        report["models"][key] = res
        print(f"  [{time.time()-t0:.0f}s]", flush=True)
        del tr, g_te, s_te
        gc.collect()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
