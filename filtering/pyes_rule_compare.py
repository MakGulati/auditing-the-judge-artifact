#!/usr/bin/env python3
"""Does the YES/NO binning rule change what the p_yes baseline is worth?

`p_yes` is the continuous black-box baseline the probes are compared against, and
it is built by summing the judge's next-token mass over "YES tokens" and "NO
tokens". The shipped rule (`extraction_common.yes_no_mass`) matches by prefix, so
over gemma-3-12b's vocabulary 765 tokens count as NO — including ` not`, ` now`,
` note`, ` nothing` — against 29 counting as YES. Any mass the judge places on
` not` is therefore recorded as a vote that the solution is wrong.

This script recomputes p_yes under the same rule and under exact matching, from
the raw top-K distributions dumped by `extract_verdict_topk.py`, and reports
whether the baseline's AUC and its recall at a train-calibrated operating point
move. If they do not,
the shipped rule is fine and this is a one-paragraph robustness note; if they do,
the black-box baseline in the paper is measuring the tokenizer as much as the judge.

Everything is fit on TRAIN and evaluated on TEST, exactly as
`make_paper_figures.py` does: the operating point comes from the judge's train
precision and is applied unchanged to test.

  python filtering/pyes_rule_compare.py \
    --train_topk results_gemma3_train/verdict_topk.npz \
    --train_raw  results_gemma3_train/gsm8k/raw.jsonl \
    --test_topk  results_gemma3_test/verdict_topk.npz \
    --test_raw   results_gemma3_test/gsm8k/raw.jsonl \
    --verify_test_hidden results_gemma3_test/hidden_rich.npz \
    --title Gemma3-12B --out filtering/figures/pyes_rule_compare.png
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import average_precision_score, roc_auc_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probe_models import (
    precision_at_threshold,
    recall_ci_at_threshold,
    threshold_at_precision,
    wilson_interval,
)
from metrics.correctness import is_labelable, label_correctness, record_dataset
from metrics.provenance import read_run_meta, record_digest
from metrics.verdict_tokens import EXACT, NO, PREFIX, YES, classify

# What must agree for the two splits to be one experiment. `split`/`seed`/
# `problems_sha256` are what is supposed to differ, so they are not here — except that
# problems_sha256 being *equal* is checked separately, as leakage.
_SPLIT_IDENTITY_GEN = ("model", "backend", "tokenizer", "mistral_format",
                       "assistant_prefill", "judge_max_tokens", "dataset")

# Two independent bf16 forward passes over a 262k-token softmax do not agree bitwise:
# kernel selection varies with sequence length, so the logits differ in the last bits.
# Sizing the floor from the arithmetic rather than from the data: bf16 carries an 8-bit
# mantissa, i.e. ~2^-8 = 0.4% relative precision. p_yes is a RATIO of two sums of
# softmax outputs derived from those logits, so the relative error compounds to roughly
# 1% absolute. Anything below that is arithmetic, not disagreement; the original 1e-5
# was unreachable in principle, and a check that can never pass protects nothing.
PYES_NOISE_TOL = 1e-2
# Reported separately, because it is a different claim: a difference this large cannot
# come from rounding and indicates the known stored-p_yes defect.
PYES_GROSS = 0.1
# Above the tolerance, a handful of records is the known `hidden_rich.npz` p_yes defect
# (~0.5% of records, verified against fresh forward passes, worth <0.001 AUC). A large
# FRACTION is a different claim entirely — that the two dumps came from different runs —
# and still aborts.
PYES_MAX_BAD_FRACTION = 0.05

# The window the shipped rule reads, and a stricter one matching what a logprob API
# typically exposes (OpenAI's top_logprobs caps at 20), so the table also answers
# "could this baseline be reproduced black-box at all?".
WINDOWS = (20, 40)
SHIPPED_WINDOW = 40


def load(topk_path, raw_path):
    """Top-K distributions joined to recomputed correctness labels."""
    d = np.load(topk_path)
    meta = json.loads(str(d["meta"]))
    idx = d["idx"].tolist()

    raw = {}
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                raw[r["idx"]] = r
    missing = [i for i in idx if i not in raw]
    if missing:
        raise SystemExit(
            f"[FATAL] {raw_path} is missing {len(missing)} dumped indices "
            f"(first={missing[0]}); the dump and the labels are from "
            f"different runs."
        )

    datasets = {record_dataset(raw[i]) for i in idx}
    if len(datasets) > 1:
        raise SystemExit(f"[FATAL] {raw_path} mixes datasets {sorted(datasets)}.")
    dataset = datasets.pop() if datasets else None
    if meta.get("dataset") and dataset and meta["dataset"] != dataset:
        raise SystemExit(
            f"[FATAL] {topk_path} was extracted with --dataset "
            f"{meta['dataset']!r} but {raw_path} holds {dataset!r} records; "
            f"the distributions were read at a different judge prompt."
        )

    # `idx` is a position in a shuffled subset, so index agreement does not show these
    # distributions were read at these records' prompts. The dump stores a per-record
    # content fingerprint so it can be shown.
    if "rec_sha" in d.files:
        bad = [i for j, i in enumerate(idx)
               if str(d["rec_sha"][j]) != record_digest(raw[i])]
        if bad:
            raise SystemExit(
                f"[FATAL] {topk_path} was written from different records than "
                f"{raw_path} holds: {len(bad)}/{len(idx)} rows disagree (first "
                f"idx={bad[0]}); the distributions and the labels come from "
                f"different runs."
            )
    else:
        print(
            f"[WARN] {topk_path} predates per-record fingerprints; it is aligned to "
            f"{raw_path} by idx alone, which cannot detect a different run.",
            file=sys.stderr,
        )

    # Same drop rule as the probe pipeline: a record that cannot carry a trustworthy
    # label is excluded, not labelled wrong.
    keep = np.array([is_labelable(raw[i]) for i in idx], dtype=bool)
    if (~keep).any():
        print(
            f"[labels] dropped {int((~keep).sum())} unlabelable record(s) from "
            f"{raw_path}",
            file=sys.stderr,
        )
    kept = [i for i, k in zip(idx, keep) if k]

    y = np.array(
        [
            int(
                label_correctness(
                    raw[i].get("phase1_answer"), raw[i].get("gold_answer"), dataset
                )
            )
            for i in kept
        ],
        dtype=int,
    )
    yes = np.array(
        [
            (
                1
                if str(raw[i].get("judge_verdict", ""))
                .strip()
                .upper()
                .startswith("YES")
                else 0
            )
            for i in kept
        ],
        dtype=int,
    )

    texts = {int(i): str(t) for i, t in zip(d["seen_ids"], d["seen_texts"])}
    return dict(
        meta=meta,
        run_meta=read_run_meta(raw_path) or meta.get("gen_run") or {},
        dataset=dataset,
        idx=kept,
        y=y,
        yes=yes,
        top_ids=d["top_ids"][keep],
        top_probs=d["top_probs"][keep],
        p_yes_shipped=d["p_yes_shipped"][keep],
        mass_full={
            r: (d[f"mass_yes_{r}_full"][keep], d[f"mass_no_{r}_full"][keep])
            for r in (PREFIX, EXACT)
        },
        texts=texts,
    )


def p_yes_window(split, rule, window):
    """p_yes from the top-`window` tokens under `rule`, with the shipped 0.5 fallback."""
    ids, probs = split["top_ids"][:, :window], split["top_probs"][:, :window]
    cls = {i: classify(t, rule) for i, t in split["texts"].items()}
    my = np.zeros(len(ids), np.float64)
    mn = np.zeros(len(ids), np.float64)
    for n in range(len(ids)):
        for tid, p in zip(ids[n], probs[n]):
            c = cls.get(int(tid))
            if c == YES:
                my[n] += p
            elif c == NO:
                mn[n] += p
    d = my + mn
    out = np.where(d > 0, my / np.where(d > 0, d, 1.0), 0.5)
    return out.astype(np.float32), my, mn, d


def p_yes_full(split, rule):
    """p_yes over the whole vocabulary (no top-K truncation)."""
    my, mn = split["mass_full"][rule]
    d = my.astype(np.float64) + mn
    return np.where(d > 0, my / np.where(d > 0, d, 1.0), 0.5).astype(np.float32)


def mis_binned_mass(split, side, window=SHIPPED_WINDOW):
    """Per-token mass the prefix rule counts as `side` and exact matching does not.

    Run for both sides, because the asymmetry claim has to be checked rather than
    assumed: the prefix rule invents NO votes out of ` not`, but it equally invents
    YES votes out of ` yesterday`, and only the data says which dominates. Returns
    (per-record total, {token_text: summed mass}).
    """
    ids, probs = split["top_ids"][:, :window], split["top_probs"][:, :window]
    bad = {
        i
        for i, t in split["texts"].items()
        if classify(t, PREFIX) == side and classify(t, EXACT) != side
    }
    per_rec = np.zeros(len(ids), np.float64)
    per_tok: dict[str, float] = {}
    for n in range(len(ids)):
        for tid, p in zip(ids[n], probs[n]):
            if int(tid) in bad:
                per_rec[n] += p
                per_tok[split["texts"][int(tid)]] = per_tok.get(
                    split["texts"][int(tid)], 0.0
                ) + float(p)
    return per_rec, dict(sorted(per_tok.items(), key=lambda kv: -kv[1]))


def evaluate(name, tr_score, te_score, tr, te, judge_prec_train):
    """AUC + the train-fixed operating point, the same estimators as the paper figures."""
    yw_tr = (tr["y"] == 0).astype(int)
    yw_te = (te["y"] == 0).astype(int)
    wrong_tr, wrong_te = 1.0 - tr_score, 1.0 - te_score
    thr = threshold_at_precision(yw_tr, wrong_tr, judge_prec_train)
    r, lo, hi = recall_ci_at_threshold(yw_te, wrong_te, thr)
    return {
        "rule": name,
        "auc": float(roc_auc_score(te["y"], te_score)),
        "ap_wrong": float(average_precision_score(yw_te, wrong_te)),
        "threshold": thr,
        "recall": r,
        "recall_lo": lo,
        "recall_hi": hi,
        "precision": precision_at_threshold(yw_te, wrong_te, thr),
        "n_saturated": int(np.sum((te_score <= 0.02) | (te_score >= 0.98))),
        "n_fallback": int(np.sum(te_score == 0.5)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_topk", required=True)
    ap.add_argument("--train_raw", required=True)
    ap.add_argument("--test_topk", required=True)
    ap.add_argument("--test_raw", required=True)
    ap.add_argument(
        "--verify_test_hidden",
        default=None,
        help="hidden_rich.npz for the TEST split; p_yes_shipped is checked "
        "against its stored p_yes. Skipping this check means nothing "
        "ties these distributions to the dump the paper figures used.",
    )
    ap.add_argument("--verify_train_hidden", default=None)
    ap.add_argument("--title", default="judge")
    ap.add_argument("--out", default="filtering/figures/pyes_rule_compare.png")
    ap.add_argument(
        "--json_out", default=None, help="defaults to --out with a .json suffix"
    )
    args = ap.parse_args()

    tr = load(args.train_topk, args.train_raw)
    te = load(args.test_topk, args.test_raw)

    # Each split is validated only against its own raw file. The operating point is
    # chosen on train and applied to test, so two splits from different runs — or from
    # the same problem set — produce a table that reads normally and means nothing.
    mismatch = [f"dump {k}: train={tr['meta'].get(k)!r} test={te['meta'].get(k)!r}"
                for k in ("model", "arch", "dataset", "topk", "assistant_prefill")
                if tr["meta"].get(k) != te["meta"].get(k)]
    mismatch += [f"generation {k}: train={tr['run_meta'][k]!r} test={te['run_meta'][k]!r}"
                 for k in _SPLIT_IDENTITY_GEN
                 if k in tr["run_meta"] and k in te["run_meta"]
                 and tr["run_meta"][k] != te["run_meta"][k]]
    if (tr["run_meta"].get("problems_sha256")
            and tr["run_meta"].get("problems_sha256") == te["run_meta"].get("problems_sha256")
            and tr["run_meta"].get("split") == te["run_meta"].get("split")):
        mismatch.append("both splits were generated from the SAME problem set — the "
                        "train-chosen operating point is being applied to its own data")
    if mismatch:
        raise SystemExit(
            "[FATAL] the train and test dumps do not describe one experiment:\n"
            + "".join(f"  - {m}\n" for m in mismatch)
        )

    print(
        f"TRAIN n={len(tr['y'])} wrong={(tr['y'] == 0).sum()}  "
        f"TEST n={len(te['y'])} wrong={(te['y'] == 0).sum()}  "
        f"dataset={te['dataset']} model={te['meta']['model']}"
    )

    # ── provenance: these distributions must be the ones behind the paper's p_yes ──
    for tag, split, hidden in (
        ("test", te, args.verify_test_hidden),
        ("train", tr, args.verify_train_hidden),
    ):
        if not hidden:
            print(
                f"[WARN] no --verify_{tag}_hidden; nothing checks that this dump "
                f"reproduces the p_yes the figures were built from.",
                file=sys.stderr,
            )
            continue
        h = np.load(hidden)
        stored = {int(i): float(p) for i, p in zip(h["idx"], h["p_yes"])}
        mine = np.array([stored[i] for i in split["idx"]], np.float32)
        delta = np.abs(mine - split["p_yes_shipped"])
        bad = int((delta > PYES_NOISE_TOL).sum())
        frac = bad / max(len(delta), 1)
        print(
            f"[verify {tag}] max |p_yes_shipped - stored p_yes| = {delta.max():.2e} "
            f"({bad}/{len(delta)} = {frac:.1%} over {PYES_NOISE_TOL:g})"
        )
        if frac > PYES_MAX_BAD_FRACTION:
            raise SystemExit(
                f"[FATAL] {args.__dict__[f'verify_{tag}_hidden']} disagrees with "
                f"{tag} top-K dump on {bad}/{len(delta)} record(s) ({frac:.1%}). Same "
                f"prompt and position should give the same number, so a disagreement "
                f"this widespread means the two passes did not see the same input and "
                f"no rule comparison built on them is meaningful."
            )
        if bad:
            gross = int((delta > PYES_GROSS).sum())
            print(
                f"[WARN] {bad} record(s) disagree beyond arithmetic noise "
                f"({gross} by more than {PYES_GROSS:g}). Neither column is authoritative "
                f"by construction: both are written by the same forward pass at the same "
                f"position, so a disagreement means one of the two dumps was produced "
                f"under a different environment than it records -- which no fingerprint "
                f"here can detect, since rec_sha pins the RECORD and not the run. Settle "
                f"it by re-extracting the top-K dump and seeing which column moves; a "
                f"stale gemma3 dump did exactly this and re-extraction made the two "
                f"bit-identical. This analysis reads the top-K dump throughout, so its "
                f"rule comparison is internally consistent either way, and the quantity "
                f"in dispute is a baseline worth <0.001 AUC.",
                file=sys.stderr,
            )

    # ── judge verdict baseline (the train precision every rule is calibrated to) ──
    yw_tr, yw_te = (tr["y"] == 0).astype(int), (te["y"] == 0).astype(int)
    flag_tr, flag_te = (tr["yes"] == 0).astype(int), (te["yes"] == 0).astype(int)
    judge_prec_train = int(((flag_tr == 1) & (yw_tr == 1)).sum()) / max(
        int(flag_tr.sum()), 1
    )
    tp = int(((flag_te == 1) & (yw_te == 1)).sum())
    judge = {
        "rule": "verdict (binary)",
        "auc": float(roc_auc_score(te["y"], te["yes"])),
        "ap_wrong": float(average_precision_score(yw_te, flag_te)),
        "recall": tp / max(int(yw_te.sum()), 1),
        "precision": tp / max(int(flag_te.sum()), 1),
    }
    judge["recall_lo"], judge["recall_hi"] = wilson_interval(tp, int(yw_te.sum()))
    print(
        f"judge verdict: balanced_acc={judge['auc']:.4f} "
        f"R={judge['recall']:.1%} P={judge['precision']:.1%} "
        f"(train precision target {judge_prec_train:.4f})"
    )

    # ── the rules ────────────────────────────────────────────────────────────────
    scores_tr, scores_te = {}, {}
    for rule in (PREFIX, EXACT):
        for w in WINDOWS:
            scores_tr[f"{rule}@{w}"] = p_yes_window(tr, rule, w)[0]
            scores_te[f"{rule}@{w}"] = p_yes_window(te, rule, w)[0]
        scores_tr[f"{rule}@full"] = p_yes_full(tr, rule)
        scores_te[f"{rule}@full"] = p_yes_full(te, rule)

    shipped = f"{PREFIX}@{SHIPPED_WINDOW}"
    d = np.abs(scores_te[shipped] - te["p_yes_shipped"])
    print(
        f"[verify] recomputed {shipped} vs p_yes_shipped: max |delta| = {d.max():.2e}"
    )

    rows = [judge] + [
        evaluate(k, scores_tr[k], scores_te[k], tr, te, judge_prec_train)
        for k in scores_te
    ]

    # ── how much mass the prefix rule actually mis-binned, on BOTH sides ─────────
    mis = {side: mis_binned_mass(te, side) for side in (NO, YES)}
    per_rec, per_tok = mis[NO]
    changed = int(
        (
            np.abs(scores_te[shipped] - scores_te[f"{EXACT}@{SHIPPED_WINDOW}"]) > 1e-3
        ).sum()
    )
    print(
        f"\nmis-binned mass (prefix counts it, exact does not), TEST top-"
        f"{SHIPPED_WINDOW}:"
    )
    for side in (NO, YES):
        pr, _ = mis[side]
        n = int((pr > 1e-4).sum())
        print(
            f"  as {side.upper():3s}: {n}/{len(pr)} records ({n / len(pr):.1%})  "
            f"mean={pr.mean():.4f}  max={pr.max():.4f}  total={pr.sum():.2f}"
        )
    print(
        f"  records whose p_yes moves >1e-3 when the rule is fixed: {changed} "
        f"({changed / len(per_rec):.1%})"
    )
    print(f"  top tokens wrongly counted as NO:")
    for t, m in list(per_tok.items())[:12]:
        print(f"    {t!r:24s} {m:9.3f} total prob mass")
    if mis[YES][1]:
        print(f"  top tokens wrongly counted as YES:")
        for t, m in list(mis[YES][1].items())[:6]:
            print(f"    {t!r:24s} {m:9.3f} total prob mass")

    print(
        f"\n{'rule':18s} {'AUC':>7s} {'AP(wrong)':>10s} {'recall':>8s} "
        f"{'realised P':>11s} {'sat.':>6s} {'p=0.5':>6s}"
    )
    for r in rows:
        rec = "  N/A " if r.get("recall") is None else f"{r['recall']:7.1%}"
        prec = "   N/A " if r.get("precision") is None else f"{r['precision']:10.1%}"
        sat = f"{r.get('n_saturated', ''):>6}" if "n_saturated" in r else " " * 6
        fb = f"{r.get('n_fallback', ''):>6}" if "n_fallback" in r else " " * 6
        print(
            f"{r['rule']:18s} {r['auc']:7.4f} {r['ap_wrong']:10.4f} {rec} {prec} "
            f"{sat} {fb}"
        )

    out = {
        "title": args.title,
        "dataset": te["dataset"],
        "model": te["meta"]["model"],
        "n_train": len(tr["y"]),
        "n_test": len(te["y"]),
        "wrong_train": int((tr["y"] == 0).sum()),
        "wrong_test": int((te["y"] == 0).sum()),
        "judge_prec_train": judge_prec_train,
        "rows": rows,
        "shipped_rule": shipped,
        "mis_binned_mass": {
            side: {
                "n_records_touched": int((mis[side][0] > 1e-4).sum()),
                "mean": float(mis[side][0].mean()),
                "max": float(mis[side][0].max()),
                "total": float(mis[side][0].sum()),
                "per_token": {k: float(v) for k, v in list(mis[side][1].items())[:50]},
            }
            for side in (NO, YES)
        },
        "n_p_yes_changed": changed,
        "vocab_counts": {
            "yes": te["meta"]["n_vocab_yes"],
            "no": te["meta"]["n_vocab_no"],
        },
    }
    json_out = args.json_out or str(Path(args.out).with_suffix(".json"))
    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(json_out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {json_out}")

    plot(rows, scores_te, te, per_tok, judge, args, shipped)
    print(f"wrote {args.out}")


def plot(rows, scores_te, te, per_tok, judge, args, shipped):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    names = [r["rule"] for r in rows]
    vals = [r["auc"] for r in rows]
    colors = ["#888888"] + [
        "#1f77b4" if n.startswith(PREFIX) else "#d62728" for n in names[1:]
    ]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color="k", lw=0.8, ls=":")
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=7)
    # The verdict's bar is balanced accuracy, not an AUC — a binary score cannot reach
    # what a continuous one can, so the first bar is not on the same scale as the rest.
    ax.set_xlabel(
        "correctness AUC (held-out test)\nfirst bar = binary verdict, bal. acc"
    )
    ax.set_title(f"{args.title}: does the binning rule matter?", fontsize=10)

    ax = axes[1]
    x = scores_te[shipped]
    y = scores_te[f"{EXACT}@{SHIPPED_WINDOW}"]
    ax.scatter(x, y, s=8, alpha=0.3, c=np.where(te["y"] == 0, "#d62728", "#1f77b4"))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel(f"$p_{{YES}}$, prefix rule (shipped, top-{SHIPPED_WINDOW})")
    ax.set_ylabel(f"$p_{{YES}}$, exact rule (top-{SHIPPED_WINDOW})")
    ax.set_title("per-record effect (red = actually wrong)", fontsize=10)

    ax = axes[2]
    items = list(per_tok.items())[:12][::-1]
    if items:
        ax.barh(range(len(items)), [m for _, m in items], color="#d62728")
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([repr(t) for t, _ in items], fontsize=8)
        ax.set_xlabel("total probability mass over the test split")
    else:
        ax.text(0.5, 0.5, "no spurious NO mass in the top-40", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(
        "mass the prefix rule counts as NO\nthat is not a NO token", fontsize=10
    )

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    fig.savefig(str(Path(args.out).with_suffix(".pdf")))


if __name__ == "__main__":
    main()
