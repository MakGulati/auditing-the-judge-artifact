"""Shared filtering-side probe helpers.

All fitting steps here consume TRAIN data only, then score TEST data. That includes
*operating-point selection*: a decision threshold read off the test set is a
best-of-all-thresholds statistic on the reporting split, which is not comparable to
the judge's single fixed verdict. `threshold_at_precision` therefore picks the
threshold on train; `recall_at_threshold` applies it to test.
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics.correctness import (audit_stored_labelable, is_labelable,
                                 label_correctness, label_policy_for,
                                 record_dataset)
from metrics.provenance import read_run_meta, record_digest


@dataclass
class Split:
    """One evaluated split, aligned by record index."""
    Z: np.ndarray          # (n, n_heads, head_dim) o_proj input per head
    y: np.ndarray          # 1 = solution correct
    S: np.ndarray          # shallow text features of the solution
    yes: np.ndarray        # judge verdict, 1 = YES  (binary operating point)
    p_yes: np.ndarray      # judge P(YES) token mass (continuous black-box score)
    idx: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)


def shallow_feats(s):
    s = s or ""
    m = re.findall(r"####\s*\$?(-?[\d,]+\.?\d*)", s) or re.findall(r"(-?[\d,]+\.?\d*)", s)
    val = 0.0
    if m:
        try:
            val = float(m[-1].replace(",", ""))
        except Exception:
            val = 0.0
    if not np.isfinite(val):
        val = 0.0
    return [
        len(s),
        len(s.split()),
        s.count("\n") + 1,
        sum(c.isdigit() for c in s),
        sum(s.count(c) for c in "+-*/="),
        1.0 if "####" in s else 0.0,
        np.log1p(abs(val)),
    ]


def lda_auc(X, yt):
    if len(np.unique(yt)) < 2:
        return 0.5
    m1 = X[yt == 1].mean(0)
    m0 = X[yt == 0].mean(0)
    Xc = X.copy()
    Xc[yt == 1] -= m1
    Xc[yt == 0] -= m0
    cov = (Xc.T @ Xc) / (len(yt) - 2) + 1e-2 * np.eye(X.shape[1], dtype=np.float32)
    w = np.linalg.solve(cov, (m1 - m0).astype(np.float32))
    s = X @ w
    a = roc_auc_score(yt, s)
    return max(a, 1 - a)


def exclusion_summary(records: list[dict]) -> dict:
    """Which records leave the analysed population, why, and what it does to prevalence.

    The analysed population is the NON-TRUNCATED generations with a usable gold answer.
    A budget-truncated solution was cut off before it could state a final answer, so its
    correctness is unknown rather than wrong -- and because the parser returns nothing
    for it, every one of them would otherwise be counted as an error. That is a claim
    about our token budget, not about the model, and it is the easy kind of error for a
    judge to spot (the text stops mid-sentence), so leaving them in donates free true
    negatives and inflates observed error prevalence.

    `prevalence_included` is what the wrong-rate would be if they were kept and counted
    wrong, so the size of that inflation is visible rather than asserted.
    """
    n = len(records)
    err = [r for r in records if r.get("error") is not None]
    trunc = [r for r in records if r.get("error") is None and r.get("truncated") is True]
    kept = [r for r in records if is_labelable(r)]
    no_gold = n - len(err) - len(trunc) - len(kept)
    wrong_kept = sum(1 for r in kept if not r.get("solve_correct"))
    wrong_all = sum(1 for r in records if not r.get("solve_correct"))
    return {
        "n_generated": n,
        "n_analysed": len(kept),
        "n_excluded_truncated": len(trunc),
        "n_excluded_error": len(err),
        "n_excluded_no_gold": no_gold,
        # Every excluded truncation is marked wrong, which is exactly why keeping them
        # would bias prevalence upward rather than merely adding noise.
        "truncated_marked_wrong": sum(1 for r in trunc if not r.get("solve_correct")),
        "prevalence_analysed": wrong_kept / max(len(kept), 1),
        "prevalence_if_truncations_included": wrong_all / max(n, 1),
    }


def missing_provenance(dump, meta: dict) -> list[str]:
    """Which provenance fields a dump does not carry.

    Dumps written before the provenance work carry none of these, and nothing can add
    them after the fact: `rec_sha` has to be computed while the record is in hand, and
    `gen_run`/`prompt_identity` come from the run_meta.json that was beside the raw
    file at extraction time. Such a dump is not *wrong* — it is unverifiable, which is
    a different claim and one the caller should have to make deliberately.
    """
    absent = []
    if "rec_sha" not in getattr(dump, "files", []):
        absent.append("rec_sha")
    for key in ("gen_run", "prompt_identity"):
        value = (meta or {}).get(key)
        # A present-but-empty dict is the shape an extraction with no run_meta.json
        # leaves behind: `prompt_identity` is built by key from `gen_run`, so it comes
        # out as {model: None, tokenizer: None, ...} — truthy, and authenticating
        # nothing. Absent and all-None are the same claim.
        if isinstance(value, dict):
            value = {k: v for k, v in value.items() if v is not None}
        if not value:
            absent.append(key)
    return absent


def check_p_yes_against_topk(hidden_path, idx, dump, stored_p_yes):
    """Cross-check the dump's `p_yes` against the sibling `verdict_topk.npz`.

    Both files are written by the same forward pass at the same position, so the two
    columns must be equal. They are computed by different scripts on different days,
    which is exactly what makes the comparison worth making: nothing else in the
    pipeline can notice that a dump was produced under a different environment.

    That is not hypothetical. A stale gemma3 `verdict_topk.npz` disagreed with its
    `hidden_rich.npz` on 2-3% of records (max 0.41), and every existing check passed
    it, because `rec_sha` fingerprints the RECORD it was extracted from and matched
    perfectly -- the records were right, the environment was not. Re-extracting the
    top-K dump made the two columns bit-identical across all 20,969 gemma3 records.

    So the value is always the activation dump's own column: it is the one aligned
    with the activations by construction. This only reports, returning a dict for
    `label_info` and warning loudly on disagreement, because the affected quantity is
    a baseline worth <0.001 AUC -- real enough to surface, too small to abort on.
    """
    topk_path = Path(hidden_path).with_name("verdict_topk.npz")
    if not topk_path.exists():
        return {"status": "no_topk_dump"}
    t = np.load(topk_path)
    if "p_yes_shipped" not in t.files or "idx" not in t.files:
        return {"status": "topk_dump_lacks_p_yes"}
    by_idx = {int(i): float(p) for i, p in zip(t["idx"], t["p_yes_shipped"])}
    absent = [i for i in idx if int(i) not in by_idx]
    if absent:
        return {"status": "partial_coverage", "n_missing": len(absent),
                "n": len(idx), "first_missing_idx": int(absent[0])}
    # idx is a position in a shuffled subset, so matching indices do not show the two
    # dumps describe the same records. Both carry a per-record content fingerprint.
    if "rec_sha" in getattr(dump, "files", []) and "rec_sha" in t.files:
        sha = {int(i): str(s) for i, s in zip(t["idx"], t["rec_sha"])}
        bad = [i for j, i in enumerate(idx) if sha[int(i)] != str(dump["rec_sha"][j])]
        if bad:
            raise ValueError(
                f"{topk_path} was extracted from different records than {hidden_path}: "
                f"{len(bad)}/{len(idx)} fingerprints disagree (first idx={bad[0]}). "
                f"Re-extract the top-K dump from the same raw.jsonl.")
    other = np.array([by_idx[int(i)] for i in idx], dtype=np.float32)
    delta = np.abs(np.asarray(stored_p_yes, dtype=np.float32) - other)
    n_disagree = int(np.count_nonzero(delta > 0.01))
    info = {"status": "agrees" if n_disagree == 0 else "disagrees",
            "n": len(idx), "n_disagree": n_disagree, "max_delta": float(delta.max())}
    if n_disagree:
        print(f"[WARN] {Path(hidden_path).name} and {topk_path.name} disagree on p_yes "
              f"for {n_disagree}/{len(idx)} record(s) (max {delta.max():.2e}). Both are "
              f"written by the same forward pass, so one of them was produced under a "
              f"different environment than it records. The activation dump's column is "
              f"used (it is the one aligned with the activations); re-extract the top-K "
              f"dump to clear this.", file=sys.stderr, flush=True)
    return info


def load_split(hidden_path, raw_path, head_dim, label_policy="numeric") -> Split:
    """Load activations and align labels/metadata by record index.

    ``numeric`` (default) recomputes correctness from raw answers, so historical
    activation dumps with exact-string labels are repaired without re-extraction.
    ``stored`` is retained only for explicit sensitivity comparisons.

    Records that cannot carry a trustworthy label (generation errors, unparseable
    gold answers) are dropped and counted rather than labelled ``wrong``.
    """
    if label_policy not in {"numeric", "stored"}:
        raise ValueError("label_policy must be 'numeric' or 'stored'")
    d = np.load(hidden_path)
    Z = d["Z_head"]
    stored_y = d["y"].astype(int)
    idx = d["idx"].tolist()
    p_yes_all = d["p_yes"] if "p_yes" in d.files else np.full(len(idx), np.nan, np.float32)
    p_yes_check = check_p_yes_against_topk(hidden_path, idx, d, p_yes_all)
    N, L, Hdim = Z.shape

    # --- head_dim must match the model that produced the dump -------------------
    # Any divisor of Hdim reshapes without error, silently splitting real heads into
    # fragments (head_dim=128 on gemma3 yields 1536 half-heads instead of 768 heads),
    # which also inflates the head-selection search space. Verify against the dump's
    # recorded geometry when present.
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    if meta:
        true_head_dim = meta.get("head_dim")
        if true_head_dim is not None and int(true_head_dim) != int(head_dim):
            raise ValueError(
                f"--head_dim {head_dim} does not match {hidden_path}: that dump was "
                f"extracted from {meta.get('model')!r} with head_dim={true_head_dim} "
                f"({meta.get('n_heads')} heads x {true_head_dim}). Reshaping at "
                f"{head_dim} would fragment real attention heads."
            )
    else:
        print(f"[WARN] {hidden_path} has no geometry metadata (pre-fix dump); cannot "
              f"verify --head_dim={head_dim}. Re-extract to enable the check.",
              file=sys.stderr, flush=True)
    if Hdim % head_dim:
        raise ValueError(f"head_dim {head_dim} does not divide o_proj width {Hdim}")
    Zh = Z.reshape(N, L * (Hdim // head_dim), head_dim)

    # --- align raw records ------------------------------------------------------
    raw_by_idx = {}
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                raw_by_idx[r["idx"]] = r

    missing = [i for i in idx if i not in raw_by_idx]
    if missing:
        raise ValueError(f"raw data is missing {len(missing)} activation indices; first={missing[0]}")

    # `idx` is a position in a shuffled subset, not a stable dataset ID, so index
    # agreement alone does not show that these activations were read at these records'
    # prompts — two runs of different problems share the same index range. The extractor
    # stores a per-row content fingerprint precisely so this can be checked.
    if "rec_sha" in d.files:
        mismatched = [i for j, i in enumerate(idx)
                      if str(d["rec_sha"][j]) != record_digest(raw_by_idx[i])]
        if mismatched:
            raise ValueError(
                f"{hidden_path} was extracted from different records than {raw_path} "
                f"holds: {len(mismatched)}/{len(idx)} rows disagree (first idx="
                f"{mismatched[0]}). The activations and the labels come from different "
                f"runs.")
    else:
        print(f"[WARN] {hidden_path} predates per-record fingerprints; it is aligned to "
              f"{raw_path} by idx alone, which cannot detect that the two come from "
              f"different runs. Re-extract to enable the check.",
              file=sys.stderr, flush=True)

    # The recomputed value decides the analysed population. Where the stored snapshot
    # disagrees, say so rather than let a hand-inspection of raw.jsonl imply a
    # different population than the figures were built from.
    # Over the WHOLE raw file, not the dump's index list: extraction already dropped
    # unlabelable records, so a dump-aligned view sees only survivors and reports zero
    # exclusions -- which is exactly the population question being asked.
    all_generated = list(raw_by_idx.values())
    stored_audit = audit_stored_labelable(all_generated)
    if stored_audit["n_disagree"]:
        print(f"[labels] stored `labelable` disagrees with the recomputed value on "
              f"{stored_audit['n_disagree']}/{stored_audit['n']} record(s) "
              f"({stored_audit['stored_true_recomputed_false']} stored-true but "
              f"excluded). The recomputed value is authoritative; the stored field is "
              f"an advisory snapshot from generation time.",
              file=sys.stderr, flush=True)

    keep = np.array([is_labelable(raw_by_idx[i]) for i in idx], dtype=bool)
    n_dropped = int((~keep).sum())
    if n_dropped:
        dropped = [raw_by_idx[i] for i, k in zip(idx, keep) if not k]
        n_err = sum(1 for r in dropped if r.get("error") is not None)
        n_trunc = sum(1 for r in dropped
                      if r.get("error") is None and r.get("truncated") is True)
        print(f"[labels] dropped {n_dropped} unlabelable record(s): {n_err} generation "
              f"error(s), {n_trunc} budget-truncated, "
              f"{n_dropped - n_err - n_trunc} unparseable gold answer(s)",
              file=sys.stderr, flush=True)

    kept_idx = [i for i, k in zip(idx, keep) if k]

    # Equivalence is dataset-specific, so the labels must be recomputed under the
    # policy the records were generated with. Comparing MATH's LaTeX answers
    # numerically would label nearly every one of them wrong — and, unlike a crash,
    # the probe would train happily on the result.
    datasets = {record_dataset(raw_by_idx[i]) for i in kept_idx}
    if len(datasets) > 1:
        raise ValueError(
            f"{raw_path} mixes records from {', '.join(sorted(datasets))}; their "
            f"answers obey different equivalence policies and cannot share one probe.")
    dataset = datasets.pop() if datasets else None
    if meta and meta.get("dataset") and dataset and meta["dataset"] != dataset:
        raise ValueError(
            f"{hidden_path} was extracted from dataset {meta['dataset']!r} but "
            f"{raw_path} holds {dataset!r} records; the activations and the labels "
            f"come from different runs.")

    numeric_y = np.array([int(label_correctness(raw_by_idx[i].get("phase1_answer"),
                                                raw_by_idx[i].get("gold_answer"),
                                                dataset))
                          for i in kept_idx], dtype=int)
    stored_y = stored_y[keep]
    changed = int(np.count_nonzero(numeric_y != stored_y))
    y = numeric_y if label_policy == "numeric" else stored_y
    policy_name = (label_policy_for(dataset) if label_policy == "numeric"
                   else "stored_npz_legacy")
    label_info = {
        "policy": policy_name,
        "dataset": dataset,
        "stored_disagreements": changed,
        "n": len(kept_idx),
        "n_dropped_unlabelable": n_dropped,
        "dump_meta": meta,
        # Generation fingerprint of the run behind raw_path, when it is still beside it.
        # Preferred over the copy the dump carries: it describes the file the labels
        # actually came from. `check_split_provenance` falls back to the dump's copy.
        "run_meta": read_run_meta(str(raw_path)),
        "paths": {"hidden": str(hidden_path), "raw": str(raw_path)},
        "unauthenticated": missing_provenance(d, meta),
        # p_yes always comes from the activation dump. This records whether the
        # independently-written top-K dump agrees, which is the only signal that one
        # of them was produced under a different environment than it records.
        "p_yes_check": p_yes_check,
        # The analysed population, stated numerically so a caption can quote it.
        "exclusions": exclusion_summary(all_generated),
        "stored_labelable_audit": stored_audit,
    }
    if label_policy == "numeric":
        print(f"[labels] {policy_name}: {changed}/{len(kept_idx)} differ from stored NPZ labels",
              file=sys.stderr, flush=True)

    S = np.array([shallow_feats(raw_by_idx[i].get("phase1_solution", "")) for i in kept_idx],
                 np.float32)
    yes = np.array([1 if str(raw_by_idx[i].get("judge_verdict", "")).strip().upper()
                    .startswith("YES") else 0 for i in kept_idx])
    return Split(Z=Zh[keep], y=y, S=S, yes=yes, p_yes=np.asarray(p_yes_all)[keep],
                 idx=kept_idx, info=label_info)


# Dump metadata that must be identical across the two splits of one experiment: the
# probe is fitted on train features and applied to test features, so a difference in
# any of these makes the two feature spaces incomparable rather than merely noisy.
_SPLIT_IDENTITY_META = ("model", "n_layers", "hidden_size", "n_heads", "o_proj_in",
                        "head_dim", "dataset")
# Generation settings that must match for the two splits to be the same experiment.
# `split`, `seed` and `problems_sha256` are excluded on purpose — they are exactly what
# is *supposed* to differ.
_SPLIT_IDENTITY_GEN = ("model", "backend", "tokenizer", "mistral_format",
                       "assistant_prefill", "judge_max_tokens", "dataset")


def _gen_identity(info: dict) -> dict:
    """Generation fingerprint for a split: run_meta.json if present, else the dump's."""
    return info.get("run_meta") or (info.get("dump_meta") or {}).get("gen_run") or {}


def check_split_provenance(train: "Split", test: "Split", *, strict: bool = True) -> None:
    """Refuse a train/test pair that does not describe one experiment.

    Each split is loaded independently, and every check up to here is *within* a split
    (the dump matches its own raw file). Nothing compares the two, so a GSM8K train dump
    and a MATH test dump — or two different models that happen to share a geometry —
    train and evaluate without a single error, and the reported held-out AUC is a
    number about nothing.

    Also fatal: two splits built from the *same* problems. That is not a mismatch, it is
    leakage, and it inflates the headline result rather than corrupting it, so it is the
    one failure mode nobody notices.
    """
    tr_meta = train.info.get("dump_meta") or {}
    te_meta = test.info.get("dump_meta") or {}
    problems = []

    if not tr_meta or not te_meta:
        print("[WARN] a dump predates metadata; train/test provenance cannot be "
              "compared. Re-extract to enable the check.", file=sys.stderr, flush=True)
    for key in _SPLIT_IDENTITY_META:
        a, b = tr_meta.get(key), te_meta.get(key)
        if a is not None and b is not None and a != b:
            problems.append(f"dump {key}: train={a!r} test={b!r}")

    if train.info.get("dataset") != test.info.get("dataset"):
        problems.append(f"raw records: train={train.info.get('dataset')!r} "
                        f"test={test.info.get('dataset')!r}")

    gtr, gte = _gen_identity(train.info), _gen_identity(test.info)
    if not gtr or not gte:
        print("[WARN] no generation fingerprint for at least one split (no run_meta.json "
              "beside raw.jsonl and no gen_run in the dump); the two splits' prompts and "
              "judge settings could not be compared.", file=sys.stderr, flush=True)
    for key in _SPLIT_IDENTITY_GEN:
        if key in gtr and key in gte and gtr[key] != gte[key]:
            problems.append(f"generation {key}: train={gtr[key]!r} test={gte[key]!r}")

    same_problems = (gtr.get("problems_sha256") and gte.get("problems_sha256")
                     and gtr["problems_sha256"] == gte["problems_sha256"]
                     and gtr.get("split") == gte.get("split"))
    if same_problems:
        problems.append(
            f"both splits were generated from the SAME problem set "
            f"(split={gtr.get('split')!r}, problems_sha256="
            f"{gtr['problems_sha256'][:12]}...) — the held-out evaluation is not held out")

    if problems:
        detail = "".join(f"  - {p}\n" for p in problems)
        message = (f"[FATAL] the train and test splits do not describe one experiment:\n"
                   f"{detail}"
                   f"        train: {train.info.get('paths', {}).get('hidden')}\n"
                   f"        test:  {test.info.get('paths', {}).get('hidden')}\n")
        if strict:
            raise SystemExit(message)
        print(message.replace("[FATAL]", "[WARN]"), file=sys.stderr, flush=True)


def check_split_usable(name: str, split: "Split") -> None:
    """Refuse an empty or single-class split here, where the cause is still visible.

    Otherwise it surfaces several hundred LDA fits later, inside sklearn, as an error
    that names neither the split nor the reason it is degenerate.
    """
    if len(split) == 0:
        raise SystemExit(
            f"[FATAL] the {name} split has no labelable records: every record was "
            f"dropped as a generation error or an unparseable gold answer.")
    classes = np.unique(split.y)
    if len(classes) < 2:
        raise SystemExit(
            f"[FATAL] the {name} split is single-class (all "
            f"{'correct' if classes[0] == 1 else 'wrong'}, n={len(split)}): there is no "
            f"correctness signal to fit or to score.")


def prepare_head_features(Ztr, ytr, Zte, topk):
    """Select top-k heads by train-only LDA AUC and flatten train/test features."""
    nH = Ztr.shape[1]
    aucs = np.array([lda_auc(Ztr[:, h, :].astype(np.float32), ytr) for h in range(nH)])
    top = np.argsort(-aucs)[:topk]
    Ftr = Ztr[:, top, :].reshape(len(ytr), -1).astype(np.float32)
    Fte = Zte[:, top, :].reshape(Zte.shape[0], -1).astype(np.float32)
    return Ftr, Fte, top, aucs


def head_table(top, aucs, head_dim, n_layers):
    """Human-readable record of which heads the probe selected (layer, head, train AUC).

    ``n_layers`` of 0 means the dump carried no geometry metadata. Falling back to
    ``max(n_layers, 1)`` would put every head in layer 0 with ``head_in_layer`` equal
    to the flat index — a confidently wrong attribution. Report None instead.
    """
    heads_per_layer = len(aucs) // n_layers if n_layers else 0
    return [{"rank": r, "flat_head": int(h),
             "layer": int(h) // heads_per_layer if heads_per_layer else None,
             "head_in_layer": int(h) % heads_per_layer if heads_per_layer else None,
             "train_lda_auc": float(aucs[h])}
            for r, h in enumerate(np.asarray(top).tolist())]


def head_ranking(top, aucs, n_layers, topk):
    """Full candidate-head record, not just the selected slice.

    Persisting only the top-k answers "which heads?" but not "by how much?": without
    the whole distribution you cannot tell whether rank 32 is meaningfully better than
    rank 400, i.e. whether "top-32 heads" is a real structure or an arbitrary cut
    through a flat ranking. It also makes topk a tunable in post-hoc analysis instead
    of a value frozen at compute time. The full vector is a few thousand floats.
    """
    aucs = np.asarray(aucs, dtype=float)
    order = np.argsort(-aucs)
    heads_per_layer = len(aucs) // n_layers if n_layers else 0
    q = {f"p{p}": float(np.percentile(aucs, p)) for p in (50, 75, 90, 95, 99)}
    return {
        "topk": int(topk),
        "n_candidate_heads": int(len(aucs)),
        "n_layers": int(n_layers) if n_layers else None,
        "heads_per_layer": int(heads_per_layer) if heads_per_layer else None,
        "selected": head_table(top, aucs, None, n_layers),
        # Ranked, so `ranking[k]` is the head that a top-(k+1) cut would have added.
        "ranking": [{"rank": r, "flat_head": int(h),
                     "layer": int(h) // heads_per_layer if heads_per_layer else None,
                     "head_in_layer": int(h) % heads_per_layer if heads_per_layer else None,
                     "train_lda_auc": float(aucs[h])}
                    for r, h in enumerate(order.tolist())],
        "auc_summary": {"max": float(aucs.max()), "min": float(aucs.min()),
                        "mean": float(aucs.mean()), **q,
                        "at_topk_cut": float(aucs[order[min(topk, len(aucs)) - 1]])},
    }


def _can_use_mlp_early_stopping(y, validation_fraction):
    counts = np.bincount(np.asarray(y, dtype=int))
    counts = counts[counts > 0]
    if len(counts) < 2 or counts.min() < 2:
        return False
    n_val = math.ceil(len(y) * validation_fraction)
    n_train = len(y) - n_val
    return n_val >= len(counts) and n_train >= len(counts)


def make_classifier(model_type, seed=0, y=None):
    if model_type == "lr":
        return LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced")
    if model_type == "mlp":
        validation_fraction = 0.15
        early_stopping = True if y is None else _can_use_mlp_early_stopping(y, validation_fraction)
        return MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=seed,
        )
    raise ValueError(f"unknown model type: {model_type}")


def is_stochastic(model_type: str) -> bool:
    """LR here is deterministic; the MLP depends on its init seed."""
    return model_type == "mlp"


def balance_train(X, y, seed=0):
    """Oversample the minority class so the objective is balanced. TRAIN ONLY.

    LogisticRegression takes ``class_weight="balanced"``; sklearn's MLPClassifier has
    no equivalent and offers no sample_weight either, so without this the two probes
    are not being compared on equal terms -- LR optimises a balanced objective and the
    MLP does not. The handicap scales inversely with minority size, so it is invisible
    on a balanced dataset and severe on a skewed one.

    Measured on GSM8K (5.7% wrong): the unbalanced MLP scores 0.760 +/- 0.106 with
    individual seeds collapsing to 0.60, and balancing lifts it to 0.800 +/- 0.018 --
    from losing to LR to beating it, with six times less variance. On MATH (17.7%
    wrong) the same change is worth +0.004, which is the expected pattern if imbalance
    is the cause.

    Oversampling rather than undersampling keeps every majority example; the RNG is
    seeded so a given (seed, data) pair is reproducible.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    target = counts.max()
    rng = np.random.default_rng(seed)
    idx = np.concatenate([
        np.where(y == c)[0] if n == target
        else rng.choice(np.where(y == c)[0], target, replace=True)
        for c, n in zip(classes, counts)
    ])
    idx.sort()          # keep row order stable for anything that inspects it
    return X[idx], y[idx]


def fit_scores(Ftr, ytr, Fte, model_type="lr", seed=0):
    """Shared scaler + classifier path. Returns P(correct) on (train, test).

    The scaler is fitted on the ORIGINAL train rows, not the rebalanced ones, so
    oversampling cannot shift the feature standardisation.
    """
    sc = StandardScaler().fit(Ftr)
    Xtr = sc.transform(Ftr)
    # LR carries class_weight="balanced" itself; the MLP cannot, so it is given a
    # balanced training set instead. Same treatment, different mechanism.
    Xfit, yfit = (balance_train(Xtr, ytr, seed=seed) if model_type == "mlp"
                  else (Xtr, ytr))
    clf = make_classifier(model_type, seed=seed, y=yfit).fit(Xfit, yfit)
    return clf.predict_proba(Xtr)[:, 1], clf.predict_proba(sc.transform(Fte))[:, 1]


def shallow_only_scores(Str, ytr, Ste, model_type="lr", seed=0):
    """The baseline the `controlled` variant residualizes against, scored on its own.

    `controlled` removes whatever the shallow text features can predict, on the theory
    that they are surface artifacts. Whether that is true is an empirical question, and
    it is answered by this number: if the shallow features rank correctness well by
    themselves, they are signal rather than nuisance, and removing them costs the probe
    something real rather than stripping a cheat.

    On MATH they reach ~0.80 AUC on their own, because a solution's length tracks the
    problem's difficulty: the longest fifth of solutions are wrong 51% of the time
    against 4% for the shortest fifth. Reporting the probe without this baseline beside
    it leaves the reader unable to see how much of its score is a ruler.
    """
    return fit_scores(Str, ytr, Ste, model_type=model_type, seed=seed)


def fit_raw_and_controlled(Ftr, ytr, Fte, Str, Ste, model_type="lr", seed=0):
    """Fit raw and shallow-controlled variants for one model type.

    Returns train and test scores for each variant; the train scores exist so that
    operating points can be chosen without touching the test split.
    """
    tr_raw, te_raw = fit_scores(Ftr, ytr, Fte, model_type=model_type, seed=seed)

    ssc = StandardScaler().fit(Str)
    Str_s = ssc.transform(Str)
    Ste_s = ssc.transform(Ste)
    reg = LinearRegression().fit(Str_s, Ftr)
    Rtr = Ftr - reg.predict(Str_s)
    Rte = Fte - reg.predict(Ste_s)
    tr_ctrl, te_ctrl = fit_scores(Rtr, ytr, Rte, model_type=model_type, seed=seed)
    return {
        "raw": te_raw, "controlled": te_ctrl,
        "raw_train": tr_raw, "controlled_train": tr_ctrl,
    }


def fit_ensemble(Ftr, ytr, Fte, Str, Ste, model_type="lr", seed=0, n_seeds=1):
    """Fit at several initialisations and AVERAGE the predicted probabilities.

    One score vector per variant, used for EVERYTHING downstream -- curves, operating
    points, bars, annotations and tables. The previous arrangement fitted `n_seeds`
    models, plotted the FIRST one's predictions, and annotated the plot with the MEAN
    of the per-seed AUCs, so the curve on the page and the number beside it described
    different fits. Averaging the probabilities removes the choice entirely: there is
    only one prediction to plot and one AUC to quote, and they are the same object.

    Ensembling also is not merely a reporting convenience -- averaging over
    initialisations is a better estimator than an arbitrary single draw, which matters
    here because the MLP's init spread reached +/-0.106 AUC on GSM8K before the
    training set was balanced.

    Deterministic models ignore `n_seeds`: refitting LR at another seed returns the
    same fit, so averaging would just be the same number.

    Returns (scores, seeds, runs): `scores` has the same keys as
    `fit_raw_and_controlled` and is the ensemble; `runs` holds the individual fits so
    the caller can report the initialisation-only spread via `ensemble_seed_aucs`,
    separately from the train-resample spread rather than blended with it.
    """
    seeds = list(range(seed, seed + n_seeds)) if is_stochastic(model_type) else [seed]
    runs = [fit_raw_and_controlled(Ftr, ytr, Fte, Str, Ste, model_type=model_type,
                                   seed=s) for s in seeds]
    scores = {k: np.mean([r[k] for r in runs], axis=0) for k in runs[0]}
    return scores, seeds, runs


def ensemble_seed_aucs(runs, y_te, variant):
    """Test AUC of each individual fit -- the initialisation-only spread.

    Kept distinct from the ensemble's own AUC: the ensemble is the reported estimate,
    and this is a statement about how much an arbitrary single draw would have moved
    it. It is NOT an error bar on the ensemble.
    """
    return [float(roc_auc_score(y_te, r[variant])) for r in runs]


# ── operating points ─────────────────────────────────────────────────────────────

def threshold_at_precision(y_wrong, score, target_prec):
    """Lowest threshold on THIS split reaching precision >= target (max recall).

    Intended for the TRAIN split. Returns None when the target is unreachable.
    """
    prec, rec, thr = precision_recall_curve(y_wrong, score)
    # precision_recall_curve returns len(thr) == len(prec) - 1; prec[i] is the
    # precision obtained by flagging everything with score >= thr[i].
    ok = np.where(prec[:-1] >= target_prec)[0]
    if ok.size == 0:
        return None
    return float(thr[ok[0]])


def recall_at_threshold(y_wrong, score, thr):
    """Recall of a FIXED decision rule (score >= thr). Comparable to a fixed verdict.

    Returns None when no operating point exists (`thr is None`, i.e. the precision
    target was unreachable on train). Returning 0.0 there would be indistinguishable
    from a detector that genuinely caught nothing, and a literal 0 in a results table
    is a claim about the probe rather than about the experiment's setup.
    """
    if thr is None:
        return None
    flag = score >= thr
    denom = int(np.sum(y_wrong))
    return float(np.sum(flag & (y_wrong == 1)) / denom) if denom else None


def precision_at_threshold(y_wrong, score, thr):
    """Realised precision of the fixed rule. None when there is no operating point."""
    if thr is None:
        return None
    flag = score >= thr
    denom = int(np.sum(flag))
    return float(np.sum(flag & (y_wrong == 1)) / denom) if denom else None


def wilson_interval(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion. (lo, hi), or (None, None).

    Recall at a *fixed* threshold is k successes out of n wrong examples — a binomial
    proportion, so it has an exact interval. Preferred over the bootstrap here because
    it is guaranteed to contain the point estimate: percentile bootstrap bounds can
    fall on the wrong side of it, and clamping the resulting negative error bar to
    zero silently redraws "the interval excludes the estimate" as "the interval
    touches it". The bootstrap is still needed for the paired probe-minus-judge
    difference, which is not a single proportion.
    """
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    lo, hi = centre - half, centre + half
    # Clamp to [0,1] and to the point estimate. At p=0 or p=1 the closed form lands a
    # few ulps off the boundary (k=n gives hi=0.9999999999999998), which makes the
    # error bar `hi - p` a tiny negative number and matplotlib refuses to render it.
    # Callers rely on exact bracketing precisely so that a negative arm means a real
    # bug, so enforce it here rather than clamping at every call site.
    return min(max(0.0, lo), p), max(min(1.0, hi), p)


def recall_ci_at_threshold(y_wrong, score, thr):
    """(recall, lo, hi) for a fixed rule, via the Wilson interval. None when no rule."""
    if thr is None:
        return None, None, None
    wrong = (np.asarray(y_wrong) == 1)
    n = int(wrong.sum())
    k = int(np.sum((score >= thr) & wrong))
    if not n:
        return None, None, None
    lo, hi = wilson_interval(k, n)
    return k / n, lo, hi


def oracle_recall_at_precision(y_wrong, score, target_prec):
    """Best recall over ALL thresholds on THIS split — an optimistic upper bound.

    Reported only alongside the train-selected operating point, never on its own:
    maximising over thresholds on the reporting split is not comparable to the
    judge's single fixed verdict. None when the target is unreachable.
    """
    prec, rec, thr = precision_recall_curve(y_wrong, score)
    # Drop sklearn's appended (precision=1, recall=0) endpoint: it has no threshold
    # behind it, so an unreachable target would otherwise match that sentinel and
    # return a spurious 0.0 — indistinguishable from a real "caught nothing".
    prec, rec = prec[:len(thr)], rec[:len(thr)]
    ok = prec >= target_prec
    return float(rec[ok].max()) if ok.any() else None


def risk_coverage(score, y):
    """Selective risk of a filtering gate: accept highest P(correct) first.

    coverage[i] = fraction of items accepted; risk[i] = fraction of ACCEPTED items
    that are actually wrong. At coverage 1.0 the risk is exactly the split's
    wrong-answer rate, so the "no filtering" reference line is the same quantity
    plotted on the same axis.

    (The earlier version ranked by |score - 0.5| and scored the probe's 0/1
    classification error, which is selective *classification*, not filtering: items
    the probe confidently called WRONG counted as "accepted".)
    """
    order = np.argsort(-np.asarray(score))
    n = len(y)
    wrong = (np.asarray(y)[order] == 0).astype(int)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(wrong) / np.arange(1, n + 1)
    return cov, risk


def train_resample_auc(Ztr, ytr, Str, Zte, yte, Ste, topk, model_type="lr",
                       n_reps=8, seed=0, log=None, n_seeds=1):
    """Test-AUC spread under resampling of the TRAIN set, re-selecting heads each time.

    Refitting a stochastic probe at several initialisations holds the *data* and the
    *selected heads* fixed, so its spread is init noise alone. With a few hundred
    minority training examples choosing 32 heads out of several hundred candidates,
    head selection is plausibly the larger term — and it is exactly the one a
    seed-only estimate cannot see. Each replicate here resamples train with
    replacement and redoes selection, residualisation and fitting inside the
    replicate, so the spread covers train sampling + selection + initialisation.

    The test split is held fixed: this is the uncertainty of the *fitted probe*, not
    of the test estimate (the Wilson/bootstrap intervals cover that separately).

    `n_seeds` must match the ensemble the point estimate uses. Fitting ONE model per
    replicate while quoting an ensemble as the estimate attaches a spread to a
    different estimator than the number it decorates: a single draw is noisier than an
    average of five, so the interval would describe a probe nobody is using. It costs
    `n_seeds` fits per replicate, which dominates this function's runtime.
    """
    rng = np.random.default_rng(seed)
    n = len(ytr)
    aucs = []
    for rep in range(n_reps):
        for _ in range(10):                       # redraw if a replicate is degenerate
            s = rng.integers(0, n, n)
            if len(np.unique(ytr[s])) >= 2:
                break
        else:
            if log:
                log(f"    [WARN] train resample {rep}: single-class draw, skipped")
            continue
        Ftr, Fte, _, _ = prepare_head_features(Ztr[s], ytr[s], Zte, topk)
        sc, _, _ = fit_ensemble(Ftr, ytr[s], Fte, Str[s], Ste,
                                model_type=model_type, seed=seed + rep,
                                n_seeds=n_seeds)
        aucs.append(float(roc_auc_score(yte, sc["controlled"])))
        if log:
            log(f"    train resample {rep + 1}/{n_reps}: controlled test AUC={aucs[-1]:.4f}")
    return aucs


def parse_models(models_arg):
    models = [m.strip().lower() for m in models_arg.split(",") if m.strip()]
    if not models:
        raise ValueError("--models must include at least one model")
    bad = [m for m in models if m not in {"lr", "mlp"}]
    if bad:
        raise ValueError(f"unsupported --models values: {', '.join(bad)}")
    return list(dict.fromkeys(models))


def model_label(model_type):
    return {"lr": "LR", "mlp": "MLP"}[model_type]
