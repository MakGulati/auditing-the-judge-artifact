#!/usr/bin/env python3
"""Reproducible IEEE TPS figure package for the filtering experiments.

The primary method is fixed *before* looking at any test budget: a five-seed
ensemble of the raw top-32-head MLP probe.  The judge's continuous p(YES) mass is
the primary black-box baseline.  Shallow-controlled and linear probes appear only
as ablations.  No figure performs per-budget or per-dataset best-of selection.

Slow path (rebuild aligned predictions from activation dumps, then render):

    MPLCONFIGDIR=/tmp .venv/bin/python filtering/make_ieee_tps_figures.py --compute

Fast path (render again from the aligned prediction files):

    MPLCONFIGDIR=/tmp .venv/bin/python filtering/make_ieee_tps_figures.py

All generated artifacts are isolated in filtering/figures/ieee_tps_2026/.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import sys
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_models import (check_split_provenance, fit_ensemble, load_split,
                          prepare_head_features, shallow_only_scores)


ROOT = Path(__file__).resolve().parent.parent
FILTERING = Path(__file__).resolve().parent
PAPER_CACHE = FILTERING / "figures" / "paper"
OUT = FILTERING / "figures" / "ieee_tps_2026"

TOPK = 32
N_SEEDS = 5
DEFAULT_N_BOOT = 2000
BOOT_SEED = 20260810
BUDGETS = np.array([0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20])

RUNS = OrderedDict([
    ("gemma3", dict(model="gemma3", model_label="Gemma 3 12B", short="Gemma 3",
                    dataset="gsm8k", head_dim=256)),
    ("qwen25_7b", dict(model="qwen25_7b", model_label="Qwen2.5 7B", short="Qwen2.5",
                       dataset="gsm8k", head_dim=128)),
    ("llama31_8b", dict(model="llama31_8b", model_label="Llama 3.1 8B", short="Llama 3.1",
                        dataset="gsm8k", head_dim=128)),
    ("gemma3_math", dict(model="gemma3", model_label="Gemma 3 12B", short="Gemma 3",
                         dataset="math", head_dim=256)),
    ("qwen25_7b_math", dict(model="qwen25_7b", model_label="Qwen2.5 7B", short="Qwen2.5",
                            dataset="math", head_dim=128)),
    ("llama31_8b_math", dict(model="llama31_8b", model_label="Llama 3.1 8B", short="Llama 3.1",
                             dataset="math", head_dim=128)),
])

MODEL_ORDER = ["gemma3", "qwen25_7b", "llama31_8b"]
MODEL_NAME = {v["model"]: v["model_label"] for v in RUNS.values()}
MODEL_SHORT = {v["model"]: v["short"] for v in RUNS.values()}

# Fixed, color-vision-deficiency-safe palette derived from Okabe--Ito.  Every
# semantic category also has a marker/linestyle so the figures remain legible in
# grayscale and when reproduced on a monochrome office printer.
PALETTE = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "bluish_green": "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "sky_blue": "#56B4E9",
    "charcoal": "#3F3F3F",
    "mid_gray": "#777777",
    "light_gray": "#B8B8B8",
}

C_RAW = PALETTE["blue"]
C_CTRL = PALETTE["vermillion"]
C_JUDGE = PALETTE["charcoal"]
C_SHALLOW = PALETTE["bluish_green"]
C_GSM = PALETTE["blue"]
C_MATH = PALETTE["vermillion"]
C_ZERO = PALETTE["mid_gray"]

METHOD_STYLE = {
    "raw": dict(color=C_RAW, linestyle="-", marker="o"),
    "controlled": dict(color=C_CTRL, linestyle="-.", marker="^"),
    "judge": dict(color=C_JUDGE, linestyle="--", marker="s"),
    "shallow": dict(color=C_SHALLOW, linestyle=":", marker="D"),
}
METHOD_LABEL = {
    "raw": "Raw MLP",
    "controlled": "Controlled MLP",
    "judge": r"Judge $p(\mathrm{YES})$",
    "shallow": "Shallow features",
}
DATASET_STYLE = {
    "gsm8k": dict(color=C_GSM, marker="o"),
    "math": dict(color=C_MATH, marker="s"),
}
MODEL_STYLE = {
    "gemma3": dict(color=PALETTE["blue"], linestyle="-", marker="o"),
    "qwen25_7b": dict(color=PALETTE["vermillion"], linestyle="--", marker="s"),
    "llama31_8b": dict(color=PALETTE["bluish_green"], linestyle="-.", marker="^"),
}

LW_MAIN = 1.55
LW_SECONDARY = 1.30
LW_REFERENCE = 0.80
MARKER_SIZE = 4.2
MARKER_EDGE_WIDTH = 0.75


def log(message: str) -> None:
    print(message, flush=True)


def dump_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path: Path, hash_file: bool = False) -> dict:
    s = path.stat()
    out = {"path": str(path.relative_to(ROOT)), "bytes": s.st_size,
           "mtime_ns": s.st_mtime_ns}
    if hash_file:
        out["sha256"] = sha256(path)
    return out


def run_paths(tag: str, cfg: dict) -> tuple[Path, Path, Path, Path]:
    tr = ROOT / f"results_{tag}_train"
    te = ROOT / f"results_{tag}_test"
    dataset = cfg["dataset"]
    return (tr / "hidden_rich.npz", tr / dataset / "raw.jsonl",
            te / "hidden_rich.npz", te / dataset / "raw.jsonl")


def prediction_path(tag: str) -> Path:
    return OUT / "aligned_predictions" / f"{tag}.npz"


def metadata_path(tag: str) -> Path:
    return OUT / "aligned_predictions" / f"{tag}.metadata.json"


def read_k5_majority(indices: list[int]) -> tuple[np.ndarray, int, Path]:
    raw_path = ROOT / "results_llama31_8b_k5_test" / "gsm8k" / "raw.jsonl"
    records = {}
    with raw_path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[int(r["idx"])] = r
    missing = [i for i in indices if i not in records]
    if missing:
        raise ValueError(f"k=5 raw file lacks {len(missing)} aligned indices")
    first = records[indices[0]]
    k = len(first.get("k_answers") or [])
    if k <= 1:
        raise ValueError("routing artifact does not contain multiple answers")
    majority = np.array([int(bool(records[i].get("majority_correct"))) for i in indices],
                        dtype=np.int8)
    return majority, k, raw_path


def compute_aligned_predictions(force: bool = False) -> None:
    """Fit the frozen primary probe and save one aligned score per held-out item."""
    (OUT / "aligned_predictions").mkdir(parents=True, exist_ok=True)
    for tag, cfg in RUNS.items():
        out_path = prediction_path(tag)
        if out_path.exists() and metadata_path(tag).exists() and not force:
            log(f"[predictions] {tag}: reuse {out_path.relative_to(ROOT)}")
            continue

        started = time.time()
        tr_hidden, tr_raw, te_hidden, te_raw = run_paths(tag, cfg)
        log(f"[predictions] {tag}: load train/test dumps")
        tr = load_split(str(tr_hidden), str(tr_raw), cfg["head_dim"])
        te = load_split(str(te_hidden), str(te_raw), cfg["head_dim"])
        check_split_provenance(tr, te)
        Ftr, Fte, top, head_aucs = prepare_head_features(tr.Z, tr.y, te.Z, TOPK)

        route = None
        route_raw_path = None
        if tag == "llama31_8b":
            route_dir = ROOT / "results_llama31_8b_k5_test"
            log("[predictions] llama31_8b: also score held-out k=5 routing run")
            route = load_split(str(route_dir / "hidden_rich.npz"),
                               str(route_dir / "gsm8k" / "raw.jsonl"), cfg["head_dim"])
            # The selected head set is train-only.  Concatenating the flattened held-out
            # targets lets one set of fitted MLPs score both without refitting.
            Froute = route.Z[:, top, :].reshape(len(route.y), -1).astype(np.float32)
            Fall = np.concatenate([Fte, Froute], axis=0)
            Sall = np.concatenate([te.S, route.S], axis=0)
        else:
            Fall, Sall = Fte, te.S

        log(f"[predictions] {tag}: fit {N_SEEDS}-seed raw/controlled MLP ensemble")
        scores, seeds, _ = fit_ensemble(Ftr, tr.y, Fall, tr.S, Sall,
                                        model_type="mlp", seed=0, n_seeds=N_SEEDS)
        _, shallow = shallow_only_scores(tr.S, tr.y, Sall, model_type="lr", seed=0)
        n_test = len(te.y)
        arrays = {
            "idx": np.asarray(te.idx, dtype=np.int64),
            "y": te.y.astype(np.int8),
            "p_yes": te.p_yes.astype(np.float32),
            "verdict_yes": te.yes.astype(np.int8),
            "mlp_raw": np.asarray(scores["raw"][:n_test], dtype=np.float32),
            "mlp_controlled": np.asarray(scores["controlled"][:n_test], dtype=np.float32),
            "shallow": np.asarray(shallow[:n_test], dtype=np.float32),
        }

        route_meta = None
        if route is not None:
            majority, k, route_raw_path = read_k5_majority(route.idx)
            arrays.update({
                "route_idx": np.asarray(route.idx, dtype=np.int64),
                "route_greedy_correct": route.y.astype(np.int8),
                "route_majority_correct": majority,
                "route_p_yes": route.p_yes.astype(np.float32),
                "route_verdict_yes": route.yes.astype(np.int8),
                "route_mlp_raw": np.asarray(scores["raw"][n_test:], dtype=np.float32),
                "route_mlp_controlled": np.asarray(scores["controlled"][n_test:],
                                                     dtype=np.float32),
                "route_shallow": np.asarray(shallow[n_test:], dtype=np.float32),
            })
            route_meta = {"n": len(route.y), "wrong": int(np.sum(route.y == 0)), "k": k,
                          "label_info": route.info}

        # Compressed score files are small enough to audit and share; the multi-GB
        # activation dumps remain the immutable source evidence.
        np.savez_compressed(out_path, **arrays)
        cache_path = PAPER_CACHE / f"metrics_{tag}.json"
        cache = json.loads(cache_path.read_text())
        if int(cache["n_test"]) != len(te.y) or int(cache["wrong_test"]) != int(np.sum(te.y == 0)):
            raise AssertionError(f"{tag}: aligned population differs from paper cache")

        meta = {
            "tag": tag, "primary_method": "raw_mlp_top32_5_seed_probability_ensemble",
            "topk": TOPK, "seeds": seeds, "n_train": len(tr.y), "n_test": len(te.y),
            "wrong_train": int(np.sum(tr.y == 0)), "wrong_test": int(np.sum(te.y == 0)),
            "selected_flat_heads": [int(x) for x in top],
            "selected_head_train_lda_auc": [float(head_aucs[x]) for x in top],
            "train_label_info": tr.info, "test_label_info": te.info,
            "route": route_meta,
            "sources": [file_record(tr_hidden), file_record(tr_raw, True),
                        file_record(te_hidden), file_record(te_raw, True),
                        file_record(cache_path, True)],
            "software": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                         "matplotlib": matplotlib.__version__},
            "elapsed_seconds": time.time() - started,
        }
        if route is not None and route_raw_path is not None:
            route_hidden = ROOT / "results_llama31_8b_k5_test" / "hidden_rich.npz"
            meta["sources"].extend([file_record(route_hidden),
                                    file_record(route_raw_path, True)])
        dump_json(metadata_path(tag), meta)
        log(f"[predictions] {tag}: wrote {out_path.relative_to(ROOT)} "
            f"({time.time() - started:.0f}s)")

        del tr, te, route, Ftr, Fte, Fall, Sall, scores, shallow, arrays
        gc.collect()


def _groups(sorted_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    ends = np.r_[starts[1:], len(sorted_scores)]
    return starts, ends


def detector_metrics(y_correct: np.ndarray, score_correct: np.ndarray,
                     include_curve: bool = False) -> dict:
    """Tie-exact AUROC/AP and tie-averaged selective risk from correctness scores."""
    y = np.asarray(y_correct, dtype=np.int8)
    score = np.asarray(score_correct, dtype=float)
    if not np.isfinite(score).all() or len(np.unique(y)) != 2:
        raise ValueError("detector metrics require finite scores and two correctness classes")
    wrong = 1 - y
    n = len(y)
    n_wrong, n_correct = int(wrong.sum()), int(y.sum())
    order = np.argsort(score, kind="mergesort")       # most likely wrong first
    ss, yy, ww = score[order], y[order], wrong[order]
    starts, ends = _groups(ss)

    # Average ranks make AUROC invariant to arbitrary ordering within tied p(YES)
    # blocks. Ranks are one-indexed in the Mann-Whitney identity.
    correct_rank_sum = 0.0
    for a, b in zip(starts, ends):
        avg_rank = ((a + 1) + b) / 2.0
        correct_rank_sum += float(yy[a:b].sum()) * avg_rank
    auroc = ((correct_rank_sum - n_correct * (n_correct + 1) / 2.0)
             / (n_correct * n_wrong))

    # Error-detection AP: ascending correctness is descending error score. sklearn's
    # non-interpolated AP evaluates precision after an entire tied threshold block.
    seen = tp = 0
    ap = 0.0
    for a, b in zip(starts, ends):
        group_wrong = int(ww[a:b].sum())
        seen += b - a
        tp += group_wrong
        if group_wrong:
            ap += (group_wrong / n_wrong) * (tp / seen)

    # Expected number of wrong items after selecting m rows when a cutoff bisects a
    # tie. This is the exact expectation under uniform random tie-breaking and avoids
    # silently favoring the stable file order when p(YES) saturates.
    flagged_wrong = np.empty(n, dtype=float)
    cumulative = 0.0
    for a, b in zip(starts, ends):
        group_wrong = float(ww[a:b].sum())
        group_n = b - a
        flagged_wrong[a:b] = cumulative + np.arange(1, group_n + 1) * group_wrong / group_n
        cumulative += group_wrong

    # Accept highest scores first. Reverse the group order but preserve the same
    # within-tie expectation.
    accepted_wrong = np.empty(n, dtype=float)
    cumulative = 0.0
    pos = 0
    for a, b in zip(starts[::-1], ends[::-1]):
        group_wrong = float(ww[a:b].sum())
        group_n = b - a
        accepted_wrong[pos:pos + group_n] = (
            cumulative + np.arange(1, group_n + 1) * group_wrong / group_n)
        cumulative += group_wrong
        pos += group_n
    risk_curve = accepted_wrong / np.arange(1, n + 1)

    budgets = {}
    for budget in BUDGETS:
        m = max(1, min(n - 1, int(np.floor(float(budget) * n + 0.5))))
        caught = float(flagged_wrong[m - 1])
        budgets[f"{budget:.4f}"] = {
            "n_reviewed": m,
            "recall": caught / n_wrong,
            "residual_risk": (n_wrong - caught) / (n - m),
        }
    out = {"auroc": float(auroc), "ap_error": float(ap),
           "aurc": float(risk_curve.mean()), "budgets": budgets}
    if include_curve:
        out["coverage"] = (np.arange(1, n + 1) / n).tolist()
        out["risk_curve"] = risk_curve.tolist()
    return out


def interval(draws: list[float] | np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(np.asarray(draws), [2.5, 97.5])]


def paired_bootstrap(y: np.ndarray, raw: np.ndarray, judge: np.ndarray,
                     n_boot: int, seed: int) -> dict:
    """Paired, class-stratified test-item bootstrap conditional on fitted scores."""
    rng = np.random.default_rng(seed)
    correct = np.flatnonzero(y == 1)
    wrong = np.flatnonzero(y == 0)
    names = [f"{b:.4f}" for b in BUDGETS]
    draws = {"delta_auroc": [], "delta_ap_error": [], "delta_aurc": []}
    for name in names:
        draws[f"risk_gain_{name}"] = []
        draws[f"recall_gain_{name}"] = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(correct, len(correct), replace=True),
                              rng.choice(wrong, len(wrong), replace=True)])
        mr = detector_metrics(y[idx], raw[idx])
        mj = detector_metrics(y[idx], judge[idx])
        draws["delta_auroc"].append(mr["auroc"] - mj["auroc"])
        draws["delta_ap_error"].append(mr["ap_error"] - mj["ap_error"])
        draws["delta_aurc"].append(mj["aurc"] - mr["aurc"])
        for name in names:
            draws[f"risk_gain_{name}"].append(
                mj["budgets"][name]["residual_risk"] -
                mr["budgets"][name]["residual_risk"])
            draws[f"recall_gain_{name}"].append(
                mr["budgets"][name]["recall"] - mj["budgets"][name]["recall"])
    return {name: {"ci95": interval(values)} for name, values in draws.items()}


def expected_route_gain(score: np.ndarray, gain: np.ndarray,
                        fracs: np.ndarray) -> np.ndarray:
    """Gain over random routing, averaging exactly over score ties."""
    order = np.argsort(np.asarray(score), kind="mergesort")
    ss = np.asarray(score, dtype=float)[order]
    gg = np.asarray(gain, dtype=float)[order]
    starts, ends = _groups(ss)
    selected_gain = np.empty(len(score), dtype=float)
    cumulative = 0.0
    for a, b in zip(starts, ends):
        group_gain = float(gg[a:b].sum())
        group_n = b - a
        selected_gain[a:b] = cumulative + np.arange(1, group_n + 1) * group_gain / group_n
        cumulative += group_gain
    out = []
    n = len(score)
    for frac in fracs:
        m = int(np.floor(float(frac) * n + 0.5))
        chosen = 0.0 if m == 0 else float(selected_gain[m - 1])
        # The implemented policy routes exactly m=round(frac*n) items.  Its random
        # null therefore routes m/n, not the nominal fraction (the two differ by up
        # to half an item).  This keeps "same compute" exact even for finite n.
        effective_frac = m / n
        out.append(chosen / n - effective_frac * float(gain.mean()))
    return np.asarray(out)


def compute_route_statistics(pred: np.lib.npyio.NpzFile, n_boot: int) -> dict:
    y = pred["route_greedy_correct"].astype(int)
    majority = pred["route_majority_correct"].astype(int)
    gain = majority - y
    fracs = np.linspace(0, 1, 21)
    scores = {"mlp_raw": pred["route_mlp_raw"],
              "mlp_controlled": pred["route_mlp_controlled"],
              "p_yes": pred["route_p_yes"]}
    point = {name: expected_route_gain(score, gain, fracs) for name, score in scores.items()}
    rng = np.random.default_rng(BOOT_SEED + 99)
    draws = {name: np.empty((n_boot, len(fracs))) for name in scores}
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for name, score in scores.items():
            draws[name][b] = expected_route_gain(score[idx], gain[idx], fracs)
    methods = {}
    for name in scores:
        methods[name] = {
            "gain_over_random": point[name].tolist(),
            "ci95_lo": np.percentile(draws[name], 2.5, axis=0).tolist(),
            "ci95_hi": np.percentile(draws[name], 97.5, axis=0).tolist(),
        }
    delta = draws["mlp_raw"] - draws["p_yes"]
    return {"n": len(y), "k": 5, "greedy_accuracy": float(y.mean()),
            "majority_accuracy": float(majority.mean()), "fractions": fracs.tolist(),
            "solve_calls_per_item": (1 + 4 * fracs).tolist(), "methods": methods,
            "raw_minus_p_yes": {
                "point": (point["mlp_raw"] - point["p_yes"]).tolist(),
                "ci95_lo": np.percentile(delta, 2.5, axis=0).tolist(),
                "ci95_hi": np.percentile(delta, 97.5, axis=0).tolist(),
            },
            "bootstrap": {"n": n_boot, "seed": BOOT_SEED + 99,
                          "unit": "held-out routing item", "paired": True}}


def cache_comparison(tag: str, metrics: dict) -> dict:
    cache = json.loads((PAPER_CACHE / f"metrics_{tag}.json").read_text())
    values = {
        "n_test": [int(metrics["n"]), int(cache["n_test"])],
        "wrong_test": [metrics["wrong"], int(cache["wrong_test"])],
        "auc_mlp_raw": [metrics["raw"]["auroc"], float(cache["auc_mlp_raw"])],
        "auc_mlp_controlled": [metrics["controlled"]["auroc"],
                               float(cache["auc_mlp_controlled"])],
        "auc_p_yes": [metrics["judge"]["auroc"], float(cache["auc_p_yes"])],
        "auc_shallow": [metrics["shallow"]["auroc"], float(cache["auc_shallow"])],
    }
    # Population and the non-fitted judge score must match exactly. MLP values can
    # move slightly across scikit-learn versions, so preserve both rather than hide it.
    if values["n_test"][0] != values["n_test"][1] or values["wrong_test"][0] != values["wrong_test"][1]:
        raise AssertionError(f"{tag}: population mismatch with source metric cache")
    if abs(values["auc_p_yes"][0] - values["auc_p_yes"][1]) > 1e-10:
        raise AssertionError(f"{tag}: p_yes AUROC mismatch with source metric cache")
    return {name: {"recomputed": pair[0], "paper_cache": pair[1],
                   "difference": pair[0] - pair[1]} for name, pair in values.items()}


def compute_statistics(n_boot: int) -> dict:
    report = {
        "schema_version": 1,
        "primary_method": "raw MLP, top 32 train-selected heads, five-seed ensemble",
        "baseline": "judge p(YES)",
        "score_semantics": "scores rank correctness; MLP outputs are not calibrated probabilities",
        "bootstrap": {"n": n_boot, "seed": BOOT_SEED,
                      "unit": "held-out test item", "paired": True,
                      "stratified_by_correctness": True,
                      "scope": "conditional on the fitted five-seed ensemble"},
        "tie_policy": "exact expectation under uniform random tie-breaking at the cutoff",
        "runs": {},
    }
    for run_index, (tag, cfg) in enumerate(RUNS.items()):
        path = prediction_path(tag)
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run with --compute")
        log(f"[statistics] {tag}: point metrics and {n_boot} paired bootstrap draws")
        with np.load(path) as p:
            y = p["y"].astype(int)
            raw = p["mlp_raw"].astype(float)
            controlled = p["mlp_controlled"].astype(float)
            judge = p["p_yes"].astype(float)
            shallow = p["shallow"].astype(float)
            mr = detector_metrics(y, raw, include_curve=True)
            mj = detector_metrics(y, judge, include_curve=True)
            mc = detector_metrics(y, controlled, include_curve=True)
            ms = detector_metrics(y, shallow, include_curve=True)
            boot = paired_bootstrap(y, raw, judge, n_boot,
                                    BOOT_SEED + 1009 * run_index)
            deltas = {
                "auroc": {"point": mr["auroc"] - mj["auroc"],
                          **boot["delta_auroc"]},
                "ap_error": {"point": mr["ap_error"] - mj["ap_error"],
                             **boot["delta_ap_error"]},
                "aurc_improvement": {"point": mj["aurc"] - mr["aurc"],
                                     **boot["delta_aurc"]},
                "budgets": {},
            }
            for budget in BUDGETS:
                key = f"{budget:.4f}"
                deltas["budgets"][key] = {
                    "residual_risk_improvement": {
                        "point": (mj["budgets"][key]["residual_risk"] -
                                  mr["budgets"][key]["residual_risk"]),
                        **boot[f"risk_gain_{key}"]},
                    "recall_improvement": {
                        "point": (mr["budgets"][key]["recall"] -
                                  mj["budgets"][key]["recall"]),
                        **boot[f"recall_gain_{key}"]},
                }
            run = {"tag": tag, "model": cfg["model"], "model_label": cfg["model_label"],
                   "dataset": cfg["dataset"], "n": len(y), "wrong": int(np.sum(y == 0)),
                   "wrong_prevalence": float(np.mean(y == 0)), "raw": mr,
                   "controlled": mc, "judge": mj, "shallow": ms, "delta_raw_vs_judge": deltas,
                   "prediction_file": file_record(path, True)}
            run["cache_comparison"] = cache_comparison(tag, run)
            report["runs"][tag] = run
            if tag == "llama31_8b" and "route_mlp_raw" in p.files:
                report["routing"] = compute_route_statistics(p, n_boot)

    dump_json(OUT / "statistics.json", report)
    return report


def load_statistics() -> dict:
    return json.loads((OUT / "statistics.json").read_text())


def set_ieee_style() -> None:
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "serif",
        # Liberation Serif is a metrically Times-compatible TrueType font.  Keeping
        # it first avoids the CID Type-0C mismatch warnings emitted by some IEEE PDF
        # preflight tools for the previously selected Nimbus Roman OTF files.
        "font.serif": ["Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.0, "axes.titlesize": 8.2,
        "axes.titleweight": "bold", "axes.labelsize": 8.0,
        "legend.fontsize": 7.0, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "axes.linewidth": 0.75, "lines.linewidth": LW_MAIN,
        "lines.markersize": MARKER_SIZE, "lines.markeredgewidth": MARKER_EDGE_WIDTH,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "axes.axisbelow": True,
        "grid.color": "#D2D2D2", "grid.alpha": 0.62, "grid.linewidth": 0.45,
        "legend.frameon": False, "legend.handlelength": 2.4,
        "legend.handletextpad": 0.55, "legend.columnspacing": 1.25,
        "axes.prop_cycle": plt.cycler(color=[C_RAW, C_CTRL, C_SHALLOW,
                                               PALETTE["purple"], PALETTE["orange"],
                                               PALETTE["sky_blue"]]),
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
        # Constrained layout keeps all artists inside the declared IEEE canvas.
        # Do not tight-crop: that silently changes the nominal 3.5/7.16-inch width.
        "savefig.facecolor": "white", "savefig.bbox": None,
    })


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    """Apply the shared low-ink IEEE axis treatment."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis)
    ax.tick_params(which="both", top=False, right=False)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    log(f"[figure] {stem}.pdf/.png")


def condition_tags() -> list[str]:
    return [f"{m}{'_math' if d == 'math' else ''}"
            for d in ("gsm8k", "math") for m in MODEL_ORDER]


def condition_label(run: dict) -> str:
    dataset = "GSM8K" if run["dataset"] == "gsm8k" else "MATH"
    return f"{run['model_label']} — {dataset}"


def plot_selective_risk(report: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.05), sharex=True, sharey=True,
                             constrained_layout=True)
    for row, dataset in enumerate(("gsm8k", "math")):
        for col, model in enumerate(MODEL_ORDER):
            tag = f"{model}{'_math' if dataset == 'math' else ''}"
            run = report["runs"][tag]
            ax = axes[row, col]
            points, lo, hi = [], [], []
            for b in BUDGETS:
                m = run["delta_raw_vs_judge"]["budgets"][f"{b:.4f}"]["residual_risk_improvement"]
                points.append(100 * m["point"])
                lo.append(100 * m["ci95"][0])
                hi.append(100 * m["ci95"][1])
            x = 100 * BUDGETS
            dataset_style = DATASET_STYLE[dataset]
            ax.axhline(0, color=C_ZERO, lw=LW_REFERENCE, ls="--", zorder=1)
            ax.fill_between(x, lo, hi, color=C_RAW, alpha=0.16, lw=0)
            ax.plot(x, points, color=C_RAW, ls=METHOD_STYLE["raw"]["linestyle"],
                    marker=dataset_style["marker"], ms=3.5, lw=LW_MAIN,
                    mfc=C_RAW, mec="white", mew=0.45, zorder=3)
            j = list(BUDGETS).index(0.05)
            ax.plot(x[j], points[j], marker=dataset_style["marker"], ms=5.0,
                    color=C_RAW, mec="white", mew=0.7, zorder=4)
            ax.set_title(MODEL_SHORT[model])
            ax.text(0.03, 0.92, "GSM8K" if dataset == "gsm8k" else "MATH",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
                    fontweight="bold", color=dataset_style["color"])
            ax.text(0.97, 0.06, f"({chr(97 + row * 3 + col)})", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.0, fontweight="bold")
            if row == 1:
                ax.set_xlabel("Answers reviewed (%)")
            if col == 0:
                ax.set_ylabel("Residual wrong-rate reduction\nvs. judge (percentage points)")
            style_axis(ax, "y")
    save_figure(fig, "fig1_selective_risk")


def plot_discrimination(report: dict) -> None:
    tags = condition_tags()
    labels = [condition_label(report["runs"][t]) for t in tags]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), sharey=True,
                             constrained_layout=True)
    for ax, metric, title in zip(axes, ("auroc", "ap_error"),
                                 ("Correctness AUROC", "Error-detection AP")):
        ax.axvline(0, color=C_ZERO, lw=LW_REFERENCE, ls="--")
        for i, tag in enumerate(tags):
            run = report["runs"][tag]
            d = run["delta_raw_vs_judge"][metric]
            dataset_style = DATASET_STYLE[run["dataset"]]
            color = dataset_style["color"]
            ax.errorbar(d["point"], i, xerr=[[d["point"] - d["ci95"][0]],
                                              [d["ci95"][1] - d["point"]]],
                        fmt=dataset_style["marker"], color=color, ecolor=color,
                        mfc=color, mec="white", mew=0.55, ms=MARKER_SIZE,
                        capsize=2.3, capthick=0.9, elinewidth=1.05)
        ax.set_title(title)
        ax.set_xlabel(r"Raw MLP $-$ judge $p(\mathrm{YES})$")
        ax.set_yticks(range(len(tags)), labels)
        ax.invert_yaxis()
        style_axis(ax, "x")
    axes[0].text(0.02, 0.04, "(a)", transform=axes[0].transAxes,
                 fontweight="bold")
    axes[1].text(0.02, 0.04, "(b)", transform=axes[1].transAxes,
                 fontweight="bold")
    save_figure(fig, "fig2_discrimination")


def plot_routing(report: dict) -> None:
    route = report["routing"]
    x = np.asarray(route["solve_calls_per_item"])
    fig, ax = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    ax.axhline(0, color=C_ZERO, lw=LW_REFERENCE, ls="--")
    styles = [("mlp_raw", METHOD_LABEL["raw"], "raw", LW_MAIN),
              ("p_yes", METHOD_LABEL["judge"], "judge", LW_SECONDARY),
              ("mlp_controlled", METHOD_LABEL["controlled"],
               "controlled", LW_SECONDARY)]
    for name, label, style_name, lw in styles:
        style = METHOD_STYLE[style_name]
        d = route["methods"][name]
        y = 100 * np.asarray(d["gain_over_random"])
        lo = 100 * np.asarray(d["ci95_lo"])
        hi = 100 * np.asarray(d["ci95_hi"])
        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.10, lw=0)
        ax.plot(x, y, color=style["color"], ls=style["linestyle"], lw=lw,
                marker=style["marker"], ms=3.3, markevery=2,
                mfc=style["color"], mec="white", mew=0.4, label=label)
    j = route["fractions"].index(0.1)
    ax.axvline(x[j], color=PALETTE["light_gray"], lw=LW_REFERENCE,
               ls=(0, (2, 2)))
    raw10 = 100 * route["methods"]["mlp_raw"]["gain_over_random"][j]
    ax.plot(x[j], raw10, METHOD_STYLE["raw"]["marker"], color=C_RAW,
            ms=5.0, mec="white", mew=0.7)
    ax.set_xlabel("Solve calls per problem")
    ax.set_ylabel("Accuracy gain over random routing\n(percentage points)")
    ax.set_xlim(1, 5)
    ax.set_xticks([1, 1.4, 2, 3, 4, 5])
    # The intervals occupy most of the panel; an opaque, borderless key keeps the
    # labels readable without covering the center lines visually.
    ax.legend(loc="upper right", frameon=True, framealpha=0.92,
              facecolor="white", edgecolor="none")
    style_axis(ax, "both")
    save_figure(fig, "fig3_selective_routing")


def plot_transfer() -> None:
    d = json.loads((PAPER_CACHE / "transfer_compare.json").read_text())
    panels = [("mlp_raw", METHOD_LABEL["raw"], "raw"),
              ("mlp_controlled", "Controlled MLP (ablation)", "controlled"),
              ("shallow", "Shallow features (ablation)", "shallow"),
              ("judge_p_yes", r"Judge $p(\mathrm{YES})$ (target-only)", "judge")]
    targets = [(m, ds) for ds in ("gsm8k", "math") for m in MODEL_ORDER]
    labels = [f"{MODEL_SHORT[m]} — {'GSM8K' if ds == 'gsm8k' else 'MATH'}"
              for m, ds in targets]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.35), sharex=True, sharey=True,
                             constrained_layout=True)
    for panel_i, (metric, title, style_name) in enumerate(panels):
        ax = axes.flat[panel_i]
        method_color = METHOD_STYLE[style_name]["color"]
        for i, (model, target) in enumerate(targets):
            model_data = d["models"][model]
            if target == "gsm8k":
                in_key, cross_key = "gsm8k->gsm8k", "math->gsm8k"
            else:
                in_key, cross_key = "math->math", "gsm8k->math"
            in_value = float(model_data[in_key][metric])
            cross_value = float(model_data[cross_key][metric])
            if metric == "judge_p_yes":
                # A diamond is deliberately distinct from the filled square used
                # for a cross-dataset-trained probe: p(YES) is target-only and no
                # probe training-domain comparison exists in this panel.
                ax.plot(in_value, i, marker="D",
                        color=C_JUDGE, mfc=C_JUDGE, mec="white", mew=0.5,
                        ls="none", ms=MARKER_SIZE)
            else:
                ax.plot([cross_value, in_value], [i, i], color=PALETTE["light_gray"],
                        lw=1.0, solid_capstyle="round")
                ax.plot(in_value, i, "o", mfc="white", mec=method_color,
                        mew=1.1, ms=4.6)
                ax.plot(cross_value, i, "s", color=method_color, mfc=method_color,
                        mec="white", mew=0.45, ms=4.2)
        ax.set_title(title)
        ax.set_yticks(range(len(targets)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0.35, 0.96)
        ax.set_xlabel("Held-out AUROC")
        ax.text(0.02, 0.04, f"({chr(97 + panel_i)})", transform=ax.transAxes,
                fontweight="bold")
        style_axis(ax, "x")
    handles = [plt.Line2D([], [], marker="o", mfc="white", mec=C_JUDGE, color="none",
                          label="in-domain probe"),
               plt.Line2D([], [], marker="s", mfc=C_JUDGE, mec="white", color=C_JUDGE,
                          linestyle="none",
                          label="cross-dataset probe"),
               plt.Line2D([], [], marker="D", mfc=C_JUDGE, mec="white", color=C_JUDGE,
                          linestyle="none", label="target-only judge score")]
    fig.legend(handles=handles, ncols=3, loc="outside upper center")
    save_figure(fig, "fig4_cross_dataset_transfer")


def plot_risk_coverage(report: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.20), sharex=True,
                             constrained_layout=True)
    for row, dataset in enumerate(("gsm8k", "math")):
        for col, model in enumerate(MODEL_ORDER):
            tag = f"{model}{'_math' if dataset == 'math' else ''}"
            run = report["runs"][tag]
            ax = axes[row, col]
            for name in ("raw", "judge", "controlled"):
                m = run[name]
                cov = 100 * np.asarray(m["coverage"])
                risk = 100 * np.asarray(m["risk_curve"])
                mask = cov >= 10
                style = METHOD_STYLE[name]
                ax.plot(cov[mask], risk[mask], color=style["color"],
                        ls=style["linestyle"], lw=(LW_MAIN if name == "raw" else LW_SECONDARY),
                        label=METHOD_LABEL[name])
            ax.axhline(100 * run["wrong_prevalence"], color=C_ZERO,
                       lw=LW_REFERENCE, ls=(0, (2, 2)))
            ax.set_title(f"{MODEL_SHORT[model]} — {'GSM8K' if dataset == 'gsm8k' else 'MATH'}")
            ax.set_xlim(10, 100)
            if row == 1:
                ax.set_xlabel("Coverage retained (%)")
            if col == 0:
                ax.set_ylabel("Wrong among retained (%)")
            ax.text(0.97, 0.06, f"({chr(97 + row * 3 + col)})",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.0, fontweight="bold")
            style_axis(ax, "both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Figure-level outside placement is included in constrained layout and prevents
    # the key from colliding with the middle top-row title at two-column width.
    fig.legend(handles, labels, ncols=3, loc="outside upper center")
    save_figure(fig, "figA1_risk_coverage")


def load_sweeps() -> dict:
    runs = {}
    for path in sorted((FILTERING / "figures").glob("sweep_defaults*.json")):
        data = json.loads(path.read_text())
        for tag, value in data["runs"].items():
            if tag in runs and runs[tag] != value:
                raise ValueError(f"conflicting sweep cache for {tag}")
            runs[tag] = value
    expected = set(RUNS)
    if set(runs) != expected:
        raise ValueError(f"sweep coverage mismatch: expected {expected}, got {set(runs)}")
    return runs


def plot_sensitivity() -> None:
    sweeps = load_sweeps()
    tags = condition_tags()
    labels = [f"{RUNS[t]['model_label']} — {'GSM8K' if RUNS[t]['dataset'] == 'gsm8k' else 'MATH'}"
              for t in tags]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), sharey=True,
                             constrained_layout=True)
    for panel_i, (ax, variant, title) in enumerate(
            zip(axes, ("raw", "controlled"),
                ("Raw features", "Shallow-controlled features"))):
        ax.axvline(0, color=C_ZERO, lw=LW_REFERENCE, ls="--")
        for i, tag in enumerate(tags):
            rows = sweeps[tag]["rows"]
            gaps = np.array([r[f"mlp_{variant}"] - r[f"lr_{variant}"] for r in rows])
            default = next(r for r in rows if r["axis"] == "default")
            point = default[f"mlp_{variant}"] - default[f"lr_{variant}"]
            dataset_style = DATASET_STYLE[RUNS[tag]["dataset"]]
            color = dataset_style["color"]
            ax.plot([gaps.min(), gaps.max()], [i, i], color=color, lw=2.2,
                    solid_capstyle="round")
            ax.plot(point, i, marker=dataset_style["marker"], mfc="white", mec=color,
                    mew=1.05, ms=4.4, ls="none")
        ax.set_title(title)
        ax.set_xlabel("MLP AUROC $-$ LR AUROC")
        ax.set_yticks(range(len(tags)), labels)
        ax.invert_yaxis()
        ax.text(0.02, 0.04, f"({chr(97 + panel_i)})",
                transform=ax.transAxes, fontweight="bold")
        style_axis(ax, "x")
    save_figure(fig, "figA2_probe_sensitivity")


def plot_symbolic() -> None:
    d = json.loads((PAPER_CACHE / "symbolic_compare.json").read_text())
    fig, ax = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)
    ax.axvline(0, color=C_ZERO, lw=LW_REFERENCE, ls="--")
    y = np.arange(len(MODEL_ORDER))
    for offset, metric, label, style_name in (
            (-0.12, "mlp_raw", "Raw MLP", "raw"),
            (0.12, "mlp_controlled", "Controlled MLP", "controlled")):
        style = METHOD_STYLE[style_name]
        for i, model in enumerate(MODEL_ORDER):
            v = d["models"][model]["delta_symbolic_minus_full"][metric]
            ax.errorbar(v["delta"], i + offset,
                        xerr=[[v["delta"] - v["ci_lo"]], [v["ci_hi"] - v["delta"]]],
                        fmt=style["marker"], color=style["color"], ecolor=style["color"],
                        mfc=style["color"], mec="white", mew=0.5,
                        ms=MARKER_SIZE, capsize=2.3, capthick=0.9, elinewidth=1.05,
                        label=label if i == 0 else None)
    ax.set_yticks(y, [MODEL_NAME[m] for m in MODEL_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("AUROC change: GSM-Symbolic $-$ GSM8K")
    # The three long intervals occupy every interior corner. Put the two-entry key
    # above the axes so it cannot obscure a result at single-column size.
    ax.legend(ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              columnspacing=1.2, handletextpad=0.5)
    style_axis(ax, "x")
    save_figure(fig, "figA3_symbolic_perturbation")


def plot_math_levels() -> None:
    d = json.loads((PAPER_CACHE / "level_breakdown.json").read_text())
    fig, ax = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)
    for model in MODEL_ORDER:
        levels = d["models"][model]["levels"]
        x = np.arange(1, 6)
        y = 100 * np.array([levels[f"Level {i}"]["mlp_minus_judge"] for i in x])
        style = MODEL_STYLE[model]
        ax.plot(x, y, marker=style["marker"], ms=4.0, color=style["color"],
                ls=style["linestyle"], lw=LW_MAIN, mfc=style["color"],
                mec="white", mew=0.45, label=MODEL_SHORT[model])
    ax.axhline(0, color=C_ZERO, lw=LW_REFERENCE, ls="--")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("MATH difficulty level")
    ax.set_ylabel("Raw MLP AUROC $-$ judge AUROC\n(percentage points)")
    # Match the other compact appendix plot and keep the key outside the data area.
    ax.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              columnspacing=1.0, handletextpad=0.5)
    style_axis(ax, "y")
    save_figure(fig, "figA4_math_level_breakdown")


def write_csvs(report: dict) -> None:
    with (OUT / "main_metrics.csv").open("w", newline="") as f:
        fields = ["tag", "model", "dataset", "n", "wrong", "wrong_prevalence",
                  "raw_auroc", "judge_auroc", "delta_auroc", "delta_auroc_ci_lo",
                  "delta_auroc_ci_hi", "raw_ap_error", "judge_ap_error", "delta_ap_error",
                  "delta_ap_ci_lo", "delta_ap_ci_hi", "raw_aurc", "judge_aurc",
                  "aurc_improvement"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tag in condition_tags():
            r = report["runs"][tag]
            da, dp = r["delta_raw_vs_judge"]["auroc"], r["delta_raw_vs_judge"]["ap_error"]
            w.writerow({"tag": tag, "model": r["model_label"], "dataset": r["dataset"],
                        "n": r["n"], "wrong": r["wrong"],
                        "wrong_prevalence": r["wrong_prevalence"],
                        "raw_auroc": r["raw"]["auroc"], "judge_auroc": r["judge"]["auroc"],
                        "delta_auroc": da["point"], "delta_auroc_ci_lo": da["ci95"][0],
                        "delta_auroc_ci_hi": da["ci95"][1],
                        "raw_ap_error": r["raw"]["ap_error"],
                        "judge_ap_error": r["judge"]["ap_error"],
                        "delta_ap_error": dp["point"], "delta_ap_ci_lo": dp["ci95"][0],
                        "delta_ap_ci_hi": dp["ci95"][1], "raw_aurc": r["raw"]["aurc"],
                        "judge_aurc": r["judge"]["aurc"],
                        "aurc_improvement": r["delta_raw_vs_judge"]["aurc_improvement"]["point"]})

    with (OUT / "fixed_budget_metrics.csv").open("w", newline="") as f:
        fields = ["tag", "dataset", "budget", "n_reviewed", "raw_recall", "judge_recall",
                  "recall_gain", "recall_gain_ci_lo", "recall_gain_ci_hi",
                  "raw_residual_risk", "judge_residual_risk", "residual_risk_improvement",
                  "risk_improvement_ci_lo", "risk_improvement_ci_hi"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tag in condition_tags():
            r = report["runs"][tag]
            for b in BUDGETS:
                key = f"{b:.4f}"
                rb, jb = r["raw"]["budgets"][key], r["judge"]["budgets"][key]
                dg = r["delta_raw_vs_judge"]["budgets"][key]
                rg, qg = dg["recall_improvement"], dg["residual_risk_improvement"]
                w.writerow({"tag": tag, "dataset": r["dataset"], "budget": b,
                            "n_reviewed": rb["n_reviewed"], "raw_recall": rb["recall"],
                            "judge_recall": jb["recall"], "recall_gain": rg["point"],
                            "recall_gain_ci_lo": rg["ci95"][0],
                            "recall_gain_ci_hi": rg["ci95"][1],
                            "raw_residual_risk": rb["residual_risk"],
                            "judge_residual_risk": jb["residual_risk"],
                            "residual_risk_improvement": qg["point"],
                            "risk_improvement_ci_lo": qg["ci95"][0],
                            "risk_improvement_ci_hi": qg["ci95"][1]})


def write_manifest() -> None:
    cache_names = ["route_compare.json", "transfer_compare.json", "symbolic_compare.json",
                   "level_breakdown.json"]
    sources = [file_record(PAPER_CACHE / name, True) for name in cache_names]
    sources.extend(file_record(path, True)
                   for path in sorted((FILTERING / "figures").glob("sweep_defaults*.json")))
    outputs = [file_record(path, True) for path in sorted(OUT.iterdir())
               if path.is_file() and path.name != "artifact_manifest.json"]
    dump_json(OUT / "artifact_manifest.json", {
        "generated_utc_epoch": time.time(),
        "script": file_record(Path(__file__), True),
        "cached_source_artifacts": sources,
        "generated_outputs_before_manifest": outputs,
        "note": "Activation-dump paths and raw JSON hashes are recorded per run in aligned_predictions/*.metadata.json.",
    })


def write_readme(report: dict) -> None:
    text = f"""# IEEE TPS 2026 figure package

This directory is generated from the repository's held-out experiment artifacts. The
primary method is fixed to the raw top-{TOPK}-head, {N_SEEDS}-seed MLP ensemble. The
controlled MLP, shallow-feature probe, and LR are ablations only.

## Reproduce

```bash
MPLCONFIGDIR=/tmp .venv/bin/python filtering/make_ieee_tps_figures.py --compute
```

Use `--force-predictions` only when the immutable source dumps or fitting code change.
Aligned predictions are retained so re-rendering does not require loading ~20 GB of
activations.

## Main figures

- `fig1_selective_risk.pdf`: residual wrong-rate reduction at fixed review budgets;
  paired 95% bootstrap intervals.
- `fig2_discrimination.pdf`: raw-MLP minus p(YES) AUROC and error-detection AP;
  paired 95% bootstrap intervals.
- `fig3_selective_routing.pdf`: accuracy gain over random compute allocation on the
  Llama/GSM8K k=5 run, including the raw primary probe.
- `fig4_cross_dataset_transfer.pdf`: in-domain versus cross-dataset AUROC; point
  estimates only because the source transfer cache has no aligned bootstrap draws.

## Appendix figures

- `figA1_risk_coverage.pdf`: full tie-averaged risk–coverage curves.
- `figA2_probe_sensitivity.pdf`: descriptive min–max gaps over the stored sweep grid;
  extrema are test-selected and are not confidence intervals.
- `figA3_symbolic_perturbation.pdf`: paired GSM-Symbolic AUROC changes.
- `figA4_math_level_breakdown.pdf`: descriptive MATH difficulty strata; no intervals.

Every figure is supplied as vector PDF and 600-dpi PNG. Main multi-panel figures use
IEEE double-column width (7.16 in); the routing and compact appendix figures use
single-column width (3.5 in). `statistics.json`, CSV tables, per-run metadata, and
`artifact_manifest.json` trace the plotted values to source files. Manuscript prose,
caption drafts, and LaTeX sources are intentionally excluded from this repository.

The figures use a fixed color-vision-deficiency-safe Okabe--Ito palette and embedded
Liberation Serif typography (a Times-compatible TrueType face). Method, dataset, and
model colors are paired with consistent line styles and markers so distinctions
survive grayscale printing.

## Metric definitions

- AUROC ranks correctness; AP ranks the minority error class.
- Risk–coverage accepts the highest correctness scores first. AURC is the mean
  selective risk over all non-empty coverages (lower is better).
- Fixed-budget recall is the fraction of all wrong answers captured among the lowest
  scoring `round(budget*n)` items.
- Fixed-budget residual risk is the wrong fraction among unreviewed items.
- Routing gain subtracts the expected accuracy of random routing at equal solve cost.
- Cutoffs that bisect score ties use the exact expectation under uniform random
  tie-breaking; this matters for saturated p(YES) values.

The MLP outputs are called *scores*, not calibrated probabilities. No ECE or Brier
score is reported because the current MLP oversamples the minority class and the LR
uses class weights without a separate train-only calibration stage.

## Reproducibility note

The rebuilt MLP scores use scikit-learn {sklearn.__version__}; per-run metadata records
the full analysis stack and preserves comparisons with the older paper caches. The
paper should disclose substantive generative-AI assistance as required by the IEEE
TPS 2026 call for papers.
"""
    (OUT / "README.md").write_text(text)


def render_all(report: dict) -> None:
    set_ieee_style()
    plot_selective_risk(report)
    plot_discrimination(report)
    plot_routing(report)
    plot_transfer()
    plot_risk_coverage(report)
    plot_sensitivity()
    plot_symbolic()
    plot_math_levels()
    write_csvs(report)
    write_readme(report)
    write_manifest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compute", action="store_true",
                    help="rebuild aligned predictions before computing statistics")
    ap.add_argument("--force-predictions", action="store_true",
                    help="overwrite existing aligned prediction caches")
    ap.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--render-only", action="store_true",
                    help="render from the existing statistics.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.compute or args.force_predictions:
        compute_aligned_predictions(force=args.force_predictions)
    if args.render_only:
        report = load_statistics()
    else:
        report = compute_statistics(args.n_bootstrap)
    render_all(report)
    log(f"done: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
