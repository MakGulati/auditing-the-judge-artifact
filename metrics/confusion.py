from __future__ import annotations

from typing import Any

import numpy as np

from metrics.correctness import is_labelable, relabel_result


def _is_positive(val: Any, positive_marker: Any) -> bool:
    if isinstance(val, str):
        return val.strip().upper() == str(positive_marker).strip().upper()
    return bool(val) == bool(positive_marker)


def build_matrix(
    results: list[dict[str, Any]],
    truth_key: str,
    pred_key: str,
    pred_positive: Any = "YES",
) -> np.ndarray:
    """
    Build [[TP, FN], [FP, TN]] confusion matrix.

    truth_key  : key of a bool field (True = positive class)
    pred_key   : key of a str or bool field
    pred_positive : value that counts as a positive prediction
    """
    tp = fn = fp = tn = 0
    for r in results:
        truth = bool(r[truth_key])
        pred = _is_positive(r[pred_key], pred_positive)
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif not truth and pred:
            fp += 1
        else:
            tn += 1
    return np.array([[tp, fn], [fp, tn]], dtype=int)


def _truncation_line(results: list[dict[str, Any]], n: int) -> str:
    """How many solutions the token budget cut off before they could finish.

    Worth its own line next to the parse-failure rate because the two are easy to
    confuse and mean opposite things. A parse failure is the model ignoring the output
    contract; a truncation is us not letting it finish, and it is recorded as a wrong
    answer either way. On MATH at the GSM8K-sized budget this reached 24%.
    """
    known = [r for r in results if r.get("truncated") is not None]
    if not known:
        return ("Phase 1 truncated (budget)       : unknown "
                "(backend did not report finish_reason)")
    n_trunc = sum(1 for r in known if r["truncated"])
    suffix = "" if len(known) == n else f"  [{len(known)}/{n} records report it]"
    warn = "   <-- RAISE --solve_max_tokens" if n_trunc / max(len(known), 1) > 0.02 else ""
    return (f"Phase 1 truncated (budget)       : {n_trunc}/{len(known)} = "
            f"{n_trunc/max(len(known),1)*100:.1f}%{suffix}{warn}")


def _metrics(m: np.ndarray) -> dict[str, float]:
    tp, fn = int(m[0, 0]), int(m[0, 1])
    fp, tn = int(m[1, 0]), int(m[1, 1])
    total = tp + fn + fp + tn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _fmt_matrix(
    m: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
) -> str:
    col_w = 22
    lines = [" " * 22 + "".join(f"{c:>{col_w}}" for c in col_labels)]
    for label, row in zip(row_labels, m):
        row_total = int(row.sum())
        cells = "".join(
            f"{int(v):>10} ({int(v)/row_total*100:5.1f}%)" if row_total else f"{'0':>10} (  0.0%)"
            for v in row
        )
        lines.append(f"{label:<22}{cells}")
    return "\n".join(lines)


def generate_report(results: list[dict[str, Any]]) -> str:
    # Historical JSONL files may contain exact-string correctness booleans.
    # Recompute on copies so reports always use the current numeric label policy.
    results = [relabel_result(dict(record)) for record in results]
    n_loaded = len(results)

    # Drop records that carry no trustworthy label: generation failures and
    # budget-truncated solutions (both of which would otherwise donate a free true
    # negative to the judge) and unparseable golds. Counted and reported by reason
    # rather than silently filtered — the three mean very different things.
    excluded = [r for r in results if not is_labelable(r)]
    n_errors = sum(1 for r in excluded if r.get("error") is not None)
    n_truncated = sum(1 for r in excluded
                      if r.get("error") is None and r.get("truncated") is True)
    n_no_gold = len(excluded) - n_errors - n_truncated
    results = [r for r in results if is_labelable(r)]

    n = len(results)
    if n == 0:
        return (f"No labelable results to report ({n_loaded} loaded, {n_errors} "
                f"generation errors, {n_truncated} budget-truncated, {n_no_gold} "
                f"unparseable gold).")

    k = max((len(r.get("k_answers", [])) for r in results), default=0)

    n_gold_correct = sum(r["solve_correct"] for r in results)
    n_judge_yes = sum(r["judge_verdict"] == "YES" for r in results)
    n_majority_correct = sum(r["majority_correct"] for r in results)
    n_parse_fail_p1 = sum(r["phase1_answer"] is None for r in results)
    n_parse_fail_k = sum(
        sum(1 for a in r["k_answers"] if a is None) for r in results
    )
    total_k = n * k

    lines: list[str] = []
    sep = "=" * 68

    lines += [sep, "LLM SELF-JUDGE EVALUATION REPORT", sep]
    lines += [
        f"\nProblems evaluated : {n}  (of {n_loaded} loaded)",
        f"Excluded, unlabelable: {n_errors} generation error(s), "
        f"{n_truncated} budget-truncated, "
        f"{n_no_gold} unparseable gold answer(s)",
        f"k samples (Phase 2): {k}",
        "",
        "--- Base Rates ---",
        f"Phase 1 correct (greedy vs gold) : {n_gold_correct}/{n} = {n_gold_correct/n*100:.1f}%",
        f"Judge says YES                   : {n_judge_yes}/{n} = {n_judge_yes/n*100:.1f}%",
        f"Majority vote correct            : {n_majority_correct}/{n} = {n_majority_correct/n*100:.1f}%",
        f"Phase 1 parse failure rate       : {n_parse_fail_p1}/{n} = {n_parse_fail_p1/n*100:.1f}%",
        _truncation_line(results, n),
        f"k-sample parse failure rate      : {n_parse_fail_k}/{total_k} = {n_parse_fail_k/total_k*100:.1f}%" if total_k else "k-sample parse failure rate      : N/A",
    ]

    # ── Matrix A ──────────────────────────────────────────────────────────────
    mat_a = build_matrix(results, "solve_correct", "judge_verdict", "YES")
    m_a = _metrics(mat_a)
    lines += [
        "", sep,
        "MATRIX A — Judge vs. Gold",
        "Is the judge calibrated to reality?",
        "",
        _fmt_matrix(mat_a, ["Gold Correct", "Gold Incorrect"], ["Judge YES", "Judge NO"]),
        "",
        f"Accuracy : {m_a['accuracy']:.3f}",
        f"Precision: {m_a['precision']:.3f}  (of judge-YES verdicts, fraction that are truly correct)",
        f"Recall   : {m_a['recall']:.3f}  (of truly correct answers, fraction judge endorsed)",
        f"F1       : {m_a['f1']:.3f}",
        f"\nFP (overconfidence) : {int(mat_a[1,0])}  — judge endorses wrong answers",
        f"FN (underconfidence): {int(mat_a[0,1])}  — judge rejects correct answers",
    ]

    # ── Matrices B and C ──────────────────────────────────────────────────────
    # With k=1 there is no vote: `majority_answer` is a single extra stochastic
    # sample, so B and C measure sampling noise, not "stable model knowledge".
    # run_full.sh deliberately uses k=1 (the probe never reads the majority), so
    # this is the common case and must not be presented as a result.
    if k < 2:
        lines += [
            "", sep,
            f"MATRIX B / MATRIX C — SKIPPED (k={k})",
            "",
            "Majority vote needs k>=2. With k=1 `majority_answer` is one extra",
            "stochastic sample, so 'majority correct' measures sampling noise and",
            "the judge-vs-majority agreement below would be uninterpretable.",
            "Re-run with --k_samples 5 if you need these matrices.",
            "", sep,
        ]
        return "\n".join(lines)

    mat_b = build_matrix(results, "solve_correct", "majority_correct", True)
    m_b = _metrics(mat_b)
    lines += [
        "", sep,
        "MATRIX B — Majority Vote vs. Gold",
        "How faithfully does majority vote track ground truth? (sanity check)",
        "",
        _fmt_matrix(mat_b, ["Gold Correct", "Gold Incorrect"], ["Majority Correct", "Majority Incorrect"]),
        "",
        f"Accuracy : {m_b['accuracy']:.3f}",
        f"Precision: {m_b['precision']:.3f}",
        f"Recall   : {m_b['recall']:.3f}",
        f"F1       : {m_b['f1']:.3f}",
        f"\nNote: high off-diagonal → k={k} insufficient or parser is lossy — fix before reading Matrix C.",
    ]

    # ── Matrix C ──────────────────────────────────────────────────────────────
    mat_c = build_matrix(results, "majority_correct", "judge_verdict", "YES")
    m_c = _metrics(mat_c)
    lines += [
        "", sep,
        "MATRIX C — Judge vs. Majority Vote",
        "Does the judge track stable model knowledge?",
        "",
        _fmt_matrix(mat_c, ["Majority Correct", "Majority Wrong"], ["Judge YES", "Judge NO"]),
        "",
        f"Accuracy : {m_c['accuracy']:.3f}",
        f"Precision: {m_c['precision']:.3f}",
        f"Recall   : {m_c['recall']:.3f}",
        f"F1       : {m_c['f1']:.3f}",
        f"\nFP: {int(mat_c[1,0])}  — judge endorses answers the model can't reliably reproduce",
        f"FN: {int(mat_c[0,1])}  — judge doubts things the model consistently gets right",
        "", sep,
    ]

    return "\n".join(lines)
