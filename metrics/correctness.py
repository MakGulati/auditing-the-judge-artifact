"""Canonical numeric-answer handling shared by generation and filtering."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


# v2 adds explicit "unlabelable" handling: a record whose gold answer does not
# parse, or whose generation raised, no longer silently becomes `correct=False`.
LABEL_POLICY = "numeric_equivalence_v2"

# MATH answers are LaTeX expressions, about half of which never parse as a number.
# Comparing them numerically would mark every one of those unlabelable, so they get
# their own policy — see metrics/latex_answer.py for what it does and does not accept.
MATH_LABEL_POLICY = "latex_numeric_equivalence_v1"

# A record written before the `dataset` field existed can only be GSM8K: it is the
# only dataset this pipeline ever generated. Defaulting keeps every historical
# raw.jsonl and activation dump readable without a migration.
DEFAULT_DATASET = "gsm8k"

_LABEL_POLICIES = {"gsm8k": LABEL_POLICY, "math": MATH_LABEL_POLICY}


def label_policy_for(dataset: str | None) -> str:
    """Policy string recorded on records and in fingerprints for ``dataset``."""
    return _LABEL_POLICIES.get(dataset or DEFAULT_DATASET, LABEL_POLICY)


def record_dataset(record: dict[str, Any]) -> str:
    """Which dataset a raw record came from, defaulting to GSM8K for legacy files."""
    return record.get("dataset") or DEFAULT_DATASET


def parse_numeric_answer(value: Any) -> Decimal | None:
    """Parse a finite decimal answer after removing harmless formatting."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text.startswith("$"):
        text = text[1:].strip()
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def canonical_numeric_answer(value: Any) -> str | None:
    """Return one canonical string for numerically equivalent parsed answers."""
    parsed = parse_numeric_answer(value)
    if parsed is None:
        return None
    if parsed == 0:
        return "0"
    return format(parsed.normalize(), "f")


def answers_numerically_equal(answer: Any, gold: Any) -> bool:
    """Compare two parsed answers by exact decimal value, not string spelling."""
    parsed_answer = parse_numeric_answer(answer)
    parsed_gold = parse_numeric_answer(gold)
    return parsed_answer is not None and parsed_gold is not None and parsed_answer == parsed_gold


def answers_equal(answer: Any, gold: Any, dataset: str | None = None) -> bool:
    """Dataset-aware answer comparison. GSM8K keeps the exact numeric semantics."""
    if (dataset or DEFAULT_DATASET) == "math":
        from metrics.latex_answer import latex_answers_equal
        return latex_answers_equal(answer, gold)
    return answers_numerically_equal(answer, gold)


def canonical_answer(value: Any, dataset: str | None = None) -> str | None:
    """One canonical string per equivalence class, under ``dataset``'s policy."""
    if (dataset or DEFAULT_DATASET) == "math":
        from metrics.latex_answer import canonical_latex_answer
        return canonical_latex_answer(value)
    return canonical_numeric_answer(value)


def gold_is_parsable(gold: Any, dataset: str | None = None) -> bool:
    """True when ``gold`` can anchor a correctness label under ``dataset``."""
    if (dataset or DEFAULT_DATASET) == "math":
        from metrics.latex_answer import gold_is_parsable as _latex_gold_is_parsable
        return _latex_gold_is_parsable(gold)
    return parse_numeric_answer(gold) is not None


def label_correctness(answer: Any, gold: Any, dataset: str | None = None) -> bool | None:
    """Three-valued correctness: True, False, or None when gold is unusable.

    ``answers_numerically_equal`` collapses "gold did not parse" into False, which
    silently mislabels a correct answer as wrong (GSM8K negative golds such as
    ``#### -12`` used to hit this). Callers that build training labels must use
    this and drop the ``None``s instead.

    ``dataset`` selects the equivalence policy; omitting it keeps the numeric one,
    so every existing caller is unchanged.
    """
    if not gold_is_parsable(gold, dataset):
        return None
    return answers_equal(answer, gold, dataset)


def is_labelable(record: dict[str, Any]) -> bool:
    """True when a raw record can carry a trustworthy correctness label.

    Excludes (a) records whose gold answer never parsed and (b) records written by
    ``run_problem_safe`` after an exception — those carry a synthetic
    ``solve_correct=False`` / ``judge_verdict="NO"`` that would otherwise count as
    a free true negative for the judge and a trivially separable probe example.
    """
    if record.get("error") is not None:
        return False
    # A solution the token budget cut off never got to state its final answer, so its
    # correctness is unknown — not wrong. Labelling it wrong asserts something about
    # the model when the cause was our budget, and it is the *easy* kind of wrong: the
    # judge sees text stopping mid-sentence and rejects it, donating a free true
    # negative and a trivially separable probe example, exactly as a generation error
    # would. `is True` on purpose: None means the backend could not report truncation,
    # which is not the same claim, and every record written before that field existed
    # must keep its previous labelability.
    if record.get("truncated") is True:
        return False
    return gold_is_parsable(record.get("gold_answer"), record_dataset(record))


def audit_stored_labelable(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare each record's STORED `labelable` against the recomputed value.

    The stored field is advisory: it is a snapshot taken at generation time, while
    `is_labelable` is evaluated against the current policy at analysis time. They can
    legitimately diverge when the policy moves, and older files were written by an
    expression (`gold_answer is not None`) that ignored truncation entirely -- it
    called budget-truncated records labelable on 7 GSM8K-train, 197 MATH-train and 117
    MATH-test records that the analysis correctly excluded.

    Divergence is therefore not an error, but it must not be silent: a reader
    inspecting raw.jsonl by hand would otherwise get a different analysed population
    than any figure in the paper.
    """
    disagree = [r for r in records
                if "labelable" in r and bool(r["labelable"]) != is_labelable(r)]
    stale_true = sum(1 for r in disagree if r.get("labelable"))
    return {
        "n": len(records),
        "n_with_field": sum(1 for r in records if "labelable" in r),
        "n_disagree": len(disagree),
        "stored_true_recomputed_false": stale_true,
        "stored_false_recomputed_true": len(disagree) - stale_true,
        "idx": [r.get("idx") for r in disagree[:10]],
    }


def relabel_result(record: dict[str, Any]) -> dict[str, Any]:
    """Update stored correctness fields using the record's dataset policy.

    ``solve_correct``/``majority_correct`` stay boolean for backwards compatibility;
    ``labelable`` records whether those booleans are meaningful for this record.
    """
    dataset = record_dataset(record)
    gold = record.get("gold_answer")
    record["solve_correct"] = answers_equal(record.get("phase1_answer"), gold, dataset)
    record["majority_correct"] = answers_equal(record.get("majority_answer"), gold, dataset)
    record["labelable"] = is_labelable(record)
    record["label_policy"] = label_policy_for(dataset)
    return record
