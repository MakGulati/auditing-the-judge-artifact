"""LaTeX answer handling for the MATH dataset — the non-numeric sibling of correctness.py.

GSM8K golds are always numbers, so `metrics.correctness` compares them as `Decimal`s.
MATH golds are LaTeX expressions (``\\frac{2}{3}``, ``2\\sqrt{3}``, ``\\text{even}``,
``(3,\\frac{\\pi}{2})``), and roughly half of them do not parse as a number at all.
Feeding those through the numeric path makes every one of them *unlabelable*, which
silently deletes half the dataset at extraction and probe time — so MATH needs its own
canonical form.

Equivalence policy (deliberately the strict, Hendrycks-comparable one):

  * normalize both sides, then compare as strings;
  * additionally accept decimal equality when BOTH sides are plain numeric literals,
    so ``2`` == ``2.00`` and ``.5`` == ``0.5``.

``\\frac{1}{2}`` therefore does NOT equal ``0.5``. That is a real judgement call: it
costs some true positives, but it keeps accuracy numbers comparable to published
MATH results, which is what the filtering study is measured against.
"""
from __future__ import annotations

import re

from metrics.correctness import parse_numeric_answer

# Commands whose argument IS the answer. `\fbox` appears in a handful of MATH
# solutions in place of `\boxed`.
_BOX_COMMANDS = ("\\boxed", "\\fbox")

# Wrappers that carry no mathematical content: their argument replaces the whole
# call. `\text{even}` is an answer; the `\text` is packaging.
_UNWRAP_COMMANDS = ("\\text", "\\mbox", "\\textbf", "\\textit", "\\mathrm", "\\mathbf")

# Stripped outright — spacing, currency and unit decoration that never changes the
# value. ORDER MATTERS: the escaped forms must come off before the bare ones, or `\$5`
# loses its `$` first and leaves a stray backslash. Operators (`\cdot`, `\times`) are
# deliberately NOT here: removing them turns `2\cdot3` into the number 23.
_NOISE = ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "~",
          "^{\\circ}", "^\\circ",
          "\\$", "$", "\\%", "%")

_LATEX_DELIMITERS = (("\\(", "\\)"), ("\\[", "\\]"), ("$$", "$$"), ("$", "$"))

# A bare decimal literal, and nothing else — notably no comma, so a normalized answer
# that still contains one is a list (``1,2``) rather than a number. See
# `canonical_latex_answer`. Exponents are included because `Decimal` accepts them and
# `1e3` really is the number 1000.
_PLAIN_DECIMAL = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def _find_matching_brace(text: str, open_pos: int) -> int | None:
    """Index of the `}` closing the `{` at ``open_pos``, or None if unbalanced.

    A regex such as ``\\\\boxed\\{([^}]*)\\}`` truncates ``\\boxed{\\frac{1}{2}}`` to
    ``\\frac{1`` — a gold answer that then compares unequal to every model output.
    Nested braces are the common case in MATH, not the exception, so the scan has to
    count depth.
    """
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_boxed(text: str | None) -> str | None:
    """Return the content of the LAST ``\\boxed{...}`` / ``\\fbox{...}`` in ``text``.

    The last one wins because MATH solutions occasionally box an intermediate result
    before boxing the final answer, and model outputs that restate their conclusion do
    the same. Also accepts the brace-less ``\\boxed 5`` form that shows up in a few
    solutions. Returns None when nothing is boxed — for a gold answer that means
    "unlabelable", for a model answer it means "wrong".
    """
    if not text:
        return None
    found: list[tuple[int, str]] = []
    for command in _BOX_COMMANDS:
        start = 0
        while True:
            hit = text.find(command, start)
            if hit == -1:
                break
            after = hit + len(command)
            probe = after
            while probe < len(text) and text[probe] == " ":
                probe += 1
            if probe < len(text) and text[probe] == "{":
                close = _find_matching_brace(text, probe)
                if close is not None:
                    found.append((hit, text[probe + 1:close]))
                    start = close + 1
                    continue
            else:
                # `\boxed 5` / `\boxed5`: the argument is the next token.
                token = re.match(r"\s*(-?[\d.]+|\\?[A-Za-z]+)", text[after:])
                if token:
                    found.append((hit, token.group(1)))
            start = after
    if not found:
        return None
    # Positions are absolute, so this picks the last box in the string regardless of
    # which command produced it.
    return max(found, key=lambda pair: pair[0])[1]


def _strip_trailing_units(text: str) -> str:
    """Drop a trailing unit block: ``12\\text{ cm}`` -> ``12``.

    The leading space inside the braces is what distinguishes a unit from an answer.
    MATH writes units as ``\\text{ cm}`` and word answers as ``\\text{even}``, so the
    space is the same discriminator the reference grader uses. Without this, a model
    answering ``12`` is marked wrong against a gold of ``12\\text{ cm}``; with it,
    ``\\text{even}`` is still preserved intact by ``_unwrap_commands`` below.
    """
    for command in _UNWRAP_COMMANDS:
        marker = command + "{ "
        head, sep, _ = text.partition(marker)
        # Only when a unit block trails actual content — `\text{ cm}` alone is the
        # answer, not a unit on nothing.
        if sep and head.strip():
            text = head.strip()
    return text


def _unwrap_commands(text: str) -> str:
    """Replace ``\\text{even}`` with ``even``, repeatedly and brace-aware."""
    for command in _UNWRAP_COMMANDS:
        while True:
            hit = text.find(command + "{")
            if hit == -1:
                break
            open_pos = hit + len(command)
            close = _find_matching_brace(text, open_pos)
            if close is None:
                break
            text = text[:hit] + text[open_pos + 1:close] + text[close + 1:]
    return text


def _brace_single_char_args(text: str) -> str:
    """``\\frac12`` -> ``\\frac{1}{2}``, ``\\sqrt3`` -> ``\\sqrt{3}``.

    Models and MATH solutions both write these interchangeably, so without this the
    same answer compares unequal to itself.
    """
    text = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", text)
    text = re.sub(r"\\frac\{([^{}]+)\}(\d)", r"\\frac{\1}{\2}", text)
    text = re.sub(r"\\frac(\d)\{([^{}]+)\}", r"\\frac{\1}{\2}", text)
    text = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", text)
    return text


def _strip_math_delimiters(text: str) -> str:
    for opener, closer in _LATEX_DELIMITERS:
        while text.startswith(opener) and text.endswith(closer) and \
                len(text) >= len(opener) + len(closer):
            text = text[len(opener):len(text) - len(closer)].strip()
    return text


def normalize_latex(value) -> str:
    """Canonical spelling of a LaTeX answer. Never returns None; may return ''.

    The steps are ordered: delimiters and wrappers come off first so the noise list
    and the fraction rules see the bare expression.
    """
    if value is None:
        return ""
    text = str(value).strip()
    text = _strip_math_delimiters(text)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = _strip_trailing_units(text)
    text = _unwrap_commands(text)
    text = _strip_math_delimiters(text)

    # `1{,}000` is how MATH writes a thousands separator inside math mode.
    text = text.replace("{,}", "")
    for token in _NOISE:
        text = text.replace(token, "")
    text = _brace_single_char_args(text)

    # `x = 5` / `y=-3`: the answer is the right-hand side. Only for a single-variable
    # left side, so `(1,2)=(a,b)`-style answers and inequalities are left alone.
    lhs_rhs = re.match(r"^\s*[A-Za-z]\s*=\s*(.+)$", text)
    if lhs_rhs:
        text = lhs_rhs.group(1)

    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")

    # Thousands separators only inside something that is otherwise a plain number:
    # stripping them unconditionally would turn the tuple `(1,2)` into `(12)`.
    if re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", text):
        text = text.replace(",", "")

    # `.5` -> `0.5`, so it meets `0.5` in the numeric comparison below.
    text = re.sub(r"^([+-]?)\.(\d)", r"\g<1>0.\2", text)
    return text


def canonical_latex_answer(value) -> str | None:
    """One canonical string per equivalence class, or None when there is no answer.

    Numeric literals are routed through the numeric canonicaliser so ``2``, ``2.0``
    and ``2.00`` collapse to one vote in majority voting; everything else is compared
    on its normalized LaTeX spelling.

    The gate is a *plain decimal literal*, not "whatever the numeric parser accepts".
    ``parse_numeric_answer`` strips every comma before parsing (correct for GSM8K,
    where a comma is always a thousands separator), so handing it a MATH answer such
    as ``1,2`` — two roots, or a coordinate list — silently returns the single number
    12, making ``1,2`` compare equal to ``12``. ``normalize_latex`` has already
    removed the thousands separators that a real number can carry, so anything with a
    comma left in it is a list and must be compared as a string.
    """
    text = normalize_latex(value)
    if not text:
        return None
    if _PLAIN_DECIMAL.fullmatch(text):
        parsed = parse_numeric_answer(text)
        if parsed is not None:
            # Mirror metrics.correctness.canonical_numeric_answer exactly.
            return "0" if parsed == 0 else format(parsed.normalize(), "f")
    return text


def latex_answers_equal(answer, gold) -> bool:
    """True when both sides denote the same MATH answer under the strict policy."""
    canonical_answer = canonical_latex_answer(answer)
    canonical_gold = canonical_latex_answer(gold)
    return (canonical_answer is not None and canonical_gold is not None
            and canonical_answer == canonical_gold)


def gold_is_parsable(gold) -> bool:
    """True when ``gold`` can anchor a correctness label (i.e. normalizes non-empty)."""
    return canonical_latex_answer(gold) is not None
