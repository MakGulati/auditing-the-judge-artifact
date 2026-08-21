"""How the judge's YES/NO probability mass is read off the verdict distribution.

`p_yes` is the continuous black-box baseline: the judge's own probability of YES at
the verdict position, normalised over the YES/NO mass. Turning a next-token
distribution into that scalar requires deciding which *tokens* count as YES and
which count as NO, and that decision is not neutral.

The rule shipped in `extraction_common.yes_no_mass` (reproduced here as
``PREFIX``) matches by prefix: any token whose normalised form starts with "NO"
counts as NO. Over gemma-3-12b's 262k vocabulary that binds 765 tokens, including
` not`, ` now`, ` note`, ` nothing`, ` normal`, ` North` and ` November` — all of
them plausible continuations for a judge about to write prose. The YES side binds
only 29. The bias is therefore both real and asymmetric: spurious mass lands on NO
roughly twenty-six times more often than on YES.

``EXACT`` is the same rule with prefix matching replaced by exact matching, so
` not` no longer counts as a NO vote. The two differ in one predicate and nothing
else, which is what makes them comparable.

Both rules share the shipped normalisation (strip whitespace, then the `":*`
characters, then upper-case) so a difference between them can only come from the
matching predicate.
"""
from __future__ import annotations

PREFIX = "prefix"
EXACT = "exact"
RULES = (PREFIX, EXACT)

# The shipped rule's YES set is not simply "starts with YES": bare "Y" and the
# fragment "YE" are accepted as whole tokens too. Kept verbatim so EXACT differs
# from PREFIX in the predicate alone, not in the vocabulary it recognises.
YES_TOKENS = ("YES", "Y", "YE")
NO_TOKENS = ("NO", "N")

YES = "yes"
NO = "no"


def normalize(text: str) -> str:
    """Normalise a decoded token exactly as `extraction_common.yes_no_mass` does."""
    return text.strip().strip('":*').upper()


def classify(text: str, rule: str = PREFIX) -> str | None:
    """Bin one decoded token as a YES vote, a NO vote, or neither.

    ``text`` is the raw decoded token; normalisation happens here so callers cannot
    apply one rule's normalisation to the other's predicate.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    t = normalize(text)
    if not t:
        return None
    if rule == PREFIX:
        # Verbatim from extraction_common.yes_no_mass, including the elif: a token
        # matching both sides (impossible here, but the ordering is part of the rule)
        # would count as YES.
        if t.startswith("YES") or t in ("Y", "YE"):
            return YES
        if t.startswith("NO") or t == "N":
            return NO
        return None
    if t in YES_TOKENS:
        return YES
    if t in NO_TOKENS:
        return NO
    return None


def mass(pairs, rule: str = PREFIX) -> tuple[float, float, float]:
    """(yes_mass, no_mass, other_mass) over ``pairs`` of (decoded_token, prob)."""
    my = mn = other = 0.0
    for text, p in pairs:
        c = classify(text, rule)
        if c == YES:
            my += p
        elif c == NO:
            mn += p
        else:
            other += p
    return my, mn, other


def p_yes(pairs, rule: str = PREFIX) -> float:
    """Normalised P(YES) over the YES/NO mass, or 0.5 when neither side appears.

    The 0.5 fallback is the shipped behaviour. It is not a neutral prior so much as
    a missing value: a record scored 0.5 means "the rule found no YES/NO tokens",
    not "the judge was undecided". Callers that care should count them.
    """
    my, mn, _ = mass(pairs, rule)
    d = my + mn
    return my / d if d > 0 else 0.5


def vocab_classes(tok, rule: str = PREFIX, vocab_size: int | None = None):
    """Token ids a rule bins as YES / NO, over a tokenizer's whole vocabulary.

    Decoding 262k ids takes a few seconds, so callers should do this once. Returns
    ``(yes_ids, no_ids)`` as plain lists of int.
    """
    n = vocab_size if vocab_size is not None else len(tok)
    yes_ids, no_ids = [], []
    for i in range(n):
        c = classify(tok.decode([i]), rule)
        if c == YES:
            yes_ids.append(i)
        elif c == NO:
            no_ids.append(i)
    return yes_ids, no_ids
