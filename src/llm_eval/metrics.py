"""Deterministic, reference-based and reference-free metrics.

All metrics return a float in [0, 1] so they can be averaged and gated in CI.
They are intentionally cheap: run them on every PR, and use the LLM judge
(see ``judge.py``) for the nuanced, slower signal.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

from .models import EvalCase, TargetResponse
from .text import normalize, numbers, sentences, tokens

REFUSAL_PATTERNS = [
    r"\bi (?:do not|don't) know\b",
    r"\bnot (?:enough|sufficient) information\b",
    r"\b(?:cannot|can't|unable to) (?:answer|determine|find)\b",
    r"\bno (?:relevant )?information\b",
    r"\bnot (?:covered|mentioned|stated) in\b",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer))


def exact_match(pred: str, ref: str) -> float:
    return float(normalize(pred) == normalize(ref))


def token_f1(pred: str, ref: str) -> float:
    p, r = tokens(pred, drop_stopwords=True), tokens(ref, drop_stopwords=True)
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    lcs = _lcs(p, r)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def sentence_supported(sentence: str, contexts: list[str], threshold: float = 0.6) -> bool:
    """A sentence is supported if most of its content words appear in one context
    and every number it states appears in that same context."""
    sent_toks = set(tokens(sentence, drop_stopwords=True))
    sent_nums = numbers(sentence)
    if not sent_toks:
        return True
    for ctx in contexts:
        ctx_toks = set(tokens(ctx, drop_stopwords=True))
        coverage = len(sent_toks & ctx_toks) / len(sent_toks)
        if coverage >= threshold and sent_nums <= numbers(ctx):
            return True
    return False


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Share of answer sentences grounded in the retrieved contexts (hallucination proxy)."""
    if is_refusal(answer):
        return 1.0
    sents = sentences(answer)
    if not sents:
        return 0.0
    if not contexts:
        return 0.0
    return sum(sentence_supported(s, contexts) for s in sents) / len(sents)


def context_recall(reference: str, contexts: list[str]) -> float:
    """How much of the reference answer is recoverable from the retrieved contexts."""
    ref = set(tokens(reference, drop_stopwords=True))
    if not ref:
        return 1.0
    ctx = set(tokens(" ".join(contexts), drop_stopwords=True))
    return len(ref & ctx) / len(ref)


def refusal_correct(case: EvalCase, answer: str) -> float:
    return float(is_refusal(answer) == case.expect_refusal)


MetricFn = Callable[[EvalCase, TargetResponse], float | None]


def _ref(fn: Callable[[str, str], float]) -> MetricFn:
    def wrapped(case: EvalCase, resp: TargetResponse) -> float | None:
        if case.expect_refusal or not case.reference:
            return None
        return fn(resp.answer, case.reference)

    return wrapped


METRICS: dict[str, MetricFn] = {
    "exact_match": _ref(exact_match),
    "token_f1": _ref(token_f1),
    "rouge_l": _ref(rouge_l),
    "faithfulness": lambda c, r: faithfulness(r.answer, r.contexts),
    "context_recall": lambda c, r: (
        None if c.expect_refusal or not c.reference else context_recall(c.reference, r.contexts)
    ),
    "refusal_correct": lambda c, r: refusal_correct(c, r.answer),
}


def compute_all(case: EvalCase, resp: TargetResponse, names: list[str] | None = None) -> dict:
    out: dict[str, float] = {}
    for name in names or list(METRICS):
        value = METRICS[name](case, resp)
        if value is not None:
            out[name] = round(float(value), 4)
    return out
