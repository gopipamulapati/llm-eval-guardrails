"""Small text utilities (no heavy NLP dependencies)."""

from __future__ import annotations

import re
import string

_STOPWORDS = frozenset(
    """a an the and or of to in on for with by at from is are was were be been being it its
    this that these those as not no do does did has have had can could should would will may
    what which who whom how when where why than then there their they them you your i we our""".split()
)

_PUNCT = str.maketrans("", "", string.punctuation.replace("%", "").replace("$", ""))


def normalize(text: str) -> str:
    text = text.lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str, drop_stopwords: bool = False) -> list[str]:
    toks = normalize(text).split()
    if drop_stopwords:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def numbers(text: str) -> set[str]:
    """Numbers as written (commas stripped), e.g. '43%', '620', '3.5'."""
    return {m.replace(",", "") for m in re.findall(r"\d[\d,]*(?:\.\d+)?%?", text)}
