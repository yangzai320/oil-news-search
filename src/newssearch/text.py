"""Tokenization shared by indexing and querying (they must agree exactly)."""

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i if in into is it its
    of on or our s she so than that the their them then there these they this to up
    was we were what when which while who will with would you your after over says said
    amid new""".split()
)


def _stem(token: str) -> str:
    # Deliberately light: fold simple plurals ("prices" -> "price") without the
    # surprises of a full stemmer. Skips -ss/-us/-is endings ("gas", "crisis").
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        if token.endswith("ies"):
            return token[:-3] + "y"
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with stopwords removed and plurals folded."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def normalize_title(text: str) -> str:
    """Canonical form used to detect syndicated duplicates."""
    return " ".join(_TOKEN_RE.findall(_WHITESPACE_RE.sub(" ", text).lower()))
