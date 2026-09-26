"""BM25 ranking over the memory-mapped inverted index."""

import bisect
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .index import Index
from .text import tokenize

K1 = 1.2
B = 0.75


@dataclass
class Hit:
    doc: int
    score: float
    ts: int
    title: str
    url: str


@dataclass
class SearchResult:
    query: str
    terms: list[str]
    total: int
    hits: list[Hit]
    took_ms: float
    unknown_terms: list[str] = field(default_factory=list)


class SearchEngine:
    def __init__(self, index: Index, cache_size: int = 4096):
        self.index = index
        self.num_docs = index.num_docs
        self.avg_len = float(index.meta.get("avg_doc_len") or 1.0)
        # Per-document BM25 length norm is query-independent: precompute once.
        self.len_norm = (K1 * (1 - B + B * np.asarray(index.doc_len, dtype=np.float32) / self.avg_len)).astype(
            np.float32
        )
        df = np.diff(index.offsets)
        self.idf = np.log1p((self.num_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        # Cache finished pages (a few hits each), never the full match arrays:
        # a common term like "oil" can match most of the corpus.
        self._page = lru_cache(maxsize=cache_size)(self._compute_page)
        self._timeline = lru_cache(maxsize=cache_size)(self._compute_timeline)

    # -- lookups ---------------------------------------------------------------

    def term_id(self, term: str) -> int | None:
        i = bisect.bisect_left(self.index.vocab, term)
        return i if i < len(self.index.vocab) and self.index.vocab[i] == term else None

    def doc_range(self, start_ts: int | None, end_ts: int | None) -> tuple[int, int]:
        """Docs are time-ordered, so [start_ts, end_ts) maps to a contiguous id range."""
        ts = self.index.doc_ts
        lo = 0 if start_ts is None else int(np.searchsorted(ts, start_ts, side="left"))
        hi = self.num_docs if end_ts is None else int(np.searchsorted(ts, end_ts, side="left"))
        return lo, max(lo, hi)

    # -- ranking ---------------------------------------------------------------

    def _rank(self, terms: tuple[str, ...], lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
        """All matching doc ids in [lo, hi) and their BM25 scores (OR semantics)."""
        docs_parts, score_parts = [], []
        for term in terms:
            tid = self.term_id(term)
            if tid is None:
                continue
            start, end = int(self.index.offsets[tid]), int(self.index.offsets[tid + 1])
            docs = self.index.postings[start:end]
            if lo > 0 or hi < self.num_docs:  # postings are sorted: slice the date range
                i, j = np.searchsorted(docs, [lo, hi])
                docs, start = docs[i:j], start + i
                end = start + len(docs)
            tf = self.index.tfs[start:end].astype(np.float32)
            docs = np.asarray(docs)
            score_parts.append(self.idf[tid] * tf * (K1 + 1) / (tf + self.len_norm[docs]))
            docs_parts.append(docs)

        if not docs_parts:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)
        if len(docs_parts) == 1:
            return docs_parts[0], score_parts[0]

        # Merge per-term postings: sort by doc id, then sum each doc's partial scores.
        docs = np.concatenate(docs_parts)
        scores = np.concatenate(score_parts)
        order = np.argsort(docs, kind="stable")
        docs, scores = docs[order], scores[order]
        starts = np.flatnonzero(np.r_[True, docs[1:] != docs[:-1]])
        return docs[starts], np.add.reduceat(scores, starts)

    def _compute_page(self, terms: tuple[str, ...], lo: int, hi: int, k: int, offset: int):
        docs, scores = self._rank(terms, lo, hi)
        top = min(offset + k, len(docs))
        if top < len(docs):
            # Top-k selection in O(n) instead of sorting every match. argpartition picks
            # an arbitrary subset of docs tied at the k-th score, so keep every doc at or
            # above that score and let the sort below apply the tie-break.
            kth = np.partition(scores, len(scores) - top)[len(scores) - top]
            cand = np.flatnonzero(scores >= kth)
        else:
            cand = np.arange(len(docs))
        # Highest score first; ties go to the newer headline (larger doc id).
        cand = cand[np.lexsort((-docs[cand], -scores[cand]))][offset:top]

        hits = tuple(
            Hit(
                doc=int(docs[i]),
                score=round(float(scores[i]), 4),
                ts=int(self.index.doc_ts[docs[i]]),
                title=self.index.title(int(docs[i])),
                url=self.index.url(int(docs[i])),
            )
            for i in cand
        )
        return len(docs), hits

    def search(
        self,
        query: str,
        k: int = 10,
        offset: int = 0,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> SearchResult:
        started = time.perf_counter()
        terms = tuple(dict.fromkeys(tokenize(query)))  # dedupe, keep order
        lo, hi = self.doc_range(start_ts, end_ts)
        total, hits = self._page(terms, lo, hi, k, offset)
        return SearchResult(
            query=query,
            terms=list(terms),
            total=total,
            hits=list(hits),
            took_ms=round((time.perf_counter() - started) * 1000, 3),
            unknown_terms=[t for t in terms if self.term_id(t) is None],
        )

    # -- extras ----------------------------------------------------------------

    def suggest(self, prefix: str, limit: int = 8) -> list[str]:
        """Most frequent vocabulary terms starting with ``prefix``."""
        prefix = prefix.strip().lower()
        if not prefix:
            return []
        vocab = self.index.vocab
        lo = bisect.bisect_left(vocab, prefix)
        hi = bisect.bisect_left(vocab, prefix + "￿")
        if lo == hi:
            return []
        df = np.diff(self.index.offsets[lo : hi + 1])
        best = np.argsort(-df, kind="stable")[:limit]
        return [vocab[lo + int(i)] for i in best]

    def timeline(self, query: str, start_ts: int | None = None, end_ts: int | None = None) -> dict:
        """Matching-headline counts per day (spans up to ~6 months) or per month."""
        terms = tuple(dict.fromkeys(tokenize(query)))
        lo, hi = self.doc_range(start_ts, end_ts)
        granularity, buckets = self._timeline(terms, lo, hi)
        return {"granularity": granularity, "buckets": list(buckets)}

    def _compute_timeline(self, terms: tuple[str, ...], lo: int, hi: int) -> tuple[str, tuple[dict, ...]]:
        if hi <= lo:
            return "day", ()
        # Pick the unit from the searched window, not the matches, so charts are comparable.
        span_days = (int(self.index.doc_ts[hi - 1]) - int(self.index.doc_ts[lo])) / 86_400
        granularity, unit = ("day", "D") if span_days <= 183 else ("month", "M")
        docs, _ = self._rank(terms, lo, hi)
        if len(docs) == 0:
            return granularity, ()

        periods = np.asarray(self.index.doc_ts[docs], dtype="datetime64[s]").astype(f"datetime64[{unit}]")
        periods.sort()
        # Emit every period between the first and last match, including empty ones,
        # so bars sit at their true positions on the time axis.
        axis = np.arange(periods[0], periods[-1] + 1)
        counts = np.searchsorted(periods, axis, side="right") - np.searchsorted(periods, axis, side="left")
        return granularity, tuple({"period": str(p), "count": int(c)} for p, c in zip(axis, counts, strict=True))

    def cache_info(self):
        return self._page.cache_info()
