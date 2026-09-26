"""Building, saving, and memory-mapping the inverted index.

On-disk layout (one directory):

    vocab.txt         sorted terms, one per line; line i is term id i
    offsets.npy       int64[V+1]  postings for term i live in [offsets[i], offsets[i+1])
    postings.npy      int32[P]    doc ids, ascending within each term
    tfs.npy           uint16[P]   term frequency aligned with postings
    doc_len.npy       uint16[N]   tokens per document
    doc_ts.npy        int64[N]    publish time (unix seconds), ascending
    title_offsets.npy int64[N+1]  byte ranges into titles.bin
    titles.bin        UTF-8 headlines, concatenated
    url_offsets.npy   int64[N+1]  byte ranges into urls.bin (empty string if unknown)
    urls.bin          UTF-8 article URLs, concatenated
    meta.json         corpus statistics

Doc ids are assigned in publish-time order, so a date range is a contiguous id range.
Arrays are loaded with mmap, so every server worker shares one copy in the page cache.
"""

import json
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .text import normalize_title, tokenize

# (unix_seconds, title) or (unix_seconds, title, url)
Record = tuple[int, str] | tuple[int, str, str]


@dataclass(frozen=True)
class Index:
    vocab: list[str]
    offsets: np.ndarray
    postings: np.ndarray
    tfs: np.ndarray
    doc_len: np.ndarray
    doc_ts: np.ndarray
    title_offsets: np.ndarray
    titles: np.ndarray
    url_offsets: np.ndarray
    urls: np.ndarray
    meta: dict

    @property
    def num_docs(self) -> int:
        return len(self.doc_len)

    def title(self, doc: int) -> str:
        return _unpack(self.titles, self.title_offsets, doc)

    def url(self, doc: int) -> str:
        return _unpack(self.urls, self.url_offsets, doc)


def _pack(strings: list[str]) -> tuple[np.ndarray, np.ndarray]:
    encoded = [s.encode("utf-8") for s in strings]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    np.cumsum([len(b) for b in encoded], out=offsets[1:])
    return offsets, np.frombuffer(b"".join(encoded), dtype=np.uint8)


def _unpack(blob: np.ndarray, offsets: np.ndarray, doc: int) -> str:
    return bytes(blob[offsets[doc] : offsets[doc + 1]]).decode("utf-8")


def build_index(records: Iterable[Record], source: str = "") -> Index:
    """Build an index from ``(unix_seconds, title[, url])`` records.

    Syndicated copies (same normalized title) collapse to the earliest one.
    """
    started = time.perf_counter()
    earliest: dict[str, tuple[int, str, str]] = {}
    seen = 0
    for ts, title, *rest in records:
        seen += 1
        title = title.strip()
        key = normalize_title(title)
        if not key:
            continue
        if key not in earliest or ts < earliest[key][0]:
            earliest[key] = (ts, title, rest[0].strip() if rest else "")

    docs = sorted(earliest.values())  # by timestamp, then title
    term_ids: dict[str, int] = {}
    post_term, post_doc, post_tf = [], [], []
    doc_len = np.zeros(len(docs), dtype=np.uint16)

    for doc_id, (_, title, _) in enumerate(docs):
        tokens = tokenize(title)
        doc_len[doc_id] = min(len(tokens), np.iinfo(np.uint16).max)
        for term, tf in Counter(tokens).items():
            post_term.append(term_ids.setdefault(term, len(term_ids)))
            post_doc.append(doc_id)
            post_tf.append(tf)

    # Renumber terms alphabetically so prefix lookups are a binary search.
    vocab = sorted(term_ids)
    remap = np.empty(len(vocab), dtype=np.int64)
    for new_id, term in enumerate(vocab):
        remap[term_ids[term]] = new_id

    term_arr = remap[np.asarray(post_term, dtype=np.int64)] if post_term else np.zeros(0, np.int64)
    doc_arr = np.asarray(post_doc, dtype=np.int32)
    tf_arr = np.asarray(post_tf, dtype=np.uint16)
    order = np.lexsort((doc_arr, term_arr))  # group by term, doc ids ascending inside
    offsets = np.zeros(len(vocab) + 1, dtype=np.int64)
    np.cumsum(np.bincount(term_arr, minlength=len(vocab)), out=offsets[1:])

    title_offsets, titles = _pack([title for _, title, _ in docs])
    url_offsets, urls = _pack([url for _, _, url in docs])

    meta = {
        "source": source,
        "records_seen": seen,
        "num_docs": len(docs),
        "vocab_size": len(vocab),
        "num_postings": int(len(doc_arr)),
        "avg_doc_len": float(doc_len.mean()) if len(docs) else 0.0,
        "min_ts": int(docs[0][0]) if docs else 0,
        "max_ts": int(docs[-1][0]) if docs else 0,
        "build_seconds": round(time.perf_counter() - started, 2),
    }
    return Index(
        vocab=vocab,
        offsets=offsets,
        postings=doc_arr[order],
        tfs=tf_arr[order],
        doc_len=doc_len,
        doc_ts=np.asarray([ts for ts, _, _ in docs], dtype=np.int64),
        title_offsets=title_offsets,
        titles=titles,
        url_offsets=url_offsets,
        urls=urls,
        meta=meta,
    )


_ARRAYS = ["offsets", "postings", "tfs", "doc_len", "doc_ts", "title_offsets", "url_offsets"]
_BLOBS = ["titles", "urls"]


def save_index(index: Index, directory: str | Path) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(index.vocab), encoding="utf-8")
    for name in _ARRAYS:
        np.save(path / f"{name}.npy", getattr(index, name))
    for name in _BLOBS:
        getattr(index, name).tofile(path / f"{name}.bin")
    (path / "meta.json").write_text(json.dumps(index.meta, indent=2))


def _load_blob(path: Path) -> np.ndarray:
    # np.memmap refuses empty files.
    return np.memmap(path, dtype=np.uint8, mode="r") if path.stat().st_size else np.zeros(0, np.uint8)


def load_index(directory: str | Path) -> Index:
    path = Path(directory)
    vocab_text = (path / "vocab.txt").read_text(encoding="utf-8")
    return Index(
        vocab=vocab_text.split("\n") if vocab_text else [],
        meta=json.loads((path / "meta.json").read_text()),
        **{name: np.load(path / f"{name}.npy", mmap_mode="r") for name in _ARRAYS},
        **{name: _load_blob(path / f"{name}.bin") for name in _BLOBS},
    )
