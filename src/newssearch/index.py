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
    meta: dict

    @property
    def num_docs(self) -> int:
        return len(self.doc_len)

    def title(self, doc: int) -> str:
        start, end = self.title_offsets[doc], self.title_offsets[doc + 1]
        return bytes(self.titles[start:end]).decode("utf-8")


def build_index(records: Iterable[tuple[int, str]], source: str = "") -> Index:
    """Build an index from ``(unix_seconds, title)`` pairs.

    Syndicated copies (same normalized title) collapse to the earliest one.
    """
    started = time.perf_counter()
    earliest: dict[str, tuple[int, str]] = {}
    seen = 0
    for ts, title in records:
        seen += 1
        title = title.strip()
        key = normalize_title(title)
        if not key:
            continue
        if key not in earliest or ts < earliest[key][0]:
            earliest[key] = (ts, title)

    docs = sorted(earliest.values())  # by timestamp, then title
    term_ids: dict[str, int] = {}
    post_term, post_doc, post_tf = [], [], []
    doc_len = np.zeros(len(docs), dtype=np.uint16)

    for doc_id, (_, title) in enumerate(docs):
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

    encoded = [title.encode("utf-8") for _, title in docs]
    title_offsets = np.zeros(len(docs) + 1, dtype=np.int64)
    np.cumsum([len(b) for b in encoded], out=title_offsets[1:])

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
        doc_ts=np.asarray([ts for ts, _ in docs], dtype=np.int64),
        title_offsets=title_offsets,
        titles=np.frombuffer(b"".join(encoded), dtype=np.uint8),
        meta=meta,
    )


_ARRAYS = ["offsets", "postings", "tfs", "doc_len", "doc_ts", "title_offsets"]


def save_index(index: Index, directory: str | Path) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(index.vocab), encoding="utf-8")
    for name in _ARRAYS:
        np.save(path / f"{name}.npy", getattr(index, name))
    index.titles.tofile(path / "titles.bin")
    (path / "meta.json").write_text(json.dumps(index.meta, indent=2))


def load_index(directory: str | Path) -> Index:
    path = Path(directory)
    vocab_text = (path / "vocab.txt").read_text(encoding="utf-8")
    arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in _ARRAYS}
    titles_path = path / "titles.bin"
    titles = np.memmap(titles_path, dtype=np.uint8, mode="r") if titles_path.stat().st_size else np.zeros(0, np.uint8)
    return Index(
        vocab=vocab_text.split("\n") if vocab_text else [],
        titles=titles,
        meta=json.loads((path / "meta.json").read_text()),
        **arrays,
    )
