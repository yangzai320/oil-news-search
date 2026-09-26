"""Deterministic benchmark query log drawn from the indexed headlines themselves.

Each query is 1-3 content words from a randomly chosen headline, which mirrors how
people search news (short keyword queries, skewed toward frequent terms).
"""

import random

from newssearch.index import Index
from newssearch.text import tokenize


def sample_queries(index: Index, n: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    queries = []
    while len(queries) < n:
        words = tokenize(index.title(rng.randrange(index.num_docs)))
        if not words:
            continue
        size = rng.choice([1, 2, 2, 3])
        queries.append(" ".join(rng.sample(words, min(size, len(words)))))
    return queries
