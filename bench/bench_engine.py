"""In-process engine latency: no HTTP, no cache.

python bench/bench_engine.py --index index --queries 2000
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from queries import sample_queries

from newssearch.index import load_index
from newssearch.search import SearchEngine


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="index")
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--out", default="bench/results/engine.json")
    args = parser.parse_args()

    load_started = time.perf_counter()
    index = load_index(args.index)
    engine = SearchEngine(index, cache_size=0)  # measure the real work, not cache hits
    load_ms = (time.perf_counter() - load_started) * 1000
    queries = sample_queries(index, args.queries)

    for q in queries[:100]:  # warm the page cache
        engine.search(q)

    latencies = []
    started = time.perf_counter()
    for q in queries:
        t = time.perf_counter()
        engine.search(q, k=10)
        latencies.append((time.perf_counter() - t) * 1000)
    wall = time.perf_counter() - started

    report = {
        "machine": f"{platform.processor() or platform.machine()} / {platform.system()}",
        "num_docs": index.num_docs,
        "vocab_size": len(index.vocab),
        "num_postings": int(index.meta["num_postings"]),
        "index_load_ms": round(load_ms, 1),
        "queries": len(queries),
        "single_thread_qps": round(len(queries) / wall, 1),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3),
            "p50": round(percentile(latencies, 50), 3),
            "p95": round(percentile(latencies, 95), 3),
            "p99": round(percentile(latencies, 99), 3),
            "max": round(max(latencies), 3),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
