"""HTTP API and static UI.

INDEX_DIR=index uvicorn newssearch.api:app --workers 4
"""

import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .index import load_index
from .search import SearchEngine

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Oil News Search", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_engine: SearchEngine | None = None


def get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        index = load_index(os.environ.get("INDEX_DIR", "index"))
        _engine = SearchEngine(index, cache_size=int(os.environ.get("CACHE_SIZE", "4096")))
    return _engine


def set_engine(engine: SearchEngine) -> None:
    """Inject an engine (tests, embedding)."""
    global _engine
    _engine = engine


def _to_ts(day: date | None) -> int | None:
    return None if day is None else int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


def _check_range(start: date | None, end: date | None) -> None:
    if start and end and start >= end:
        raise HTTPException(status_code=422, detail="start must be before end")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "docs": get_engine().num_docs}


@app.get("/api/stats")
def stats() -> dict:
    engine = get_engine()
    cache = engine.cache_info()
    return {**engine.index.meta, "cache": {"hits": cache.hits, "misses": cache.misses, "size": cache.currsize}}


@app.get("/api/search")
def search(
    q: Annotated[str, Query(min_length=1, max_length=200)],
    k: Annotated[int, Query(ge=1, le=50)] = 10,
    offset: Annotated[int, Query(ge=0, le=1000)] = 0,
    start: date | None = None,
    end: Annotated[date | None, Query(description="exclusive")] = None,
) -> dict:
    _check_range(start, end)
    result = get_engine().search(q, k=k, offset=offset, start_ts=_to_ts(start), end_ts=_to_ts(end))
    return {
        "query": result.query,
        "terms": result.terms,
        "unknown_terms": result.unknown_terms,
        "total": result.total,
        "took_ms": result.took_ms,
        "hits": [
            {
                "doc": h.doc,
                "score": h.score,
                "published": datetime.fromtimestamp(h.ts, tz=UTC).isoformat(),
                "title": h.title,
                "url": h.url,
            }
            for h in result.hits
        ],
    }


@app.get("/api/suggest")
def suggest(
    prefix: Annotated[str, Query(min_length=1, max_length=40)],
    limit: Annotated[int, Query(ge=1, le=20)] = 8,
):
    return {"prefix": prefix, "suggestions": get_engine().suggest(prefix, limit=limit)}


@app.get("/api/timeline")
def timeline(
    q: Annotated[str, Query(min_length=1, max_length=200)],
    start: date | None = None,
    end: date | None = None,
):
    _check_range(start, end)
    return {"query": q, **get_engine().timeline(q, start_ts=_to_ts(start), end_ts=_to_ts(end))}
