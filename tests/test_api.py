import pytest
from fastapi.testclient import TestClient

from newssearch import api
from newssearch.index import build_index
from newssearch.search import SearchEngine

T0 = 1_767_225_600  # 2026-01-01 UTC


@pytest.fixture
def client():
    records = [
        (T0, "Oil prices jump after tanker attack near Strait of Hormuz", "https://example.com/oil"),
        (T0 + 40 * 86_400, "Crude inventories fall for third week, EIA says"),
        (T0 + 41 * 86_400, "Oil slips as ceasefire talks resume"),
    ]
    api.set_engine(SearchEngine(build_index(records)))
    return TestClient(api.app)


def test_search_returns_ranked_hits(client):
    body = client.get("/api/search", params={"q": "oil tanker"}).json()
    assert body["total"] == 2
    assert body["hits"][0]["title"].startswith("Oil prices jump")
    assert body["hits"][0]["published"] == "2026-01-01T00:00:00+00:00"
    assert body["hits"][0]["url"] == "https://example.com/oil"
    assert body["took_ms"] >= 0


def test_search_date_filter(client):
    body = client.get("/api/search", params={"q": "oil", "start": "2026-02-01"}).json()
    assert [h["title"] for h in body["hits"]] == ["Oil slips as ceasefire talks resume"]


@pytest.mark.parametrize(
    "params",
    [
        {"q": ""},
        {"q": "oil", "k": 0},
        {"q": "oil", "k": 51},
        {"q": "oil", "start": "2026-03-01", "end": "2026-02-01"},
        {"q": "oil", "start": "not-a-date"},
    ],
)
def test_search_validates_input(client, params):
    assert client.get("/api/search", params=params).status_code == 422


def test_suggest_and_timeline(client):
    assert client.get("/api/suggest", params={"prefix": "tan"}).json()["suggestions"] == ["tanker"]
    body = client.get("/api/timeline", params={"q": "oil"}).json()
    assert body["granularity"] == "day"
    buckets = body["buckets"]
    assert len(buckets) == 42  # 2026-01-01 .. 2026-02-11, empty days included
    assert (buckets[0], buckets[-1]) == ({"period": "2026-01-01", "count": 1}, {"period": "2026-02-11", "count": 1})
    assert sum(b["count"] for b in buckets) == 2


def test_home_page_and_health(client):
    assert "Oil News Search" in client.get("/").text
    assert client.get("/healthz").json() == {"status": "ok", "docs": 3}
    assert client.get("/api/stats").json()["num_docs"] == 3
