import math
from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from newssearch.index import build_index, load_index, save_index
from newssearch.search import K1, B, SearchEngine
from newssearch.text import tokenize

DAY = 86_400
T0 = 1_767_225_600  # 2026-01-01 UTC

CORPUS = [
    (T0 + 0 * DAY, "Oil prices jump after tanker attack near Strait of Hormuz"),
    (T0 + 1 * DAY, "OPEC+ agrees to extend production cuts through June"),
    (T0 + 2 * DAY, "Crude inventories fall for third week, EIA says"),
    (T0 + 3 * DAY, "Iran ceasefire talks lift hopes, oil prices slip"),
    (T0 + 4 * DAY, "Tanker rates soar as Hormuz shipping risk grows"),
    (T0 + 5 * DAY, "US sanctions target Iranian oil exports to Asia"),
    (T0 + 6 * DAY, "Oil Prices Jump After Tanker Attack Near Strait Of Hormuz!"),  # syndicated copy
]


@pytest.fixture
def engine():
    return SearchEngine(build_index(CORPUS))


def test_tokenize_folds_case_plurals_and_stopwords():
    assert tokenize("The Tankers and PRICES of the Gas crisis") == ["tanker", "price", "gas", "crisis"]


def test_syndicated_copies_collapse_to_earliest(engine):
    assert engine.num_docs == 6
    hits = engine.search("tanker attack").hits
    assert hits[0].ts == T0  # the earlier copy survives


def test_ranking_prefers_documents_matching_more_terms(engine):
    hits = engine.search("hormuz tanker").hits
    titles = [h.title for h in hits]
    assert len(titles) == 2
    assert all("Hormuz" in t for t in titles)


def test_date_range_filter_is_half_open(engine):
    res = engine.search("oil", start_ts=T0 + 1 * DAY, end_ts=T0 + 5 * DAY)
    assert {h.ts for h in res.hits} == {T0 + 3 * DAY}
    assert engine.search("oil", start_ts=T0 + 5 * DAY).total == 1


def test_pagination_is_consistent(engine):
    full = [h.doc for h in engine.search("oil", k=10).hits]
    paged = [h.doc for off in range(0, 4) for h in engine.search("oil", k=1, offset=off).hits]
    assert paged == full


def test_ties_break_toward_newest_on_every_page():
    # 60 equally scored headlines: selection must not depend on argpartition's tie order.
    records = [(T0 + i, f"Oil update number{i:03d}") for i in range(60)]
    engine = SearchEngine(build_index(records))
    newest_first = list(range(59, -1, -1))
    for k in (1, 7, 10):
        paged = [h.doc for off in range(0, 60, k) for h in engine.search("oil", k=k, offset=off).hits]
        assert paged == newest_first


def test_unknown_terms_are_reported(engine):
    res = engine.search("oil zzzqx")
    assert res.unknown_terms == ["zzzqx"]
    assert res.total > 0
    assert engine.search("zzzqx").total == 0


def test_suggest_orders_by_document_frequency(engine):
    assert engine.suggest("t")[0] == "tanker"  # df 2; every other "t" term has df 1
    assert engine.suggest("hor") == ["hormuz"]
    assert engine.suggest("qqq") == []


def test_timeline_counts_by_month(engine):
    assert engine.timeline("oil") == [{"month": "2026-01", "count": 3}]


def test_index_round_trips_through_disk(tmp_path, engine):
    save_index(engine.index, tmp_path)
    loaded = SearchEngine(load_index(tmp_path))
    assert [h.title for h in loaded.search("oil prices").hits] == [h.title for h in engine.search("oil prices").hits]


# ---------------------------------------------------------------------------
# Property test: the vectorized engine must agree with a naive BM25 reference.
# ---------------------------------------------------------------------------

WORDS = ["oil", "crude", "tanker", "hormuz", "opec", "cut", "iran", "sanction", "price", "talk", "drone"]


def reference_bm25(docs: list[list[str]], query: list[str]) -> dict[int, float]:
    n = len(docs)
    avg = sum(len(d) for d in docs) / n
    df = Counter(t for d in docs for t in set(d))
    scores: dict[int, float] = {}
    for i, doc in enumerate(docs):
        tf = Counter(doc)
        s = 0.0
        for term in dict.fromkeys(query):
            if tf[term]:
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                s += idf * tf[term] * (K1 + 1) / (tf[term] + K1 * (1 - B + B * len(doc) / avg))
        if s > 0:
            scores[i] = s
    return scores


@settings(max_examples=150, deadline=None)
@given(
    titles=st.lists(st.lists(st.sampled_from(WORDS), min_size=1, max_size=8), min_size=1, max_size=40),
    query=st.lists(st.sampled_from(WORDS), min_size=1, max_size=4),
)
def test_matches_reference_bm25(titles, query):
    # unique suffix per doc so none are collapsed as duplicates
    records = [(T0 + i, " ".join(words) + f" story{i}") for i, words in enumerate(titles)]
    engine = SearchEngine(build_index(records))
    docs = [tokenize(title) for _, title in records]

    expected = reference_bm25(docs, query)
    result = engine.search(" ".join(query), k=len(records))

    assert result.total == len(expected)
    for hit in result.hits:
        assert hit.score == pytest.approx(expected[hit.doc], rel=1e-4, abs=1e-4)
    ranked = [h.score for h in result.hits]
    assert ranked == sorted(ranked, reverse=True)
