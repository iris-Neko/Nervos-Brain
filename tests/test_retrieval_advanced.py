"""Tests for the dual-layer + multi-path retrieval modules (M4B).

Covers:
  - config.py        (load_retrieval_config)
  - embedding.py     (get_embedding — deterministic path)
  - dual_layer.py    (ArchiveStore CRUD, DualLayerWriter sync-write)
  - bm25_index.py    (tokenize, BM25Index build + search)
  - fuzzy_search.py  (fuzzy_search — matches, threshold, CJK)
  - rank_fusion.py   (reciprocal_rank_fusion — scores, dedup, provenance)
  - multi_retriever.py (MultiRetriever — vector, BM25, fuzzy, exact paths)

All tests run fully offline (no API calls, no persistent disk state).
Each test that touches Qdrant or SQLite uses pytest's tmp_path fixture.
"""

import pytest

from nervos_brain.retrieval import (
    ArchiveRecord,
    ArchiveStore,
    BM25Index,
    DualLayerWriter,
    FusedResult,
    FuzzyResult,
    MultiRetriever,
    QdrantStore,
    RetrievalConfig,
    fuzzy_search,
    get_embedding,
    load_retrieval_config,
    reciprocal_rank_fusion,
    tokenize,
)
from nervos_brain.retrieval.qdrant_writer import _stable_point_id


# ── shared fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def cfg(tmp_path):
    return RetrievalConfig(
        qdrant_path=str(tmp_path / "qdrant"),
        archive_db=str(tmp_path / "archive.db"),
        vector_size=64,
        top_k_per_path=5,
        final_top_k=5,
        fuzzy_threshold=0.4,
        enable_bm25=True,
        enable_fuzzy=True,
        enable_exact=True,
    )


@pytest.fixture()
def archive(cfg):
    return ArchiveStore(db_path=cfg.archive_db, config=cfg)


@pytest.fixture()
def qdrant(cfg):
    return QdrantStore(config=cfg, qdrant_location=cfg.qdrant_path)


@pytest.fixture()
def writer(cfg, qdrant, archive):
    return DualLayerWriter(qdrant_store=qdrant, archive_store=archive, config=cfg)


def _write_fiber(writer):
    writer.write(
        source="fiber",
        doc_type="doc",
        url="https://example.com/open-channel",
        anchor="doc:open-channel#chunk:0",
        title="OpenChannel Protocol",
        summary="How to open a Fiber payment channel between two nodes.",
        keywords="OpenChannel,Fiber,payment,channel",
        raw_text="To open a Fiber channel, call openChannel with the peer node ID and the capacity.",
    )


def _write_htlc(writer):
    writer.write(
        source="rfcs",
        doc_type="doc",
        url="https://example.com/htlc",
        anchor="doc:htlc-spec#chunk:0",
        title="HTLC Timeout Mechanism",
        summary="Hash Time-Lock Contract timeout and on-chain resolution.",
        keywords="HTLC,timeout,on-chain,locktime",
        raw_text="An HTLC expires when the locktime is exceeded and the pre-image has not been revealed.",
    )


def _write_channel_manager(writer):
    writer.write(
        source="fiber",
        doc_type="code",
        url="https://github.com/example/fiber",
        anchor="doc:channel-manager#chunk:0",
        title="ChannelManager class",
        summary="Core class for managing Fiber channel state.",
        keywords="ChannelManager,class,Python,open_channel",
        raw_text="class ChannelManager:\n    def open_channel(self, peer_id): ...",
    )


def _write_nervos_brain(writer):
    writer.write(
        source="nervos_talk",
        doc_type="forum_post",
        url="https://talk.nervos.org/t/spark-program-nervos-brain/9995/1",
        anchor="doc:nervos-talk-9995#post:1",
        title="Spark Program | Nervos Brain - Agentic RAG",
        summary="Nervos Brain project proposal and weekly updates.",
        keywords="Nervos Brain,Spark Program,Agentic RAG",
        raw_text="Nervos Brain is a developer onboarding engine powered by Agentic RAG.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# config
# ═══════════════════════════════════════════════════════════════════════════


def test_load_retrieval_config_returns_correct_type():
    cfg = load_retrieval_config()
    assert isinstance(cfg, RetrievalConfig)


def test_load_retrieval_config_has_rrf_k():
    cfg = load_retrieval_config()
    # config.yaml sets rrf_k=60; default also 60
    assert cfg.rrf_k == 60


def test_retrieval_config_overrides_apply(cfg):
    assert cfg.vector_size == 64
    assert cfg.fuzzy_threshold == 0.4


def test_qdrant_stable_point_id_prefers_hash_then_anchor():
    assert _stable_point_id("text", {"hash": "h1", "anchor": "a1"}) == _stable_point_id(
        "other text",
        {"hash": "h1", "anchor": "a2"},
    )
    assert _stable_point_id("text", {"anchor": "a1"}) == _stable_point_id(
        "other text",
        {"anchor": "a1"},
    )
    assert _stable_point_id("text", {}) == _stable_point_id("text", {})


def test_qdrant_store_uses_server_url_when_configured(monkeypatch):
    calls: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def get_collections(self):
            class Collections:
                collections = []

            return Collections()

        def create_collection(self, **kwargs):
            calls["create_collection"] = kwargs

    monkeypatch.setattr("nervos_brain.retrieval.qdrant_writer.QdrantClient", FakeClient)
    cfg = RetrievalConfig(qdrant_url="http://127.0.0.1:6333", collection_name="test")

    store = QdrantStore(config=cfg)

    assert store.mode == "server"
    assert calls["kwargs"]["url"] == "http://127.0.0.1:6333"
    assert calls["create_collection"]["collection_name"] == "test"


# ═══════════════════════════════════════════════════════════════════════════
# embedding
# ═══════════════════════════════════════════════════════════════════════════


def test_get_embedding_returns_correct_dimension(cfg):
    vec = get_embedding("open channel", cfg)
    assert len(vec) == 64


def test_get_embedding_is_deterministic(cfg):
    v1 = get_embedding("fiber channel", cfg)
    v2 = get_embedding("fiber channel", cfg)
    assert v1 == v2


def test_get_embedding_different_texts_differ(cfg):
    v1 = get_embedding("fiber channel", cfg)
    v2 = get_embedding("HTLC timeout", cfg)
    assert v1 != v2


def test_get_embedding_values_in_range(cfg):
    vec = get_embedding("hello world", cfg)
    assert all(-1.0 <= x <= 1.0 for x in vec)


# ═══════════════════════════════════════════════════════════════════════════
# ArchiveStore
# ═══════════════════════════════════════════════════════════════════════════


def _make_record(anchor="doc:test#chunk:0", content_hash="abc123"):
    import uuid
    return ArchiveRecord(
        id=str(uuid.uuid4()),
        source="test",
        doc_type="doc",
        url="https://example.com",
        anchor=anchor,
        title="Test Title",
        summary="A short summary.",
        keywords="test,keyword",
        raw_text="This is the full raw text of the document.",
        raw_format="text",
        lang="en",
        version="latest",
        topic="testing",
        content_hash=content_hash,
    )


def test_archive_upsert_and_count(archive):
    assert archive.count() == 0
    archive.upsert(_make_record())
    assert archive.count() == 1


def test_archive_get_by_anchor(archive):
    rec = _make_record(anchor="doc:fiber#chunk:0", content_hash="hash001")
    archive.upsert(rec)
    found = archive.get_by_anchor("doc:fiber#chunk:0")
    assert found is not None
    assert found.title == "Test Title"


def test_archive_get_by_anchor_missing(archive):
    assert archive.get_by_anchor("doc:nonexistent#chunk:99") is None


def test_archive_get_by_hash(archive):
    rec = _make_record(content_hash="unique_hash_xyz")
    archive.upsert(rec)
    found = archive.get_by_hash("unique_hash_xyz")
    assert found is not None


def test_archive_upsert_is_idempotent(archive):
    rec = _make_record(content_hash="dup_hash")
    archive.upsert(rec)
    archive.upsert(rec)  # second upsert of same hash → no error, count stays 1
    assert archive.count() == 1


def test_archive_upsert_updates_title(archive):
    rec = _make_record(content_hash="update_hash")
    archive.upsert(rec)
    rec.title = "Updated Title"
    archive.upsert(rec)
    found = archive.get_by_hash("update_hash")
    assert found.title == "Updated Title"


def test_archive_list_all(archive):
    for i in range(3):
        archive.upsert(_make_record(anchor=f"doc:x#chunk:{i}", content_hash=f"hash_{i}"))
    assert len(archive.list_all()) == 3


# ═══════════════════════════════════════════════════════════════════════════
# DualLayerWriter
# ═══════════════════════════════════════════════════════════════════════════


def test_writer_populates_both_layers(writer, archive):
    _write_fiber(writer)
    # deep layer
    assert archive.count() == 1
    rec = archive.get_by_anchor("doc:open-channel#chunk:0")
    assert rec is not None
    assert "openChannel" in rec.raw_text


def test_writer_returns_content_hash(writer):
    h = writer.write(
        source="test",
        doc_type="doc",
        url="https://x.com",
        anchor="doc:x#chunk:0",
        title="T",
        summary="S",
        keywords="k",
        raw_text="raw",
    )
    assert isinstance(h, str) and len(h) == 64  # sha-256 hex


def test_writer_idempotent_on_same_content(writer, archive):
    _write_fiber(writer)
    _write_fiber(writer)  # exact same payload → dedup
    assert archive.count() == 1


def test_writer_batch(writer, archive):
    records = [
        dict(source="s", doc_type="doc", url="u", anchor=f"doc:x#chunk:{i}",
             title=f"T{i}", summary="S", keywords="k", raw_text=f"text {i}")
        for i in range(4)
    ]
    written = writer.write_batch(records)
    assert written == 4
    assert archive.count() == 4


# ═══════════════════════════════════════════════════════════════════════════
# BM25Index
# ═══════════════════════════════════════════════════════════════════════════


def test_tokenize_english():
    tokens = tokenize("open channel payment")
    assert "open" in tokens
    assert "channel" in tokens
    assert "payment" in tokens


def test_tokenize_camel_case():
    tokens = tokenize("OpenChannel")
    assert "open" in tokens
    assert "channel" in tokens


def test_tokenize_snake_case():
    tokens = tokenize("open_channel_protocol")
    assert "open" in tokens
    assert "channel" in tokens
    assert "protocol" in tokens


def test_tokenize_cjk():
    tokens = tokenize("开放通道协议")
    assert "开" in tokens
    assert "通" in tokens


def test_tokenize_mixed():
    tokens = tokenize("Fiber通道 openChannel")
    assert "fiber" in tokens
    assert "通" in tokens
    assert "open" in tokens


def test_bm25_empty_index_returns_nothing():
    idx = BM25Index()
    assert idx.search("anything") == []


def test_bm25_build_and_search(archive, writer):
    _write_fiber(writer)
    _write_htlc(writer)
    _write_channel_manager(writer)

    idx = BM25Index()
    idx.build_from_store(archive)
    assert idx.size == 3

    hits = idx.search("HTLC timeout locktime", top_k=3)
    assert len(hits) > 0
    assert hits[0].anchor == "doc:htlc-spec#chunk:0"


def test_bm25_top_k_respected(archive, writer):
    for i in range(5):
        writer.write(
            source="s", doc_type="doc", url="u",
            anchor=f"doc:x#chunk:{i}",
            title=f"Document {i}",
            summary="test document",
            keywords="test",
            raw_text=f"test document number {i}",
        )
    idx = BM25Index()
    idx.build_from_store(archive)
    hits = idx.search("test document", top_k=2)
    assert len(hits) <= 2


def test_bm25_irrelevant_query_returns_empty(archive, writer):
    _write_fiber(writer)
    idx = BM25Index()
    idx.build_from_store(archive)
    hits = idx.search("zzzzzzz_no_match_whatsoever")
    assert hits == []


def test_bm25_scores_are_positive(archive, writer):
    _write_htlc(writer)
    idx = BM25Index()
    idx.build_from_store(archive)
    hits = idx.search("HTLC")
    assert all(h.score > 0 for h in hits)


# ═══════════════════════════════════════════════════════════════════════════
# fuzzy_search
# ═══════════════════════════════════════════════════════════════════════════


_CANDIDATES = [
    {"anchor": "doc:open-channel#chunk:0", "title": "OpenChannel Protocol",  "source": "fiber", "keywords": "OpenChannel,Fiber,payment"},
    {"anchor": "doc:htlc-spec#chunk:0",    "title": "HTLC Timeout Mechanism","source": "rfcs",  "keywords": "HTLC,timeout,locktime"},
    {"anchor": "doc:channel-mgr#chunk:0",  "title": "ChannelManager class",  "source": "fiber", "keywords": "ChannelManager,Python"},
]


def test_fuzzy_exact_title_hits():
    hits = fuzzy_search("OpenChannel Protocol", _CANDIDATES, threshold=0.5)
    anchors = [h.anchor for h in hits]
    assert "doc:open-channel#chunk:0" in anchors


def test_fuzzy_typo_hits():
    # "chanell" (double-l typo) should still surface ChannelManager
    hits = fuzzy_search("chanell manager", _CANDIDATES, threshold=0.4)
    assert len(hits) > 0


def test_fuzzy_abbreviation_variant():
    # "fibre" vs "Fiber" — close enough
    hits = fuzzy_search("fibre channel", _CANDIDATES, threshold=0.3)
    assert len(hits) > 0


def test_fuzzy_threshold_filters():
    # Very high threshold should eliminate noisy matches
    hits = fuzzy_search("xyz_no_match", _CANDIDATES, threshold=0.9)
    assert hits == []


def test_fuzzy_top_k_respected():
    hits = fuzzy_search("channel", _CANDIDATES, threshold=0.3, top_k=1)
    assert len(hits) <= 1


def test_fuzzy_results_sorted_descending():
    hits = fuzzy_search("OpenChannel", _CANDIDATES, threshold=0.3)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_fuzzy_keyword_match():
    # "locktime" is a keyword of the HTLC record
    hits = fuzzy_search("locktime", _CANDIDATES, threshold=0.7)
    anchors = [h.anchor for h in hits]
    assert "doc:htlc-spec#chunk:0" in anchors


def test_fuzzy_result_has_matched_field(capsys):
    hits = fuzzy_search("ChannelManager", _CANDIDATES, threshold=0.5)
    assert any(h.matched_field for h in hits)  # matched_field is non-empty


# ═══════════════════════════════════════════════════════════════════════════
# reciprocal_rank_fusion
# ═══════════════════════════════════════════════════════════════════════════


_LIST_A = [
    {"anchor": "a1", "title": "Alpha One",   "source": "x", "score": 0.9},
    {"anchor": "a2", "title": "Alpha Two",   "source": "x", "score": 0.7},
    {"anchor": "a3", "title": "Alpha Three", "source": "x", "score": 0.5},
]
_LIST_B = [
    {"anchor": "a2", "title": "Alpha Two",   "source": "y", "score": 0.8},
    {"anchor": "a4", "title": "Beta Four",   "source": "y", "score": 0.6},
    {"anchor": "a1", "title": "Alpha One",   "source": "x", "score": 0.4},
]


def test_rrf_merges_two_lists():
    fused = reciprocal_rank_fusion([_LIST_A, _LIST_B])
    anchors = [r.anchor for r in fused]
    # all unique anchors should appear
    assert set(anchors) == {"a1", "a2", "a3", "a4"}


def test_rrf_deduplicates():
    # a1 and a2 appear in both lists → each should appear only once
    fused = reciprocal_rank_fusion([_LIST_A, _LIST_B])
    anchors = [r.anchor for r in fused]
    assert len(anchors) == len(set(anchors))


def test_rrf_scores_sorted_descending():
    fused = reciprocal_rank_fusion([_LIST_A, _LIST_B])
    scores = [r.rrf_score for r in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_top_ranked_appeared_in_both_lists():
    fused = reciprocal_rank_fusion([_LIST_A, _LIST_B])
    # a1 is rank-1 in A and rank-2 in B; a2 is rank-2 in A and rank-1 in B
    # both have contribution from 2 paths → should beat single-list items
    top_anchors = {r.anchor for r in fused[:2]}
    assert "a1" in top_anchors or "a2" in top_anchors


def test_rrf_single_list_passthrough():
    fused = reciprocal_rank_fusion([_LIST_A])
    assert len(fused) == len(_LIST_A)
    assert fused[0].anchor == "a1"  # rank 1 in the only list


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []


def test_rrf_provenance_recorded():
    fused = reciprocal_rank_fusion([_LIST_A, _LIST_B], path_names=["vector", "bm25"])
    # a2 appears in both paths
    a2 = next(r for r in fused if r.anchor == "a2")
    assert "vector" in a2.contributions
    assert "bm25" in a2.contributions


def test_rrf_k_smoothing_effect():
    # Higher k → smaller score difference between top and bottom
    fused_low_k  = reciprocal_rank_fusion([_LIST_A], k=1)
    fused_high_k = reciprocal_rank_fusion([_LIST_A], k=1000)
    spread_low  = fused_low_k[0].rrf_score  - fused_low_k[-1].rrf_score
    spread_high = fused_high_k[0].rrf_score - fused_high_k[-1].rrf_score
    assert spread_low > spread_high


def test_rrf_skips_empty_anchor():
    dirty_list = [{"anchor": "", "title": "ghost", "source": "x", "score": 1.0}]
    fused = reciprocal_rank_fusion([dirty_list])
    assert fused == []


# ═══════════════════════════════════════════════════════════════════════════
# MultiRetriever (integration)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def populated_retriever(cfg, qdrant, archive, writer):
    _write_fiber(writer)
    _write_htlc(writer)
    _write_channel_manager(writer)
    retriever = MultiRetriever(qdrant_store=qdrant, archive_store=archive, config=cfg)
    retriever.rebuild_bm25()
    return retriever


def test_retriever_rebuild_bm25_returns_count(cfg, qdrant, archive, writer):
    _write_fiber(writer)
    _write_htlc(writer)
    retriever = MultiRetriever(qdrant_store=qdrant, archive_store=archive, config=cfg)
    assert retriever.rebuild_bm25() == 2


def test_retriever_vector_path(populated_retriever):
    results = populated_retriever.search(
        "how to open a payment channel", filters={"source": "fiber"}
    )
    assert len(results) > 0
    # Evidence conforms to protocol
    e = results[0]
    assert "anchor" in e and "title" in e and "score" in e and "snippet" in e


def test_retriever_broad_resource_query_uses_vector_without_filters(monkeypatch, populated_retriever):
    calls: list[dict] = []

    def fake_vector_search(query, filters, top_k):
        calls.append({"query": query, "filters": filters, "top_k": top_k})
        return [
            {
                "anchor": "doc:resources#chunk:0",
                "title": "CKB learning resources",
                "source": "docs",
                "score": 0.99,
            }
        ]

    def fail_slow_path(*_args, **_kwargs):
        raise AssertionError("broad resource query should not run slow archive paths")

    monkeypatch.setattr(populated_retriever, "_vector_search", fake_vector_search)
    monkeypatch.setattr(populated_retriever, "_bm25_search", fail_slow_path)
    monkeypatch.setattr(populated_retriever, "_fuzzy_search", fail_slow_path)
    monkeypatch.setattr(populated_retriever, "_exact_search", fail_slow_path)

    results = populated_retriever.search("CKB 入门有没有比较靠谱的资料可以看？", top_k=3)

    assert results
    assert calls[0]["filters"] == {}


def test_retriever_llm_regex_query_hard_recalls_named_project(
    monkeypatch, cfg, qdrant, archive, writer
):
    _write_nervos_brain(writer)
    retriever = MultiRetriever(qdrant_store=qdrant, archive_store=archive, config=cfg)
    retriever.rebuild_bm25()
    monkeypatch.setattr(retriever, "_vector_search", lambda *_args, **_kwargs: [])
    results = retriever.search(
        "我听说有一个nervos brain的项目，有没有文档可以借鉴一下",
        regex_queries=[
            {
                "label": "Nervos Brain",
                "pattern": r"(?i)\bnervos[\s_-]+brain\b",
                "fields": ["title", "keywords", "anchor", "url", "summary"],
            }
        ],
        top_k=3,
    )

    assert results[0]["anchor"] == "doc:nervos-talk-9995#post:1"
    assert results[0]["payload"]["regex_valid_count"] == 1


def test_retriever_rejects_dangerous_regex_and_continues(monkeypatch, populated_retriever):
    monkeypatch.setattr(populated_retriever, "_vector_search", lambda *_args, **_kwargs: [])
    results = populated_retriever.search(
        "anything",
        regex_queries=[{"label": "broad", "pattern": r".*", "fields": ["title"]}],
        top_k=3,
    )

    assert isinstance(results, list)
    if results:
        assert results[0]["payload"]["regex_dropped_count"] == 1


def test_retriever_returns_evidence_protocol(populated_retriever):
    results = populated_retriever.search("HTLC timeout")
    for e in results:
        assert isinstance(e["score"], float)
        assert isinstance(e["snippet"], str)
        assert isinstance(e["anchor"], str)
        assert isinstance(e["title"], str)


def test_retriever_bm25_path_finds_keyword(populated_retriever):
    results = populated_retriever.search("locktime on-chain HTLC")
    anchors = [e["anchor"] for e in results]
    assert "doc:htlc-spec#chunk:0" in anchors


def test_retriever_fuzzy_path_handles_typo(populated_retriever):
    # "chanell" typo should still surface ChannelManager
    results = populated_retriever.search("chanell manager")
    assert len(results) > 0


def test_retriever_evidence_snippet_from_archive(populated_retriever):
    # Snippet should come from the deep archive (raw_text), not the Qdrant summary
    results = populated_retriever.search(
        "open channel capacity", filters={"source": "fiber"}
    )
    fiber_results = [e for e in results if "open" in e["anchor"].lower() or "fiber" in e["source"]]
    if fiber_results:
        # raw_text contains "openChannel" — verify it made it into the snippet
        assert any("openChannel" in e["snippet"] or "Fiber" in e["snippet"]
                   for e in fiber_results)


def test_retriever_top_k_cap(cfg, qdrant, archive, writer):
    for i in range(8):
        writer.write(
            source="s", doc_type="doc", url="u",
            anchor=f"doc:item#chunk:{i}",
            title=f"Item {i}",
            summary="test",
            keywords="test",
            raw_text=f"test item {i}",
        )
    retriever = MultiRetriever(qdrant_store=qdrant, archive_store=archive, config=cfg)
    retriever.rebuild_bm25()
    results = retriever.search("test item", top_k=3)
    assert len(results) <= 3


def test_retriever_no_results_on_empty_store(cfg, qdrant, archive):
    retriever = MultiRetriever(qdrant_store=qdrant, archive_store=archive, config=cfg)
    # BM25 not built, no Qdrant data, no filters → nothing should crash
    results = retriever.search("anything")
    assert isinstance(results, list)


def test_retriever_rrf_provenance_in_payload(populated_retriever):
    results = populated_retriever.search("Fiber channel", filters={"source": "fiber"})
    for e in results:
        # payload must carry at least the rrf_score key
        assert "rrf_score" in e["payload"]


def test_retriever_hydrated_payload_keeps_archive_metadata(populated_retriever):
    results = populated_retriever.search("open channel capacity", filters={"source": "fiber"})
    hit = next(e for e in results if e["anchor"] == "doc:open-channel#chunk:0")

    assert hit["payload"]["source"] == "fiber"
    assert hit["payload"]["type"] == "doc"
    assert hit["payload"]["topic"] == "unknown"
    assert hit["payload"]["url"] == "https://example.com/open-channel"
