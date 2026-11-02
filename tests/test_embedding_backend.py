"""Embedding backend, vector-path switches, dimension checks and the vector cache.

No real model is loaded: the hash backend stands in for it, and a counting fake
embedder is injected where the cache has to be observed.
"""
from __future__ import annotations

import math

import pytest

from nervos_brain.retrieval import (
    ArchiveStore,
    DualLayerWriter,
    EmbeddingCache,
    EmbeddingError,
    MultiRetriever,
    QdrantStore,
    RetrievalConfig,
    embed_documents,
    embed_query,
    model_label,
    resolve_backend,
    vector_path_status,
)
import nervos_brain.retrieval.embedding as embedding_module


def _cfg(tmp_path, **overrides) -> RetrievalConfig:
    base = dict(
        qdrant_path=str(tmp_path / "qdrant"),
        archive_db=str(tmp_path / "archive.db"),
        vector_size=64,
        embedding_cache_db=str(tmp_path / "cache" / "emb.db"),
    )
    base.update(overrides)
    return RetrievalConfig(**base)


def _norm(vec) -> float:
    return math.sqrt(sum(x * x for x in vec))


class CountingEmbedder:
    """Deterministic fake model that records every text it is asked to encode."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t) % 7 + 1 + i) for i in range(self.dim)] for t in texts]


@pytest.fixture(autouse=True)
def _clear_query_cache():
    embedding_module._QUERY_CACHE.clear()
    yield
    embedding_module._QUERY_CACHE.clear()


@pytest.fixture()
def counting(monkeypatch):
    embedder = CountingEmbedder()
    monkeypatch.setattr(embedding_module, "get_embedder", lambda cfg: embedder)
    return embedder


# ── backend resolution (new setting + legacy switch) ────────────────────────


def test_empty_backend_without_legacy_switch_is_hash(tmp_path):
    assert resolve_backend(_cfg(tmp_path)) == "hash"


def test_legacy_use_real_embedding_maps_to_api(tmp_path):
    assert resolve_backend(_cfg(tmp_path, use_real_embedding=True)) == "api"


@pytest.mark.parametrize("name", ["local", "api", "hash", "none"])
def test_explicit_backend_wins_over_legacy_switch(tmp_path, name):
    cfg = _cfg(tmp_path, embedding_backend=name, use_real_embedding=True)
    assert resolve_backend(cfg) == name


def test_backend_name_is_case_insensitive(tmp_path):
    assert resolve_backend(_cfg(tmp_path, embedding_backend=" Local ")) == "local"


def test_unknown_backend_is_rejected(tmp_path):
    with pytest.raises(EmbeddingError, match="unknown embedding_backend"):
        resolve_backend(_cfg(tmp_path, embedding_backend="word2vec"))


def test_model_label_uses_model_for_real_backends_and_dim_for_hash(tmp_path):
    assert model_label(_cfg(tmp_path, embedding_backend="local", embedding_model="BAAI/bge-m3")) == "BAAI/bge-m3"
    assert model_label(_cfg(tmp_path, embedding_backend="hash", vector_size=64)) == "hash-64"
    assert model_label(_cfg(tmp_path, embedding_backend="none", vector_size=64)) == "hash-64"


# ── vector path switch ──────────────────────────────────────────────────────


def test_hash_vectors_are_allowed_under_pytest(tmp_path):
    assert vector_path_status(_cfg(tmp_path, embedding_backend="hash")) == (True, "")


def test_hash_vectors_are_blocked_outside_pytest(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("NERVOS_ALLOW_HASH_EMBEDDING", raising=False)
    enabled, reason = vector_path_status(_cfg(tmp_path, embedding_backend="hash"))
    assert enabled is False
    assert "test-only" in reason


def test_legacy_default_config_is_blocked_outside_pytest(tmp_path, monkeypatch):
    """The old default (use_real_embedding false) resolved to hash vectors."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("NERVOS_ALLOW_HASH_EMBEDDING", raising=False)
    assert vector_path_status(_cfg(tmp_path))[0] is False


def test_hash_vectors_can_be_allowed_explicitly(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("NERVOS_ALLOW_HASH_EMBEDDING", raising=False)
    assert vector_path_status(_cfg(tmp_path, embedding_backend="hash", embedding_allow_hash=True))[0]
    monkeypatch.setenv("NERVOS_ALLOW_HASH_EMBEDDING", "1")
    assert vector_path_status(_cfg(tmp_path, embedding_backend="hash"))[0]


def test_enable_vector_false_turns_the_path_off(tmp_path):
    enabled, reason = vector_path_status(_cfg(tmp_path, embedding_backend="hash", enable_vector=False))
    assert (enabled, reason) == (False, "enable_vector is false")


def test_backend_none_turns_the_path_off(tmp_path):
    enabled, reason = vector_path_status(_cfg(tmp_path, embedding_backend="none"))
    assert enabled is False
    assert "none" in reason


def test_real_backend_is_enabled_without_loading_a_model(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    cfg = _cfg(tmp_path, embedding_backend="local", embedding_model="BAAI/bge-m3", vector_size=1024)
    assert vector_path_status(cfg) == (True, "")


def _populated(tmp_path, **overrides):
    cfg = _cfg(tmp_path, embedding_backend="hash", **overrides)
    qdrant = QdrantStore(config=cfg, qdrant_location=cfg.qdrant_path)
    archive = ArchiveStore(db_path=cfg.archive_db, config=cfg)
    writer = DualLayerWriter(qdrant_store=qdrant, archive_store=archive, config=cfg)
    writer.write(
        source="fiber",
        doc_type="doc",
        url="https://example.com/open-channel",
        anchor="doc:open-channel#chunk:0",
        title="OpenChannel Protocol",
        summary="How to open a Fiber payment channel between two nodes.",
        keywords="OpenChannel,Fiber,payment,channel",
        raw_text="OpenChannel starts a Fiber payment channel.",
        raw_format="text",
        lang="en",
        version="latest",
        topic="channel",
    )
    return cfg, qdrant, archive


def test_retriever_skips_vector_path_when_hash_is_blocked(tmp_path, monkeypatch):
    cfg, qdrant, archive = _populated(tmp_path)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("NERVOS_ALLOW_HASH_EMBEDDING", raising=False)

    retriever = MultiRetriever(qdrant_store=qdrant, archive_store=archive, config=cfg)
    retriever.rebuild_bm25()
    called: list[str] = []
    monkeypatch.setattr(retriever, "_vector_search", lambda *a, **k: called.append("x") or [])

    results = retriever.search("OpenChannel")

    assert called == []
    assert retriever.vector_status["enabled"] is False
    assert results, "lexical paths still answer"
    assert all("vector" not in (r.get("payload", {}).get("retrieval_paths") or []) for r in results)


def test_retriever_uses_vector_path_when_enabled(tmp_path):
    cfg, qdrant, archive = _populated(tmp_path)
    retriever = MultiRetriever(qdrant_store=qdrant, archive_store=archive, config=cfg)
    assert retriever.vector_status == {"enabled": True, "reason": ""}
    assert retriever._vector_search("open a channel", {}, 5)


def test_retriever_with_enable_vector_false_never_calls_vector_search(tmp_path, monkeypatch):
    cfg, qdrant, archive = _populated(tmp_path, enable_vector=False)
    retriever = MultiRetriever(qdrant_store=qdrant, archive_store=archive, config=cfg)
    retriever.rebuild_bm25()
    monkeypatch.setattr(
        retriever, "_vector_search", lambda *a, **k: pytest.fail("vector path must stay off")
    )
    assert retriever.search("OpenChannel")


# ── collection compatibility ───────────────────────────────────────────────


def test_vector_compatibility_ok_for_matching_collection(tmp_path):
    _, qdrant, _ = _populated(tmp_path)
    assert qdrant.vector_compatibility() == (True, "")


def test_vector_compatibility_rejects_dimension_mismatch(tmp_path):
    cfg, _, _ = _populated(tmp_path)
    mismatched = QdrantStore.__new__(QdrantStore)
    mismatched.__dict__.update(QdrantStore(config=cfg, qdrant_location=cfg.qdrant_path).__dict__)
    mismatched.config = RetrievalConfig(**{**cfg.__dict__, "vector_size": 1024})

    ok, reason = mismatched.vector_compatibility()

    assert ok is False
    assert "64-dim" in reason and "1024" in reason and "--recreate" in reason


def test_vector_compatibility_rejects_model_label_mismatch(tmp_path):
    """Same dimension, different model: vectors from different models are not comparable."""
    cfg, qdrant, _ = _populated(tmp_path)
    qdrant.config = RetrievalConfig(
        **{**cfg.__dict__, "embedding_backend": "local", "embedding_model": "some/other-model"}
    )

    ok, reason = qdrant.vector_compatibility()

    assert ok is False
    assert "hash-64" in reason and "some/other-model" in reason


def test_vector_compatibility_rejects_legacy_collection_for_real_backend(tmp_path):
    """Collections written by the old hash writer have no embedding_model label."""
    cfg, qdrant, _ = _populated(tmp_path)
    points, _ = qdrant.client.scroll(cfg.collection_name, limit=10, with_payload=True)
    for point in points:
        qdrant.client.delete_payload(cfg.collection_name, keys=["embedding_model"], points=[point.id])
    qdrant.config = RetrievalConfig(
        **{**cfg.__dict__, "embedding_backend": "local", "embedding_model": "BAAI/bge-m3"}
    )

    ok, reason = qdrant.vector_compatibility()

    assert ok is False
    assert "no embedding_model metadata" in reason


def test_retriever_disables_vector_path_on_mismatch_instead_of_returning_empty_silently(
    tmp_path, caplog
):
    cfg, qdrant, archive = _populated(tmp_path)
    qdrant.config = RetrievalConfig(**{**cfg.__dict__, "vector_size": 1024})
    retriever = MultiRetriever(qdrant_store=qdrant, archive_store=archive, config=cfg)
    retriever._cfg = qdrant.config

    with caplog.at_level("ERROR"):
        enabled, reason = retriever.check_vector_path()

    assert enabled is False
    assert "vector path DISABLED" in caplog.text
    assert "64-dim" in reason


def test_upsert_records_model_label_in_payload(tmp_path):
    cfg, qdrant, _ = _populated(tmp_path)
    points, _ = qdrant.client.scroll(cfg.collection_name, limit=1, with_payload=True)
    assert points[0].payload["embedding_model"] == "hash-64"


# ── L2 normalisation ───────────────────────────────────────────────────────


def test_embed_query_is_l2_normalised(tmp_path):
    vec = embed_query("open a channel", _cfg(tmp_path, embedding_backend="hash"))
    assert len(vec) == 64
    assert _norm(vec) == pytest.approx(1.0, abs=1e-9)


def test_embed_documents_are_l2_normalised(counting, tmp_path):
    cfg = _cfg(tmp_path, embedding_backend="local", embedding_model="m", vector_size=8)
    vecs = embed_documents(["alpha", "a much longer document text"], cfg)
    assert all(_norm(v) == pytest.approx(1.0, abs=1e-9) for v in vecs)


def test_normalisation_can_be_switched_off(counting, tmp_path):
    cfg = _cfg(
        tmp_path, embedding_backend="local", embedding_model="m", vector_size=8, embedding_normalize=False
    )
    (vec,) = embed_documents(["alpha"], cfg)
    assert _norm(vec) > 1.5


def test_zero_vector_is_left_alone():
    assert embedding_module._l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_query_and_doc_prefixes_are_applied(counting, tmp_path):
    cfg = _cfg(
        tmp_path,
        embedding_backend="local",
        embedding_model="m",
        vector_size=8,
        embedding_query_prefix="query: ",
        embedding_doc_prefix="passage: ",
        embedding_cache_db="",
    )
    embed_query("hello", cfg)
    embed_documents(["hello"], cfg)
    assert counting.calls == [["query: hello"], ["passage: hello"]]


# ── document vector cache ──────────────────────────────────────────────────


def _real_cfg(tmp_path, **overrides):
    return _cfg(tmp_path, embedding_backend="local", embedding_model="m", vector_size=8, **overrides)


def test_cache_hit_skips_re_encoding(counting, tmp_path):
    cfg = _real_cfg(tmp_path)

    first = embed_documents(["alpha", "beta"], cfg)
    second = embed_documents(["alpha", "beta"], cfg)

    assert counting.calls == [["alpha", "beta"]]
    assert first == second


def test_cache_encodes_only_the_missing_texts(counting, tmp_path):
    cfg = _real_cfg(tmp_path)

    embed_documents(["alpha"], cfg)
    vecs = embed_documents(["alpha", "beta", "beta"], cfg)

    assert counting.calls == [["alpha"], ["beta"]]
    assert len(vecs) == 3 and vecs[1] == vecs[2]


def test_cache_is_keyed_by_model(counting, tmp_path):
    embed_documents(["alpha"], _real_cfg(tmp_path))
    other = _cfg(tmp_path, embedding_backend="local", embedding_model="other", vector_size=8)

    embed_documents(["alpha"], other)

    assert counting.calls == [["alpha"], ["alpha"]]


def test_cache_survives_a_new_connection(counting, tmp_path):
    cfg = _real_cfg(tmp_path)
    embed_documents(["alpha"], cfg)
    embedding_module._CACHES.clear()  # simulate a new process

    embed_documents(["alpha"], cfg)

    assert counting.calls == [["alpha"]]


def test_empty_cache_path_disables_the_cache(counting, tmp_path):
    cfg = _real_cfg(tmp_path, embedding_cache_db="")
    embed_documents(["alpha"], cfg)
    embed_documents(["alpha"], cfg)
    assert counting.calls == [["alpha"], ["alpha"]]


def test_hash_backend_does_not_touch_the_cache(tmp_path):
    cfg = _cfg(tmp_path, embedding_backend="hash")
    embed_documents(["alpha"], cfg)
    assert not (tmp_path / "cache").exists()


def test_embedding_cache_round_trip(tmp_path):
    cache = EmbeddingCache(str(tmp_path / "c.db"))
    cache.put_many([("h1", [0.5, 0.25]), ("h2", [1.0, 2.0])], "model-a")

    found = cache.get_many(["h1", "h2", "missing"], "model-a")

    assert found == {"h1": [0.5, 0.25], "h2": [1.0, 2.0]}
    assert cache.get_many(["h1"], "model-b") == {}
    assert cache.count("model-a") == 2
    cache.close()


# ── query vector cache ─────────────────────────────────────────────────────


def test_repeated_query_is_encoded_once(counting, tmp_path):
    cfg = _real_cfg(tmp_path)

    first = embed_query("open a channel", cfg)
    second = embed_query("open a channel", cfg)  # the other two collections of one search

    assert counting.calls == [["open a channel"]]
    assert first == second
    second[0] = -1.0
    assert embed_query("open a channel", cfg)[0] != -1.0, "callers get a copy"


def test_query_cache_is_keyed_by_text_prefix_and_model(counting, tmp_path):
    base = _real_cfg(tmp_path)
    embed_query("a", base)
    embed_query("b", base)
    embed_query("a", _cfg(tmp_path, embedding_backend="local", embedding_model="m", vector_size=8,
                          embedding_query_prefix="query: "))
    embed_query("a", _cfg(tmp_path, embedding_backend="local", embedding_model="other", vector_size=8))

    assert len(counting.calls) == 4


def test_query_cache_is_bounded(counting, tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_module, "_QUERY_CACHE_SIZE", 2)
    cfg = _real_cfg(tmp_path)
    for text in ("a", "b", "c"):
        embed_query(text, cfg)

    embed_query("a", cfg)  # evicted, encoded again

    assert len(counting.calls) == 4
