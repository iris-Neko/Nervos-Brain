"""Regression tests for the small retrieval fixes made alongside the embedding work."""
from __future__ import annotations

from nervos_brain.graph_engine.full_nodes import evidence_merger
from nervos_brain.retrieval import (
    ArchiveStore,
    DualLayerWriter,
    MultiRetriever,
    QdrantStore,
    RetrievalConfig,
)

_REGEX = [
    {
        "label": "Fiber",
        "pattern": r"(?i)\bfiber\b",
        "fields": ["title", "keywords", "summary"],
    }
]


def _retriever(tmp_path, count: int = 3):
    cfg = RetrievalConfig(
        qdrant_path=str(tmp_path / "qdrant"),
        archive_db=str(tmp_path / "archive.db"),
        vector_size=64,
        embedding_backend="hash",
    )
    qdrant = QdrantStore(config=cfg, qdrant_location=cfg.qdrant_path)
    archive = ArchiveStore(db_path=cfg.archive_db, config=cfg)
    writer = DualLayerWriter(qdrant_store=qdrant, archive_store=archive, config=cfg)
    for i in range(count):
        writer.write(
            source="docs",
            doc_type="doc",
            url=f"https://example.com/{i}",
            anchor=f"doc:{i}#chunk:0",
            title=f"Fiber note {i}" if i % 2 == 0 else f"Other note {i}",
            summary="about fiber payment channels" if i % 2 == 0 else "unrelated",
            keywords="fiber" if i % 2 == 0 else "misc",
            raw_text=f"body {i}",
            raw_format="text",
            lang="en",
            version="latest",
            topic="t",
        )
    retriever = MultiRetriever(qdrant_store=qdrant, archive_store=archive, config=cfg)
    retriever.rebuild_bm25()
    return retriever, archive


# ── regex path reuses the cached records ────────────────────────────────────


def test_regex_path_does_not_reread_the_archive_per_call(tmp_path, monkeypatch):
    retriever, archive = _retriever(tmp_path)
    reads: list[int] = []
    original = archive.list_all
    monkeypatch.setattr(archive, "list_all", lambda: reads.append(1) or original())

    normalized = [{**q, "_compiled": __import__("re").compile(q["pattern"])} for q in _REGEX]
    for _ in range(3):
        rows = retriever._regex_exact_search(normalized, 10)

    assert reads == []
    assert {row["anchor"] for row in rows} == {"doc:0#chunk:0", "doc:2#chunk:0"}


def test_regex_path_falls_back_to_the_archive_without_an_index(tmp_path):
    retriever, _ = _retriever(tmp_path)
    retriever._indexed_records = []  # rebuild_bm25() was never called
    normalized = [{**q, "_compiled": __import__("re").compile(q["pattern"])} for q in _REGEX]

    rows = retriever._regex_exact_search(normalized, 10)

    assert {row["anchor"] for row in rows} == {"doc:0#chunk:0", "doc:2#chunk:0"}


# ── evidence_merger dedup key ───────────────────────────────────────────────


def _evidence(anchor: str, *, score: float, rrf: str, source: str = "docs", **extra) -> dict:
    return {
        "id": anchor,
        "source": source,
        "title": "T",
        "url": "https://example.com",
        "anchor": anchor,
        "snippet": "s",
        "score": score,
        "payload": {"rrf_score": rrf, "retrieved_at": f"2026-11-02T00:00:{int(score * 100):02d}Z"},
        "hash": "",
        **extra,
    }


def test_same_document_from_two_hops_is_deduplicated():
    hop1 = _evidence("doc:a#chunk:0", score=0.03, rrf="0.03")
    hop2 = _evidence("doc:a#chunk:0", score=0.05, rrf="0.05")  # new score and timestamp

    merged = evidence_merger({"evidence": [hop1, hop2]})

    assert len(merged["evidence"]) == 1


def test_same_anchor_in_different_sources_is_kept_apart():
    docs = _evidence("doc:a#chunk:0", score=0.03, rrf="0.03", source="docs")
    forum = _evidence("doc:a#chunk:0", score=0.03, rrf="0.03", source="forum")

    assert len(evidence_merger({"evidence": [docs, forum]})["evidence"]) == 2


def test_different_anchors_are_kept_apart():
    a = _evidence("doc:a#chunk:0", score=0.03, rrf="0.03")
    b = _evidence("doc:b#chunk:0", score=0.03, rrf="0.03")

    assert len(evidence_merger({"evidence": [a, b]})["evidence"]) == 2


def test_existing_content_hash_still_wins():
    a = _evidence("doc:a#chunk:0", score=0.03, rrf="0.03", hash="same")
    b = _evidence("doc:b#chunk:0", score=0.03, rrf="0.03", hash="same")

    assert len(evidence_merger({"evidence": [a, b]})["evidence"]) == 1


def test_fallback_hash_is_stable_and_written_back():
    ev = _evidence("doc:a#chunk:0", score=0.03, rrf="0.03")
    again = _evidence("doc:a#chunk:0", score=0.9, rrf="0.9")

    evidence_merger({"evidence": [ev]})
    evidence_merger({"evidence": [again]})

    assert ev["hash"] and ev["hash"] == again["hash"]


def test_evidence_without_anchor_falls_back_to_content_hash():
    one = {"source": "x", "title": "a", "snippet": "one"}
    two = {"source": "x", "title": "a", "snippet": "two"}
    same = {"source": "x", "title": "a", "snippet": "one"}

    assert len(evidence_merger({"evidence": [one, two, same]})["evidence"]) == 2
