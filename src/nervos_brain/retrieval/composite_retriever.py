"""Configurable multi-backend retrieval runtime.

This module lets the online runtime query multiple independent retrieval
stores, for example the GitHub docs corpus and the Nervos Talk forum corpus,
without merging their Qdrant directories or SQLite archives.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Iterable

from nervos_brain.core_protocols import Evidence

from .config import RetrievalConfig, load_retrieval_backend_configs
from .dual_layer import ArchiveRecord, ArchiveStore
from .multi_retriever import MultiRetriever
from .qdrant_writer import QdrantStore


@dataclass(frozen=True)
class RetrievalBackend:
    """One configured retrieval backend."""

    name: str
    config: RetrievalConfig
    qdrant_store: QdrantStore
    archive_store: ArchiveStore
    retriever: MultiRetriever


class CompositeArchiveStore:
    """Read-only archive facade over multiple ``ArchiveStore`` instances."""

    def __init__(self, stores: Iterable[ArchiveStore] = ()) -> None:
        self._stores = list(stores)

    @property
    def stores(self) -> list[ArchiveStore]:
        return list(self._stores)

    def get_by_anchor(self, anchor: str) -> ArchiveRecord | None:
        for store in self._stores:
            record = store.get_by_anchor(anchor)
            if record is not None:
                return record
        return None

    def list_all(self) -> list[ArchiveRecord]:
        records: list[ArchiveRecord] = []
        for store in self._stores:
            records.extend(store.list_all())
        return records

    def count(self) -> int:
        return sum(store.count() for store in self._stores)


class CompositeRetriever:
    """Merge results from multiple ``MultiRetriever`` backends."""

    def __init__(
        self,
        backends: Iterable[RetrievalBackend],
        *,
        final_top_k: int | None = None,
    ) -> None:
        self.backends = list(backends)
        self._archive = CompositeArchiveStore(
            backend.archive_store for backend in self.backends
        )
        self._final_top_k = final_top_k
        self._cfg = self.backends[0].config if self.backends else RetrievalConfig()
        self._last_regex_summary: dict[str, Any] = {
            "regex_queries_count": 0,
            "regex_valid_count": 0,
            "regex_dropped_count": 0,
        }

    def rebuild_bm25(self) -> int:
        """Rebuild all backend BM25 indexes and return the total indexed size."""
        return sum(backend.retriever.rebuild_bm25() for backend in self.backends)

    @property
    def backend_names(self) -> list[str]:
        return [backend.name for backend in self.backends]

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        regex_queries: list[dict[str, Any]] | None = None,
    ) -> list[Evidence]:
        """Search every backend, merge duplicate anchors, and return best hits."""
        if not self.backends:
            return []

        final_top_k = top_k or self._final_top_k or self._cfg.final_top_k
        per_backend_k = max(final_top_k, self._cfg.final_top_k)

        merged: dict[tuple[str, str], Evidence] = {}
        order: dict[tuple[str, str], int] = {}
        sequence = 0
        regex_summary: dict[str, Any] = {
            "regex_queries_count": len(regex_queries or []),
            "regex_valid_count": 0,
            "regex_dropped_count": 0,
        }
        dropped_reasons: set[str] = set()
        for backend in self.backends:
            search_kwargs: dict[str, Any] = {
                "query": query,
                "filters": filters,
                "top_k": per_backend_k,
            }
            if _search_accepts_regex_queries(backend.retriever):
                search_kwargs["regex_queries"] = regex_queries
            results = backend.retriever.search(**search_kwargs)
            backend_regex_summary = getattr(backend.retriever, "last_regex_summary", {})
            if not isinstance(backend_regex_summary, dict):
                backend_regex_summary = {}
            regex_summary["regex_valid_count"] = max(
                int(regex_summary.get("regex_valid_count", 0) or 0),
                int(backend_regex_summary.get("regex_valid_count", 0) or 0),
            )
            regex_summary["regex_dropped_count"] = max(
                int(regex_summary.get("regex_dropped_count", 0) or 0),
                int(backend_regex_summary.get("regex_dropped_count", 0) or 0),
            )
            reasons = str(backend_regex_summary.get("regex_dropped_reasons", "") or "")
            dropped_reasons.update(reason for reason in reasons.split(",") if reason)
            for row in results:
                key = _evidence_key(row)
                if key not in order:
                    order[key] = sequence
                    sequence += 1
                existing = merged.get(key)
                candidate = dict(row)
                payload = candidate.get("payload")
                if isinstance(payload, dict):
                    candidate["payload"] = {**payload, "backend": backend.name}
                else:
                    candidate["payload"] = {"backend": backend.name}
                candidate["source"] = str(candidate.get("source") or backend.name)
                if existing is None or _score(candidate) > _score(existing):
                    merged[key] = candidate  # type: ignore[assignment]

        ranked = sorted(
            merged.values(),
            key=lambda row: (_score(row), -order[_evidence_key(row)]),
            reverse=True,
        )
        if dropped_reasons:
            regex_summary["regex_dropped_reasons"] = ",".join(sorted(dropped_reasons))
        self._last_regex_summary = regex_summary
        return ranked[:final_top_k]

    @property
    def last_regex_summary(self) -> dict[str, Any]:
        return dict(self._last_regex_summary)


def build_configured_retriever(
    *,
    sections: list[str] | None = None,
    rebuild_bm25: bool = True,
) -> CompositeRetriever | MultiRetriever:
    """Build the retriever described by ``config.yaml``.

    With one configured backend this returns a plain ``MultiRetriever`` for
    backward compatibility. With two or more it returns ``CompositeRetriever``.
    """
    backends: list[RetrievalBackend] = []
    for name, cfg in load_retrieval_backend_configs(sections=sections):
        qdrant = QdrantStore(config=cfg, qdrant_location=cfg.qdrant_path)
        archive = ArchiveStore(db_path=cfg.archive_db, config=cfg)
        retriever = MultiRetriever(
            qdrant_store=qdrant,
            archive_store=archive,
            config=cfg,
        )
        backends.append(
            RetrievalBackend(
                name=name,
                config=cfg,
                qdrant_store=qdrant,
                archive_store=archive,
                retriever=retriever,
            )
        )

    if not backends:
        cfg = RetrievalConfig()
        qdrant = QdrantStore(config=cfg, qdrant_location=cfg.qdrant_path)
        archive = ArchiveStore(db_path=cfg.archive_db, config=cfg)
        retriever = MultiRetriever(qdrant_store=qdrant, archive_store=archive, config=cfg)
        if rebuild_bm25:
            retriever.rebuild_bm25()
        return retriever

    if len(backends) == 1:
        retriever = backends[0].retriever
        if rebuild_bm25:
            retriever.rebuild_bm25()
        return retriever

    composite = CompositeRetriever(backends)
    if rebuild_bm25:
        composite.rebuild_bm25()
    return composite


def _evidence_key(row: dict[str, Any]) -> tuple[str, str]:
    anchor = str(row.get("anchor") or row.get("id") or "")
    source = str(row.get("source") or "")
    if anchor:
        return source, anchor
    return source, str(id(row))


def _score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _search_accepts_regex_queries(retriever: Any) -> bool:
    try:
        signature = inspect.signature(retriever.search)
    except (TypeError, ValueError, AttributeError):
        return False
    return "regex_queries" in signature.parameters
