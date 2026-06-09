"""Multi-path retrieval orchestrator.

Pipeline (mirrors the design spec):
  1. Vector search  — semantic recall via Qdrant shallow index
  2. BM25 search    — keyword / term recall via in-memory BM25 index
  3. Fuzzy match    — naming variants, abbreviations, typos
  4. Exact match    — hard anchors: function names, class names, titles
  5. RRF fusion     — merge ranked lists with Reciprocal Rank Fusion
  6. Evidence hydration — attach full raw_text from the archive layer

Each path is independently gated by the corresponding ``enable_*`` flag in
RetrievalConfig, so you can turn off any path without touching the code.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from nervos_brain.core_protocols import Evidence

from .bm25_index import BM25Index
from .config import RetrievalConfig, load_retrieval_config
from .dual_layer import ArchiveRecord, ArchiveStore
from .embedding import get_embedding
from .evidence_adapter import qdrant_result_to_evidence
from .fuzzy_search import fuzzy_search
from .qdrant_writer import QdrantStore
from .rank_fusion import FusedResult, reciprocal_rank_fusion
from .search import _build_filter_must


_REGEX_FIELDS = {"title", "keywords", "anchor", "url", "summary", "raw_text"}
_DEFAULT_REGEX_FIELDS = ("title", "keywords", "anchor", "url", "summary")
_REGEX_MAX_QUERIES = 3
_REGEX_MAX_PATTERN_CHARS = 160
_REGEX_RAW_TEXT_CHARS = 4000
_REGEX_DANGEROUS_PATTERNS = (
    r"\.\*",
    r"\.\+",
    r"\(\?[:=!<P]",
    r"\\[1-9]",
    r"\(\.\*\)",
    r"\(\.\+\)",
    r"\{[0-9,]{3,}\}",
)


def _is_broad_resource_query(query: str) -> bool:
    text = str(query or "").lower()
    compact = "".join(text.split())
    markers = (
        "资料",
        "文档",
        "教程",
        "学习",
        "入门",
        "推荐",
        "靠谱",
        "可以看",
        "resources",
        "docs",
        "documentation",
        "tutorial",
        "learning",
        "recommended",
        "getting started",
    )
    return any(marker in text or marker in compact for marker in markers)


def _normalize_regex_queries(raw_queries: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not isinstance(raw_queries, list):
        return [], []

    normalized: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    seen: set[str] = set()

    for raw_query in raw_queries[:_REGEX_MAX_QUERIES]:
        if not isinstance(raw_query, dict):
            dropped.append({"reason": "regex_query_not_object"})
            continue
        label = str(raw_query.get("label", "") or "").strip()[:80]
        pattern = str(raw_query.get("pattern", "") or "").strip()
        if not pattern:
            dropped.append({"label": label, "reason": "regex_empty"})
            continue
        if len(pattern) > _REGEX_MAX_PATTERN_CHARS:
            dropped.append({"label": label, "reason": "regex_too_long"})
            continue
        if pattern in seen:
            dropped.append({"label": label, "reason": "regex_duplicate"})
            continue
        if _looks_dangerous_regex(pattern):
            dropped.append({"label": label, "reason": "regex_too_broad"})
            continue
        try:
            compiled = re.compile(pattern)
        except re.error:
            dropped.append({"label": label, "reason": "regex_compile_error"})
            continue

        raw_fields = raw_query.get("fields", _DEFAULT_REGEX_FIELDS)
        fields: list[str] = []
        if isinstance(raw_fields, list):
            for raw_field in raw_fields:
                field = str(raw_field or "").strip()
                if field in _REGEX_FIELDS and field not in fields:
                    fields.append(field)
        if not fields:
            fields = list(_DEFAULT_REGEX_FIELDS)

        seen.add(pattern)
        normalized.append(
            {
                "label": label,
                "pattern": pattern,
                "fields": fields,
                "reason": str(raw_query.get("reason", "") or "").strip()[:160],
                "_compiled": compiled,
            }
        )

    return normalized, dropped


def _looks_dangerous_regex(pattern: str) -> bool:
    stripped = pattern.strip()
    if stripped in {".*", ".+", "(?i).*", "(?i).+"}:
        return True
    for marker in _REGEX_DANGEROUS_PATTERNS:
        if re.search(marker, stripped):
            return True
    return False


class MultiRetriever:
    """Orchestrate all retrieval paths and return fused Evidence.

    Usage
    -----
    retriever = MultiRetriever(qdrant_store, archive_store)
    # on first use (or after new data is ingested) build the BM25 index:
    retriever.rebuild_bm25()
    # then query:
    evidence = retriever.search("how to open a Fiber channel?", filters={"topic": "fiber"})
    """

    def __init__(
        self,
        qdrant_store: QdrantStore | None = None,
        archive_store: ArchiveStore | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._cfg = config or load_retrieval_config()
        self._qdrant = qdrant_store or QdrantStore(
            config=self._cfg, qdrant_location=self._cfg.qdrant_path
        )
        self._archive = archive_store or ArchiveStore(config=self._cfg)
        self._bm25 = BM25Index()
        self._last_regex_summary: dict[str, Any] = {
            "regex_queries_count": 0,
            "regex_valid_count": 0,
            "regex_dropped_count": 0,
        }

    # ── index management ───────────────────────────────────────────────────

    def rebuild_bm25(self) -> int:
        """Rebuild the BM25 index from the current archive contents."""
        self._bm25.build_from_store(self._archive)
        return self._bm25.size

    # ── main search entry point ────────────────────────────────────────────

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, str]] = None,
        top_k: Optional[int] = None,
        regex_queries: Optional[list[dict[str, Any]]] = None,
    ) -> List[Evidence]:
        """Run all enabled retrieval paths, fuse, and return Evidence.

        Args:
            query:    Natural-language query or hard symbol (e.g. ``OpenChannel``).
            filters:  Qdrant payload filters applied to the vector path.
                      Must be provided (non-empty) to enable vector search;
                      if omitted, only BM25/fuzzy/exact paths run.
            top_k:    Override ``config.final_top_k``.
        """
        cfg = self._cfg
        final_top_k = top_k or cfg.final_top_k
        per_path_k = cfg.top_k_per_path

        ranked_lists: List[List[dict]] = []
        path_names: List[str] = []
        normalized_regex, dropped_regex = _normalize_regex_queries(regex_queries)
        regex_meta: dict[str, Any] = {
            "regex_queries_count": len(regex_queries or []),
            "regex_valid_count": len(normalized_regex),
            "regex_dropped_count": len(dropped_regex),
        }
        if dropped_regex:
            regex_meta["regex_dropped_reasons"] = ",".join(
                sorted({row.get("reason", "") for row in dropped_regex if row.get("reason")})
            )
        self._last_regex_summary = regex_meta

        # ── 1. Vector search ───────────────────────────────────────────────
        # Broad official-doc queries should still hit the fast Qdrant path.
        vec_results = self._vector_search(query, filters or {}, per_path_k)
        if vec_results:
            ranked_lists.append(vec_results)
            path_names.append("vector")

        run_slow_paths = not _is_broad_resource_query(query)

        # ── 2. BM25 keyword search ─────────────────────────────────────────
        if cfg.enable_bm25 and self._bm25.size > 0 and run_slow_paths:
            bm25_results = self._bm25_search(query, per_path_k, filters)
            if bm25_results:
                ranked_lists.append(bm25_results)
                path_names.append("bm25")

        # ── 3. Fuzzy match ─────────────────────────────────────────────────
        if cfg.enable_fuzzy and run_slow_paths:
            fuzzy_results = self._fuzzy_search(
                query, per_path_k, cfg.fuzzy_threshold, filters
            )
            if fuzzy_results:
                ranked_lists.append(fuzzy_results)
                path_names.append("fuzzy")

        # ── 4. Exact match ─────────────────────────────────────────────────
        if cfg.enable_exact and run_slow_paths:
            exact_results = self._exact_search(query, per_path_k, filters)
            if exact_results:
                ranked_lists.append(exact_results)
                path_names.append("exact")
        if cfg.enable_exact and normalized_regex:
            regex_results = self._regex_exact_search(normalized_regex, per_path_k, filters)
            if regex_results:
                ranked_lists.append(regex_results)
                path_names.append("regex_exact")

        if not ranked_lists:
            return []

        # ── 5. RRF fusion ──────────────────────────────────────────────────
        fused = reciprocal_rank_fusion(
            ranked_lists, k=cfg.rrf_k, path_names=path_names
        )[:final_top_k]

        # ── 6. Hydrate Evidence with archive raw_text ──────────────────────
        evidence = [self._to_evidence(r) for r in fused]
        for row in evidence:
            payload = row.get("payload")
            if isinstance(payload, dict):
                payload.update(regex_meta)
        return evidence

    @property
    def last_regex_summary(self) -> dict[str, Any]:
        return dict(self._last_regex_summary)

    # ── private path implementations ───────────────────────────────────────

    def _vector_search(
        self, query: str, filters: Dict[str, str], top_k: int
    ) -> List[dict]:
        query_vec = get_embedding(query, self._cfg)
        query_filter = _build_filter_must(filters) if filters else None
        try:
            response = self._qdrant.client.query_points(
                collection_name=self._qdrant.config.collection_name,
                query=query_vec,
                query_filter=query_filter,
                limit=top_k,
            )
            return [
                {
                    "anchor": r.payload.get("anchor", ""),
                    "title": r.payload.get("title", ""),
                    "source": r.payload.get("source", ""),
                    "score": float(r.score or 0.0),
                    "payload": dict(r.payload),
                    "qdrant_id": str(r.id),
                }
                for r in response.points
                if r.payload
            ]
        except Exception:
            return []

    def _bm25_search(
        self, query: str, top_k: int, filters: Optional[Dict[str, str]] = None
    ) -> List[dict]:
        if filters:
            records = [
                record
                for record in self._archive.list_all()
                if self._record_matches_filters(record, filters)
            ]
            if not records:
                return []
            index = BM25Index()
            index.build(records)
            hits = index.search(query, top_k=top_k)
        else:
            hits = self._bm25.search(query, top_k=top_k)
        return [
            {
                "anchor": h.anchor,
                "title": h.title,
                "source": h.source,
                "score": h.score,
            }
            for h in hits
        ]

    def _record_matches_filters(self, record: ArchiveRecord, filters: Dict[str, str]) -> bool:
        if not filters:
            return True
        payload = {
            "source": record.source,
            "type": record.doc_type,
            "doc_type": record.doc_type,
            "version": record.version,
            "lang": record.lang,
            "url": record.url,
            "anchor": record.anchor,
            "topic": record.topic,
            "title": record.title,
            "keywords": record.keywords,
        }
        for key, expected in filters.items():
            if expected in (None, ""):
                continue
            actual = payload.get(str(key), "")
            if str(actual) != str(expected):
                return False
        return True

    def _fuzzy_search(
        self, query: str, top_k: int, threshold: float, filters: Optional[Dict[str, str]] = None
    ) -> List[dict]:
        # Build candidate list from archive (titles + keywords only, cheap)
        candidates = [
            {
                "anchor": r.anchor,
                "title": r.title,
                "source": r.source,
                "keywords": r.keywords,
            }
            for r in self._archive.list_all()
            if self._record_matches_filters(r, filters or {})
        ]
        hits = fuzzy_search(query, candidates, threshold=threshold, top_k=top_k)
        return [
            {
                "anchor": h.anchor,
                "title": h.title,
                "source": h.source,
                "score": h.score,
            }
            for h in hits
        ]

    def _regex_exact_search(
        self,
        regex_queries: list[dict[str, Any]],
        top_k: int,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[dict]:
        if not regex_queries:
            return []

        weights = {
            "title": 10.0,
            "keywords": 8.0,
            "anchor": 7.0,
            "url": 6.0,
            "summary": 4.0,
            "raw_text": 1.5,
        }
        best: dict[str, dict] = {}
        for record in self._archive.list_all():
            if not self._record_matches_filters(record, filters or {}):
                continue
            for idx, query in enumerate(regex_queries, start=1):
                compiled = query.get("_compiled")
                if compiled is None:
                    continue
                score = 0.0
                matched_fields: list[str] = []
                for field in query.get("fields", _DEFAULT_REGEX_FIELDS):
                    text = self._regex_field_text(record, str(field))
                    if text and compiled.search(text):
                        score += weights.get(str(field), 1.0)
                        matched_fields.append(str(field))
                if score <= 0.0:
                    continue
                score += max(0.0, 0.1 - (idx * 0.01))
                existing = best.get(record.anchor)
                if existing is None or score > float(existing.get("score", 0.0)):
                    best[record.anchor] = {
                        "anchor": record.anchor,
                        "title": record.title,
                        "source": record.source,
                        "score": score,
                        "payload": {
                            "regex_label": query.get("label", ""),
                            "regex_pattern": query.get("pattern", ""),
                            "regex_fields": ",".join(matched_fields),
                        },
                    }

        results = list(best.values())
        results.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return results[:top_k]

    @staticmethod
    def _regex_field_text(record: ArchiveRecord, field: str) -> str:
        if field == "title":
            return record.title or ""
        if field == "keywords":
            return record.keywords or ""
        if field == "anchor":
            return record.anchor or ""
        if field == "url":
            return record.url or ""
        if field == "summary":
            return record.summary or ""
        if field == "raw_text":
            return (record.raw_text or "")[:_REGEX_RAW_TEXT_CHARS]
        return ""

    def _exact_search(
        self, query: str, top_k: int, filters: Optional[Dict[str, str]] = None
    ) -> List[dict]:
        """Exact-match against title, keywords, and anchor fields in Qdrant.

        Used for hard anchors like function names, class names, file paths.
        Tries three payload fields in order; merges and deduplicates.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchText

        results: List[dict] = []
        seen: set[str] = set()

        for field_name in ("title", "keywords", "anchor"):
            try:
                cond = Filter(
                    must=[FieldCondition(key=field_name, match=MatchText(text=query))]
                )
                resp = self._qdrant.client.scroll(
                    collection_name=self._qdrant.config.collection_name,
                    scroll_filter=cond,
                    limit=top_k,
                    with_payload=True,
                )
                for point in resp[0]:
                    payload = dict(point.payload or {})
                    if filters:
                        matched = True
                        for key, expected in filters.items():
                            if expected in (None, ""):
                                continue
                            if str(payload.get(str(key), "")) != str(expected):
                                matched = False
                                break
                        if not matched:
                            continue
                    anchor = payload.get("anchor", "")
                    if anchor and anchor not in seen:
                        seen.add(anchor)
                        results.append(
                            {
                                "anchor": anchor,
                                "title": payload.get("title", ""),
                                "source": payload.get("source", ""),
                                "score": 1.0,  # exact match = max score
                            }
                        )
            except Exception:
                continue

        return results[:top_k]

    def _to_evidence(self, fused: FusedResult) -> Evidence:
        """Convert a FusedResult to the protocol Evidence TypedDict.

        Tries to fetch the full raw_text from the archive for the snippet.
        Falls back to whatever is available in the Qdrant payload.
        """
        archive_rec = self._archive.get_by_anchor(fused.anchor)
        snippet = ""
        url = "unknown"
        payload = {
            "rrf_score": str(fused.rrf_score),
            **{f"contrib_{k}": str(v) for k, v in fused.contributions.items()},
        }
        if archive_rec:
            snippet = archive_rec.raw_text[:1200]
            url = archive_rec.url
            payload.update(
                {
                    "source": archive_rec.source,
                    "type": archive_rec.doc_type,
                    "doc_type": archive_rec.doc_type,
                    "version": archive_rec.version,
                    "lang": archive_rec.lang,
                    "url": archive_rec.url,
                    "anchor": archive_rec.anchor,
                    "topic": archive_rec.topic,
                    "title": archive_rec.title,
                    "keywords": archive_rec.keywords,
                }
            )

        return {
            "id": fused.anchor,
            "source": fused.source or "multi",
            "title": fused.title,
            "url": url,
            "anchor": fused.anchor,
            "snippet": snippet,
            "score": fused.rrf_score,
            "payload": payload,
            "hash": "",
            "retrieved_ts_ms": int(time.time() * 1000),
        }
