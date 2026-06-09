"""工具实际实现绑定：qdrant_search / memory_fetch 等。"""

from __future__ import annotations

import sqlite3
import time
import re
import inspect
from typing import Any

from nervos_brain.core_protocols import ToolCallRequest
from nervos_brain.retrieval import ArchiveStore, BM25Index, CompositeArchiveStore, tokenize


def handle_qdrant_search(request: ToolCallRequest) -> dict[str, Any]:
    """qdrant_search 工具处理器（需要外部注入 store）。"""
    from nervos_brain.retrieval import search_with_filters

    args = request["args"]
    retriever = args.get("_multi_retriever")
    if retriever is not None:
        search_kwargs: dict[str, Any] = {
            "query": str(args["query"]),
            "filters": args.get("filters") or None,
            "top_k": int(args.get("top_k", 5)),
        }
        if _search_accepts_regex_queries(retriever):
            search_kwargs["regex_queries"] = (
                args.get("regex_queries") if isinstance(args.get("regex_queries"), list) else None
            )
        results = retriever.search(**search_kwargs)
        regex_meta = _summarize_regex_payloads(results, retriever)
        return {
            "evidence": results,
            "data": {"hit_count": len(results), "backend": "multi_retriever", **regex_meta},
            "raw_size_bytes": sum(len(e.get("snippet", "")) for e in results),
            "redactions_applied": [],
        }

    store = args.get("_store")
    if store is None:
        return {"evidence": [], "data": {}, "raw_size_bytes": 0, "redactions_applied": []}

    results = search_with_filters(
        store=store,
        query=str(args["query"]),
        filters=args.get("filters", {}),
        top_k=int(args.get("top_k", 5)),
    )
    return {
        "evidence": results,
        "data": {"hit_count": len(results)},
        "raw_size_bytes": sum(len(e.get("snippet", "")) for e in results),
        "redactions_applied": [],
    }


def _summarize_regex_payloads(
    results: list[dict[str, Any]],
    retriever: Any | None = None,
) -> dict[str, int | str]:
    if retriever is not None:
        summary = getattr(retriever, "last_regex_summary", None)
        if isinstance(summary, dict):
            data: dict[str, int | str] = {
                "regex_queries_count": int(summary.get("regex_queries_count", 0) or 0),
                "regex_valid_count": int(summary.get("regex_valid_count", 0) or 0),
                "regex_dropped_count": int(summary.get("regex_dropped_count", 0) or 0),
            }
            reasons = str(summary.get("regex_dropped_reasons", "") or "").strip()
            if reasons:
                data["regex_dropped_reasons"] = reasons
            return data
    for row in results:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        if "regex_queries_count" not in payload:
            continue
        summary: dict[str, int | str] = {
            "regex_queries_count": int(payload.get("regex_queries_count", 0) or 0),
            "regex_valid_count": int(payload.get("regex_valid_count", 0) or 0),
            "regex_dropped_count": int(payload.get("regex_dropped_count", 0) or 0),
        }
        reasons = str(payload.get("regex_dropped_reasons", "") or "").strip()
        if reasons:
            summary["regex_dropped_reasons"] = reasons
        return summary
    return {"regex_queries_count": 0, "regex_valid_count": 0, "regex_dropped_count": 0}


def _search_accepts_regex_queries(retriever: Any) -> bool:
    try:
        signature = inspect.signature(retriever.search)
    except (TypeError, ValueError, AttributeError):
        return False
    return "regex_queries" in signature.parameters


def handle_memory_fetch(request: ToolCallRequest) -> dict[str, Any]:
    """memory_fetch 工具处理器（需要外部注入 memory_service）。"""
    args = request["args"]
    svc = args.get("_memory_service")
    if svc is None:
        return {"evidence": [], "data": {}, "raw_size_bytes": 0, "redactions_applied": []}

    namespace = args.get("namespace", "user")
    platform = args.get("platform", "")

    if namespace == "user":
        key = {"platform": platform, "user_id": args.get("user_id", "")}
        facts = svc.list_user_facts(key=key)
    else:
        key = {
            "platform": platform,
            "guild_id": args.get("guild_id", ""),
            "channel_id": args.get("channel_id", ""),
        }
        facts = svc.list_channel_facts(key=key)

    evidence = [
        {
            "id": f["id"],
            "source": "memory",
            "title": f"fact:{f['key']}",
            "url": f"memory://{namespace}/{f['id']}",
            "anchor": f"kind:fact",
            "snippet": f"{f['key']}={f['value']}",
            "score": f["confidence"],
            "payload": {"source": "memory", "type": "fact", "namespace": namespace},
            "hash": f["id"],
            "retrieved_ts_ms": f["updated_ts_ms"],
        }
        for f in facts
    ]
    return {
        "evidence": evidence,
        "data": {"fact_count": len(facts)},
        "raw_size_bytes": sum(len(str(f)) for f in facts),
        "redactions_applied": [],
    }


def handle_discourse_query(request: ToolCallRequest) -> dict[str, Any]:
    """discourse_query 处理器，优先走 transport，其次本地 archive 回退。"""
    args = request["args"]
    transport = args.get("_transport")
    if transport is not None:
        raw = transport.send(
            {
                "tool": "discourse_query",
                "request_id": request["request_id"],
                "step_id": request["step_id"],
                "query": str(args["query"]),
                "category": str(args.get("category", "")),
                "time_range": str(args.get("time_range", "")),
                "top_k": int(args.get("top_k", 5)),
            }
        )
        if isinstance(raw, dict):
            return _normalize_external_tool_result(raw, source="discourse")

    store = args.get("_archive_store")
    if isinstance(store, (ArchiveStore, CompositeArchiveStore)):
        return _search_archive_records(
            store=store,
            source="discourse",
            query=str(args["query"]),
            top_k=int(args.get("top_k", 5)),
            category=str(args.get("category", "")),
        )
    return {"evidence": [], "data": {}, "raw_size_bytes": 0, "redactions_applied": []}


def handle_github_search(request: ToolCallRequest) -> dict[str, Any]:
    """github_search 处理器，优先走 transport，其次本地 archive 回退。"""
    args = request["args"]
    transport = args.get("_transport")
    if transport is not None:
        raw = transport.send(
            {
                "tool": "github_search",
                "request_id": request["request_id"],
                "step_id": request["step_id"],
                "query": str(args["query"]),
                "repo": str(args.get("repo", "")),
                "path": str(args.get("path", "")),
                "top_k": int(args.get("top_k", 5)),
            }
        )
        if isinstance(raw, dict):
            return _normalize_external_tool_result(raw, source="github")

    store = args.get("_archive_store")
    if isinstance(store, (ArchiveStore, CompositeArchiveStore)):
        return _search_archive_records(
            store=store,
            source="github",
            query=str(args["query"]),
            top_k=int(args.get("top_k", 5)),
            repo=str(args.get("repo", "")),
            path=str(args.get("path", "")),
        )
    return {"evidence": [], "data": {}, "raw_size_bytes": 0, "redactions_applied": []}


def _normalize_external_tool_result(raw: dict[str, Any], *, source: str) -> dict[str, Any]:
    evidence = raw.get("evidence", [])
    normalized: list[dict[str, Any]] = []
    if isinstance(evidence, list):
        for row in evidence:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item["source"] = source
            normalized.append(item)
    return {
        "evidence": normalized,
        "data": raw.get("data", {}),
        "raw_size_bytes": int(raw.get("raw_size_bytes", 0) or 0),
        "redactions_applied": list(raw.get("redactions_applied", [])),
    }


def _search_archive_records(
    *,
    store: ArchiveStore | CompositeArchiveStore,
    source: str,
    query: str,
    top_k: int,
    category: str = "",
    repo: str = "",
    path: str = "",
) -> dict[str, Any]:
    if source == "discourse":
        category = _normalize_discourse_category(category)
    query_variants = _expand_archive_queries(query, source=source)
    sqlite_result = _search_archive_records_sql(
        store=store,
        source=source,
        query_variants=query_variants,
        top_k=max(1, min(int(top_k), 20)),
        category=category,
        repo=repo,
        path=path,
    )
    if sqlite_result is not None:
        return sqlite_result

    records = []
    for record in store.list_all():
        if source == "discourse" and record.source not in {"nervos_talk", "discourse"}:
            continue
        if source == "github" and not record.source.startswith("github"):
            continue
        if category and not _record_matches(record, category):
            continue
        if repo and not _record_matches(record, repo):
            continue
        if path and not _record_matches(record, path):
            continue
        records.append(record)

    if not records:
        return {"evidence": [], "data": {"hit_count": 0}, "raw_size_bytes": 0, "redactions_applied": []}

    index = BM25Index()
    index.build(records)
    hits = _search_bm25_variants(
        index=index,
        queries=query_variants,
        top_k=max(1, min(int(top_k), 20)),
    )
    by_anchor = {record.anchor: record for record in records}
    ranked_records: list[tuple[float, Any]] = []
    for hit in hits:
        record = by_anchor.get(hit.anchor)
        if record is None:
            continue
        ranked_records.append((float(hit.score), record))

    if not ranked_records:
        ranked_records = _fallback_archive_rank(
            records,
            query_variants,
            max(1, min(int(top_k), 20)),
        )

    evidence = []
    for score, record in ranked_records:
        evidence.append(
            {
                "id": record.anchor,
                "source": source,
                "title": record.title,
                "url": record.url,
                "anchor": record.anchor,
                "snippet": record.raw_text[:1200],
                "score": float(score),
                "payload": {
                    "source": record.source,
                    "type": record.doc_type,
                    "version": record.version,
                    "topic": record.topic,
                },
                "hash": record.content_hash,
                "retrieved_ts_ms": int(time.time() * 1000),
            }
        )

    return {
        "evidence": evidence,
        "data": {"hit_count": len(evidence), "backend": "archive_bm25"},
        "raw_size_bytes": sum(len(row.get("snippet", "")) for row in evidence),
        "redactions_applied": [],
    }


def _search_archive_records_sql(
    *,
    store: ArchiveStore | CompositeArchiveStore,
    source: str,
    query_variants: list[str],
    top_k: int,
    category: str = "",
    repo: str = "",
    path: str = "",
) -> dict[str, Any] | None:
    db_paths = _archive_db_paths(store)
    if not db_paths:
        return None

    terms = _archive_sql_terms(query_variants)
    if not terms:
        return None

    rows: list[dict[str, Any]] = []
    per_db_limit = max(50, top_k * 40)
    for db_path in db_paths:
        rows.extend(
            _query_archive_db(
                db_path=db_path,
                source=source,
                terms=terms,
                category=category,
                repo=repo,
                path=path,
                limit=per_db_limit,
            )
        )

    ranked = _rank_archive_rows(rows, query_variants, terms, top_k)
    evidence = [_archive_row_to_evidence(row, source=source, score=score) for score, row in ranked]
    return {
        "evidence": evidence,
        "data": {"hit_count": len(evidence), "backend": "archive_sql_prefilter"},
        "raw_size_bytes": sum(len(row.get("snippet", "")) for row in evidence),
        "redactions_applied": [],
    }


def _archive_db_paths(store: ArchiveStore | CompositeArchiveStore) -> list[str]:
    stores = store.stores if isinstance(store, CompositeArchiveStore) else [store]
    paths: list[str] = []
    seen: set[str] = set()
    for item in stores:
        db_path = str(getattr(item, "db_path", "") or "")
        if db_path and db_path not in seen:
            seen.add(db_path)
            paths.append(db_path)
    return paths


def _archive_sql_terms(query_variants: list[str]) -> list[str]:
    text = " ".join(str(item) for item in query_variants if str(item).strip())
    lowered = text.lower()
    terms: list[str] = []

    def add(term: str) -> None:
        cleaned = re.sub(r"\s+", " ", str(term or "").strip())
        if cleaned and cleaned.lower() not in {item.lower() for item in terms}:
            terms.append(cleaned)

    for word in re.findall(r"[A-Za-z0-9_.+-]{2,}", text):
        lower = word.lower()
        if lower in {
            "the",
            "and",
            "for",
            "with",
            "about",
            "forum",
            "talk",
            "community",
            "nervos",
            "ckb",
        }:
            continue
        add(word)
        if lower.endswith("s") and len(lower) > 4:
            add(word[:-1])

    known_phrases = (
        "Nervos Brain",
        "Nervos.Land",
        "prompt injection",
        "开发进度",
        "周报",
        "真实数据",
        "游戏",
        "项目",
        "讨论",
        "社区",
        "钱包",
        "交易",
        "转账",
        "通道",
        "节点",
        "报错",
        "错误",
    )
    for phrase in known_phrases:
        if phrase.lower() in lowered or phrase in text:
            add(phrase)

    if "游戏" in text:
        for term in ("game", "gaming", "GameFi", "Nervos.Land"):
            add(term)
    if "开发" in text or "进度" in text:
        for term in ("开发进度", "周报", "progress"):
            add(term)

    return terms[:16]


def _query_archive_db(
    *,
    db_path: str,
    source: str,
    terms: list[str],
    category: str,
    repo: str,
    path: str,
    limit: int,
) -> list[dict[str, Any]]:
    source_sql = "source in ('nervos_talk', 'discourse')" if source == "discourse" else "source like 'github%'"
    fields = ("title", "summary", "keywords", "raw_text", "topic", "url", "anchor")
    like_clauses: list[str] = []
    where_params: list[Any] = []
    score_clauses: list[str] = []
    score_params: list[Any] = []
    for term in terms:
        title_weight = 6 if " " in term or any("\u4e00" <= ch <= "\u9fff" for ch in term) else 3
        body_weight = 3 if " " in term or any("\u4e00" <= ch <= "\u9fff" for ch in term) else 1
        for field, weight in (
            ("title", title_weight),
            ("keywords", title_weight),
            ("summary", body_weight),
            ("raw_text", body_weight),
            ("topic", 2),
            ("url", 1),
            ("anchor", 1),
        ):
            score_clauses.append(f"case when {field} like ? then {weight} else 0 end")
            score_params.append(f"%{term}%")
        for field in fields:
            like_clauses.append(f"{field} like ?")
            where_params.append(f"%{term}%")

    where = f"{source_sql} and (" + " or ".join(like_clauses) + ")"
    score_sql = " + ".join(score_clauses) if score_clauses else "0"
    params = [*score_params, *where_params, int(limit)]
    sql = f"""
        select id, source, doc_type, url, anchor, title, summary, keywords,
               raw_text, raw_format, lang, version, topic, content_hash,
               created_ts, updated_ts,
               ({score_sql}) as match_score
        from archive_records
        where {where}
        order by match_score desc, updated_ts desc, created_ts desc
        limit ?
    """

    rows: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        for row in con.execute(sql, params):
            item = dict(row)
            if category and not _archive_row_matches(item, category):
                continue
            if repo and not _archive_row_matches(item, repo):
                continue
            if path and not _archive_row_matches(item, path):
                continue
            rows.append(item)
    return rows


def _archive_row_matches(row: dict[str, Any], needle: str) -> bool:
    if not needle:
        return True
    needles = _expand_filter_needles(needle)
    haystack = " ".join(
        str(row.get(key, ""))
        for key in ("title", "topic", "url", "keywords", "summary", "anchor")
        if row.get(key)
    ).lower()
    return any(item and item in haystack for item in needles)


def _rank_archive_rows(
    rows: list[dict[str, Any]],
    query_variants: list[str],
    terms: list[str],
    top_k: int,
) -> list[tuple[float, dict[str, Any]]]:
    by_anchor: dict[str, tuple[float, dict[str, Any]]] = {}
    query_texts = [query.strip().lower() for query in query_variants if query.strip()]
    lowered_terms = [term.lower() for term in terms if term.strip()]
    for row in rows:
        title = str(row.get("title", "") or "")
        haystack = " ".join(
            str(row.get(key, ""))
            for key in ("title", "keywords", "summary", "raw_text", "topic", "url", "anchor")
            if row.get(key)
        ).lower()
        title_lower = title.lower()
        score = 0.0
        for query_text in query_texts:
            if query_text and query_text in haystack:
                score += 8.0
        for term in lowered_terms:
            if term in title_lower:
                score += 3.0
            elif term in haystack:
                score += 1.0
        try:
            score += min(float(row.get("updated_ts") or 0) / 1_000_000_000_000_000, 1.0)
        except (TypeError, ValueError):
            pass
        if score <= 0:
            continue
        anchor = str(row.get("anchor", ""))
        existing = by_anchor.get(anchor)
        if existing is None or score > existing[0]:
            by_anchor[anchor] = (score, row)
    ranked = sorted(by_anchor.values(), key=lambda item: item[0], reverse=True)
    return ranked[:top_k]


def _archive_row_to_evidence(row: dict[str, Any], *, source: str, score: float) -> dict[str, Any]:
    return {
        "id": row.get("anchor", ""),
        "source": source,
        "title": row.get("title", ""),
        "url": row.get("url", ""),
        "anchor": row.get("anchor", ""),
        "snippet": str(row.get("raw_text", "") or "")[:1200],
        "score": float(score),
        "payload": {
            "source": row.get("source", ""),
            "type": row.get("doc_type", ""),
            "version": row.get("version", ""),
            "topic": row.get("topic", ""),
        },
        "hash": row.get("content_hash", ""),
        "retrieved_ts_ms": int(time.time() * 1000),
    }


def _record_matches(record: Any, needle: str) -> bool:
    if not needle:
        return True
    needles = _expand_filter_needles(needle)
    haystacks = (
        getattr(record, "title", ""),
        getattr(record, "topic", ""),
        getattr(record, "url", ""),
        getattr(record, "keywords", ""),
        getattr(record, "summary", ""),
        getattr(record, "anchor", ""),
    )
    text = " ".join(str(value).lower() for value in haystacks if value)
    return any(item and item in text for item in needles)


def _fallback_archive_rank(records: list[Any], queries: list[str], top_k: int) -> list[tuple[float, Any]]:
    ranked: list[tuple[float, Any]] = []
    query_texts = [query.strip().lower() for query in queries if query.strip()]
    query_tokens: set[str] = set()
    for query in queries:
        query_tokens.update(token for token in tokenize(query) if token)
    for record in records:
        haystack = " ".join(
            str(value)
            for value in (
                getattr(record, "title", ""),
                getattr(record, "keywords", ""),
                getattr(record, "summary", ""),
                getattr(record, "raw_text", "")[:2000],
            )
            if value
        ).lower()
        overlap = sum(1 for token in query_tokens if token in haystack)
        if any(query_text and query_text in haystack for query_text in query_texts):
            overlap += max(2, len(query_tokens))
        if overlap > 0:
            ranked.append((float(overlap), record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:top_k]


def _search_bm25_variants(
    *,
    index: BM25Index,
    queries: list[str],
    top_k: int,
) -> list[Any]:
    by_anchor: dict[str, Any] = {}
    for rank_bias, query in enumerate(queries):
        hits = index.search(query, top_k=top_k)
        for hit in hits:
            adjusted_score = float(hit.score) * (1.0 - min(rank_bias, 5) * 0.04)
            existing = by_anchor.get(hit.anchor)
            if existing is None or adjusted_score > float(existing.score):
                hit.score = adjusted_score
                by_anchor[hit.anchor] = hit
    ordered = sorted(by_anchor.values(), key=lambda item: float(item.score), reverse=True)
    return ordered[:top_k]


def _normalize_discourse_category(category: str) -> str:
    text = re.sub(r"\s+", " ", str(category or "").strip().lower())
    platform_only = {
        "nervos talk",
        "talk",
        "talk.nervos.org",
        "forum",
        "forums",
        "community",
        "社区",
        "论坛",
        "讨论",
        "帖子",
    }
    if text in platform_only:
        return ""
    if text and all(part in platform_only for part in re.split(r"[,/| ]+", text) if part):
        return ""
    return str(category or "").strip()


def _expand_archive_queries(query: str, *, source: str) -> list[str]:
    base = re.sub(r"\s+", " ", str(query or "").strip())
    if not base:
        return [""]

    normalized = _strip_platform_words(base)
    variants = [base]
    if normalized and normalized != base:
        variants.append(normalized)

    lowered = base.lower()
    expansions: list[str] = []
    if source == "discourse":
        expansions.extend(["community discussion", "forum discussion"])
    if "游戏" in base:
        expansions.extend(["game", "gaming", "decentralized game", "NFT game", "Nervos.Land"])
    if "去中心化" in base:
        expansions.append("decentralized")
    if "做" in base or "构建" in base or "开发" in base:
        expansions.extend(["build", "development"])
    if "讨论" in base or "帖子" in base or "论坛" in base:
        expansions.extend(["discussion", "thread"])
    if "交易" in base:
        expansions.extend(["transaction", "transfer"])
    if "钱包" in base:
        expansions.append("wallet")
    if "报错" in base or "错误" in base:
        expansions.extend(["error", "issue"])
    if "ckb" in lowered:
        expansions.append("CKB")
    if "nervos" in lowered:
        expansions.append("Nervos")

    if expansions:
        variants.append(" ".join([normalized or base, *expansions]))

    out: list[str] = []
    seen: set[str] = set()
    for item in variants:
        text = item.strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out or [base]


def _expand_filter_needles(needle: str) -> list[str]:
    raw = str(needle or "").strip()
    stripped = _strip_platform_words(raw)
    needles = [raw.lower()]
    if stripped:
        needles.append(stripped.lower())
    if "游戏" in raw:
        needles.extend(["game", "gaming", "nervos.land"])
    if "去中心化" in raw:
        needles.append("decentralized")
    if "交易" in raw:
        needles.extend(["transaction", "transfer"])
    if "钱包" in raw:
        needles.append("wallet")
    seen: set[str] = set()
    out: list[str] = []
    for item in needles:
        value = item.strip().lower()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _strip_platform_words(text: str) -> str:
    cleaned = str(text or "")
    for pattern in (
        r"@NBCKB_Bot",
        r"(?i)nervos\s*talk",
        r"(?i)talk\.nervos\.org",
        r"(?i)\bforum\b",
        r"(?i)\bforums\b",
        r"论坛",
        r"帖子",
        r"讨论",
        r"找一下",
        r"查一下",
        r"搜索",
        r"关于",
        r"上",
    ):
        cleaned = re.sub(pattern, " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


TOOL_HANDLERS: dict[str, Any] = {
    "qdrant_search": handle_qdrant_search,
    "discourse_query": handle_discourse_query,
    "github_search": handle_github_search,
    "memory_fetch": handle_memory_fetch,
}
