"""Dual-layer storage model.

浅层索引 (Shallow index)  →  Qdrant
    每个文档/切片存向量 + 浓缩 payload（标题、摘要、关键词、锚点）。
    用于语义检索和精确字段过滤。

深层原件 (Archive layer)  →  SQLite via SQLAlchemy
    存放原始材料：markdown/html 全文、代码源文件、日志、JSON 等。
    由 anchor 与浅层索引一一对应，供精读和 BM25 倒排索引使用。
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import List, Optional

from sqlalchemy import String, Text, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .anchor_hash import build_stable_hash
from .config import RetrievalConfig, load_retrieval_config
from nervos_brain.pathing import resolve_project_path
from .qdrant_writer import QdrantStore


# ── SQLAlchemy model ────────────────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class ArchiveRecord(_Base):
    """深层原件表。每行对应一个可被 anchor 唯一定位的原始文本片段。"""

    __tablename__ = "archive_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(64))
    doc_type: Mapped[str] = mapped_column(String(32), default="doc")
    url: Mapped[str] = mapped_column(String(512), default="unknown")
    anchor: Mapped[str] = mapped_column(String(256), index=True)
    title: Mapped[str] = mapped_column(String(512), default="")
    # 短摘要 (≤300 chars) — 写入 Qdrant payload 的 snippet 字段
    summary: Mapped[str] = mapped_column(Text, default="")
    # 逗号分隔的关键词 / 函数名 / 类名
    keywords: Mapped[str] = mapped_column(Text, default="")
    # 原始全文 (markdown, html, code, plain text …)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    raw_format: Mapped[str] = mapped_column(String(32), default="text")
    lang: Mapped[str] = mapped_column(String(16), default="unknown")
    version: Mapped[str] = mapped_column(String(64), default="unknown")
    topic: Mapped[str] = mapped_column(String(128), default="unknown")
    # SHA-256 of (url, anchor, raw_text) — 用于去重
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_ts: Mapped[int] = mapped_column(default=0)
    updated_ts: Mapped[int] = mapped_column(default=0)

    def __repr__(self) -> str:
        return f"<ArchiveRecord anchor={self.anchor!r} title={self.title!r}>"


# ── ArchiveStore ────────────────────────────────────────────────────────────

class ArchiveStore:
    """SQLite 深层原件仓库的读写接口。"""

    def __init__(self, db_path: str | None = None, config: RetrievalConfig | None = None) -> None:
        cfg = config or load_retrieval_config()
        path = resolve_project_path(db_path or cfg.archive_db)
        self.db_path = str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{path}", echo=False)
        _Base.metadata.create_all(engine)
        # expire_on_commit=False keeps attributes readable after the session
        # closes, so callers can safely re-pass the same record object.
        self._factory = sessionmaker(engine, expire_on_commit=False)

    # ── write ──────────────────────────────────────────────────────────────

    def upsert(self, record: ArchiveRecord) -> None:
        """Insert or update by content_hash (idempotent)."""
        with self._factory() as session:
            existing = session.scalar(
                select(ArchiveRecord).where(ArchiveRecord.content_hash == record.content_hash)
            )
            now = int(time.time() * 1000)
            if existing is None:
                record.created_ts = now
                record.updated_ts = now
                session.add(record)
            else:
                existing.title = record.title
                existing.summary = record.summary
                existing.keywords = record.keywords
                existing.raw_text = record.raw_text
                existing.updated_ts = now
            session.commit()

    # ── read ───────────────────────────────────────────────────────────────

    def get_by_anchor(self, anchor: str) -> Optional[ArchiveRecord]:
        with self._factory() as session:
            return session.scalar(
                select(ArchiveRecord).where(ArchiveRecord.anchor == anchor)
            )

    def list_anchors_with_prefix(self, prefix: str) -> set[str]:
        """Return all anchors starting with ``prefix``.

        Used by incremental forum crawlers to skip already-ingested posts
        for a given topic without loading the full archive into memory.
        """
        if not prefix:
            return set()
        pattern = f"{prefix}%"
        with self._factory() as session:
            rows = session.scalars(
                select(ArchiveRecord.anchor).where(ArchiveRecord.anchor.like(pattern))
            ).all()
            return set(rows)

    def get_by_hash(self, content_hash: str) -> Optional[ArchiveRecord]:
        with self._factory() as session:
            return session.scalar(
                select(ArchiveRecord).where(ArchiveRecord.content_hash == content_hash)
            )

    def list_all(self) -> List[ArchiveRecord]:
        """Return all records (used for building BM25 index in memory)."""
        with self._factory() as session:
            rows = session.scalars(select(ArchiveRecord)).all()
            # detach from session before returning
            session.expunge_all()
            return list(rows)

    def list_by_source_topic(self, *, source: str, topic: str) -> List[ArchiveRecord]:
        """Return records for one source/topic pair."""
        with self._factory() as session:
            rows = session.scalars(
                select(ArchiveRecord).where(
                    ArchiveRecord.source == source,
                    ArchiveRecord.topic == topic,
                )
            ).all()
            session.expunge_all()
            return list(rows)

    def delete_by_hashes(self, hashes: list[str]) -> int:
        """Delete archive records by content_hash and return deleted count."""
        values = [str(value).strip() for value in hashes if str(value).strip()]
        if not values:
            return 0
        with self._factory() as session:
            result = session.execute(
                delete(ArchiveRecord).where(ArchiveRecord.content_hash.in_(values))
            )
            session.commit()
            return int(result.rowcount or 0)

    def count(self) -> int:
        with self._factory() as session:
            return session.query(ArchiveRecord).count()


# ── DualLayerWriter ─────────────────────────────────────────────────────────

class DualLayerWriter:
    """浅层（Qdrant）+ 深层（SQLite）同步写入器。

    入库流程:
    1. 计算 content_hash（去重键）。
    2. 把原文写入 ArchiveStore。
    3. 用 title+summary 的 embedding 向量 + 浓缩 payload 写入 Qdrant。
    """

    def __init__(
        self,
        qdrant_store: QdrantStore | None = None,
        archive_store: ArchiveStore | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        cfg = config or load_retrieval_config()
        self._cfg = cfg
        self._qdrant = qdrant_store or QdrantStore(config=cfg, qdrant_location=cfg.qdrant_path)
        self._archive = archive_store or ArchiveStore(config=cfg)

    def write(
        self,
        *,
        source: str,
        doc_type: str,
        url: str,
        anchor: str,
        title: str,
        summary: str,
        keywords: str,
        raw_text: str,
        raw_format: str = "text",
        lang: str = "unknown",
        version: str = "unknown",
        topic: str = "unknown",
    ) -> str:
        """Write one document unit to both layers.

        Returns the content_hash (dedup key).
        """
        content_hash = build_stable_hash(
            url=url,
            anchor=anchor,
            snippet=raw_text[:200],
            payload={"source": source, "anchor": anchor},
        )

        # ── deep layer ─────────────────────────────────────────────────────
        record = ArchiveRecord(
            id=str(uuid.uuid4()),
            source=source,
            doc_type=doc_type,
            url=url,
            anchor=anchor,
            title=title,
            summary=summary,
            keywords=keywords,
            raw_text=raw_text,
            raw_format=raw_format,
            lang=lang,
            version=version,
            topic=topic,
            content_hash=content_hash,
        )
        self._archive.upsert(record)

        # ── shallow index ──────────────────────────────────────────────────
        # The vector encodes the title + summary (not the full raw_text),
        # keeping the index lean and semantically focused.
        index_text = f"{title}\n{summary}"
        shallow_payload = {
            "source": source,
            "type": doc_type,
            "version": version,
            "lang": lang,
            "url": url,
            "anchor": anchor,
            "topic": topic,
            "title": title,
            "snippet": summary[:self._cfg.snippet_max_chars],
            "keywords": keywords,
            "hash": content_hash,
        }
        self._qdrant.upsert_chunks(
            [{"text": index_text, "payload": shallow_payload}]
        )
        return content_hash

    def write_batch(self, records: List[dict]) -> int:
        """Convenience wrapper — ``records`` is a list of kwargs dicts for ``write()``."""
        return sum(1 for r in records if self.write(**r) is not None)
