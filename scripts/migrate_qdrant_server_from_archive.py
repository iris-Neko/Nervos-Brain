#!/usr/bin/env python3
"""Rebuild Qdrant server collections from SQLite archive stores.

This is the migration path from embedded Qdrant local directories to a shared
Qdrant server. The SQLite archive remains the source of truth.

Vectors come from the configured embedding backend (``embedding_backend`` in
config.yaml, or the ``--embedding-*`` flags). Document vectors are cached in
``embedding_cache_db``, so a second rebuild does not re-encode unchanged rows.
``--local`` rebuilds embedded (path-mode) Qdrant directories instead of a
server, e.g. for offline evaluation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nervos_brain.retrieval import ArchiveStore, RetrievalConfig
from nervos_brain.retrieval.config import (
    has_retrieval_section,
    load_retrieval_backend_configs,
    load_retrieval_config,
)
from nervos_brain.retrieval.embedding import index_vectors, model_label, resolve_backend
from nervos_brain.pathing import resolve_project_path


PUBLIC_BACKEND_OVERRIDES = {
    "retrieval": {
        "qdrant_path": "data/qdrant_local",
        "collection_name": "nervos_docs",
        "archive_db": "data/archive.db",
    },
    "retrieval_forum_talk": {
        "qdrant_path": "data/qdrant_talk_forum",
        "collection_name": "nervos_talk_user_discussions",
        "archive_db": "data/forum_talk/archive.db",
    },
    "retrieval_github_code": {
        "qdrant_path": "data/qdrant_github_code",
        "collection_name": "nervos_github_code",
        "archive_db": "data/github_code/archive.db",
    },
}

PUBLIC_BACKENDS = tuple(PUBLIC_BACKEND_OVERRIDES)


def _point_id(record) -> str:
    return str(uuid5(NAMESPACE_URL, str(record.content_hash or record.anchor)))


def _payload(record, cfg: RetrievalConfig) -> dict[str, str]:
    return {
        "source": record.source,
        "type": record.doc_type,
        "version": record.version,
        "lang": record.lang,
        "url": record.url,
        "anchor": record.anchor,
        "topic": record.topic,
        "title": record.title,
        "snippet": record.summary[: cfg.snippet_max_chars],
        "keywords": record.keywords,
        "hash": record.content_hash,
        "embedding_model": model_label(cfg),
    }


def _index_text(record) -> str:
    return f"{record.title}\n{record.summary}"


def _batched(rows: list, size: int):
    for idx in range(0, len(rows), size):
        yield rows[idx : idx + size]


def _assert_archive_ready(path: str) -> Path:
    archive_path = resolve_project_path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"archive DB not found: {archive_path}. "
            "Run `git lfs pull` if this file is LFS-backed, or ingest data first."
        )
    with archive_path.open("rb") as f:
        prefix = f.read(128)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
        raise RuntimeError(
            f"archive DB is still a Git LFS pointer, not real SQLite data: {archive_path}. "
            "Run `git lfs pull` and retry."
        )
    return archive_path


def load_public_backend_configs() -> list[tuple[str, RetrievalConfig]]:
    """Return the publishable three-corpus layout without requiring config.yaml.

    Embedding and ranking settings come from config.yaml's ``retrieval``
    section when it exists; paths and collection names are always the
    publishable defaults.
    """
    base = load_retrieval_config("retrieval") if has_retrieval_section("retrieval") else RetrievalConfig()
    configs: list[tuple[str, RetrievalConfig]] = []
    for name in PUBLIC_BACKENDS:
        merged = {**base.__dict__, **PUBLIC_BACKEND_OVERRIDES[name]}
        configs.append((name, RetrievalConfig(**merged)))
    return configs


def migrate_backend(
    *,
    client: QdrantClient,
    name: str,
    cfg: RetrievalConfig,
    recreate: bool,
    batch_size: int,
) -> int:
    _assert_archive_ready(cfg.archive_db)
    archive = ArchiveStore(db_path=cfg.archive_db, config=cfg)
    records = archive.list_all()
    existing = {c.name for c in client.get_collections().collections}

    if recreate and cfg.collection_name in existing:
        client.delete_collection(cfg.collection_name)
        existing.remove(cfg.collection_name)

    if cfg.collection_name not in existing:
        if not recreate:
            print(
                f"[check] {name}: collection={cfg.collection_name} missing; "
                f"archive_rows={len(records)}"
            )
            return 0
        client.create_collection(
            collection_name=cfg.collection_name,
            vectors_config=VectorParams(size=cfg.vector_size, distance=Distance.COSINE),
        )

    if not recreate:
        info = client.get_collection(cfg.collection_name)
        print(
            f"[check] {name}: collection={cfg.collection_name} "
            f"points={info.points_count} archive_rows={len(records)}"
        )
        return int(info.points_count or 0)

    written = 0
    encode_s = 0.0
    started = time.perf_counter()
    for batch in _batched(records, batch_size):
        encode_started = time.perf_counter()
        vectors = index_vectors([_index_text(record) for record in batch], cfg)
        encode_s += time.perf_counter() - encode_started
        points = [
            PointStruct(id=_point_id(record), vector=vector, payload=_payload(record, cfg))
            for record, vector in zip(batch, vectors)
        ]
        if points:
            client.upsert(collection_name=cfg.collection_name, points=points)
            written += len(points)
            print(f"  {name}: {written}/{len(records)} encode={encode_s:.1f}s", flush=True)

    info = client.get_collection(cfg.collection_name)
    print(
        f"[migrated] {name}: collection={cfg.collection_name} "
        f"written={written} points={info.points_count} archive_rows={len(records)} "
        f"embedding={model_label(cfg)} encode_s={encode_s:.1f} "
        f"total_s={time.perf_counter() - started:.1f}"
    )
    return written


def _apply_overrides(cfg: RetrievalConfig, args: argparse.Namespace) -> RetrievalConfig:
    data = dict(cfg.__dict__)
    data["qdrant_url"] = "" if args.local else args.url
    if args.local and args.qdrant_root:
        data["qdrant_path"] = str(Path(args.qdrant_root) / Path(cfg.qdrant_path).name)
    for field in ("embedding_backend", "embedding_model", "embedding_device", "embedding_cache_db"):
        value = getattr(args, field)
        if value is not None:
            data[field] = value
    if args.vector_size is not None:
        data["vector_size"] = int(args.vector_size)
    if args.embedding_batch_size is not None:
        data["embedding_batch_size"] = int(args.embedding_batch_size)
    return RetrievalConfig(**data)


def _local_client(cfg: RetrievalConfig) -> QdrantClient:
    path = resolve_project_path(cfg.qdrant_path)
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild Qdrant server collections from SQLite archive stores.",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:6333",
        help="Qdrant server URL.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional Qdrant API key. Empty for localhost/no-auth deployments.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete/recreate target collections and upsert all archive rows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Qdrant upsert batch size.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        default=[],
        help=(
            "Config section to migrate. Repeatable. Defaults to config retrieval_backends. "
            "Ignored when --public-default-backends is set."
        ),
    )
    parser.add_argument(
        "--public-default-backends",
        action="store_true",
        help=(
            "Migrate the publishable docs/forum/github_code layout without requiring "
            "a local config.yaml. Intended for fresh clone deployment bootstrap."
        ),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Rebuild embedded Qdrant path-mode directories (each backend's qdrant_path) "
            "instead of a server."
        ),
    )
    parser.add_argument(
        "--qdrant-root",
        default="",
        help="With --local: write each backend to <root>/<basename of its qdrant_path> instead.",
    )
    parser.add_argument("--embedding-backend", choices=["local", "api", "hash", "none"], default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--vector-size", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-cache-db", default=None)
    args = parser.parse_args()

    client = None
    if not args.local:
        client = QdrantClient(url=args.url, api_key=args.api_key or None, timeout=30.0)
        try:
            client.get_collections()
        except Exception as exc:
            print(
                f"Failed to connect to Qdrant server at {args.url}. "
                "Start it with: docker compose -f docker-compose.qdrant.yml up -d",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc

    if args.public_default_backends:
        backend_configs = load_public_backend_configs()
    else:
        backend_configs = load_retrieval_backend_configs(sections=args.backend or None)

    total = 0
    for name, cfg in backend_configs:
        cfg = _apply_overrides(cfg, args)
        if args.recreate and resolve_backend(cfg) not in ("local", "api"):
            print(
                f"[warn] {name}: embedding backend {resolve_backend(cfg)!r} writes hash "
                "placeholders; the vector path stays off outside tests.",
                file=sys.stderr,
            )
        backend_client = _local_client(cfg) if args.local else client
        total += migrate_backend(
            client=backend_client,
            name=name,
            cfg=cfg,
            recreate=bool(args.recreate),
            batch_size=max(1, int(args.batch_size)),
        )
        if args.local:
            backend_client.close()

    mode = "migrated" if args.recreate else "checked"
    print(f"Qdrant server archive migration {mode}: total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
