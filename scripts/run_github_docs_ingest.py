#!/usr/bin/env python3
"""Crawl GitHub docs and ingest them into the dual-layer retrieval DB.

Examples:
  python scripts/run_github_docs_ingest.py
  python scripts/run_github_docs_ingest.py --no-ingest --max-repos-per-owner 10
  python scripts/run_github_docs_ingest.py --github-token "$GITHUB_TOKEN"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

# Allow running from repo root without package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nervos_brain.ingestion import (
    GitHubDocsCrawler,
    IngestionPipeline,
    RawDocument,
    run_incremental_github_ingest,
)
from nervos_brain.retrieval import (
    ArchiveStore,
    MultiRetriever,
    QdrantStore,
    RetrievalConfig,
    load_retrieval_config,
)
from nervos_brain.retrieval.dual_layer import DualLayerWriter

DEFAULT_CORPUS = "github_docs"
DEFAULT_STATE_FILE = "data/ingest_state/github_docs_state.json"
DEFAULT_MANIFEST_OUT = "data/manifests/github_docs_manifest.json"
DEFAULT_INCREMENTAL_JSONL_OUT = "data/tmp/github_docs_delta.jsonl"

DEFAULT_TARGETS = [
    "https://github.com/nervosnetwork",
    "https://github.com/web5fans",
    "https://github.com/ckb-devrel",
    "https://github.com/RGBPlusPlus",
    "https://github.com/nervosnetwork/fiber",
    "https://github.com/nervosnetwork/tentacle",
    "https://github.com/appfi5",
]


def _build_writer(cfg: RetrievalConfig) -> DualLayerWriter:
    qdrant = QdrantStore(config=cfg, qdrant_location=cfg.qdrant_path)
    archive = ArchiveStore(db_path=cfg.archive_db, config=cfg)
    return DualLayerWriter(qdrant_store=qdrant, archive_store=archive, config=cfg)


def _doc_to_row(doc: RawDocument) -> dict:
    return {
        "id": doc.external_id,
        "title": doc.title,
        "raw_text": doc.raw_text,
        "url": doc.url,
        "anchor": doc.anchor,
        "doc_type": doc.doc_type,
        "summary": doc.summary,
        "keywords": doc.keywords,
        "raw_format": doc.raw_format,
        "lang": doc.lang,
        "version": doc.version,
        "topic": doc.topic,
        "metadata": doc.metadata,
    }


class _JsonlExportCrawler:
    """Wrapper crawler that tees all docs to JSONL while yielding them."""

    def __init__(self, base: GitHubDocsCrawler, jsonl_path: str) -> None:
        self._base = base
        self._path = Path(jsonl_path)

    def crawl(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            for doc in self._base.crawl():
                row = _doc_to_row(doc)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                yield doc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl GitHub docs and ingest into Qdrant + SQLite.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="GitHub owner/repo URL (repeatable). Defaults to Nervos-related sources.",
    )
    parser.add_argument(
        "--github-token",
        default=os.getenv("GITHUB_TOKEN"),
        help="GitHub token to avoid API rate limits. Defaults to env GITHUB_TOKEN.",
    )
    parser.add_argument(
        "--clone-workspace",
        default="data/tmp/github_repos",
        help="Temporary workspace for git clones.",
    )
    parser.add_argument(
        "--jsonl-out",
        default="data/sources/github_docs.jsonl",
        help="Path to export crawled docs as JSONL.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.2,
        help="Delay between GitHub API requests (seconds).",
    )
    parser.add_argument("--include-forks", action="store_true", help="Include fork repos.")
    parser.add_argument("--include-archived", action="store_true", help="Include archived repos.")
    parser.add_argument("--max-repos-per-owner", type=int, default=None, help="Optional owner repo cap.")
    parser.add_argument("--max-files-per-repo", type=int, default=None, help="Optional per-repo file cap.")
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=300_000,
        help="Skip files larger than this many bytes.",
    )
    parser.add_argument("--incremental", action="store_true", help="Only crawl repos whose default-branch commit changed.")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE, help="Local incremental state path.")
    parser.add_argument("--manifest-out", default=DEFAULT_MANIFEST_OUT, help="Public commit manifest output path.")
    parser.add_argument("--reset-state", action="store_true", help="Ignore existing incremental state.")
    parser.add_argument("--no-ingest", action="store_true", help="Only crawl + export JSONL; skip DB.")
    parser.add_argument("--dry-run", action="store_true", help="Parse only; skip DB writes.")
    parser.add_argument("--archive", default=None, help="Override archive db path.")
    parser.add_argument("--qdrant", default=None, help="Override qdrant path.")
    args = parser.parse_args()

    targets = args.target or DEFAULT_TARGETS
    if args.incremental and args.jsonl_out == "data/sources/github_docs.jsonl":
        args.jsonl_out = DEFAULT_INCREMENTAL_JSONL_OUT

    crawler = GitHubDocsCrawler(
        targets=targets,
        clone_workspace=args.clone_workspace,
        github_token=args.github_token,
        request_delay=args.request_delay,
        include_forks=args.include_forks,
        include_archived=args.include_archived,
        max_repos_per_owner=args.max_repos_per_owner,
        max_files_per_repo=args.max_files_per_repo,
        max_file_bytes=args.max_file_bytes,
    )
    export_crawler = _JsonlExportCrawler(crawler, jsonl_path=args.jsonl_out)

    try:
        cfg = load_retrieval_config()
        if args.archive:
            cfg = RetrievalConfig(**{**cfg.__dict__, "archive_db": args.archive})
        if args.qdrant:
            cfg = RetrievalConfig(**{**cfg.__dict__, "qdrant_path": args.qdrant})

        if args.incremental:
            dry_incremental = args.dry_run or args.no_ingest
            writer = None if dry_incremental else _build_writer(cfg)
            result = run_incremental_github_ingest(
                corpus=DEFAULT_CORPUS,
                targets=targets,
                crawler=crawler,
                writer=writer,
                cfg=cfg,
                state_file=args.state_file,
                manifest_out=args.manifest_out,
                jsonl_out=args.jsonl_out,
                row_factory=_doc_to_row,
                reset_state=args.reset_state,
                dry_run=dry_incremental,
            )
            print("Incremental ingest finished:")
            print(f"  repos={result.repos}")
            print(f"  changed_repos={result.changed_repos}")
            print(f"  skipped_repos={result.skipped_repos}")
            print(f"  seen={result.seen}")
            print(f"  written={result.written}")
            print(f"  cleaned={result.cleaned}")
            print(f"  failed_repos={result.failed_repos}")
            print(f"  state={args.state_file}")
            print(f"  manifest={args.manifest_out}")
            print(f"  jsonl={args.jsonl_out}")
            print(f"  bm25_index_size={result.bm25_index_size}")
            return

        if args.no_ingest:
            docs = 0
            for _ in export_crawler.crawl():
                docs += 1
            print("Crawl finished:")
            print(f"  targets={len(targets)}")
            print(f"  docs={docs}")
            print(f"  jsonl={args.jsonl_out}")
            return

        writer = _build_writer(cfg)
        pipeline = IngestionPipeline(writer)
        stats = pipeline.run(export_crawler, dry_run=args.dry_run)

        print("Ingest finished:")
        print(f"  targets={len(targets)}")
        print(f"  seen={stats.seen}")
        print(f"  written={stats.written}")
        print(f"  skipped={stats.skipped}")
        print(f"  failed={stats.failed}")
        print(f"  jsonl={args.jsonl_out}")
        if stats.errors:
            print("  errors:")
            for err in stats.errors[:10]:
                print(f"    - {err}")

        if not args.dry_run:
            retriever = MultiRetriever(
                qdrant_store=writer._qdrant,
                archive_store=writer._archive,
                config=cfg,
            )
            indexed = retriever.rebuild_bm25()
            print(f"  bm25_index_size={indexed}")
    except requests.RequestException as exc:
        print(f"Network error while crawling GitHub: {exc}")
        print("Please verify DNS/network connectivity and retry.")
        raise SystemExit(2)
    except RuntimeError as exc:
        print(f"Crawl failed: {exc}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
