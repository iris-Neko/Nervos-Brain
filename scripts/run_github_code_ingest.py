#!/usr/bin/env python3
"""Crawl GitHub source code into a dedicated retrieval DB.

Examples:
  python scripts/run_github_code_ingest.py --target https://github.com/nervosnetwork/fiber --max-files-per-repo 50
  python scripts/run_github_code_ingest.py --no-ingest --max-repos-per-owner 5
  python scripts/run_github_code_ingest.py --github-token "$GITHUB_TOKEN"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import requests

# Allow running from repo root without package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nervos_brain.ingestion import (
    GitHubCodeCrawler,
    IngestionPipeline,
    RawDocument,
    run_incremental_github_ingest,
)
from nervos_brain.retrieval import (
    ArchiveStore,
    MultiRetriever,
    QdrantStore,
    RetrievalConfig,
    get_retrieval_section,
    load_retrieval_config,
)
from nervos_brain.retrieval.dual_layer import DualLayerWriter

DEFAULT_TARGETS = [
    "https://github.com/nervosnetwork",
    "https://github.com/web5fans",
    "https://github.com/ckb-devrel",
    "https://github.com/RGBPlusPlus",
    "https://github.com/nervosnetwork/fiber",
    "https://github.com/nervosnetwork/tentacle",
    "https://github.com/appfi5",
]

DEFAULT_CORPUS = "github_code"
DEFAULT_STATE_FILE = "data/ingest_state/github_code_state.json"
DEFAULT_MANIFEST_OUT = "data/manifests/github_code_manifest.json"
DEFAULT_INCREMENTAL_JSONL_OUT = "data/tmp/github_code_delta.jsonl"

DEFAULT_CODE_ARCHIVE_DB = "data/github_code/archive.db"
DEFAULT_CODE_QDRANT_PATH = "data/qdrant_github_code"
DEFAULT_CODE_COLLECTION = "nervos_github_code"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("github-code-ingest")


def _build_writer(cfg: RetrievalConfig) -> DualLayerWriter:
    qdrant = QdrantStore(config=cfg, qdrant_location=cfg.qdrant_path)
    archive = ArchiveStore(db_path=cfg.archive_db, config=cfg)
    return DualLayerWriter(qdrant_store=qdrant, archive_store=archive, config=cfg)


def _build_code_config(args: argparse.Namespace) -> RetrievalConfig:
    base = load_retrieval_config()
    merged = {
        **base.__dict__,
        "archive_db": DEFAULT_CODE_ARCHIVE_DB,
        "qdrant_path": DEFAULT_CODE_QDRANT_PATH,
        "collection_name": DEFAULT_CODE_COLLECTION,
        **get_retrieval_section("retrieval_github_code"),
    }
    if args.archive:
        merged["archive_db"] = args.archive
    if args.qdrant:
        merged["qdrant_path"] = args.qdrant
    if args.collection:
        merged["collection_name"] = args.collection
    return RetrievalConfig(**merged)


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

    def __init__(self, base: GitHubCodeCrawler, jsonl_path: str) -> None:
        self._base = base
        self._path = Path(jsonl_path)

    def crawl(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            for doc in self._base.crawl():
                f.write(json.dumps(_doc_to_row(doc), ensure_ascii=False) + "\n")
                yield doc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl GitHub source code into a dedicated Qdrant + SQLite DB.",
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
        default="data/tmp/github_code_repos",
        help="Temporary workspace for git clones.",
    )
    parser.add_argument(
        "--jsonl-out",
        default="data/sources/github_code.jsonl",
        help="Path to export crawled code rows as JSONL.",
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
        default=250_000,
        help="Skip files larger than this many bytes.",
    )
    parser.add_argument(
        "--git-timeout",
        type=int,
        default=180,
        help="Timeout in seconds for each git command.",
    )
    parser.add_argument(
        "--git-retries",
        type=int,
        default=2,
        help="Retry count for retryable git network failures.",
    )
    parser.add_argument("--incremental", action="store_true", help="Only crawl repos whose default-branch commit changed.")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE, help="Local incremental state path.")
    parser.add_argument("--manifest-out", default=DEFAULT_MANIFEST_OUT, help="Public commit manifest output path.")
    parser.add_argument("--reset-state", action="store_true", help="Ignore existing incremental state.")
    parser.add_argument("--no-ingest", action="store_true", help="Only crawl + export JSONL; skip DB.")
    parser.add_argument("--dry-run", action="store_true", help="Parse only; skip DB writes.")
    parser.add_argument("--archive", default=None, help="Override archive db path.")
    parser.add_argument("--qdrant", default=None, help="Override qdrant path.")
    parser.add_argument("--collection", default=None, help="Override Qdrant collection name.")
    args = parser.parse_args()

    targets = args.target or DEFAULT_TARGETS
    if args.incremental and args.jsonl_out == "data/sources/github_code.jsonl":
        args.jsonl_out = DEFAULT_INCREMENTAL_JSONL_OUT

    crawler = GitHubCodeCrawler(
        targets=targets,
        clone_workspace=args.clone_workspace,
        github_token=args.github_token,
        request_delay=args.request_delay,
        include_forks=args.include_forks,
        include_archived=args.include_archived,
        max_repos_per_owner=args.max_repos_per_owner,
        max_files_per_repo=args.max_files_per_repo,
        max_file_bytes=args.max_file_bytes,
        git_timeout=args.git_timeout,
        git_retries=args.git_retries,
    )
    export_crawler = _JsonlExportCrawler(crawler, jsonl_path=args.jsonl_out)

    try:
        logger.info("GitHub code ingest targets=%s no_ingest=%s dry_run=%s", len(targets), args.no_ingest, args.dry_run)
        cfg = _build_code_config(args)

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
            print("Incremental code ingest finished:")
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
            print(f"  code_files={docs}")
            print(f"  jsonl={args.jsonl_out}")
            return

        writer = _build_writer(cfg)
        pipeline = IngestionPipeline(writer)
        stats = pipeline.run(export_crawler, dry_run=args.dry_run)

        print("Code ingest finished:")
        print(f"  targets={len(targets)}")
        print(f"  seen={stats.seen}")
        print(f"  written={stats.written}")
        print(f"  skipped={stats.skipped}")
        print(f"  failed={stats.failed}")
        print(f"  archive={cfg.archive_db}")
        print(f"  qdrant={cfg.qdrant_path}")
        print(f"  collection={cfg.collection_name}")
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
