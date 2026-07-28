"""RetrievalLayer package.

M4 minimal pipeline (unchanged):
  read → chunk → payload → anchor/hash → Qdrant write → filter search → Evidence

M4B dual-layer + multi-path extensions:
  DualLayerWriter   — sync write to Qdrant (shallow index) + SQLite (archive)
  ArchiveStore      — raw content store (deep layer)
  BM25Index         — in-memory keyword index built from the archive
  FuzzyIndex        — reusable RapidFuzz prefilter + legacy-score reranker
  rank_fusion       — Reciprocal Rank Fusion across all paths
  MultiRetriever    — orchestrates all 4 paths → fused Evidence
"""

# M4 core
from .anchor_hash import build_anchor, build_stable_hash
from .chunking import chunk_text
from .config import (
    RetrievalConfig,
    get_retrieval_section,
    get_retrieval_backend_sections,
    has_retrieval_section,
    load_retrieval_backend_configs,
    load_retrieval_config,
)
from .evidence_adapter import qdrant_result_to_evidence
from .ingest import build_ingest_records
from .payload_builder import build_payload
from .qdrant_writer import QdrantStore, deterministic_embedding
from .readers import read_text_file
from .search import search_with_filters

# M4B: dual-layer storage
from .embedding import get_embedding
from .dual_layer import ArchiveRecord, ArchiveStore, DualLayerWriter

# M4B: multi-path retrieval
from .bm25_index import BM25Index, BM25Result, tokenize
from .fuzzy_search import FuzzyIndex, FuzzyResult, fuzzy_search
from .rank_fusion import FusedResult, reciprocal_rank_fusion
from .multi_retriever import MultiRetriever
from .composite_retriever import (
    CompositeArchiveStore,
    CompositeRetriever,
    RetrievalBackend,
    build_configured_retriever,
)

__all__ = [
    # M4 core
    "RetrievalConfig",
    "load_retrieval_config",
    "load_retrieval_backend_configs",
    "get_retrieval_backend_sections",
    "get_retrieval_section",
    "has_retrieval_section",
    "read_text_file",
    "chunk_text",
    "build_payload",
    "build_anchor",
    "build_stable_hash",
    "build_ingest_records",
    "deterministic_embedding",
    "QdrantStore",
    "search_with_filters",
    "qdrant_result_to_evidence",
    # M4B: storage
    "get_embedding",
    "ArchiveRecord",
    "ArchiveStore",
    "DualLayerWriter",
    # M4B: search paths
    "BM25Index",
    "BM25Result",
    "tokenize",
    "FuzzyResult",
    "FuzzyIndex",
    "fuzzy_search",
    "FusedResult",
    "reciprocal_rank_fusion",
    "MultiRetriever",
    "RetrievalBackend",
    "CompositeArchiveStore",
    "CompositeRetriever",
    "build_configured_retriever",
]
