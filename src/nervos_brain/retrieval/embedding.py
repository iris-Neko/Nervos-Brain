"""Embedding backends for the retrieval layer.

Every vector that enters or queries Qdrant goes through this module, so the
ingest path, the archive → Qdrant rebuild script and the query path always use
the same model.

Backends (``RetrievalConfig.embedding_backend``):

local  → sentence-transformers model on this machine (e.g. ``BAAI/bge-m3``).
         Optional dependency: ``pip install -e ".[local-embedding]"``.
api    → OpenAI-compatible ``/embeddings`` endpoint via ``litellm.embedding``.
hash   → SHA-256 pseudo vectors. Test-only: outside pytest the vector path is
         switched off unless ``embedding_allow_hash`` is set explicitly.
none   → no semantic vectors. Qdrant still stores payloads (the exact path
         needs them) with hash placeholders, and the vector path is off.

An empty ``embedding_backend`` keeps the legacy switch working:
``use_real_embedding: true`` → api, otherwise hash.

All vectors are L2-normalised (``embedding_normalize``) and Qdrant collections
use cosine distance.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import sqlite3
import threading
import time
from array import array
from typing import Any, Dict, List, Sequence

from nervos_brain.pathing import resolve_project_path

from .config import RetrievalConfig, _load_yaml_config, load_retrieval_config

logger = logging.getLogger(__name__)

BACKENDS = ("local", "api", "hash", "none")
REAL_BACKENDS = ("local", "api")


class EmbeddingError(RuntimeError):
    """The configured embedding backend cannot produce usable vectors."""


# ── hash vectors (tests / placeholders) ────────────────────────────────────

def deterministic_embedding(text: str, dim: int = 64) -> List[float]:
    """确定性伪向量（SHA-256 → [-1, 1]）。

    没有任何语义：只给测试和 Qdrant 占位用，不能进入线上检索。
    """
    vec: List[float] = []
    for i in range(dim):
        h = hashlib.sha256(f"{i}:{text}".encode("utf-8")).hexdigest()
        # 映射到 [-1, 1]
        vec.append(((int(h[:8], 16) / 0xFFFFFFFF) * 2.0) - 1.0)
    return vec


# ── configuration helpers ──────────────────────────────────────────────────

def resolve_backend(cfg: RetrievalConfig) -> str:
    """Return the effective backend name, honouring the legacy switch."""
    name = (cfg.embedding_backend or "").strip().lower()
    if not name:
        return "api" if cfg.use_real_embedding else "hash"
    if name not in BACKENDS:
        raise EmbeddingError(
            f"unknown embedding_backend {name!r}; expected one of {', '.join(BACKENDS)}"
        )
    return name


def hash_allowed(cfg: RetrievalConfig) -> bool:
    """Hash vectors may be searched only in tests or when explicitly allowed."""
    if cfg.embedding_allow_hash:
        return True
    flag = os.environ.get("NERVOS_ALLOW_HASH_EMBEDDING", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    return "PYTEST_CURRENT_TEST" in os.environ


def model_label(cfg: RetrievalConfig) -> str:
    """Identifier stored in each Qdrant payload as ``embedding_model``."""
    if resolve_backend(cfg) in REAL_BACKENDS:
        return str(cfg.embedding_model)
    return f"hash-{cfg.vector_size}"


def vector_path_status(cfg: RetrievalConfig) -> tuple[bool, str]:
    """Whether the vector path may run with this config, and why not."""
    if not cfg.enable_vector:
        return False, "enable_vector is false"
    try:
        backend = resolve_backend(cfg)
    except EmbeddingError as exc:
        return False, str(exc)
    if backend == "none":
        return False, "embedding_backend is none"
    if backend == "hash" and not hash_allowed(cfg):
        return False, (
            "embedding_backend resolves to hash, which is test-only "
            "(set embedding_backend to local or api and rebuild the collections)"
        )
    return True, ""


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 0.0:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]


# ── backends ───────────────────────────────────────────────────────────────

class _HashEmbedder:
    def __init__(self, cfg: RetrievalConfig) -> None:
        self.dim = int(cfg.vector_size)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        return [deterministic_embedding(text, self.dim) for text in texts]


class _LocalEmbedder:
    def __init__(self, cfg: RetrievalConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise EmbeddingError(
                "embedding_backend=local needs sentence-transformers. Install it with "
                '`pip install -e ".[local-embedding]"` (and a torch build for your GPU), '
                "or switch embedding_backend to api."
            ) from exc

        started = time.perf_counter()
        self._model = SentenceTransformer(cfg.embedding_model, device=cfg.embedding_device or None)
        if cfg.embedding_max_seq_length > 0:
            self._model.max_seq_length = int(cfg.embedding_max_seq_length)
        if str(self._model.device).startswith("cuda"):
            self._model.half()
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        dim = int(get_dim() or 0)
        if dim != int(cfg.vector_size):
            raise EmbeddingError(
                f"embedding model {cfg.embedding_model!r} outputs {dim}-dim vectors "
                f"but vector_size is {cfg.vector_size}; set vector_size: {dim}"
            )
        self.dim = dim
        self._batch_size = max(1, int(cfg.embedding_batch_size))
        logger.info(
            "loaded local embedding model %s (dim=%s, device=%s) in %.1fs",
            cfg.embedding_model,
            dim,
            self._model.device,
            time.perf_counter() - started,
        )

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        matrix = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in matrix]


class _ApiEmbedder:
    def __init__(self, cfg: RetrievalConfig) -> None:
        self._model = str(cfg.embedding_model)
        self.dim = int(cfg.vector_size)
        self._batch_size = max(1, int(cfg.embedding_batch_size))
        self._max_retries = max(0, int(cfg.embedding_max_retries))
        key_env = cfg.embedding_api_key_env
        self._api_key = (
            (os.environ.get(key_env, "") if key_env else "")
            or os.environ.get("LLM_API_KEY", "")
            or _read_llm_setting("api_key")
        )
        self._api_base = cfg.embedding_api_base or _read_llm_setting("api_base")

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        rows = list(texts)
        for start in range(0, len(rows), self._batch_size):
            vectors.extend(self._encode_batch(rows[start : start + self._batch_size]))
        return vectors

    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        import litellm  # deferred to avoid hard import at module load

        kwargs: Dict[str, Any] = {"model": self._model, "input": texts}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = litellm.embedding(**kwargs)
                break
            except Exception as exc:  # network / rate limit / gateway errors
                last_exc = exc
                if attempt >= self._max_retries:
                    raise EmbeddingError(
                        f"embedding API call failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                time.sleep(min(2.0**attempt, 10.0))
        else:  # pragma: no cover - loop always breaks or raises
            raise EmbeddingError(f"embedding API call failed: {last_exc}")

        items = list(resp.data)
        items.sort(key=lambda item: int(_item_field(item, "index", 0) or 0))
        vectors = [[float(x) for x in _item_field(item, "embedding", [])] for item in items]
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"embedding API returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        for vec in vectors:
            if len(vec) != self.dim:
                raise EmbeddingError(
                    f"embedding model {self._model!r} returned {len(vec)}-dim vectors "
                    f"but vector_size is {self.dim}; set vector_size: {len(vec)}"
                )
        return vectors


def _item_field(item: Any, name: str, default: Any) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _read_llm_setting(name: str) -> str:
    try:
        raw = _load_yaml_config()
        llm = raw.get("llm", {}) if isinstance(raw, dict) else {}
        return str(llm.get(name, "") if isinstance(llm, dict) else "")
    except Exception:
        return ""


_EMBEDDERS: Dict[tuple, Any] = {}
_EMBEDDERS_LOCK = threading.Lock()


def _embedder_key(cfg: RetrievalConfig, backend: str) -> tuple:
    if backend == "hash" or backend == "none":
        return ("hash", int(cfg.vector_size))
    return (
        backend,
        str(cfg.embedding_model),
        int(cfg.vector_size),
        cfg.embedding_device,
        int(cfg.embedding_max_seq_length),
        int(cfg.embedding_batch_size),
        cfg.embedding_api_base,
        cfg.embedding_api_key_env,
    )


def get_embedder(cfg: RetrievalConfig) -> Any:
    """Return a process-wide embedder for this config (models load once)."""
    backend = resolve_backend(cfg)
    key = _embedder_key(cfg, backend)
    with _EMBEDDERS_LOCK:
        embedder = _EMBEDDERS.get(key)
        if embedder is None:
            if backend == "local":
                embedder = _LocalEmbedder(cfg)
            elif backend == "api":
                embedder = _ApiEmbedder(cfg)
            else:
                embedder = _HashEmbedder(cfg)
            _EMBEDDERS[key] = embedder
        return embedder


# ── document vector cache ──────────────────────────────────────────────────

class EmbeddingCache:
    """SQLite cache of raw document vectors keyed by (text hash, model key)."""

    def __init__(self, path: str) -> None:
        resolved = resolve_project_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(resolved)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            " text_hash TEXT NOT NULL,"
            " model TEXT NOT NULL,"
            " dim INTEGER NOT NULL,"
            " vector BLOB NOT NULL,"
            " PRIMARY KEY (text_hash, model))"
        )
        self._conn.commit()

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, hashes: Sequence[str], model: str) -> Dict[str, List[float]]:
        found: Dict[str, List[float]] = {}
        unique = list(dict.fromkeys(hashes))
        with self._lock:
            for start in range(0, len(unique), 500):
                chunk = unique[start : start + 500]
                marks = ",".join("?" for _ in chunk)
                rows = self._conn.execute(
                    f"SELECT text_hash, vector FROM embeddings WHERE model = ? AND text_hash IN ({marks})",
                    [model, *chunk],
                ).fetchall()
                for text_hash, blob in rows:
                    vec = array("f")
                    vec.frombytes(blob)
                    found[text_hash] = vec.tolist()
        return found

    def put_many(self, items: Sequence[tuple[str, Sequence[float]]], model: str) -> None:
        rows = [
            (text_hash, model, len(vec), array("f", vec).tobytes())
            for text_hash, vec in items
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (text_hash, model, dim, vector) VALUES (?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def count(self, model: str | None = None) -> int:
        with self._lock:
            if model is None:
                return int(self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)
                ).fetchone()[0]
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_CACHES: Dict[str, EmbeddingCache] = {}
_CACHES_LOCK = threading.Lock()


def _get_cache(cfg: RetrievalConfig) -> EmbeddingCache | None:
    if not cfg.embedding_cache_db:
        return None
    path = str(resolve_project_path(cfg.embedding_cache_db))
    with _CACHES_LOCK:
        cache = _CACHES.get(path)
        if cache is None:
            cache = EmbeddingCache(path)
            _CACHES[path] = cache
        return cache


def _cache_model_key(cfg: RetrievalConfig, backend: str) -> str:
    key = f"{backend}:{cfg.embedding_model}:{cfg.vector_size}"
    if backend == "local":
        key += f":msl{cfg.embedding_max_seq_length}"
    return key


# ── public API ─────────────────────────────────────────────────────────────

def embed_documents(
    texts: Sequence[str],
    config: RetrievalConfig | None = None,
    *,
    use_cache: bool = True,
) -> List[List[float]]:
    """Encode document texts in batches (with ``embedding_doc_prefix``)."""
    cfg = config or load_retrieval_config()
    backend = resolve_backend(cfg)
    inputs = [f"{cfg.embedding_doc_prefix}{text}" for text in texts]
    if not inputs:
        return []

    embedder = get_embedder(cfg)
    cache = _get_cache(cfg) if use_cache and backend in REAL_BACKENDS else None
    if cache is None:
        raw = embedder.encode(inputs)
    else:
        model_key = _cache_model_key(cfg, backend)
        hashes = [EmbeddingCache.text_hash(text) for text in inputs]
        cached = cache.get_many(hashes, model_key)
        missing: Dict[str, str] = {}
        for text_hash, text in zip(hashes, inputs):
            if text_hash not in cached and text_hash not in missing:
                missing[text_hash] = text
        if missing:
            fresh = embedder.encode(list(missing.values()))
            new_items = list(zip(missing.keys(), fresh))
            cache.put_many(new_items, model_key)
            cached.update(new_items)
        raw = [cached[text_hash] for text_hash in hashes]

    if cfg.embedding_normalize:
        return [_l2_normalize(vec) for vec in raw]
    return [list(vec) for vec in raw]


def embed_query(text: str, config: RetrievalConfig | None = None) -> List[float]:
    """Encode one query (with ``embedding_query_prefix``)."""
    cfg = config or load_retrieval_config()
    raw = get_embedder(cfg).encode([f"{cfg.embedding_query_prefix}{text}"])[0]
    return _l2_normalize(raw) if cfg.embedding_normalize else list(raw)


def index_vectors(
    texts: Sequence[str],
    config: RetrievalConfig | None = None,
) -> List[List[float]]:
    """Vectors to store in Qdrant for these texts.

    Real backends encode the text. ``hash`` and ``none`` write hash
    placeholders so payload-only paths (exact match) keep working; the vector
    path refuses to search them outside tests (see ``vector_path_status``).
    """
    cfg = config or load_retrieval_config()
    if resolve_backend(cfg) in REAL_BACKENDS:
        return embed_documents(texts, cfg)
    vectors = _HashEmbedder(cfg).encode([f"{cfg.embedding_doc_prefix}{t}" for t in texts])
    return [_l2_normalize(vec) for vec in vectors] if cfg.embedding_normalize else vectors


def get_embedding(
    text: str,
    config: RetrievalConfig | None = None,
) -> List[float]:
    """Backward-compatible alias: return the query vector for *text*."""
    return embed_query(text, config)
