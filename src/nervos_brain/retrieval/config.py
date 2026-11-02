from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nervos_brain.pathing import load_project_config, resolve_project_path

_config_cache: dict[str, Any] | None = None


def _load_yaml_config() -> dict[str, Any]:
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    _config_cache = load_project_config()
    return _config_cache


def get_retrieval_section(section: str) -> dict[str, Any]:
    """Return a shallow copy of a config section if it exists and is a mapping."""
    root = _load_yaml_config()
    data = root.get(section, {})
    return dict(data) if isinstance(data, dict) else {}


def has_retrieval_section(section: str) -> bool:
    """Whether config.yaml contains a mapping section with this name."""
    return bool(get_retrieval_section(section))


def _bool_cfg(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _float_cfg(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_retrieval_backend_sections(default_section: str = "retrieval") -> list[str]:
    """Return configured retrieval backend section names.

    ``config.yaml`` may define a top-level ``retrieval_backends`` list:

    retrieval_backends:
      - retrieval
      - retrieval_forum_talk

    If it is absent or invalid, the runtime falls back to the primary
    ``retrieval`` section for backward compatibility.
    """
    root = _load_yaml_config()
    raw = root.get("retrieval_backends")
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, (list, tuple)):
        candidates = [str(item) for item in raw]
    else:
        candidates = [default_section]

    sections: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        section = str(item).strip()
        if not section or section in seen:
            continue
        seen.add(section)
        sections.append(section)
    return sections or [default_section]


@dataclass(frozen=True)
class RetrievalConfig:
    """检索层统一配置。

    从 config.yaml[retrieval] 读取，缺省值保持向后兼容 M4 行为。
    """

    # Qdrant 浅层索引
    vector_size: int = 64
    collection_name: str = "nervos_docs"
    qdrant_path: str = "data/qdrant_local"
    qdrant_url: str = ""
    qdrant_api_key_env: str = "QDRANT_API_KEY"
    qdrant_timeout_s: float = 10.0
    qdrant_prefer_grpc: bool = False

    # 切片参数
    chunk_size: int = 600
    chunk_overlap: int = 120
    snippet_max_chars: int = 1200

    # SQLite 深层原件库
    archive_db: str = "data/archive.db"

    # Embedding
    # embedding_backend: local | api | hash | none。为空时按旧开关
    # use_real_embedding 推断（true → api，false → hash）。
    # hash 只给测试用：非测试环境下 hash/none 都会关闭向量路。
    use_real_embedding: bool = False
    embedding_backend: str = ""
    embedding_model: str = "openai/embedding-2"
    embedding_batch_size: int = 32
    embedding_normalize: bool = True
    embedding_query_prefix: str = ""
    embedding_doc_prefix: str = ""
    embedding_api_base: str = ""
    embedding_api_key_env: str = ""
    embedding_device: str = ""
    embedding_max_seq_length: int = 512
    embedding_max_retries: int = 3
    # 文档向量缓存（SQLite，键 = 编码文本哈希 + 模型）。空字符串 = 不缓存。
    embedding_cache_db: str = "data/embedding_cache/embeddings.db"
    # 显式允许 hash 向量参与检索（pytest 运行时自动允许）。
    embedding_allow_hash: bool = False
    enable_vector: bool = True

    # 多路检索参数
    top_k_per_path: int = 10
    final_top_k: int = 10
    rrf_k: int = 60
    fuzzy_threshold: float = 0.55
    enable_bm25: bool = True
    enable_fuzzy: bool = True
    enable_exact: bool = True


def load_retrieval_config(
    section: str = "retrieval",
    inherit_from: str | None = None,
) -> RetrievalConfig:
    """从 config.yaml 指定 section 构建 RetrievalConfig。

    Args:
        section:      读取的配置段名，默认 ``retrieval``。
        inherit_from: 可选的基础段名，若提供则先读取该段，再由 ``section`` 覆盖。
    """
    cfg: dict[str, Any] = {}

    if inherit_from:
        cfg.update(get_retrieval_section(inherit_from))

    cfg.update(get_retrieval_section(section))

    return RetrievalConfig(
        vector_size=int(cfg.get("vector_size", 64)),
        collection_name=cfg.get("collection_name", "nervos_docs"),
        qdrant_path=cfg.get("qdrant_path", "data/qdrant_local"),
        qdrant_url=str(cfg.get("qdrant_url", "") or "").strip(),
        qdrant_api_key_env=str(cfg.get("qdrant_api_key_env", "QDRANT_API_KEY") or "").strip(),
        qdrant_timeout_s=_float_cfg(cfg.get("qdrant_timeout_s", 10.0), 10.0),
        qdrant_prefer_grpc=_bool_cfg(cfg.get("qdrant_prefer_grpc", False), False),
        chunk_size=int(cfg.get("chunk_size", 600)),
        chunk_overlap=int(cfg.get("chunk_overlap", 120)),
        snippet_max_chars=int(cfg.get("snippet_max_chars", 1200)),
        archive_db=cfg.get("archive_db", "data/archive.db"),
        use_real_embedding=_bool_cfg(cfg.get("use_real_embedding", False), False),
        embedding_backend=str(cfg.get("embedding_backend", "") or "").strip().lower(),
        embedding_model=cfg.get("embedding_model", "openai/embedding-2"),
        embedding_batch_size=int(cfg.get("embedding_batch_size", 32)),
        embedding_normalize=_bool_cfg(cfg.get("embedding_normalize", True), True),
        embedding_query_prefix=str(cfg.get("embedding_query_prefix", "") or ""),
        embedding_doc_prefix=str(cfg.get("embedding_doc_prefix", "") or ""),
        embedding_api_base=str(cfg.get("embedding_api_base", "") or "").strip(),
        embedding_api_key_env=str(cfg.get("embedding_api_key_env", "") or "").strip(),
        embedding_device=str(cfg.get("embedding_device", "") or "").strip(),
        embedding_max_seq_length=int(cfg.get("embedding_max_seq_length", 512)),
        embedding_max_retries=int(cfg.get("embedding_max_retries", 3)),
        embedding_cache_db=str(
            cfg.get("embedding_cache_db", "data/embedding_cache/embeddings.db") or ""
        ).strip(),
        embedding_allow_hash=_bool_cfg(cfg.get("embedding_allow_hash", False), False),
        enable_vector=_bool_cfg(cfg.get("enable_vector", True), True),
        top_k_per_path=int(cfg.get("top_k_per_path", 10)),
        final_top_k=int(cfg.get("final_top_k", 10)),
        rrf_k=int(cfg.get("rrf_k", 60)),
        fuzzy_threshold=float(cfg.get("fuzzy_threshold", 0.55)),
        enable_bm25=_bool_cfg(cfg.get("enable_bm25", True), True),
        enable_fuzzy=_bool_cfg(cfg.get("enable_fuzzy", True), True),
        enable_exact=_bool_cfg(cfg.get("enable_exact", True), True),
    )


def load_retrieval_backend_configs(
    sections: list[str] | None = None,
    *,
    inherit_from: str = "retrieval",
) -> list[tuple[str, RetrievalConfig]]:
    """Load all configured retrieval backend configs.

    Non-primary sections inherit shared knobs from ``retrieval`` by default
    (embedding mode, vector size, ranking parameters) while overriding their
    own Qdrant/archive paths and collection names.
    """
    selected = sections or get_retrieval_backend_sections(default_section=inherit_from)
    configs: list[tuple[str, RetrievalConfig]] = []
    for section in selected:
        name = str(section).strip()
        if not name:
            continue
        base = None if name == inherit_from else inherit_from
        configs.append((name, load_retrieval_config(section=name, inherit_from=base)))
    if configs:
        return configs
    return [(inherit_from, load_retrieval_config(section=inherit_from))]
