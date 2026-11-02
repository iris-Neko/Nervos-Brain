import hashlib
import os
from typing import Dict, List, Optional
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import RetrievalConfig
from .embedding import deterministic_embedding, index_vectors, model_label
from nervos_brain.pathing import resolve_project_path

__all__ = ["QdrantStore", "deterministic_embedding"]


class QdrantStore:
    """Qdrant 最小存储封装。"""

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        qdrant_location: str = "data/qdrant_local",
    ) -> None:
        self.config = config or RetrievalConfig()
        self.mode = "server" if self.config.qdrant_url else "local"
        if self.config.qdrant_url:
            api_key = None
            if self.config.qdrant_api_key_env:
                api_key = os.getenv(self.config.qdrant_api_key_env) or None
            try:
                self.client = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=api_key,
                    timeout=self.config.qdrant_timeout_s,
                    prefer_grpc=self.config.qdrant_prefer_grpc,
                )
                self.client.get_collections()
            except Exception as exc:
                raise RuntimeError(
                    "Failed to connect to Qdrant server "
                    f"{self.config.qdrant_url!r}. Start it with "
                    "`docker compose -f docker-compose.qdrant.yml up -d` "
                    "or clear qdrant_url to use local path mode."
                ) from exc
        else:
            resolved_location = resolve_project_path(qdrant_location)
            resolved_location.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(resolved_location))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        names = {c.name for c in collections}
        if self.config.collection_name not in names:
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(size=self.config.vector_size, distance=Distance.COSINE),
            )

    def upsert_chunks(self, chunks: List[Dict[str, str]]) -> int:
        """批量写入切片。向量由配置的 embedding 后端批量编码。"""
        if not chunks:
            return 0
        texts = [item["text"] for item in chunks]
        vectors = index_vectors(texts, self.config)
        label = model_label(self.config)
        points: List[PointStruct] = []
        for item, vector in zip(chunks, vectors):
            text = item["text"]
            payload = dict(item["payload"])
            payload["snippet"] = text[: self.config.snippet_max_chars]
            payload["embedding_model"] = label
            points.append(
                PointStruct(
                    id=_stable_point_id(text, payload),
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.config.collection_name, points=points)
        return len(points)

    def vector_compatibility(self) -> tuple[bool, str]:
        """Check that the collection was built by the configured embedding.

        Compares the collection's vector size with ``vector_size`` and the
        ``embedding_model`` payload label of a sample point with the backend's
        label. Collections written before the label existed were built with
        hash vectors.
        """
        name = self.config.collection_name
        expected = model_label(self.config)
        rebuild_hint = (
            "rebuild it with `python scripts/migrate_qdrant_server_from_archive.py --recreate`"
        )
        try:
            info = self.client.get_collection(name)
        except Exception as exc:
            return False, f"cannot read Qdrant collection {name!r}: {exc}"
        vectors = info.config.params.vectors
        size = getattr(vectors, "size", None)
        if size is None:
            return False, f"collection {name!r} uses named vectors, which the vector path does not support"
        if int(size) != int(self.config.vector_size):
            return False, (
                f"collection {name!r} stores {size}-dim vectors but the embedding backend "
                f"({expected}) expects {self.config.vector_size}; {rebuild_hint}"
            )
        try:
            points, _ = self.client.scroll(
                collection_name=name, limit=1, with_payload=True, with_vectors=False
            )
        except Exception as exc:
            return False, f"cannot sample Qdrant collection {name!r}: {exc}"
        if points:
            stored = (points[0].payload or {}).get("embedding_model")
            if stored is None and not expected.startswith("hash-"):
                return False, (
                    f"collection {name!r} has no embedding_model metadata (built by the old "
                    f"hash writer) but the backend is {expected}; {rebuild_hint}"
                )
            if stored is not None and str(stored) != expected:
                return False, (
                    f"collection {name!r} was built with {stored} but the backend is "
                    f"{expected}; {rebuild_hint}"
                )
        return True, ""

    def delete_by_hashes(self, hashes: List[str]) -> int:
        """Delete Qdrant points whose stable IDs are derived from content hashes."""
        ids = [str(uuid5(NAMESPACE_URL, str(value))) for value in hashes if str(value).strip()]
        if not ids:
            return 0
        self.client.delete(collection_name=self.config.collection_name, points_selector=ids)
        return len(ids)


def _stable_point_id(text: str, payload: Dict[str, str]) -> str:
    stable_key = str(
        payload.get("hash")
        or payload.get("anchor")
        or hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    return str(uuid5(NAMESPACE_URL, stable_key))
