"""Chroma PersistentClient wrapper. Embedding model/provider are Settings-driven
(Requirement 3.2) and the collection is keyed by (chunk config, embedding model)
so different ablation runs never clobber each other's index (design.md)."""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from ..ingestion.chunking import Chunk

_ADD_BATCH_SIZE = 500


def _sanitize_metadata(metadata: dict) -> dict:
    safe = {}
    for k, v in metadata.items():
        if v is None:
            continue
        safe[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return safe


def collection_name_for(chunk_strategy: str, chunk_size: int, chunk_overlap: int,
                         embedding_provider: str, embedding_model: str) -> str:
    """One collection per (chunking config, embedding model) — Requirement 3.3 /
    design.md's ablation-isolation rule."""
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", embedding_model)
    name = f"wf_{chunk_strategy}_{chunk_size}_{chunk_overlap}_{embedding_provider}_{safe_model}"
    return name[:63].strip("_-") or "wf_default"


class VectorStore:
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        persist_dir: Optional[str | Path] = None,
        collection_name: Optional[str] = None,
    ):
        from ..config import settings

        self.embedding_model = embedding_model or settings.embedding_model
        self.embedding_provider = embedding_provider or settings.embedding_provider
        self.persist_dir = str(persist_dir or settings.vector_store_dir)
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self._embed_fn = self._build_embedding_function()
        self.collection_name = collection_name or collection_name_for(
            settings.chunk_strategy, settings.chunk_size, settings.chunk_overlap,
            self.embedding_provider, self.embedding_model,
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def _build_embedding_function(self):
        if self.embedding_provider == "openai":
            from ..config import settings

            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.openai_api_key, model_name=self.embedding_model,
            )
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)

    def build(self, chunks: list[Chunk], ids: Optional[list[str]] = None, force: bool = False) -> list[str]:
        """Embed and index `chunks`. Reuses the existing collection (Requirement 3.3)
        when it already holds the same number of records, unless `force=True`."""
        ids = ids or [str(uuid.uuid4()) for _ in chunks]
        existing = self.collection.count()
        if not force and existing == len(chunks) and existing > 0:
            return ids

        if existing > 0:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"},
            )

        documents = [c.text for c in chunks]
        metadatas = [_sanitize_metadata(c.metadata) for c in chunks]
        for i in range(0, len(chunks), _ADD_BATCH_SIZE):
            self.collection.add(
                ids=ids[i:i + _ADD_BATCH_SIZE],
                documents=documents[i:i + _ADD_BATCH_SIZE],
                metadatas=metadatas[i:i + _ADD_BATCH_SIZE],
            )
        return ids

    def query(self, query_text: str, k: int = 20) -> list[dict]:
        if self.collection.count() == 0:
            return []
        k = min(k, self.collection.count())
        result = self.collection.query(query_texts=[query_text], n_results=k)
        out = []
        for cid, text, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            out.append({"id": cid, "text": text, "metadata": meta, "score": 1.0 - dist})
        return out

    def count(self) -> int:
        return self.collection.count()
