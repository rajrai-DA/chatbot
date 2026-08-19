"""HybridIndex: dense (Chroma) + sparse (BM25) retrieval, fused via RRF or
weighted-alpha (Requirement 4.2), with retrieval mode selectable (Requirement 4.1/4.3)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from ..ingestion.chunking import Chunk
from .sparse import SparseIndex
from .vector_store import VectorStore


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int | str
    score: float
    rank: int


def _minmax_norm(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _rrf_fuse(result_lists: list[list[dict]], rrf_k: int, top_k: int) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            items.setdefault(cid, item)
    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [{**items[cid], "score": scores[cid]} for cid in ranked_ids]


def _weighted_fuse(dense_results: list[dict], sparse_results: list[dict], alpha: float, top_k: int) -> list[dict]:
    dense_norm = _minmax_norm({r["id"]: r["score"] for r in dense_results})
    sparse_norm = _minmax_norm({r["id"]: r["score"] for r in sparse_results})
    items = {r["id"]: r for r in dense_results}
    for r in sparse_results:
        items.setdefault(r["id"], r)
    all_ids = set(dense_norm) | set(sparse_norm)
    combined = {cid: alpha * dense_norm.get(cid, 0.0) + (1 - alpha) * sparse_norm.get(cid, 0.0) for cid in all_ids}
    ranked_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]
    return [{**items[cid], "score": combined[cid]} for cid in ranked_ids]


class HybridIndex:
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.vector_store = VectorStore(
            embedding_model=embedding_model, embedding_provider=embedding_provider,
            persist_dir=persist_dir, collection_name=collection_name,
        )
        self.sparse_index = SparseIndex()

    def build(self, chunks: list[Chunk], force: bool = False) -> int:
        ids = [str(uuid.uuid4()) for _ in chunks]
        ids = self.vector_store.build(chunks, ids=ids, force=force)
        self.sparse_index.build(chunks, ids=ids)
        return len(chunks)

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        mode: Optional[str] = None,
        fusion_method: Optional[str] = None,
        fusion_alpha: Optional[float] = None,
        rrf_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        from ..config import settings

        k = k or settings.top_k
        mode = mode or settings.retrieval_mode
        fusion_method = fusion_method or settings.fusion_method
        fusion_alpha = settings.fusion_alpha if fusion_alpha is None else fusion_alpha
        rrf_k = rrf_k or settings.rrf_k

        if mode == "dense":
            results = self.vector_store.query(query, k=k)
        elif mode == "sparse":
            results = self.sparse_index.search(query, k=k)
        elif mode == "hybrid":
            dense_results = self.vector_store.query(query, k=k)
            sparse_results = self.sparse_index.search(query, k=k)
            if fusion_method == "rrf":
                results = _rrf_fuse([dense_results, sparse_results], rrf_k=rrf_k, top_k=k)
            else:
                results = _weighted_fuse(dense_results, sparse_results, alpha=fusion_alpha, top_k=k)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode!r} — expected dense/sparse/hybrid")

        return [
            RetrievedChunk(
                text=r["text"], source=r["metadata"].get("source", ""), page=r["metadata"].get("page", ""),
                score=r["score"], rank=i + 1,
            )
            for i, r in enumerate(results)
        ]
