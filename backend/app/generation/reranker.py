"""Cross-encoder reranking of hybrid retrieval results (Requirement 5).
Toggle via Settings.rerank_enabled — identity passthrough when off, so the
orchestrating chatbot code never has to branch on it (design.md)."""
from __future__ import annotations

from typing import Optional

from ..retrieval.hybrid import RetrievedChunk


class Reranker:
    def __init__(self, model_name: Optional[str] = None, enabled: Optional[bool] = None):
        from ..config import settings

        self.enabled = settings.rerank_enabled if enabled is None else enabled
        self.model_name = model_name or settings.rerank_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: Optional[int] = None) -> list[RetrievedChunk]:
        from ..config import settings

        top_n = top_n or settings.top_n
        if not candidates:
            return []
        if not self.enabled:
            return candidates[:top_n]

        model = self._get_model()
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_n]
        return [
            RetrievedChunk(text=c.text, source=c.source, page=c.page, score=float(s), rank=i + 1)
            for i, (c, s) in enumerate(ranked)
        ]
