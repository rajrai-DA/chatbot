"""bm25s sparse index over the same chunk text as the dense store (Requirement 4.1)."""
from __future__ import annotations

import uuid
from typing import Optional

import bm25s
from Stemmer import Stemmer

from ..ingestion.chunking import Chunk


class SparseIndex:
    def __init__(self):
        self.stemmer = Stemmer("english")
        self.bm25: Optional[bm25s.BM25] = None
        self.chunks: list[Chunk] = []
        self.ids: list[str] = []

    def _tokenize(self, texts: list[str]):
        return bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer.stemWords, show_progress=False)

    def build(self, chunks: list[Chunk], ids: Optional[list[str]] = None) -> list[str]:
        self.chunks = chunks
        self.ids = ids or [str(uuid.uuid4()) for _ in chunks]
        tokens = self._tokenize([c.text for c in chunks])
        self.bm25 = bm25s.BM25(method="lucene")
        self.bm25.index(tokens)
        return self.ids

    def search(self, query: str, k: int = 20) -> list[dict]:
        if self.bm25 is None or not self.chunks:
            return []
        k = min(k, len(self.chunks))
        query_tokens = self._tokenize([query])
        idxs, scores = self.bm25.retrieve(query_tokens, k=k, show_progress=False)
        results = []
        for idx, score in zip(idxs[0], scores[0]):
            idx = int(idx)
            chunk = self.chunks[idx]
            results.append({
                "id": self.ids[idx], "text": chunk.text, "metadata": dict(chunk.metadata), "score": float(score),
            })
        return results
