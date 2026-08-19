"""ProductionRAGChatbot — the single entry point the FastAPI layer calls.
Orchestrates: guardrails -> memory-aware query rewrite -> hybrid retrieve ->
rerank -> prompt assembly -> LLM call -> citation mapping -> memory update
(design.md Components: Generation)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..memory.session_store import SessionStore, condense_question
from ..retrieval.hybrid import HybridIndex, RetrievedChunk
from .guardrails import (
    ACCOUNT_DATA_MESSAGE,
    NO_INFO_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    check_account_data_request,
    check_conversational,
    check_groundedness,
    check_out_of_scope,
)
from .prompt import build_prompt, build_smalltalk_prompt
from .reranker import Reranker


@dataclass
class Citation:
    document: str
    page: object


@dataclass
class ChatResult:
    answer: str
    citations: list[Citation]
    refused: bool
    standalone_question: str
    chunks: list[RetrievedChunk] = field(default_factory=list)  # reranked chunks behind `citations`, incl. text


class ProductionRAGChatbot:
    """Property 1 (design.md): never calls the LLM with zero retrieved
    chunks, and never emits an answer whose claims aren't traceable to a
    retrieved chunk — enforced by guardrails running strictly before
    generation. Property 4: scope refusals never depend on the LLM."""

    def __init__(self, index: Optional[HybridIndex] = None, reranker: Optional[Reranker] = None,
                 session_store: Optional[SessionStore] = None, llm=None):
        from ..config import settings

        self.settings = settings
        self.index = index or HybridIndex()
        self.reranker = reranker or Reranker()
        self.sessions = session_store or SessionStore()
        if llm is None:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model=settings.llm_model, temperature=settings.llm_temperature)
        self.llm = llm

    def answer(self, session_id: str, message: str) -> ChatResult:
        if check_out_of_scope(message):
            return ChatResult(answer=OUT_OF_SCOPE_MESSAGE, citations=[], refused=True, standalone_question=message)
        if check_account_data_request(message):
            return ChatResult(answer=ACCOUNT_DATA_MESSAGE, citations=[], refused=True, standalone_question=message)

        if check_conversational(message):
            history_text = self.sessions.history_text(session_id)
            answer_text = self.llm.invoke(build_smalltalk_prompt(message, history_text=history_text)).content.strip()
            self.sessions.append(session_id, "user", message)
            self.sessions.append(session_id, "assistant", answer_text)
            return ChatResult(answer=answer_text, citations=[], refused=False, standalone_question=message)

        history = self.sessions.get_history(session_id)
        standalone = condense_question(message, history, llm=self.llm)

        candidates = self.index.search(standalone, k=self.settings.top_k, mode=self.settings.retrieval_mode)
        reranked: list[RetrievedChunk] = self.reranker.rerank(standalone, candidates, top_n=self.settings.top_n)

        if not check_groundedness(reranked, self.settings.min_rerank_score):
            self.sessions.append(session_id, "user", message)
            self.sessions.append(session_id, "assistant", NO_INFO_MESSAGE)
            return ChatResult(
                answer=NO_INFO_MESSAGE, citations=[], refused=True, standalone_question=standalone, chunks=reranked,
            )

        history_text = self.sessions.history_text(session_id)
        prompt = build_prompt(standalone, reranked, history_text=history_text)
        answer_text = self.llm.invoke(prompt).content.strip()

        citations = [Citation(document=c.source, page=c.page) for c in reranked]

        self.sessions.append(session_id, "user", message)
        self.sessions.append(session_id, "assistant", answer_text)

        return ChatResult(
            answer=answer_text, citations=citations, refused=False, standalone_question=standalone, chunks=reranked,
        )

    def ingest(self, force: bool = False) -> dict:
        """Build the hybrid index from the source corpus using the finalized Settings."""
        from ..ingestion.chunking import chunk_documents
        from ..ingestion.loaders import load_all_source_documents

        docs = load_all_source_documents(parser=self.settings.parser)
        chunks = chunk_documents(
            docs, strategy=self.settings.chunk_strategy, size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        n = self.index.build(chunks, force=force)
        return {"documents": len(docs), "chunks": n}
