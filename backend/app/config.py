"""Centralized, ablation-tunable settings for the RAG pipeline.

Every value that Phases 5-6 of tasks.md sweep lives here, never hardcoded in
ingestion/retrieval/generation code (Design Property 5). Defaults below are
the actual ablation-suite winners measured in ../../EVALUATION_REPORT.md
(Stages 1-8) — see that file for the numbers behind each choice.
"""
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
CHATBOT_DIR = BACKEND_DIR.parent
SOURCE_DATA_DIR = CHATBOT_DIR.parent / "source_data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(CHATBOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Secrets ──
    openai_api_key: str = Field(default="")
    gemini_api_key: Optional[str] = Field(default=None)

    # ── Paths ──
    source_data_dir: Path = SOURCE_DATA_DIR
    vector_store_dir: Path = CHATBOT_DIR / "data" / "vector_store"

    # ── Ingestion (Stage 1) ── pypdf and pymupdf tied on R@3 (0.652); pymupdf wins the
    # production default for its markdown table structure, which matters for this
    # corpus's fee tables (see EVALUATION_REPORT.md Stage 1 synthesis).
    parser: Literal["pypdf", "pdfplumber", "pymupdf"] = "pymupdf"

    # ── Chunking (Stage 2) ── fixed/300/0 won R@3=0.739, the best of 19 configs swept.
    chunk_strategy: Literal["fixed", "recursive", "semantic"] = "fixed"
    chunk_size: int = 300
    chunk_overlap: int = 0

    # ── Embeddings (Stage 3) ── text-embedding-3-small (1536-dim) beat bge-small-en-v1.5
    # (384-dim) on every retrieval metric (R@3 0.609 vs 0.478).
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: Literal["openai", "sentence-transformers"] = "openai"

    # ── Vector DB (Stage 4) ── R@3 parity with FAISS (0.609 vs 0.609); Chroma wins on
    # native metadata filtering and built-in persistence.
    vector_db: Literal["chroma", "faiss"] = "chroma"

    # ── Retrieval mode / fusion (Stages 5-6) ── hybrid beat dense-only and sparse-only
    # (R@3 0.652 vs 0.609/0.609); weighted α=0.7 edged out RRF on MRR@10/NDCG@3 at equal R@3.
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    fusion_method: Literal["rrf", "weighted"] = "weighted"
    fusion_alpha: float = 0.7
    rrf_k: int = 60
    top_k: int = 20

    # ── Reranking (Stage 7) ── cross-encoder rerank improved R@3 by +0.087 and NDCG@3 by
    # +0.130 for +142ms/query — worth it for a support bot that isn't latency-critical.
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5

    # ── Guardrails ──
    min_rerank_score: float = -9.5

    # ── Generation (Stage 8) ── gpt-4o beat gpt-4o-mini on Faithfulness/Hallucination
    # (0.759/0.522 vs 0.844/0.652 — lower hallucination is better) at $0.002/query.
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0

    # ── Memory ──
    memory_max_turns: int = 6
    memory_token_budget: int = 2000


settings = Settings()

# Libraries that construct their own OpenAI/Gemini client (langchain_openai's
# ChatOpenAI, ragas, deepeval) read credentials from the process environment,
# not from this Settings object — pydantic-settings only loads .env into
# `settings`'s own fields. Mirror the keys into os.environ once here so every
# downstream client works without threading api_key through every call site.
if settings.openai_api_key:
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
if settings.gemini_api_key:
    os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
