# Requirements Document

## Introduction

This spec covers the "Wells Fargo Customer Support Assistant" capstone: a RAG chatbot that answers a Wells Fargo customer's questions about their deposit account terms, applicable fees, and credit card agreement, grounded only in the documents provided in `../source_data/`, with mandatory source citation and refusal when the answer isn't present.

The project has two deliverables of near-equal engineering weight, and the grading rubric (`../../REQUIREMENT.md`, `../../EVALUATION_METHODOLOGY.md`) makes clear that evidence outweighs the app itself:

1. **A production chatbot app** — React frontend + Python (FastAPI) backend, Chroma vector store, OpenAI for embeddings/generation.
2. **An evaluation harness and report** — ablation experiments across 8 pipeline stages (parsing, chunking, embedding model, vector DB, retrieval mode, hybrid merge weighting, reranking, LLM choice), each backed by real measured numbers on a 20+ question labeled evaluation set, plus end-to-end RAGAS + DeepEval scores on the final chosen pipeline.

Grading weights: evaluation rigor 40%, acceptance-test correctness 30%, end-to-end RAGAS/DeepEval scores 20%, code quality/usability 10%. A working chatbot with no evaluation evidence caps at 30%. The evaluation harness is therefore not an optional add-on — it is the primary deliverable, and the production app's configuration (chunk size, embedding model, hybrid weighting, reranker on/off, LLM choice) must be *derived from* the ablation results, not chosen up front.

Reusable reference material lives in `../../notebooks/`: notebooks 02–10 teach each individual pipeline stage and its evaluation metrics; notebook 13 (`13_capstone_production_rag_chatbot_STUDENT.ipynb`) is a fill-in-the-blank scaffold whose classes (`HybridIndex`, `Reranker`, `ProductionRAGChatbot`, `load_document`, `chunk_documents`, `format_sources`) are the direct template for this app's backend; `production_rag_chatbot_memory/` (`app.py`, `rag_agent_pipeline.py`, `eval_pipeline.py`) is a worked reference implementation of a memory-enabled agent and evaluation harness built on top of that same scaffold.

**Reconciling the tutor's follow-up email with `EVALUATION_METHODOLOGY.md`:** the written methodology defines R@1, R@3, MRR@10, NDCG@3 as the confirmed retrieval metric set. The tutor's later email additionally names Recall@10 explicitly, and calls out **Scalability** alongside Latency and Cost as a required factor, plus explicitly invites dataset-specific custom metrics, and states the final output must include a **presentation** of findings, not just documentation. This spec treats the written methodology as the base and layers the email's additions on top (Requirements 10.2, 10.5, and 16 below) rather than treating them as conflicting — nothing in the email contradicts the methodology, it sharpens it.

## Requirements

### Requirement 1: Multi-Format Document Ingestion with Source Metadata

**User Story:** As the chatbot pipeline, I want to load all provided Wells Fargo documents with page/source metadata preserved, so that every downstream answer can cite exactly where it came from.

#### Acceptance Criteria
1. WHEN the ingestion pipeline runs THEN THE SYSTEM SHALL load `WellsFargo_Deposit_Account_Agreement.pdf`, `WellsFargo_Consumer_Account_Fees_Info.pdf`, and `WellsFargo_Credit_Card_Agreement.pdf` from `../source_data/`, preserving the source filename and page number as metadata on every extracted unit of text.
2. WHEN the ingestion pipeline runs THEN THE SYSTEM SHALL also parse `WellsFargo_Financial_Data_XBRL.xml` together with `WellsFargo_Financial_Labels_XBRL.xml` into human-readable fact statements (e.g. "Wells Fargo & Company reported Revenues of $X for period Y"), tagging each with the source filename and the XBRL concept name in place of a page number.
3. IF a PDF page yields no extractable text (e.g. scanned/garbled) THEN THE SYSTEM SHALL log the page as a parsing failure rather than silently dropping it, so parsing-quality can be measured in the Stage 1 ablation.
4. THE SYSTEM SHALL support swapping the PDF parsing library (pypdf / pdfplumber / PyMuPDF) behind one interface, so Stage 1 of the ablation suite can compare them without duplicating ingestion logic.

### Requirement 2: Configurable Chunking Strategy

**User Story:** As the chatbot pipeline, I want documents split into retrievable chunks using a strategy and size proven by evaluation, so that retrieval quality is maximized rather than assumed.

#### Acceptance Criteria
1. WHEN chunking a loaded document THEN THE SYSTEM SHALL retain the originating document name and page number (or XBRL concept) on every chunk's metadata.
2. THE SYSTEM SHALL expose chunk size, overlap, and strategy (fixed-size, recursive/sentence-aware, semantic) as configuration values, not hardcoded constants, so the Stage 2 ablation can sweep them.
3. WHEN the ablation suite determines a winning chunk size/overlap/strategy THEN THE SYSTEM SHALL use that configuration as the default for the production app.

### Requirement 3: Embedding Generation and Vector Store Persistence

**User Story:** As the chatbot pipeline, I want chunk embeddings computed and persisted in a vector database, so that retrieval doesn't require re-embedding the corpus on every run.

#### Acceptance Criteria
1. THE SYSTEM SHALL store chunk embeddings in a Chroma `PersistentClient` collection on local disk, with the original chunk text and metadata attached to each vector record.
2. THE SYSTEM SHALL expose the embedding model as a configuration value, so the Stage 3 ablation can compare at least 2 real models (e.g. `text-embedding-3-small` vs. a local `sentence-transformers` model) before the production default is fixed.
3. WHEN the vector store already contains an up-to-date index for the current corpus and chunking configuration THEN THE SYSTEM SHALL reuse it rather than re-embedding from scratch.

### Requirement 4: Hybrid Retrieval (Dense + Sparse + Fusion)

**User Story:** As a customer asking a question, I want the system to find the most relevant document chunks regardless of whether my wording matches the document's exact terms or is a paraphrase, so that I get accurate answers either way.

#### Acceptance Criteria
1. THE SYSTEM SHALL support dense retrieval (embedding similarity via the Chroma store) and sparse retrieval (BM25 via `bm25s`) over the same chunk corpus.
2. THE SYSTEM SHALL support fusing dense and sparse results via Reciprocal Rank Fusion (RRF) and via weighted linear combination with a configurable α, so Stage 6 of the ablation suite can compare both and sweep α.
3. WHEN the ablation suite determines a winning retrieval mode (dense-only / sparse-only / hybrid) and merge method THEN THE SYSTEM SHALL use that configuration as the production default.
4. THE SYSTEM SHALL return, for each retrieved chunk, its source document, page/concept, and retrieval score, for use in citations and guardrail checks.

### Requirement 5: Reranking of Retrieved Chunks

**User Story:** As a customer, I want the most relevant chunks prioritized before the model generates an answer, so that the answer is grounded in the best available evidence.

#### Acceptance Criteria
1. THE SYSTEM SHALL support an optional cross-encoder reranking stage (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) applied to the top-20 hybrid retrieval results, reducing to a smaller top-N (e.g. top-5) before generation.
2. THE SYSTEM SHALL allow reranking to be toggled on/off via configuration, so Stage 7 of the ablation suite can compare accuracy and latency with and without it.
3. WHEN the ablation suite shows reranking improves R@3/NDCG@3 enough to justify its added latency THEN THE SYSTEM SHALL enable it by default in production; otherwise it SHALL be disabled by default.

### Requirement 6: Grounded Answer Generation with Mandatory Citations

**User Story:** As a customer, I want every answer to cite the specific Wells Fargo document (and page, where applicable) it came from, so that I can trust and verify the information.

#### Acceptance Criteria
1. WHEN generating an answer THE SYSTEM SHALL use only the retrieved (and, if enabled, reranked) chunks as context — the system prompt SHALL explicitly forbid answering from the model's general knowledge.
2. WHEN an answer is generated THE SYSTEM SHALL include numbered inline citations (e.g. `[1]`, `[2]`) that map to a source list showing document name and page number (or XBRL concept name).
3. THE SYSTEM SHALL support at least 2 real LLMs behind one configuration switch (e.g. `gpt-4o-mini` vs. `gpt-4o` or another provider), so Stage 8 of the ablation suite can compare them with retrieval held fixed.
4. WHEN the ablation suite determines a winning LLM THEN THE SYSTEM SHALL use it as the production default for generation.

### Requirement 7: Guardrails — No Hallucination, Out-of-Scope Refusal, No Account-Data Access

**User Story:** As Wells Fargo (the deploying institution), I want the assistant to refuse to guess, refuse questions about other banks, and refuse to pretend it can access personal account data, so that the bot never misleads a customer or overstates its capabilities.

#### Acceptance Criteria
1. IF the top reranked/retrieved chunk's relevance score is below a configured minimum threshold THEN THE SYSTEM SHALL respond that it doesn't have enough information rather than generating a best-effort guess.
2. IF a user asks about a different bank or company (e.g. Chase, HDFC, Bank of America) THEN THE SYSTEM SHALL respond that the question is out of scope for this assistant, without attempting to answer it.
3. IF a user asks for personal/account-specific data (e.g. "what's my balance") THEN THE SYSTEM SHALL explain that it can only answer general policy questions and cannot access personal account data.
4. THE SYSTEM SHALL never state a specific fee amount, APR, or rule that is not present in the retrieved source chunks.

### Requirement 8: Multi-Turn Conversational Memory

**User Story:** As a customer, I want to ask follow-up questions without repeating context, so that the conversation feels natural.

#### Acceptance Criteria
1. WHEN a user sends a follow-up message in the same session THEN THE SYSTEM SHALL use prior conversation turns to resolve references (e.g. rewriting "what about the fee for that?" using the prior topic) before retrieval.
2. THE SYSTEM SHALL scope conversation history per session (session ID from the frontend), not globally across users.
3. THE SYSTEM SHALL cap or summarize conversation history so token usage doesn't grow unbounded over a long session.

### Requirement 9: Labeled Evaluation Question Set

**User Story:** As the developer, I want a labeled set of at least 20 real questions with ground-truth sources before optimizing any pipeline stage, so that every later comparison is measured against a fixed ruler.

#### Acceptance Criteria
1. THE SYSTEM'S evaluation harness SHALL include at least 20 questions, each with: question text, ground-truth source document (and page/concept if applicable), and a short ideal answer.
2. THE SYSTEM SHALL derive this set by expanding the 7 seed questions already listed in `../../REQUIREMENT.md` §7, covering all 3 PDFs plus the XBRL data.
3. THE SYSTEM SHALL classify each question by query type (exact-term/keyword-style vs. paraphrased/semantic), so Stage 5's dense-vs-sparse-vs-hybrid comparison can be broken out by query type as required.

### Requirement 10: Ablation Experiment Suite and Evaluation Report

**User Story:** As the developer being graded primarily on evidence, I want every pipeline decision backed by a measured comparison table with a written, numbers-cited justification, so the evaluation rigor requirement (40% of the grade) is fully satisfied.

#### Acceptance Criteria
1. THE SYSTEM SHALL implement runnable experiments for all 8 stages defined in `../../EVALUATION_METHODOLOGY.md` Part C: parsing strategy, chunking strategy, embedding model, vector DB, retrieval mode, hybrid merge weighting, reranking, and LLM choice.
2. WHEN each ablation experiment completes THEN THE SYSTEM SHALL produce a filled results table (not a placeholder) using the exact metrics specified per stage, extended with Recall@10 alongside R@1/R@3/MRR@10/NDCG@3 (per the tutor's explicit request), latency, and RAGAS/DeepEval scores as applicable.
3. THE SYSTEM SHALL compile all 8 tables plus a synthesis paragraph — citing the actual winning numbers per stage — into `EVALUATION_REPORT.md` at the project root of `Chatbot/`.
4. THE SYSTEM SHALL include a "what we'd try next" section in the evaluation report, tying back to failure modes not yet addressed.
5. FOR the vector DB (Stage 4) and LLM (Stage 8) comparisons, THE SYSTEM SHALL additionally include a written **Scalability** discussion alongside Latency and Cost — e.g. how index build time/query latency and API cost/throughput are expected to change if the corpus grows from 60 pages to a full bank-wide document set — since the tutor named scalability as a required factor and the actual corpus is too small to measure this empirically.
6. THE SYSTEM SHALL implement at least one dataset-specific custom metric beyond the standard set — a **numeric-fact accuracy** check that verifies every dollar amount, percentage, or APR figure in a generated answer exactly matches the value present in its cited source chunk — since silently altering a fee or APR number is this domain's most consequential failure mode, and report it alongside the standard metrics wherever generation is evaluated (Stage 8 and the end-to-end run in Requirement 11).

### Requirement 11: End-to-End RAGAS + DeepEval Scoring

**User Story:** As the developer, I want a headline quality score for the final chosen pipeline, so grading has one clear end-to-end number beyond the per-stage ablations.

#### Acceptance Criteria
1. WHEN the final pipeline configuration is fixed (post-ablation) THEN THE SYSTEM SHALL run the full evaluation question set through it and compute RAGAS Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
2. THE SYSTEM SHALL also compute DeepEval Hallucination, Answer Relevancy, Faithfulness, and a custom G-Eval criterion (e.g. "Strict Grounding") on the same runs.
3. THE SYSTEM SHALL record these end-to-end scores in `EVALUATION_REPORT.md` as the headline result.

### Requirement 12: Acceptance Test Verification

**User Story:** As the grader, I want the 5 official acceptance questions from `../../REQUIREMENT.md` §6 verified against the running app, so correctness (30% of the grade) is demonstrably met.

#### Acceptance Criteria
1. THE SYSTEM SHALL be manually or automatically exercised against all 5 acceptance questions (monthly service fee, credit card APR, account closure without notice, out-of-scope Chase question, personal balance question).
2. WHEN each acceptance question is run THEN THE SYSTEM SHALL log the question, the full answer with citations, and whether the expected behavior (correct grounded answer, correct citation, or correct refusal) was met.
3. THE SYSTEM SHALL persist these transcripts (or screenshots) as part of the submitted deliverables.

### Requirement 13: React Chat Web UI

**User Story:** As a Wells Fargo customer, I want a simple chat interface where I can ask questions and see cited sources, so that I can use the assistant without any technical knowledge.

#### Acceptance Criteria
1. THE SYSTEM SHALL provide a React single-page chat interface with a message input, scrollable message history, and a loading indicator while a response is pending.
2. WHEN a bot response includes citations THEN THE FRONTEND SHALL render them as visible source references (document name + page) alongside the answer, not just as inline bracket numbers.
3. THE FRONTEND SHALL persist a session ID (e.g. in `localStorage`) for the duration of a browser session so multi-turn memory (Requirement 8) works across page reloads within that session.
4. IF the backend returns an error or times out THEN THE FRONTEND SHALL display a clear error state rather than hanging silently.

### Requirement 14: FastAPI Backend API

**User Story:** As the React frontend, I want a documented HTTP API to send chat messages and receive grounded answers, so the frontend and RAG pipeline are cleanly decoupled.

#### Acceptance Criteria
1. THE SYSTEM SHALL expose `POST /chat` accepting `{session_id, message}` and returning `{answer, citations: [{document, page}], ...}`.
2. THE SYSTEM SHALL expose `GET /health` for liveness checks.
3. THE SYSTEM SHALL enable CORS for the local React dev server origin.
4. IF the OpenAI API call fails or times out THEN THE SYSTEM SHALL return a clear error response rather than a raw stack trace.

### Requirement 15: Configuration, Secrets, and Environment Setup

**User Story:** As the developer, I want API keys and pipeline configuration managed via environment variables and a settings module, so secrets are never hardcoded and the winning ablation configuration is easy to apply.

#### Acceptance Criteria
1. THE SYSTEM SHALL read `OPENAI_API_KEY` (and any other provider keys used in the LLM ablation) from a `.env` file, never committed to version control, with a `.env.example` documenting required variables.
2. THE SYSTEM SHALL centralize all tunable pipeline settings (chunk size/overlap, embedding model, retrieval mode, fusion method/α, reranker on/off, LLM choice, guardrail threshold) in one settings module, defaulted to the ablation-suite winners.
3. THE SYSTEM SHALL document, in a top-level `README.md` under `Chatbot/`, how to install dependencies and run both the backend and frontend locally.

### Requirement 16: Findings Presentation Deliverable

**User Story:** As the tutor grading this capstone, I want a presentation of the group's findings in addition to the written documentation, so the team can walk through and defend their design decisions live, not just submit a report.

#### Acceptance Criteria
1. THE SYSTEM'S deliverables SHALL include a presentation (slide deck or equivalent walkthrough document, e.g. `Chatbot/PRESENTATION.md` or a slides file) summarizing: the architecture, each of the 6 headline decisions the tutor asked to be justified (parsing technique, chunking strategy, retrieval methodology, embeddings, vector database, LLM), the winning number for each, and the end-to-end RAGAS/DeepEval results.
2. THE PRESENTATION SHALL explicitly answer each of the 6 decision questions from the tutor's email in the format asked — e.g. for retrieval methodology: dense vs. sparse vs. hybrid and why; if hybrid, fusion method and α and why; if reranking, which technique and why; for embeddings, model, vector dimensionality, and why.
3. THE PRESENTATION SHALL be derived from `EVALUATION_REPORT.md` and `ACCEPTANCE_TESTS.md` rather than duplicating their work — it summarizes and highlights, the full evidence lives in those files.

## Glossary

| Term | Meaning |
|---|---|
| RAG | Retrieval-Augmented Generation — answering a query by first retrieving relevant text chunks, then having an LLM generate an answer grounded in them. |
| Chunk | A retrievable unit of text produced by splitting a source document, tagged with source document/page metadata. |
| Dense retrieval | Retrieval by embedding-vector similarity (semantic match). |
| Sparse retrieval | Retrieval by lexical/keyword match (BM25). |
| Hybrid retrieval | Combining dense and sparse retrieval results via a fusion method. |
| RRF | Reciprocal Rank Fusion — merges ranked lists by summing `1/(k+rank)` per item across lists, k≈60. |
| Weighted-α fusion | Merges normalized dense and sparse scores as `α·dense + (1-α)·sparse`. |
| Reranking | A second-pass model (e.g. cross-encoder) that rescores an initial retrieval result set for better ordering before generation. |
| R@k (Recall@k) | Fraction of queries for which the correct source chunk appears in the top-k retrieved results. |
| MRR@10 (Mean Reciprocal Rank) | Average of `1/rank_of_first_correct_chunk` (within top 10) across queries. |
| NDCG@k | Normalized Discounted Cumulative Gain — rewards graded relevance and correct ordering within the top-k results. |
| RAGAS | An evaluation framework scoring generated answers on Faithfulness, Answer Relevancy, Context Precision, and Context Recall. |
| DeepEval | A second LLM-judge evaluation framework used to corroborate RAGAS scores (Hallucination, Faithfulness, Answer Relevancy, G-Eval). |
| Groundedness guardrail | A check that refuses to answer when retrieved evidence is too weak (below a minimum relevance score), instead of letting the LLM guess. |
| XBRL | eXtensible Business Reporting Language — the structured XML format used for the SEC-filed financial facts in `WellsFargo_Financial_Data_XBRL.xml`. |
| Ablation | An experiment that swaps one pipeline component/setting while holding all others fixed, to isolate that component's effect on measured metrics. |
