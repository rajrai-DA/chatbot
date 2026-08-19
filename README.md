# Wells Fargo RAG Chatbot

A citation-grounded RAG chatbot answering customer questions about Wells Fargo's deposit
account terms, fees, and credit card agreement — plus the full evaluation harness used to
justify every pipeline decision. See [`.kiro/specs/wells-fargo-rag-chatbot/`](.kiro/specs/wells-fargo-rag-chatbot/)
for the requirements/design/task spec, [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md) for
the ablation results, [`ACCEPTANCE_TESTS.md`](ACCEPTANCE_TESTS.md) for the 5 official
acceptance-test transcripts, and [`PRESENTATION.md`](PRESENTATION.md) for the findings summary.

## Architecture

```
frontend/ (React + Vite)  --POST /chat-->  backend/ (FastAPI)
                                              -> ProductionRAGChatbot
                                                 (guardrails -> hybrid retrieval
                                                  -> rerank -> grounded generation
                                                  -> citations -> memory)
                                              -> Chroma (vector store, persisted
                                                 in data/vector_store/)
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- An OpenAI API key (embeddings + generation)

## Backend setup

```bash
cd backend
python3 -m venv venv          # or reuse an existing venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `../.env.example` to `../.env` (one level up, in `Chatbot/`) and fill in your key:

```bash
cp ../.env.example ../.env
# edit ../.env: OPENAI_API_KEY=sk-...
```

Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

The first `/chat` request triggers a one-time ingest of `../../source_data/*.pdf` and the
XBRL financial data into the Chroma vector store at `../data/vector_store/` (persisted —
subsequent starts reuse it). `GET /health` returns `{"status": "ok"}` without triggering ingest.

Run the backend tests:

```bash
pytest tests/
```

## Frontend setup

```bash
cd frontend
npm install
cp .env.example .env   # VITE_API_BASE_URL defaults to http://localhost:8000
npm run dev
```

Open the printed local URL (default `http://localhost:5173`). The chat UI persists a
`session_id` in `localStorage` so multi-turn memory survives a page reload.

## Running the evaluation harness

Each ablation stage is a standalone script under `evaluation/`, run against the same
25-question labeled set in `evaluation/eval_questions.json`:

```bash
cd evaluation
python run_stage1_parsing.py
python run_stage2_chunking.py
python run_stage3_embeddings.py
python run_stage4_vector_db.py
python run_stage5_retrieval_mode.py
python run_stage6_hybrid_merge.py
python run_stage7_reranking.py
python run_stage8_llm.py
python run_end_to_end_ragas_deepeval.py
```

Every script appends a filled results table to `../EVALUATION_REPORT.md`. Stages 3+ make
real OpenAI API calls (embeddings and/or generation), and Stage 8 / the end-to-end run also
call RAGAS and DeepEval LLM judges — expect real (small) API cost and multi-minute runtimes.

## Configuration

All pipeline tunables (parser, chunk size/strategy, embedding model, retrieval mode, fusion
method/α, reranker, LLM, guardrail threshold) live in one place: `backend/app/config.py`'s
`Settings` class, loaded from `Chatbot/.env`. The defaults there are the ablation-suite
winners from `EVALUATION_REPORT.md` — changing the production pipeline is a config edit,
never a code edit.
