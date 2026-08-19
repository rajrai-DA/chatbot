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

## Getting started (new contributors)

This repo excludes anything generated or environment-specific — `backend/venv/`,
`frontend/node_modules/`, `data/vector_store/` (the Chroma DB), and `.env` (secrets) are all
git-ignored. After cloning, you need to (re)create these locally:

```bash
git clone https://github.com/rajrai-DA/chatbot.git
cd chatbot
```

Then follow **Backend setup** and **Frontend setup** below. The source corpus (the WellsFargo
`*.pdf`/`*.xml` files) ships in the repo at `source_data/`, so no separate download is needed.

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

The first `/chat` request triggers a one-time ingest of `../source_data/*.pdf` and the
XBRL financial data into the Chroma vector store at `../data/vector_store/` (persisted —
subsequent starts reuse it). `GET /health` returns `{"status": "ok"}` without triggering ingest.

### Vector database: creating and reloading

The Chroma DB lives at `data/vector_store/` (git-ignored — every developer builds their own
locally) and is not a single collection: `backend/app/retrieval/vector_store.py`'s
`collection_name_for()` derives the collection name from the active chunking config and
embedding model in `backend/app/config.py` (`chunk_strategy`, `chunk_size`, `chunk_overlap`,
`embedding_provider`, `embedding_model`), so different ablation configs get separate
collections and never clobber each other in the same DB folder.

**First-time creation** — automatic, no separate download needed since `source_data/` ships in
the repo. Just start the backend and send any `/chat` request (or call `get_chatbot()`):

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
# in another terminal: curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
#   -d '{"session_id": "init", "message": "hello"}'
```

This parses the PDFs/XBRL, chunks them, embeds each chunk, and populates the collection for
the current config. `VectorStore.build()` skips re-embedding on subsequent startups as long
as the existing collection's chunk count matches — so restarts are fast and don't re-call the
embedding API.

**Reloading / forcing a full rebuild** — needed after changing the source documents, or after
changing `chunk_strategy`/`chunk_size`/`chunk_overlap`/`embedding_model` when you want the
existing collection re-embedded rather than a new one created alongside it:

```bash
cd backend
source venv/bin/activate
python -c "from app.generation.chatbot import ProductionRAGChatbot; \
print(ProductionRAGChatbot().ingest(force=True))"
```

Or, to wipe everything and start clean (e.g. the DB got corrupted, or you want to reclaim
disk space from stale ablation collections):

```bash
rm -rf data/vector_store/
# next /chat request (or the command above) rebuilds it from source_data/
```

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
