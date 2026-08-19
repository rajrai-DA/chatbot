# Wells Fargo RAG Chatbot — Findings Presentation

_A summary of `EVALUATION_REPORT.md` and `ACCEPTANCE_TESTS.md` — the full evidence lives in
those files; this is the walkthrough version._

## 1. What we built

A citation-grounded RAG chatbot answering customer questions about Wells Fargo's deposit
account terms, fees, credit card agreement, and financial filings (XBRL) — refusing to
guess, refusing out-of-scope questions, and refusing to pretend it can see personal account
data.

```
React (Vite) frontend --POST /chat--> FastAPI backend --> ProductionRAGChatbot
                                                             guardrails -> hybrid retrieval
                                                             (dense Chroma + sparse BM25)
                                                             -> cross-encoder rerank
                                                             -> grounded gpt-4o generation
                                                             -> citations -> session memory
```

Every tunable in that pipeline — parser, chunk size/strategy, embedding model, vector DB,
retrieval mode, fusion method/α, reranker on/off, LLM — lives in one `Settings` object
(`backend/app/config.py`), and every default there is a measured ablation winner, not a
guess.

## 2. The six decisions, and the numbers behind each

### Parsing technique — `pymupdf`
`pypdf` and `pymupdf` tied on downstream R@3 (0.652 vs 0.652), both beating `pdfplumber`
(0.565). We broke the tie toward `pymupdf` because it preserves table structure as
markdown — which matters for this corpus's fee tables even though R@3 alone couldn't see it.

### Chunking strategy — fixed-size, 300 characters, no overlap
The best of 19 configurations swept (fixed/recursive/semantic × 300/500/800 × 0/50/100):
**R@3 = 0.739**, beating the next-best config (`fixed`/500/50, R@3=0.652) and every
recursive/semantic variant tried.

### Retrieval methodology — hybrid, weighted fusion at α = 0.7
Hybrid beat both dense-only and sparse-only (R@3 = 0.652 vs 0.609 / 0.609), with the
keyword vs. semantic breakdown (0.636 vs 0.667) confirming both query styles benefit.
Weighted-α fusion at **α=0.7** (R@3=0.652, MRR@10=0.617, NDCG@3=0.598) edged out RRF
(same R@3, but MRR@10=0.576, NDCG@3=0.560) on ranking quality at equal top-line recall.
Reranking (cross-encoder `ms-marco-MiniLM-L-6-v2`, top-20 → top-5): **enabled** —
+0.087 R@3, +0.130 NDCG@3 for +142ms/query, a trade worth making for a support bot.

### Embeddings — `text-embedding-3-small`, 1536 dimensions
Beat the local `bge-small-en-v1.5` (384-dim) on every retrieval metric: R@3 0.609 vs 0.478,
MRR@10 0.585 vs 0.494 — at the cost of ~32x higher per-query latency (520ms vs 16ms),
accepted because retrieval quality is this project's primary grading axis.

### Vector database — Chroma
R@3 parity with FAISS confirmed (0.609 vs 0.609, delta=0.000) — the real differentiator is
operational: Chroma bundles persistence and native metadata filtering that FAISS doesn't
have without hand-built plumbing.

### LLM — `gpt-4o`
Beat `gpt-4o-mini` on DeepEval Hallucination (0.522 vs 0.652, lower is better) despite
`gpt-4o-mini` scoring higher on RAGAS Faithfulness (0.844 vs 0.759) — both hit a perfect
1.000 numeric fact accuracy on this eval set, but hallucination is the more consequential
signal for a banking assistant, and `gpt-4o`'s ~$0.002/query is negligible at this scale.

## 3. End-to-end results (the headline number)

Full 25-question set through the live, finalized `ProductionRAGChatbot`:

| Retrieval | Generation |
|---|---|
| R@1=0.522, R@3=0.609, R@10=0.652 | Faithfulness=0.689, Answer Relevancy=0.665 |
| MRR@10=0.574, NDCG@3=0.570 | Hallucination=0.522, G-Eval Strict Grounding=0.923 |
| | **Numeric fact accuracy = 1.000** |

Both out-of-scope and account-data negative controls were correctly refused (2/2).

## 4. Acceptance tests — 3 pass, 1 partial, 1 fail (honestly reported)

| # | Question | Result |
|---|---|---|
| 1 | Monthly service fee | Partial — correctly refused rather than guess the $ figure |
| 2 | Credit card APR | Fail — correct source wasn't retrieved into top-5 |
| 3 | Account closure without notice | **Pass** — grounded, cited |
| 4 | Chase overdraft fee | **Pass** — refused, out of scope |
| 5 | Personal balance | **Pass** — refused, account-data guardrail |

The two non-passes are retrieval-recall misses consistent with the measured R@3≈0.61 (not
1.0) — not guardrail or hallucination failures. In both cases the model correctly declined
to invent a number rather than guessing wrong. Full root-cause diagnosis is in
`ACCEPTANCE_TESTS.md`.

## 5. What we'd try next

- Table-aware chunking — the #1 miss traces to a fee table's header row landing in a
  different chunk than its data row under fixed/300/0 chunking.
- Investigate why the Deposit Account Agreement's "using your card" section out-competes
  the actual Credit Card Agreement under BM25 for APR-style questions, and whether Chroma's
  approximate (HNSW) search is consistent enough across process restarts for this corpus
  size — an exact search may be affordable at ~2,500 chunks.
- Widen the LLM ablation to a 3rd model (Gemini was available but out of scope for this pass).
- Hunt for numeric drift under adversarial phrasing and multi-hop numeric questions — this
  eval set's perfect 1.000 numeric fact accuracy is encouraging but not yet stress-tested.
