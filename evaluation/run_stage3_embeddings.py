"""Stage 3 ablation — embedding model comparison.

Compares OpenAI text-embedding-3-small against a local sentence-transformers
model, reporting vector dimensionality (the tutor explicitly asks "which
model, vector size, and why"), retrieval metrics, and per-query latency
(Requirement 10.1/10.2, EVALUATION_METHODOLOGY.md Stage 3).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from _util import append_to_report, evaluate_retrieval, fmt, load_eval_questions

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings  # noqa: E402
from app.ingestion.chunking import chunk_documents  # noqa: E402
from app.ingestion.loaders import load_all_source_documents  # noqa: E402
from app.retrieval.hybrid import HybridIndex  # noqa: E402

DOCS = load_all_source_documents(parser=settings.parser)
CHUNKS = chunk_documents(DOCS, strategy=settings.chunk_strategy, size=settings.chunk_size, overlap=settings.chunk_overlap)

CANDIDATES = [
    {"provider": "openai", "model": "text-embedding-3-small", "dims": 1536},
    {"provider": "sentence-transformers", "model": "BAAI/bge-small-en-v1.5", "dims": 384},
]


def run_for_candidate(provider: str, model: str) -> dict:
    collection = f"wf_stage3_{provider}_{model}".replace("/", "_").replace(".", "_")[:63]
    index = HybridIndex(embedding_provider=provider, embedding_model=model, collection_name=collection)

    t0 = time.perf_counter()
    index.build(CHUNKS, force=True)
    build_time_s = time.perf_counter() - t0

    questions = load_eval_questions()
    metrics = evaluate_retrieval(lambda q: index.search(q, k=10, mode="dense"), questions)

    return {"provider": provider, "model": model, "build_time_s": build_time_s, **metrics}


def main():
    rows = []
    for c in CANDIDATES:
        r = run_for_candidate(c["provider"], c["model"])
        r["dims"] = c["dims"]
        rows.append(r)

    table = "## Stage 3 — Embedding model\n\n"
    table += "| Embedding model | Dimensions | R@1 | R@3 | R@10 | MRR@10 | NDCG@3 | Latency (ms/query) |\n"
    table += "|---|---|---|---|---|---|---|---|\n"
    for r in rows:
        o = r["overall"]
        table += (
            f"| {r['model']} ({r['provider']}) | {r['dims']} | {fmt(o['r_at_1'])} | {fmt(o['r_at_3'])} | "
            f"{fmt(o['r_at_10'])} | {fmt(o['mrr_at_10'])} | {fmt(o['ndcg_at_3'])} | {fmt(r['avg_latency_ms'])} |\n"
        )

    winner = max(rows, key=lambda r: (r["overall"]["r_at_3"], r["overall"]["mrr_at_10"]))
    table += (
        f"\n**Winner: `{winner['model']}` ({winner['provider']}, {winner['dims']}-dim)** — "
        f"R@3={fmt(winner['overall']['r_at_3'])}, MRR@10={fmt(winner['overall']['mrr_at_10'])} at "
        f"{fmt(winner['avg_latency_ms'])}ms/query, the best retrieval quality among the embedding models tested.\n"
    )

    print(table)
    append_to_report(table)
    return winner


if __name__ == "__main__":
    main()
