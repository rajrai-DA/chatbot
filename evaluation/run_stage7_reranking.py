"""Stage 7 ablation — reranking on vs off.

Cross-encoder reranks the top-20 hybrid results down to top-5, comparing
accuracy delta against added latency (Requirement 10.1,
EVALUATION_METHODOLOGY.md Stage 7).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from _util import append_to_report, fmt, load_eval_questions

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings  # noqa: E402
from app.ingestion.chunking import chunk_documents  # noqa: E402
from app.ingestion.loaders import load_all_source_documents  # noqa: E402
from app.generation.reranker import Reranker  # noqa: E402
from app.retrieval.hybrid import HybridIndex  # noqa: E402
from metrics import mean_mrr_at_10, mean_ndcg_at_3, mean_recall_at_k  # noqa: E402

DOCS = load_all_source_documents(parser=settings.parser)
CHUNKS = chunk_documents(DOCS, strategy=settings.chunk_strategy, size=settings.chunk_size, overlap=settings.chunk_overlap)

INDEX = HybridIndex(collection_name="wf_stage7_shared")
INDEX.build(CHUNKS, force=True)


def _as_dicts(results):
    return [{"source": c.source, "page": c.page, "score": c.score, "text": c.text} for c in results]


def run(rerank_enabled: bool) -> dict:
    reranker = Reranker(enabled=rerank_enabled)
    questions = load_eval_questions()

    all_retrieved, ground_truths, latencies = [], [], []
    for q in questions:
        t0 = time.perf_counter()
        candidates = INDEX.search(q["question"], k=20, mode="hybrid")
        final = reranker.rerank(q["question"], candidates, top_n=5)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_retrieved.append(_as_dicts(final))
        ground_truths.append((q["ground_truth_source"], q["ground_truth_page"]))

    return {
        "r_at_3": mean_recall_at_k(all_retrieved, ground_truths, 3),
        "ndcg_at_3": mean_ndcg_at_3(all_retrieved, ground_truths),
        "mrr_at_10": mean_mrr_at_10(all_retrieved, ground_truths),
        "avg_latency_ms": sum(latencies) / len(latencies),
    }


def main():
    off = run(False)
    on = run(True)

    table = "## Stage 7 — Reranking\n\n"
    table += "| Config | R@3 | NDCG@3 | Added latency (ms) |\n"
    table += "|---|---|---|---|\n"
    table += f"| No reranker | {fmt(off['r_at_3'])} | {fmt(off['ndcg_at_3'])} | 0 |\n"
    added = on["avg_latency_ms"] - off["avg_latency_ms"]
    table += f"| + Cross-encoder rerank | {fmt(on['r_at_3'])} | {fmt(on['ndcg_at_3'])} | {fmt(added)} |\n"

    delta_r3 = on["r_at_3"] - off["r_at_3"]
    delta_ndcg = on["ndcg_at_3"] - off["ndcg_at_3"]
    worth_it = delta_r3 > 0 or delta_ndcg > 0
    verdict = "enabled" if worth_it else "disabled"
    table += (
        f"\n**Winner: reranking `{verdict}` by default.** R@3 changed by {delta_r3:+.3f} and NDCG@3 by "
        f"{delta_ndcg:+.3f} for +{fmt(added)}ms/query. "
        f"{'The accuracy gain justifies the added latency for a customer-facing support bot.' if worth_it else 'The added latency is not justified by the (flat or negative) accuracy change on this corpus.'}\n"
    )

    print(table)
    append_to_report(table)
    return worth_it


if __name__ == "__main__":
    main()
