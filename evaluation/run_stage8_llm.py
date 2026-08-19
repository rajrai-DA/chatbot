"""Stage 8 ablation — LLM for generation, retrieval held fixed.

Compares gpt-4o-mini vs gpt-4o with retrieval fixed at Stage 5-7's winning
config, reporting RAGAS + DeepEval + numeric_fact_accuracy + cost/latency
(Requirement 10.1/10.2/10.6, EVALUATION_METHODOLOGY.md Stage 8).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from _util import append_to_report, fmt, load_eval_questions
from genai_metrics import evaluate_deepeval, evaluate_ragas

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings  # noqa: E402
from app.generation.prompt import NO_INFO_ANSWER, build_prompt  # noqa: E402
from app.generation.reranker import Reranker  # noqa: E402
from app.ingestion.chunking import chunk_documents  # noqa: E402
from app.ingestion.loaders import load_all_source_documents  # noqa: E402
from app.retrieval.hybrid import HybridIndex  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from metrics import numeric_fact_accuracy  # noqa: E402

# Stage 5-7 winners, fixed for this stage (design.md: "fix retrieval, vary only the LLM")
RETRIEVAL_MODE, FUSION_METHOD, RERANK_ENABLED = "hybrid", "rrf", True
MIN_RERANK_SCORE = settings.min_rerank_score

DOCS = load_all_source_documents(parser=settings.parser)
CHUNKS = chunk_documents(DOCS, strategy=settings.chunk_strategy, size=settings.chunk_size, overlap=settings.chunk_overlap)

INDEX = HybridIndex(collection_name="wf_stage8_shared")
INDEX.build(CHUNKS, force=True)
RERANKER = Reranker(enabled=RERANK_ENABLED)

CANDIDATES = ["gpt-4o-mini", "gpt-4o"]

# Approximate public list pricing (USD / 1M tokens) — used only to report an
# indicative cost/query for this ablation, not for billing.
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def _retrieve_and_answer(question: str, llm):
    candidates = INDEX.search(question, k=20, mode=RETRIEVAL_MODE, fusion_method=FUSION_METHOD)
    reranked = RERANKER.rerank(question, candidates, top_n=5)
    if not reranked or reranked[0].score < MIN_RERANK_SCORE:
        return NO_INFO_ANSWER, reranked, {}
    response = llm.invoke(build_prompt(question, reranked))
    usage = getattr(response, "usage_metadata", None) or {}
    return response.content.strip(), reranked, usage


def run_for_model(model_name: str) -> dict:
    llm = ChatOpenAI(model=model_name, temperature=0)
    questions = load_eval_questions()

    items, latencies, numeric_scores = [], [], []
    total_input_tokens = total_output_tokens = 0

    for q in questions:
        t0 = time.perf_counter()
        answer, reranked, usage = _retrieve_and_answer(q["question"], llm)
        latencies.append((time.perf_counter() - t0) * 1000)

        contexts = [c.text for c in reranked] or [""]
        reference = q.get("ideal_answer", "") or (reranked[0].text if reranked else "")
        items.append({"question": q["question"], "answer": answer, "contexts": contexts, "reference": reference})
        numeric_scores.append(numeric_fact_accuracy(answer, contexts))

        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)

    ragas_scores = evaluate_ragas(items, model="gpt-4o-mini")
    deepeval_scores = evaluate_deepeval(items, model="gpt-4o-mini")

    price = PRICING.get(model_name, {"input": 0, "output": 0})
    avg_cost = (
        (total_input_tokens / 1_000_000) * price["input"] + (total_output_tokens / 1_000_000) * price["output"]
    ) / len(questions)

    return {
        "model": model_name, "ragas": ragas_scores, "deepeval": deepeval_scores,
        "numeric_fact_accuracy": sum(numeric_scores) / len(numeric_scores) if numeric_scores else None,
        "avg_cost_per_query": avg_cost, "avg_latency_ms": sum(latencies) / len(latencies),
    }


def main():
    rows = [run_for_model(m) for m in CANDIDATES]

    table = "## Stage 8 — LLM for generation\n\n"
    table += (
        "| LLM | Faithfulness (RAGAS) | Answer Relevancy (RAGAS) | Context Precision (RAGAS) | "
        "Hallucination (DeepEval) | G-Eval Strict Grounding (DeepEval) | Numeric fact accuracy | "
        "Cost/query | Latency (ms) |\n"
    )
    table += "|---|---|---|---|---|---|---|---|---|\n"
    for r in rows:
        ra, de = r["ragas"], r["deepeval"]
        table += (
            f"| {r['model']} | {fmt(ra['faithfulness'])} | {fmt(ra['answer_relevancy'])} | "
            f"{fmt(ra['context_precision'])} | {fmt(de['hallucination'])} | {fmt(de['g_eval_strict_grounding'])} | "
            f"{fmt(r['numeric_fact_accuracy'])} | ${fmt(r['avg_cost_per_query'])} | {fmt(r['avg_latency_ms'])} |\n"
        )

    def _score(r):
        ra = r["ragas"]
        return (ra["faithfulness"] or 0) + (ra["answer_relevancy"] or 0) - (r["deepeval"]["hallucination"] or 0)

    winner = max(rows, key=_score)
    cost_ratio = PRICING["gpt-4o"]["input"] / PRICING["gpt-4o-mini"]["input"]
    table += (
        f"\n**Winner: `{winner['model']}`** — Faithfulness={fmt(winner['ragas']['faithfulness'])}, "
        f"Hallucination={fmt(winner['deepeval']['hallucination'])} (lower is better), numeric fact "
        f"accuracy={fmt(winner['numeric_fact_accuracy'])}, at ${fmt(winner['avg_cost_per_query'])}/query.\n\n"
        f"**Scalability.** Both candidates are stateless per-call OpenAI API requests, so throughput "
        f"scales horizontally with concurrent users up to OpenAI's account-level rate limits — the "
        f"bottleneck at scale is rate-limit headroom and cost, not architecture. `gpt-4o-mini` costs "
        f"roughly {cost_ratio:.1f}x less per input token than `gpt-4o`, so under high concurrent load the "
        f"cost gap between the two candidates widens linearly with query volume, making the cheaper model "
        f"materially more scalable for a high-traffic customer support deployment if its quality gap "
        f"stays small.\n"
    )

    print(table)
    append_to_report(table)
    return winner


if __name__ == "__main__":
    main()
