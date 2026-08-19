"""End-to-end scoring (Requirement 11) — runs the full 25-question eval set
through the REAL, now-finalized `ProductionRAGChatbot` (the exact code path
production traffic uses, not a script-only pipeline), computes RAGAS +
DeepEval + numeric_fact_accuracy, and records the headline result in
EVALUATION_REPORT.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from _util import append_to_report, fmt, load_eval_questions
from genai_metrics import evaluate_deepeval, evaluate_ragas

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.generation.chatbot import ProductionRAGChatbot  # noqa: E402
from metrics import mean_mrr_at_10, mean_ndcg_at_3, mean_recall_at_k, numeric_fact_accuracy  # noqa: E402


def main():
    bot = ProductionRAGChatbot()
    print("Ingesting corpus with finalized Settings...")
    stats = bot.ingest(force=False)
    print(f"Indexed {stats['chunks']} chunks from {stats['documents']} documents.")

    questions = load_eval_questions(include_negative=True)

    items = []
    all_retrieved, ground_truths = [], []
    numeric_scores = []
    latencies = []
    refusal_log = []

    for q in questions:
        session_id = f"e2e-{q['id']}"
        t0 = time.perf_counter()
        result = bot.answer(session_id, q["question"])
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved_dicts = [{"source": c.document, "page": c.page} for c in result.citations]

        if q.get("ground_truth_source"):
            all_retrieved.append(retrieved_dicts)
            ground_truths.append((q["ground_truth_source"], q["ground_truth_page"]))

            context_texts = [c.text for c in result.chunks] or [""]

            items.append({
                "question": q["question"], "answer": result.answer, "contexts": context_texts,
                "reference": q.get("ideal_answer", ""),
            })
            numeric_scores.append(numeric_fact_accuracy(result.answer, context_texts))
        else:
            # Negative controls (out-of-scope / account-data) — correctness is
            # "did it refuse", logged for ACCEPTANCE_TESTS.md, not retrieval-scored.
            refusal_log.append({"question": q["question"], "refused": result.refused, "answer": result.answer})

    retrieval_metrics = {
        "r_at_1": mean_recall_at_k(all_retrieved, ground_truths, 1),
        "r_at_3": mean_recall_at_k(all_retrieved, ground_truths, 3),
        "r_at_10": mean_recall_at_k(all_retrieved, ground_truths, 10),
        "mrr_at_10": mean_mrr_at_10(all_retrieved, ground_truths),
        "ndcg_at_3": mean_ndcg_at_3(all_retrieved, ground_truths),
    }

    print("Scoring with RAGAS...")
    ragas_scores = evaluate_ragas(items, model="gpt-4o-mini")
    print("Scoring with DeepEval...")
    deepeval_scores = evaluate_deepeval(items, model="gpt-4o-mini")

    avg_numeric = sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
    avg_latency = sum(latencies) / len(latencies)
    negative_controls_correct = sum(1 for r in refusal_log if r["refused"])

    section = "## End-to-End Scoring (Requirement 11) — Headline Result\n\n"
    section += (
        f"Full 25-question set run through the live `ProductionRAGChatbot` "
        f"(finalized `Settings`: parser=`{bot.settings.parser}`, chunking=`{bot.settings.chunk_strategy}`"
        f"/{bot.settings.chunk_size}/{bot.settings.chunk_overlap}, embeddings=`{bot.settings.embedding_model}`, "
        f"retrieval=`{bot.settings.retrieval_mode}`/`{bot.settings.fusion_method}`, "
        f"rerank={bot.settings.rerank_enabled}, LLM=`{bot.settings.llm_model}`).\n\n"
    )

    section += "### Retrieval (citations vs. ground truth)\n\n"
    section += "| R@1 | R@3 | R@10 | MRR@10 | NDCG@3 |\n|---|---|---|---|---|\n"
    section += (
        f"| {fmt(retrieval_metrics['r_at_1'])} | {fmt(retrieval_metrics['r_at_3'])} | "
        f"{fmt(retrieval_metrics['r_at_10'])} | {fmt(retrieval_metrics['mrr_at_10'])} | "
        f"{fmt(retrieval_metrics['ndcg_at_3'])} |\n\n"
    )

    section += "### Generation quality (RAGAS + DeepEval + custom metric)\n\n"
    section += (
        "| Faithfulness (RAGAS) | Answer Relevancy (RAGAS) | Context Precision (RAGAS) | "
        "Context Recall (RAGAS) | Hallucination (DeepEval) | G-Eval Strict Grounding (DeepEval) | "
        "Numeric fact accuracy | Avg latency (ms) |\n|---|---|---|---|---|---|---|---|\n"
    )
    section += (
        f"| {fmt(ragas_scores['faithfulness'])} | {fmt(ragas_scores['answer_relevancy'])} | "
        f"{fmt(ragas_scores['context_precision'])} | {fmt(ragas_scores['context_recall'])} | "
        f"{fmt(deepeval_scores['hallucination'])} | {fmt(deepeval_scores['g_eval_strict_grounding'])} | "
        f"{fmt(avg_numeric)} | {fmt(avg_latency)} |\n\n"
    )

    section += "### Negative controls (out-of-scope / account-data)\n\n"
    section += f"{negative_controls_correct}/{len(refusal_log)} correctly refused.\n\n"
    for r in refusal_log:
        section += f"- **{r['question']}** — refused={r['refused']}: \"{r['answer']}\"\n"
    section += "\n"

    print(section)
    append_to_report(section)


if __name__ == "__main__":
    main()
