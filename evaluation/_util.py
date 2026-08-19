"""Shared helpers for the run_stageN_*.py ablation scripts — not a stage
itself. Keeps each stage script focused on what it varies."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
CHATBOT_DIR = EVAL_DIR.parent
BACKEND_DIR = CHATBOT_DIR / "backend"
REPORT_PATH = CHATBOT_DIR / "EVALUATION_REPORT.md"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from metrics import mean_mrr_at_10, mean_ndcg_at_3, mean_recall_at_k  # noqa: E402


def load_eval_questions(include_negative: bool = False) -> list[dict]:
    with open(EVAL_DIR / "eval_questions.json") as f:
        questions = json.load(f)
    if not include_negative:
        questions = [q for q in questions if q.get("ground_truth_source")]
    return questions


def as_result_dicts(retrieved_chunks) -> list[dict]:
    return [{"source": c.source, "page": c.page, "score": c.score, "text": c.text} for c in retrieved_chunks]


def evaluate_retrieval(search_fn, questions: list[dict], k: int = 10) -> dict:
    """`search_fn(query) -> list[RetrievedChunk]`. Runs every question through
    it once and computes R@1/R@3/R@10/MRR@10/NDCG@3, overall and broken out
    by query_type (Requirement 9.3 / Stage 5's keyword-vs-semantic split)."""
    all_retrieved = []
    ground_truths = []
    query_types = []
    latencies = []

    for q in questions:
        t0 = time.perf_counter()
        results = as_result_dicts(search_fn(q["question"]))
        latencies.append((time.perf_counter() - t0) * 1000)
        all_retrieved.append(results)
        ground_truths.append((q["ground_truth_source"], q["ground_truth_page"]))
        query_types.append(q.get("query_type", "unknown"))

    def _metrics_for(indices):
        retrieved_subset = [all_retrieved[i] for i in indices]
        gt_subset = [ground_truths[i] for i in indices]
        if not retrieved_subset:
            return {}
        return {
            "r_at_1": mean_recall_at_k(retrieved_subset, gt_subset, 1),
            "r_at_3": mean_recall_at_k(retrieved_subset, gt_subset, 3),
            "r_at_10": mean_recall_at_k(retrieved_subset, gt_subset, 10),
            "mrr_at_10": mean_mrr_at_10(retrieved_subset, gt_subset),
            "ndcg_at_3": mean_ndcg_at_3(retrieved_subset, gt_subset),
        }

    all_idx = list(range(len(questions)))
    keyword_idx = [i for i, t in enumerate(query_types) if t == "keyword"]
    semantic_idx = [i for i, t in enumerate(query_types) if t == "semantic"]

    return {
        "overall": _metrics_for(all_idx),
        "keyword": _metrics_for(keyword_idx),
        "semantic": _metrics_for(semantic_idx),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "n_questions": len(questions),
    }


def append_to_report(markdown: str) -> None:
    with open(REPORT_PATH, "a") as f:
        f.write(markdown.rstrip() + "\n\n")


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)
