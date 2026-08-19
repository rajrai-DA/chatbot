"""Shared retrieval + generation metrics, imported by every run_stageN_*.py
script so all ablation stages measure identically (design.md).

Retrieval metrics follow EVALUATION_METHODOLOGY.md Part A exactly:
R@1, R@3, R@10 (Recall@k), MRR@10 (Mean Reciprocal Rank), NDCG@3.
"""
from __future__ import annotations

import math
import re
from typing import Optional, Sequence

_NUMERIC_PATTERN = re.compile(r"\$[\d,]+(?:\.\d+)?%?|\d[\d,]*(?:\.\d+)?%")


def _matches_ground_truth(result: dict, source: str, page) -> bool:
    return result.get("source") == source and str(result.get("page")) == str(page)


def recall_at_k(retrieved: Sequence[dict], source: str, page, k: int) -> int:
    """1 if the ground-truth (source, page) chunk appears in the top-k retrieved
    results for this query, else 0."""
    return int(any(_matches_ground_truth(r, source, page) for r in retrieved[:k]))


def reciprocal_rank(retrieved: Sequence[dict], source: str, page, k: int = 10) -> float:
    """1 / rank_of_first_correct_chunk within the top-k, else 0 (MRR@10 building block)."""
    for i, r in enumerate(retrieved[:k], start=1):
        if _matches_ground_truth(r, source, page):
            return 1.0 / i
    return 0.0


def _dcg(relevances: Sequence[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_3(retrieved: Sequence[dict], source: str, page, k: int = 3) -> float:
    """Binary-relevance NDCG@3: 1 for the ground-truth chunk, 0 otherwise,
    normalized against the ideal (best-possible) ranking."""
    relevances = [1.0 if _matches_ground_truth(r, source, page) else 0.0 for r in retrieved[:k]]
    dcg = _dcg(relevances)
    idcg = _dcg(sorted(relevances, reverse=True))
    return dcg / idcg if idcg > 0 else 0.0


def mean_recall_at_k(all_retrieved: Sequence[Sequence[dict]], ground_truths: Sequence[tuple], k: int) -> float:
    scores = [recall_at_k(r, src, page, k) for r, (src, page) in zip(all_retrieved, ground_truths)]
    return sum(scores) / len(scores) if scores else 0.0


def mean_mrr_at_10(all_retrieved: Sequence[Sequence[dict]], ground_truths: Sequence[tuple]) -> float:
    scores = [reciprocal_rank(r, src, page, k=10) for r, (src, page) in zip(all_retrieved, ground_truths)]
    return sum(scores) / len(scores) if scores else 0.0


def mean_ndcg_at_3(all_retrieved: Sequence[Sequence[dict]], ground_truths: Sequence[tuple]) -> float:
    scores = [ndcg_at_3(r, src, page, k=3) for r, (src, page) in zip(all_retrieved, ground_truths)]
    return sum(scores) / len(scores) if scores else 0.0


# ── Custom domain metric (Requirement 10.6) ──

def extract_numeric_facts(text: str) -> list[str]:
    """Pull dollar amounts, percentages, and bare numbers (APRs, fee counts,
    day counts) out of a generated answer for verbatim cross-checking."""
    return _NUMERIC_PATTERN.findall(text)


def numeric_fact_accuracy(answer: str, ground_truth_chunk: str | Sequence[str]) -> float:
    """Fraction of numeric facts (dollar amounts, percentages, APRs) in `answer`
    that appear verbatim in the cited source chunk(s) — this domain's most
    consequential failure mode is silently altering a fee or APR figure.
    Returns 1.0 when the answer makes no numeric claims (nothing to falsify).
    """
    source_text = ground_truth_chunk if isinstance(ground_truth_chunk, str) else "\n".join(ground_truth_chunk)
    facts = extract_numeric_facts(answer)
    if not facts:
        return 1.0
    verified = sum(1 for f in facts if f in source_text)
    return verified / len(facts)
