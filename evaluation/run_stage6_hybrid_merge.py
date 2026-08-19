"""Stage 6 ablation — hybrid merge method and weighting (RRF vs weighted-alpha).

Sweeps >=3 alpha values for weighted fusion (Requirement 10.1,
EVALUATION_METHODOLOGY.md Stage 6). Only meaningful if Stage 5 picked hybrid.
"""
from __future__ import annotations

import sys
from pathlib import Path

from _util import append_to_report, evaluate_retrieval, fmt, load_eval_questions

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.ingestion.chunking import chunk_documents  # noqa: E402
from app.ingestion.loaders import load_all_source_documents  # noqa: E402
from app.retrieval.hybrid import HybridIndex  # noqa: E402

# Pinned to the scaffold defaults Stages 1-8 held fixed while running (config.py's
# Settings aren't overwritten with ablation winners until task 6.5) — NOT read from
# `settings`, so re-running this script later reproduces the same numbers instead
# of silently picking up the now-finalized config.
BASE_PARSER = "pymupdf"
BASE_CHUNK_STRATEGY, BASE_CHUNK_SIZE, BASE_CHUNK_OVERLAP = "recursive", 500, 50

DOCS = load_all_source_documents(parser=BASE_PARSER)
CHUNKS = chunk_documents(DOCS, strategy=BASE_CHUNK_STRATEGY, size=BASE_CHUNK_SIZE, overlap=BASE_CHUNK_OVERLAP)

INDEX = HybridIndex(collection_name="wf_stage6_hybrid_shared")
INDEX.build(CHUNKS, force=True)

ALPHAS = [0.3, 0.5, 0.7]


def main():
    questions = load_eval_questions()
    rows = []

    rrf_metrics = evaluate_retrieval(
        lambda q: INDEX.search(q, k=10, mode="hybrid", fusion_method="rrf"), questions
    )["overall"]
    rows.append({"method": "RRF", "alpha": "-", **rrf_metrics})

    for alpha in ALPHAS:
        m = evaluate_retrieval(
            lambda q, a=alpha: INDEX.search(q, k=10, mode="hybrid", fusion_method="weighted", fusion_alpha=a),
            questions,
        )["overall"]
        rows.append({"method": "Weighted", "alpha": alpha, **m})

    table = "## Stage 6 — Hybrid merge method and weighting\n\n"
    table += "| Merge method | α (if weighted) | R@3 | R@10 | MRR@10 | NDCG@3 |\n"
    table += "|---|---|---|---|---|---|\n"
    for r in rows:
        table += (
            f"| {r['method']} | {r['alpha']} | {fmt(r['r_at_3'])} | {fmt(r['r_at_10'])} | "
            f"{fmt(r['mrr_at_10'])} | {fmt(r['ndcg_at_3'])} |\n"
        )

    winner = max(rows, key=lambda r: (r["r_at_3"], r["mrr_at_10"]))
    if winner["method"] == "RRF":
        justification = f"RRF (R@3={fmt(winner['r_at_3'])}) beat every weighted-α configuration tried."
    else:
        justification = (
            f"Weighted fusion at α={winner['alpha']} (R@3={fmt(winner['r_at_3'])}) beat RRF "
            f"(R@3={fmt(rows[0]['r_at_3'])}) and the other α values tried."
        )
    alpha_suffix = f" (α={winner['alpha']})" if winner["method"] == "Weighted" else ""
    table += f"\n**Winner: `{winner['method']}`{alpha_suffix}.** {justification}\n"

    print(table)
    append_to_report(table)
    return winner


if __name__ == "__main__":
    main()
