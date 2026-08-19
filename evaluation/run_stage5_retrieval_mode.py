"""Stage 5 ablation — retrieval mode: dense vs sparse vs hybrid.

Broken out by query_type (keyword vs semantic) — the actual proof for
whether hybrid is worth the added complexity (Requirement 10.1,
EVALUATION_METHODOLOGY.md Stage 5).
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
# Settings aren't overwritten with ablation winners until task 6.5, after every
# stage has run) — NOT read from `settings`, so re-running this script later (e.g.
# to add a metric column) reproduces the same numbers instead of silently picking
# up the now-finalized config.
BASE_PARSER = "pymupdf"
BASE_CHUNK_STRATEGY, BASE_CHUNK_SIZE, BASE_CHUNK_OVERLAP = "recursive", 500, 50

DOCS = load_all_source_documents(parser=BASE_PARSER)
CHUNKS = chunk_documents(DOCS, strategy=BASE_CHUNK_STRATEGY, size=BASE_CHUNK_SIZE, overlap=BASE_CHUNK_OVERLAP)

INDEX = HybridIndex(collection_name="wf_stage5_hybrid_shared")
INDEX.build(CHUNKS, force=True)

MODES = ["dense", "sparse", "hybrid"]
BASE_FUSION_METHOD = "rrf"  # Stage 6 hadn't run yet at the point Stage 5 was measured


def main():
    questions = load_eval_questions()
    rows = {}
    for mode in MODES:
        rows[mode] = evaluate_retrieval(
            lambda q, m=mode: INDEX.search(q, k=10, mode=m, fusion_method=BASE_FUSION_METHOD), questions
        )

    table = "## Stage 5 — Retrieval mode: dense vs sparse vs hybrid\n\n"
    table += "| Mode | R@1 (all) | R@3 (all) | R@10 (all) | R@3 (keyword) | R@3 (semantic) | MRR@10 | NDCG@3 |\n"
    table += "|---|---|---|---|---|---|---|---|\n"
    label = {"dense": "Dense only", "sparse": "Sparse (BM25) only", "hybrid": "Hybrid"}
    for mode in MODES:
        o, kw, sem = rows[mode]["overall"], rows[mode]["keyword"], rows[mode]["semantic"]
        table += (
            f"| {label[mode]} | {fmt(o['r_at_1'])} | {fmt(o['r_at_3'])} | {fmt(o['r_at_10'])} | "
            f"{fmt(kw.get('r_at_3', 0.0))} | {fmt(sem.get('r_at_3', 0.0))} | {fmt(o['mrr_at_10'])} | {fmt(o['ndcg_at_3'])} |\n"
        )

    winner_mode = max(MODES, key=lambda m: (rows[m]["overall"]["r_at_3"], rows[m]["overall"]["mrr_at_10"]))
    w = rows[winner_mode]["overall"]
    table += (
        f"\n**Winner: `{winner_mode}`** — R@3={fmt(w['r_at_3'])}, MRR@10={fmt(w['mrr_at_10'])} overall. "
        f"Keyword R@3={fmt(rows[winner_mode]['keyword'].get('r_at_3', 0.0))} vs semantic "
        f"R@3={fmt(rows[winner_mode]['semantic'].get('r_at_3', 0.0))} shows "
        f"{'a consistent split across query types' if winner_mode == 'hybrid' else 'this mode has an edge for this corpus'}.\n"
    )

    print(table)
    append_to_report(table)
    return winner_mode


if __name__ == "__main__":
    main()
