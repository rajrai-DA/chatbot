"""Stage 2 ablation — chunking strategy × size × overlap.

Sweeps strategy (fixed/recursive) × size (300/500/800) × overlap (0/50/100),
holding parsing/embedding/retrieval fixed at Stage 1's winner and hybrid
defaults (Requirement 10.1, EVALUATION_METHODOLOGY.md Stage 2).

Semantic chunking is reported separately (one row, at the default 500/50
"size/overlap" label) since it doesn't take a size/overlap parameter.
"""
from __future__ import annotations

import sys
from pathlib import Path

from _util import append_to_report, evaluate_retrieval, fmt, load_eval_questions

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings  # noqa: E402
from app.ingestion.chunking import chunk_documents  # noqa: E402
from app.ingestion.loaders import load_all_source_documents  # noqa: E402
from app.retrieval.hybrid import HybridIndex  # noqa: E402

# Stage 1 winner (parser) is read from Settings, which defaults to pymupdf
# pending Stage 1's actual result (task 6.5 finalizes this later).
DOCS = load_all_source_documents(parser=settings.parser)

SWEEP = (
    [("fixed", size, overlap) for size in (300, 500, 800) for overlap in (0, 50, 100)]
    + [("recursive", size, overlap) for size in (300, 500, 800) for overlap in (0, 50, 100)]
    + [("semantic", 500, 50)]
)


def run_for_config(strategy: str, size: int, overlap: int) -> dict:
    chunks = chunk_documents(DOCS, strategy=strategy, size=size, overlap=overlap)
    collection = f"wf_stage2_{strategy}_{size}_{overlap}"
    index = HybridIndex(collection_name=collection)
    index.build(chunks, force=True)

    questions = load_eval_questions()
    metrics = evaluate_retrieval(lambda q: index.search(q, k=10, mode="hybrid"), questions)["overall"]

    return {"strategy": strategy, "size": size, "overlap": overlap, "n_chunks": len(chunks), **metrics}


def main():
    rows = [run_for_config(*cfg) for cfg in SWEEP]

    table = "## Stage 2 — Chunking strategy\n\n"
    table += "| Strategy | Chunk size | Overlap | Chunks | R@1 | R@3 | R@10 | MRR@10 | NDCG@3 |\n"
    table += "|---|---|---|---|---|---|---|---|---|\n"
    for r in rows:
        table += (
            f"| {r['strategy']} | {r['size']} | {r['overlap']} | {r['n_chunks']} | "
            f"{fmt(r['r_at_1'])} | {fmt(r['r_at_3'])} | {fmt(r['r_at_10'])} | {fmt(r['mrr_at_10'])} | {fmt(r['ndcg_at_3'])} |\n"
        )

    winner = max(rows, key=lambda r: (r["r_at_3"], r["mrr_at_10"]))
    table += (
        f"\n**Winner: `{winner['strategy']}` at size={winner['size']}, overlap={winner['overlap']}** "
        f"— R@3={fmt(winner['r_at_3'])}, MRR@10={fmt(winner['mrr_at_10'])}, the best combination among "
        f"{len(rows)} configurations swept.\n"
    )

    print(table)
    append_to_report(table)
    return winner


if __name__ == "__main__":
    main()
