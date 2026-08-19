"""Stage 1 ablation — parsing strategy (pypdf vs pdfplumber vs pymupdf).

Reports % of pages with clean extracted text and downstream R@3 when each
parser's output is fed through the same chunking/embedding/retrieval config
(Requirement 10.1, EVALUATION_METHODOLOGY.md Stage 1).
"""
from __future__ import annotations

import sys
from pathlib import Path

from _util import append_to_report, evaluate_retrieval, fmt, load_eval_questions

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings  # noqa: E402
from app.ingestion.chunking import chunk_documents  # noqa: E402
from app.ingestion.loaders import load_document, load_xbrl  # noqa: E402
from app.retrieval.hybrid import HybridIndex  # noqa: E402

PDF_NAMES = [
    "WellsFargo_Deposit_Account_Agreement.pdf",
    "WellsFargo_Consumer_Account_Fees_Info.pdf",
    "WellsFargo_Credit_Card_Agreement.pdf",
]

# Held fixed while parsing is what's under test
CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP = "recursive", 500, 50


def run_for_parser(parser: str) -> dict:
    docs = []
    for name in PDF_NAMES:
        docs.extend(load_document(settings.source_data_dir / name, parser=parser))
    docs.extend(load_xbrl(settings.source_data_dir / "WellsFargo_Financial_Data_XBRL.xml"))

    clean_pages = sum(1 for d in docs if not d.metadata.get("parse_failed", False))
    clean_pct = 100.0 * clean_pages / len(docs) if docs else 0.0

    chunks = chunk_documents(docs, strategy=CHUNK_STRATEGY, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    index = HybridIndex(collection_name=f"wf_stage1_{parser}")
    index.build(chunks, force=True)

    questions = load_eval_questions()
    metrics = evaluate_retrieval(lambda q: index.search(q, k=10, mode="hybrid"), questions)

    return {
        "parser": parser,
        "n_pages": len(docs),
        "clean_pct": clean_pct,
        "n_chunks": len(chunks),
        "r_at_3": metrics["overall"]["r_at_3"],
    }


def main():
    rows = [run_for_parser(p) for p in ["pypdf", "pdfplumber", "pymupdf"]]

    table = "## Stage 1 — Parsing strategy\n\n"
    table += "| Parser | Clean text % | Pages | Chunks | R@3 (downstream) |\n"
    table += "|---|---|---|---|---|\n"
    for r in rows:
        table += f"| {r['parser']} | {fmt(r['clean_pct'])}% | {r['n_pages']} | {r['n_chunks']} | {fmt(r['r_at_3'])} |\n"

    winner = max(rows, key=lambda r: (r["clean_pct"], r["r_at_3"]))
    table += (
        f"\n**Winner: `{winner['parser']}`** — {fmt(winner['clean_pct'])}% clean text and "
        f"R@3={fmt(winner['r_at_3'])}, the best combination of extraction quality and downstream "
        f"retrieval among the three parsers tested.\n"
    )

    print(table)
    append_to_report(table)
    return winner


if __name__ == "__main__":
    main()
