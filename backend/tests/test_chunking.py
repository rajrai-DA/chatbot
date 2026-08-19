"""Verifies markdown table rows survive size-based chunking with their header
attached (fixes wrong "not enough information" refusals on fee-table
questions whose answer was retrieved but split across chunk boundaries), and
that large merged-cell listings fall back to plain chunking instead of
tagging every row with a misleading header."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import Document


def _chunk(text: str, size: int = 300, overlap: int = 0) -> list[str]:
    doc = Document(text=text, metadata={"source": "x.pdf", "page": 1})
    return [c.text for c in chunk_documents([doc], strategy="fixed", size=size, overlap=overlap)]


def test_small_table_pairs_every_row_with_its_header():
    table = (
        "# Fees\n\n"
        "|**Account**|**Monthly fee**|\n"
        "|---|---|\n"
        "|**Everyday Checking**|$15|\n"
        "|**Clear Access Banking**|$5|\n"
    )
    pieces = _chunk(table)
    everyday = next(p for p in pieces if "Everyday Checking" in p and "$15" in p)
    assert "**Account**" in everyday and "**Monthly fee**" in everyday

    clear_access = next(p for p in pieces if "Clear Access Banking" in p and "$5" in p)
    assert "**Account**" in clear_access and "**Monthly fee**" in clear_access


def test_large_merged_cell_table_falls_back_to_plain_chunking():
    rows = "\n".join(f"||Fee category {i}|${i} each|" for i in range(20))
    table = "||**Wells Fargo ATMs**|No fee|\n|---|---|---|\n" + rows
    pieces = _chunk(table, size=120, overlap=0)
    # None of the 20 unrelated rows should get the unrelated first row
    # ("Wells Fargo ATMs") stitched onto them as a fabricated header.
    mistagged = [p for p in pieces if "Fee category 10" in p and "Wells Fargo ATMs" in p]
    assert not mistagged
