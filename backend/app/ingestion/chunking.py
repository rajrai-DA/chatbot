"""Chunking: strategy/size/overlap are all config-driven (Requirement 2.2) so
Stage 2 of the ablation suite can sweep them without touching this code."""
from __future__ import annotations

import logging
import re
from typing import Optional

from .loaders import Document

logger = logging.getLogger(__name__)

Chunk = Document  # a chunk is just a smaller Document — same text+metadata shape

# A markdown table row/separator line, e.g. "|**Account**|**Fee**|" or "|---|---|".
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")

# Beyond this many data rows under one separator, PDF-to-markdown output is
# usually a merged-cell fee list (first column blank except where the
# category changes) rather than a uniform table — pairing every row with the
# first row above the separator would tag it with an unrelated label, so we
# fall back to ordinary size-based chunking instead.
_MAX_TABLE_ROWS_FOR_HEADER_PAIRING = 8


def _table_row_chunks(lines: list[str], size: int, overlap: int, splitter_fn) -> list[str]:
    """One chunk per data row, each paired with the header + separator line.
    Size-based splitting cuts a table's header from the row holding the
    actual fee amount, leaving the LLM a table it can't read (see
    EVALUATION_REPORT.md follow-up: wrong refusals on fee questions whose
    answer was retrieved but split across chunk boundaries)."""
    if len(lines) < 3 or not _TABLE_SEP_RE.match(lines[1]):
        return splitter_fn("\n".join(lines), size, overlap)
    header, sep = lines[0], lines[1]
    rows = [row for row in lines[2:] if row.strip()]
    if len(rows) > _MAX_TABLE_ROWS_FOR_HEADER_PAIRING:
        return splitter_fn("\n".join(lines), size, overlap)
    return [f"{header}\n{sep}\n{row}" for row in rows]


def _split_table_aware(text: str, size: int, overlap: int, splitter_fn) -> list[str]:
    """Route markdown table lines to `_table_row_chunks` (kept intact per
    row) and everything else through `splitter_fn` as before."""
    lines = text.split("\n")
    pieces: list[str] = []
    prose_buf: list[str] = []
    table_buf: list[str] = []

    def flush_prose():
        if prose_buf:
            pieces.extend(splitter_fn("\n".join(prose_buf), size, overlap))
            prose_buf.clear()

    def flush_table():
        if table_buf:
            pieces.extend(_table_row_chunks(table_buf, size, overlap, splitter_fn))
            table_buf.clear()

    for line in lines:
        if _TABLE_ROW_RE.match(line):
            flush_prose()
            table_buf.append(line)
        else:
            flush_table()
            prose_buf.append(line)
    flush_table()
    flush_prose()
    return pieces


def _fixed_size_split(text: str, size: int, overlap: int) -> list[str]:
    step = max(size - overlap, 1)
    pieces = []
    for start in range(0, len(text), step):
        piece = text[start:start + size]
        if piece.strip():
            pieces.append(piece)
        if start + size >= len(text):
            break
    return pieces


def _recursive_split(text: str, size: int, overlap: int) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return [p for p in splitter.split_text(text) if p.strip()]


def _semantic_split(text: str, size: int, overlap: int) -> list[str]:
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings

        splitter = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-small"))
        pieces = [p for p in splitter.split_text(text) if p.strip()]
        if pieces:
            return pieces
    except Exception as exc:  # pragma: no cover - optional dependency / API failure
        logger.warning("semantic chunking unavailable (%s) — falling back to recursive", exc)
    return _recursive_split(text, size, overlap)


_STRATEGIES = {"fixed": _fixed_size_split, "recursive": _recursive_split, "semantic": _semantic_split}


def chunk_documents(
    documents: list[Document],
    strategy: Optional[str] = None,
    size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[Chunk]:
    """Split parsed documents into overlapping chunks, inheriting source/page
    metadata on every chunk (Requirement 2.1). Parse-failed/empty documents
    are skipped — they carry no text to chunk."""
    if strategy is None or size is None or overlap is None:
        from ..config import settings

        strategy = strategy or settings.chunk_strategy
        size = size if size is not None else settings.chunk_size
        overlap = overlap if overlap is not None else settings.chunk_overlap

    splitter_fn = _STRATEGIES.get(strategy)
    if splitter_fn is None:
        raise ValueError(f"Unknown chunk strategy: {strategy!r} — expected one of {sorted(_STRATEGIES)}")

    chunks: list[Chunk] = []
    for doc in documents:
        if not doc.text.strip():
            continue
        for piece in _split_table_aware(doc.text, size, overlap, splitter_fn):
            chunks.append(Chunk(text=piece, metadata=dict(doc.metadata)))
    return chunks
