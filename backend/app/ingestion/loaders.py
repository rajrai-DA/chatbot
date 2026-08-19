"""Document ingestion: swappable PDF parsers + XBRL fact ingestion.

`load_document()` is the single entry point the rest of the pipeline calls —
it dispatches on file extension and, for PDFs, on `Settings.parser` (or an
explicit override), so Stage 1 of the ablation suite can call it 3 ways over
the same files and diff clean-text %% (Requirement 1.4).
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

INSTANCE_NS = "{http://www.xbrl.org/2003/instance}"
LINK_NS = "{http://www.xbrl.org/2003/linkbase}"
XLINK_NS = "{http://www.w3.org/1999/xlink}"
STANDARD_LABEL_ROLE = "http://www.xbrl.org/2003/role/label"
FALLBACK_LABEL_ROLES = (
    "http://www.xbrl.org/2003/role/terseLabel",
    "http://www.xbrl.org/2003/role/verboseLabel",
)


@dataclass
class Document:
    """One retrievable unit of source text with citation metadata."""

    text: str
    metadata: dict = field(default_factory=dict)


# ── PDF parsing (Stage 1 ablation: pypdf / pdfplumber / pymupdf) ──

def _load_pypdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            logger.warning("pypdf: no extractable text on %s page %d", path.name, i)
        docs.append(Document(text=text, metadata={"source": path.name, "page": i, "parse_failed": not text}))
    return docs


def _load_pdfplumber(path: Path) -> list[Document]:
    import pdfplumber

    docs = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                logger.warning("pdfplumber: no extractable text on %s page %d", path.name, i)
            docs.append(Document(text=text, metadata={"source": path.name, "page": i, "parse_failed": not text}))
    return docs


def _load_pymupdf(path: Path) -> list[Document]:
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    docs = []
    for i, page in enumerate(pages, start=1):
        text = (page.get("text") or "").strip()
        if not text:
            logger.warning("pymupdf: no extractable text on %s page %d", path.name, i)
        docs.append(Document(text=text, metadata={"source": path.name, "page": i, "parse_failed": not text}))
    return docs


_PDF_PARSERS = {"pypdf": _load_pypdf, "pdfplumber": _load_pdfplumber, "pymupdf": _load_pymupdf}


def load_pdf(path: str | Path, parser: Optional[Literal["pypdf", "pdfplumber", "pymupdf"]] = None) -> list[Document]:
    """Parse a PDF into one Document per page. Pages with no extractable text
    are kept (not dropped) with `metadata["parse_failed"] = True` so Stage 1
    can measure clean-text %% (Requirement 1.3)."""
    path = Path(path)
    if parser is None:
        from ..config import settings

        parser = settings.parser
    loader = _PDF_PARSERS.get(parser)
    if loader is None:
        raise ValueError(f"Unknown parser: {parser!r} — expected one of {sorted(_PDF_PARSERS)}")
    return loader(path)


# ── XBRL ingestion (Requirement 1.2) ──

def _namespace_map(path: Path) -> dict[str, str]:
    """URI -> declared prefix, read from the document's own xmlns declarations."""
    ns_map: dict[str, str] = {}
    for _, (prefix, uri) in ET.iterparse(str(path), events=("start-ns",)):
        ns_map[uri] = prefix
    return ns_map


def load_xbrl_labels(labels_path: str | Path) -> dict[str, str]:
    """Parse the labels linkbase into {concept_qname: human_label}, e.g.
    {"us-gaap:Revenues": "Revenues"}. Prefers the standard `label` role,
    falling back to terse/verbose when a concept has no standard label."""
    tree = ET.parse(str(labels_path))
    root = tree.getroot()

    loc_to_concept: dict[str, str] = {}
    for loc in root.iter(f"{LINK_NS}loc"):
        loc_label = loc.get(f"{XLINK_NS}label")
        href = loc.get(f"{XLINK_NS}href")
        if not loc_label or not href:
            continue
        fragment = href.rsplit("#", 1)[-1]
        concept = fragment.replace("_", ":", 1)
        loc_to_concept[loc_label] = concept

    label_text_by_id_role: dict[tuple[str, str], str] = {}
    for label_el in root.iter(f"{LINK_NS}label"):
        lbl_label = label_el.get(f"{XLINK_NS}label")
        role = label_el.get(f"{XLINK_NS}role", "")
        text = (label_el.text or "").strip()
        if lbl_label and text:
            label_text_by_id_role[(lbl_label, role)] = text

    concept_to_label: dict[str, str] = {}
    for arc in root.iter(f"{LINK_NS}labelArc"):
        frm = arc.get(f"{XLINK_NS}from")
        to = arc.get(f"{XLINK_NS}to")
        concept = loc_to_concept.get(frm)
        if not concept or concept in concept_to_label:
            continue
        text = label_text_by_id_role.get((to, STANDARD_LABEL_ROLE))
        if not text:
            for role in FALLBACK_LABEL_ROLES:
                text = label_text_by_id_role.get((to, role))
                if text:
                    break
        if text:
            concept_to_label[concept] = text

    return concept_to_label


def _format_fact_value(raw_value: str, unit_ref: Optional[str]) -> str:
    raw_value = raw_value.strip()
    try:
        num = float(raw_value)
    except ValueError:
        return raw_value

    if unit_ref == "usd":
        return f"${int(num):,}" if num == int(num) else f"${num:,.2f}"
    if unit_ref in ("usdPerShare", "usdPerLoan"):
        return f"${num:,.2f}"
    if unit_ref == "shares":
        return f"{int(num):,} shares"
    if unit_ref in (None, "number", "pure"):
        return raw_value
    return f"{raw_value} {unit_ref}"


def _period_phrase(start: Optional[str], end: Optional[str], instant: Optional[str]) -> str:
    if instant:
        return f"as of {instant}"
    if start and end:
        return f"on {start}" if start == end else f"for the period {start} to {end}"
    return "for the reported period"


def load_xbrl_facts(data_path: str | Path, labels_path: str | Path) -> list[Document]:
    """Walk `data_path` facts into synthetic sentence Documents, e.g.
    "Revenues was $32,846,000,000 for the period 2025-01-01 to 2025-12-31."

    Restricted to facts whose context carries no dimensional segment (i.e.
    whole-company reported figures, not broken out by product/geography axis)
    — this keeps the synthetic corpus to the ~1-2k entity-level facts a
    customer-facing question would actually target, instead of exploding into
    the full ~7.6k dimensionally-qualified fact set.

    Also skips `*TextBlock` concepts (and any other unusually long fact
    value) — these carry entire footnote/disclosure sections as HTML, not a
    reportable figure, and would otherwise turn one XBRL fact into thousands
    of chunks that swamp the corpus with disclosure boilerplate.
    """
    data_path = Path(data_path)
    labels_path = Path(labels_path)
    labels = load_xbrl_labels(labels_path)
    ns_map = _namespace_map(data_path)

    tree = ET.parse(str(data_path))
    root = tree.getroot()

    contexts: dict[str, dict] = {}
    for c in root.findall(f"{INSTANCE_NS}context"):
        entity = c.find(f"{INSTANCE_NS}entity")
        segment = entity.find(f"{INSTANCE_NS}segment") if entity is not None else None
        if segment is not None:
            continue
        period = c.find(f"{INSTANCE_NS}period")
        instant = period.find(f"{INSTANCE_NS}instant")
        start = period.find(f"{INSTANCE_NS}startDate")
        end = period.find(f"{INSTANCE_NS}endDate")
        contexts[c.get("id")] = {
            "instant": instant.text if instant is not None else None,
            "start": start.text if start is not None else None,
            "end": end.text if end is not None else None,
        }

    documents: list[Document] = []
    for el in root:
        tag = el.tag
        if tag.endswith("}context") or tag.endswith("}unit") or tag.endswith("}schemaRef"):
            continue
        ctx_ref = el.get("contextRef")
        if ctx_ref is None or ctx_ref not in contexts:
            continue  # dimensionally-qualified or footnote fact — out of scope here

        uri, _, local = tag[1:].partition("}")
        if local.endswith("TextBlock"):
            continue  # full disclosure/footnote HTML, not a reportable figure
        prefix = ns_map.get(uri, uri)
        concept = f"{prefix}:{local}"
        label = labels.get(concept, local)

        raw_value = (el.text or "").strip()
        if not raw_value:
            logger.warning("XBRL: empty value for fact %s (context %s)", concept, ctx_ref)
            continue
        if len(raw_value) > 300:
            logger.warning("XBRL: skipping unusually long fact value for %s (%d chars)", concept, len(raw_value))
            continue

        value_text = _format_fact_value(raw_value, el.get("unitRef"))
        period = contexts[ctx_ref]
        sentence = f"{label} was {value_text} {_period_phrase(period['start'], period['end'], period['instant'])}."
        documents.append(Document(
            text=sentence,
            metadata={"source": data_path.name, "page": concept, "context": ctx_ref, "parse_failed": False},
        ))

    return documents


def load_xbrl(data_path: str | Path, labels_path: Optional[str | Path] = None) -> list[Document]:
    data_path = Path(data_path)
    if labels_path is None:
        labels_path = data_path.parent / "WellsFargo_Financial_Labels_XBRL.xml"
    return load_xbrl_facts(data_path, Path(labels_path))


# ── Unified entry point ──

def load_document(path: str | Path, parser: Optional[str] = None) -> list[Document]:
    """Dispatch on file extension: `.pdf` -> swappable parser, `.xml` -> XBRL facts."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path, parser=parser)
    if suffix == ".xml":
        return load_xbrl(path)
    raise ValueError(f"Unsupported file type: {suffix!r} — use .pdf or .xml")


def load_all_source_documents(source_dir: Optional[str | Path] = None, parser: Optional[str] = None) -> list[Document]:
    """Load all 3 Wells Fargo PDFs + the XBRL financial facts from `source_dir`."""
    if source_dir is None:
        from ..config import settings

        source_dir = settings.source_data_dir
    source_dir = Path(source_dir)

    pdf_names = [
        "WellsFargo_Deposit_Account_Agreement.pdf",
        "WellsFargo_Consumer_Account_Fees_Info.pdf",
        "WellsFargo_Credit_Card_Agreement.pdf",
    ]
    documents: list[Document] = []
    for name in pdf_names:
        documents.extend(load_pdf(source_dir / name, parser=parser))
    documents.extend(load_xbrl(source_dir / "WellsFargo_Financial_Data_XBRL.xml"))
    return documents
