"""GET /evaluate/report and POST /evaluate/acceptance — lets the frontend
dashboard show real numbers instead of a hand-copied snapshot.

/evaluate/report parses the actual EVALUATION_REPORT.md (the file every
run_stageN_*.py ablation script writes into) rather than duplicating its
numbers in a second, driftable place. It's cheap (file read + regex), so it's
safe to call on every dashboard load.

/evaluate/acceptance re-runs the 5 official acceptance questions
(REQUIREMENT.md §6 / ACCEPTANCE_TESTS.md) against the live ProductionRAGChatbot
right now. It's a handful of real LLM calls (~seconds, not free), so unlike
/evaluate/report it's only triggered on demand, never automatically."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from .chat import get_chatbot

router = APIRouter(prefix="/evaluate")

REPORT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "EVALUATION_REPORT.md"

# The 5 official acceptance questions (REQUIREMENT.md §6), with the expectation
# ACCEPTANCE_TESTS.md judges each one against: grounded questions should NOT be
# refused, the two negative controls SHOULD be refused.
ACCEPTANCE_QUESTIONS = [
    {
        "question": "What's the monthly service fee on my checking account and how do I avoid it?",
        "expect_refusal": False,
    },
    {"question": "What's the APR on my Wells Fargo credit card?", "expect_refusal": False},
    {"question": "Can Wells Fargo close my account without notice?", "expect_refusal": False},
    {"question": "What's Chase's overdraft fee?", "expect_refusal": True},
    {"question": "What's my current account balance?", "expect_refusal": True},
]


def _parse_md_table(block: str) -> list[dict]:
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return []
    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(dict(zip(headers, cells)))
    return rows


def _clean_markdown(s: str) -> str:
    s = re.sub(r"^\*\*Winner:\s*", "", s)
    return s.replace("**", "").replace("`", "").strip()


def _section(text: str, start_heading: str, next_heading_prefix: str) -> str:
    start = text.find(start_heading)
    if start == -1:
        return ""
    start += len(start_heading)
    end = text.find(next_heading_prefix, start)
    return text[start : end if end != -1 else None]


def parse_evaluation_report(text: str) -> dict:
    stages = []
    for m in re.finditer(r"^## (Stage \d+ — .+)$", text, re.MULTILINE):
        stage_title = m.group(1).strip()
        section_end = text.find("\n## ", m.end())
        section = text[m.end() : section_end if section_end != -1 else None]
        winner_match = re.search(r"\*\*Winner:.+", section)
        stages.append({"stage": stage_title, "winner": _clean_markdown(winner_match.group(0)) if winner_match else ""})

    e2e = _section(text, "## End-to-End Scoring (Requirement 11) — Headline Result", "\n## ")
    retrieval_block = _section(e2e, "### Retrieval (citations vs. ground truth)", "\n### ")
    generation_block = _section(e2e, "### Generation quality (RAGAS + DeepEval + custom metric)", "\n### ")
    retrieval_rows = _parse_md_table(retrieval_block)
    generation_rows = _parse_md_table(generation_block)

    neg_match = re.search(r"\d+/\d+ correctly refused", e2e)

    return {
        "stages": stages,
        "retrieval": retrieval_rows[0] if retrieval_rows else {},
        "generation": generation_rows[0] if generation_rows else {},
        "negative_controls": neg_match.group(0) if neg_match else None,
    }


class StageResult(BaseModel):
    stage: str
    winner: str


class ReportResponse(BaseModel):
    stages: list[StageResult]
    retrieval: dict[str, str]
    generation: dict[str, str]
    negative_controls: str | None


@router.get("/report", response_model=ReportResponse)
def get_report() -> ReportResponse:
    text = REPORT_PATH.read_text()
    return ReportResponse(**parse_evaluation_report(text))


class AcceptanceCitation(BaseModel):
    document: str
    page: int | str


class AcceptanceResult(BaseModel):
    question: str
    expect_refusal: bool
    refused: bool
    passed: bool
    answer: str
    citations: list[AcceptanceCitation]


class AcceptanceRunResponse(BaseModel):
    results: list[AcceptanceResult]
    pass_count: int
    total: int


@router.post("/acceptance", response_model=AcceptanceRunResponse)
def run_acceptance_tests() -> AcceptanceRunResponse:
    bot = get_chatbot()
    results = []
    for i, item in enumerate(ACCEPTANCE_QUESTIONS):
        session_id = f"acceptance-live-{i}"
        bot.sessions.clear(session_id)
        outcome = bot.answer(session_id, item["question"])
        passed = outcome.refused == item["expect_refusal"]
        results.append(
            AcceptanceResult(
                question=item["question"],
                expect_refusal=item["expect_refusal"],
                refused=outcome.refused,
                passed=passed,
                answer=outcome.answer,
                citations=[AcceptanceCitation(document=c.document, page=c.page) for c in outcome.citations],
            )
        )
    return AcceptanceRunResponse(
        results=results, pass_count=sum(1 for r in results if r.passed), total=len(results),
    )
