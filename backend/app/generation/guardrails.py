"""Guardrails — enforced in code, never left to the model to self-police
(Requirement 7, design.md Property 4). Out-of-scope and account-data checks
run before retrieval; the groundedness check runs after retrieval but before
generation."""
from __future__ import annotations

import re
from typing import Optional

from ..retrieval.hybrid import RetrievedChunk

OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions about Wells Fargo's own products and documents — that question is "
    "out of scope for this assistant."
)
ACCOUNT_DATA_MESSAGE = (
    "I can only answer general policy questions from Wells Fargo's published documents — I can't "
    "access personal account data like your balance, transactions, or account number."
)
NO_INFO_MESSAGE = "I don't have enough information in the provided documents to answer that confidently."

# Other banks/issuers a customer might ask about by name — keyword match is
# intentionally simple and fast; it runs before any retrieval or LLM call.
COMPETITOR_BANKS = [
    "chase", "jpmorgan", "jp morgan", "bank of america", "bofa", "citibank", "citi ",
    "capital one", "us bank", "u.s. bank", "pnc bank", "td bank", "truist", "regions bank",
    "hsbc", "barclays", "discover card", "american express", "amex", "navy federal",
    "ally bank", "sofi", "charles schwab bank", "goldman sachs bank", "santander",
]

ACCOUNT_DATA_PATTERNS = [
    r"\bmy (?:current )?(?:account )?balance\b",
    r"\bmy account number\b",
    r"\bmy (?:card|debit|credit) card number\b",
    r"\bmy (?:recent |latest )?transactions?\b",
    r"\bhow much (?:money )?(?:do i have|is in my account)\b",
    r"\bmy routing number\b",
    r"\bmy statement\b",
]
_ACCOUNT_DATA_RE = re.compile("|".join(ACCOUNT_DATA_PATTERNS), re.IGNORECASE)

# Small talk and questions about the conversation itself, not about Wells
# Fargo products/documents — these should be answered from session memory
# (build_smalltalk_prompt) instead of going through document retrieval,
# which would never find anything relevant and would refuse regardless of
# what the user told the bot earlier.
CONVERSATIONAL_PATTERNS = [
    r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)\b",
    r"\bmy name is\b",
    r"\bcall me\b",
    r"\bwhat(?:'s| is) my name\b",
    r"\bwho am i\b",
    r"\bdo you remember\b",
    r"\bwhat did i (?:just )?(?:ask|say)\b",
    r"\bwhat have we (?:talked about|discussed)\b",
    r"\bhow are you\b",
]
_CONVERSATIONAL_RE = re.compile("|".join(CONVERSATIONAL_PATTERNS), re.IGNORECASE)


def check_conversational(message: str) -> bool:
    return bool(_CONVERSATIONAL_RE.search(message))


def check_out_of_scope(message: str) -> bool:
    lowered = message.lower()
    return any(bank in lowered for bank in COMPETITOR_BANKS)


def check_account_data_request(message: str) -> bool:
    return bool(_ACCOUNT_DATA_RE.search(message))


def check_groundedness(chunks: list[RetrievedChunk], min_rerank_score: Optional[float] = None) -> bool:
    """True if there's enough grounded evidence to answer; False -> refuse."""
    if min_rerank_score is None:
        from ..config import settings

        min_rerank_score = settings.min_rerank_score
    if not chunks:
        return False
    return chunks[0].score >= min_rerank_score
