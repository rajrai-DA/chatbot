"""Verifies the 5 official acceptance questions (../../REQUIREMENT.md §6) route
to the expected guardrail/answer path (Requirement 12, design.md Testing Strategy).
Retrieval and the LLM are mocked — this is a routing test, not a quality test;
retrieval quality is covered by evaluation/run_stageN_*.py."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.generation.chatbot import ProductionRAGChatbot
from app.retrieval.hybrid import RetrievedChunk


def _grounded_chatbot():
    """A chatbot whose retrieval/rerank always return one strong, on-topic chunk
    and whose LLM returns a canned cited answer — for exercising the
    grounded-answer path without live retrieval or API calls."""
    strong_chunk = RetrievedChunk(
        text="Everyday Checking has a $15 monthly service fee, waived by meeting one of several options.",
        source="WellsFargo_Consumer_Account_Fees_Info.pdf", page=4, score=5.0, rank=1,
    )

    index = MagicMock()
    index.search.return_value = [strong_chunk]

    reranker = MagicMock()
    reranker.rerank.return_value = [strong_chunk]

    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="The monthly service fee is $15. [1]")

    bot = ProductionRAGChatbot(index=index, reranker=reranker, llm=llm)
    return bot


@pytest.fixture
def grounded_bot():
    return _grounded_chatbot()


@pytest.fixture
def bare_bot():
    """No mocked LLM call expected — out-of-scope/account-data guardrails must
    short-circuit before the LLM or retrieval are ever touched."""
    index = MagicMock()
    reranker = MagicMock()
    llm = MagicMock()
    return ProductionRAGChatbot(index=index, reranker=reranker, llm=llm)


def test_monthly_service_fee_question_answers_with_citation(grounded_bot):
    result = grounded_bot.answer("s1", "What's the monthly service fee on my checking account and how do I avoid it?")
    assert result.refused is False
    assert result.citations
    assert result.citations[0].document == "WellsFargo_Consumer_Account_Fees_Info.pdf"


def test_apr_question_answers_with_citation(grounded_bot):
    result = grounded_bot.answer("s2", "What's the APR on my Wells Fargo credit card?")
    assert result.refused is False
    assert result.citations


def test_account_closure_question_answers_with_citation(grounded_bot):
    result = grounded_bot.answer("s3", "Can Wells Fargo close my account without notice?")
    assert result.refused is False
    assert result.citations


def test_competitor_bank_question_is_out_of_scope(bare_bot):
    result = bare_bot.answer("s4", "What's Chase's overdraft fee?")
    assert result.refused is True
    assert "out of scope" in result.answer.lower()
    bare_bot.index.search.assert_not_called()
    bare_bot.llm.invoke.assert_not_called()


def test_personal_balance_question_is_account_data_refusal(bare_bot):
    result = bare_bot.answer("s5", "What's my current account balance?")
    assert result.refused is True
    assert "personal account data" in result.answer.lower() or "account data" in result.answer.lower()
    bare_bot.index.search.assert_not_called()
    bare_bot.llm.invoke.assert_not_called()


def test_weak_retrieval_refuses_before_llm_call():
    weak_chunk = RetrievedChunk(text="unrelated", source="x.pdf", page=1, score=-99.0, rank=1)
    index = MagicMock()
    index.search.return_value = [weak_chunk]
    reranker = MagicMock()
    reranker.rerank.return_value = [weak_chunk]
    llm = MagicMock()

    bot = ProductionRAGChatbot(index=index, reranker=reranker, llm=llm)
    result = bot.answer("s6", "What's the fee for something obscure not in the docs?")

    assert result.refused is True
    llm.invoke.assert_not_called()


def test_session_isolation():
    bot = _grounded_chatbot()
    bot.answer("session-a", "What's the monthly service fee?")
    assert bot.sessions.get_history("session-b") == []
    assert len(bot.sessions.get_history("session-a")) == 2


def test_conversational_message_bypasses_retrieval_and_recalls_name(bare_bot):
    bare_bot.llm.invoke.side_effect = [
        MagicMock(content="Hi Raj! How can I help?"),
        MagicMock(content="Your name is Raj."),
    ]

    r1 = bare_bot.answer("s7", "Hi, my name is Raj.")
    assert r1.refused is False
    assert r1.citations == []

    r2 = bare_bot.answer("s7", "What is my name?")
    assert r2.refused is False
    assert "Raj" in r2.answer

    bare_bot.index.search.assert_not_called()
    bare_bot.reranker.rerank.assert_not_called()
    assert len(bare_bot.sessions.get_history("s7")) == 4
