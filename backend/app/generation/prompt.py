"""Grounded system prompt + citation formatting (Requirement 6.1/6.2).
Forbids outside knowledge, requires numbered [n] citations, adapted from
notebook 01/13's grounded-answer convention."""
from __future__ import annotations

from ..retrieval.hybrid import RetrievedChunk

ANSWER_PROMPT = """You are a careful Wells Fargo customer support assistant answering questions ONLY using \
the numbered sources below, which come from official Wells Fargo documents.
- Cite sources inline like [1], [2] after every claim they support.
- Never use outside knowledge or general banking knowledge not present in the sources below.
- Never state a specific fee amount, APR, or rule that is not present in the sources.
- If the sources don't contain the answer, say so plainly: "I don't have enough information in the \
provided documents to answer that." Never guess.

Sources:
{sources}

Conversation so far:
{history}

Question: {question}

Answer (with inline citations):"""

NO_INFO_ANSWER = "I don't have enough information in the provided documents to answer that confidently."

# Greetings, self-introductions, and questions about the conversation itself
# (e.g. "what's my name") aren't document questions — they route around
# retrieval entirely (see check_conversational in guardrails.py) and are
# answered from session history alone, never from outside/document knowledge.
SMALLTALK_PROMPT = """You are a friendly Wells Fargo customer support assistant. The user's message \
below is conversational — a greeting, them sharing their name, or a question about this \
conversation itself — not a question about Wells Fargo products, fees, or policies.
- Answer briefly and naturally using ONLY the conversation history below.
- Never state a Wells Fargo fee, rate, or policy detail here, even if you know it — those must \
always come from retrieved documents, never from memory or general knowledge. If the user's \
message also contains a banking question, don't answer that part; ask them to ask it directly.
- If the conversation history doesn't contain what they're asking (e.g. they ask for a name they \
never gave you), say so plainly. Never guess.

Conversation so far:
{history}

Message: {message}

Response:"""


def format_sources(chunks: list[RetrievedChunk]) -> str:
    return "\n".join(f"[{i + 1}] ({c.source}, p.{c.page}) {c.text}" for i, c in enumerate(chunks))


def build_prompt(question: str, chunks: list[RetrievedChunk], history_text: str = "(none yet)") -> str:
    return ANSWER_PROMPT.format(sources=format_sources(chunks), history=history_text, question=question)


def build_smalltalk_prompt(message: str, history_text: str = "(none yet)") -> str:
    return SMALLTALK_PROMPT.format(history=history_text, message=message)
