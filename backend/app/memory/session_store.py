"""Per-session conversation memory (Requirement 8). Sessions are strictly
isolated by `session_id` (design.md Property 6) — never global state. A
condense-question step rewrites follow-ups into standalone queries before
retrieval, and history is trimmed past a turn budget so token usage doesn't
grow unbounded (Requirement 8.3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChatTurn:
    role: str
    content: str


CONDENSE_PROMPT = """Given the recent conversation and a follow-up question, decide whether the \
follow-up depends on the conversation (uses a pronoun or implicit reference like "it", "that", \
"those", "the previous one") to make sense.

- If it DOES depend on the conversation, rewrite it as a standalone question that replaces the \
reference with the actual thing it refers to.
- If it does NOT depend on the conversation — including if it's simply a new question on a \
different topic — return it EXACTLY UNCHANGED. Do not "helpfully" relate an unrelated question \
back to the previous topic.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""


class SessionStore:
    """In-memory per-session_id turn history. Not process-shared/durable by
    design — session state living in a single backend process is sufficient
    for this app's scope (a durable store would be a drop-in swap here)."""

    def __init__(self, max_turns: Optional[int] = None):
        from ..config import settings

        self.max_turns = max_turns or settings.memory_max_turns
        self._sessions: dict[str, list[ChatTurn]] = {}

    def get_history(self, session_id: str) -> list[ChatTurn]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, role: str, content: str) -> None:
        history = self._sessions.setdefault(session_id, [])
        history.append(ChatTurn(role=role, content=content))
        if len(history) > self.max_turns * 2:
            del history[: len(history) - self.max_turns * 2]

    def history_text(self, session_id: str, last_n_turns: Optional[int] = None) -> str:
        history = self.get_history(session_id)
        n = last_n_turns or self.max_turns
        recent = history[-(n * 2):]
        return "\n".join(f"{t.role}: {t.content}" for t in recent) if recent else "(none yet)"

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def condense_question(question: str, history: list[ChatTurn], llm=None) -> str:
    if not history:
        return question
    if llm is None:
        from langchain_openai import ChatOpenAI

        from ..config import settings

        llm = ChatOpenAI(model=settings.llm_model, temperature=0)
    history_text = "\n".join(f"{t.role}: {t.content}" for t in history[-6:])
    prompt = CONDENSE_PROMPT.format(history=history_text, question=question)
    return llm.invoke(prompt).content.strip()
