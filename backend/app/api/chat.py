"""POST /chat (Requirement 14.1)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from openai import APIError, APITimeoutError
from pydantic import BaseModel

router = APIRouter()

_chatbot = None


def get_chatbot():
    global _chatbot
    if _chatbot is None:
        from ..generation.chatbot import ProductionRAGChatbot

        _chatbot = ProductionRAGChatbot()
        _chatbot.ingest()
    return _chatbot


class ChatRequest(BaseModel):
    session_id: str
    message: str


class Citation(BaseModel):
    document: str
    page: int | str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    try:
        result = get_chatbot().answer(request.session_id, request.message)
    except (APIError, APITimeoutError):
        raise HTTPException(status_code=502, detail="The AI service is temporarily unavailable. Please try again.")

    return ChatResponse(
        answer=result.answer,
        citations=[Citation(document=c.document, page=c.page) for c in result.citations],
        refused=result.refused,
    )
