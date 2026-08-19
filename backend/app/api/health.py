"""GET /health (Requirement 14.2)."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}
