"""Pydantic request/response schemas for the support assistant API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    query: str = Field(..., description="The customer's support message.")
    top_k: int = Field(3, ge=1, le=10, description="Number of documents to use as context.")


class SourceCitation(BaseModel):
    document: str
    page: int | None = None


class TicketResponse(BaseModel):
    intent: str
    priority: Literal["low", "medium", "high"]
    sentiment: str
    retrieved_documents: list[str]
    answer: str
    sources: list[SourceCitation]
    escalation_required: bool


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    embedding_backend: str
    reranker_backend: str
