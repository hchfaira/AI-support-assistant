"""FastAPI gateway exposing the RAG support assistant pipeline.

Run locally with:
    uvicorn src.api.main:app --reload
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from src.api.schemas import HealthResponse, TicketRequest, TicketResponse
from src.rag.generation import answer_query, build_default_pipeline

app = FastAPI(
    title="Enterprise AI Support Assistant",
    description="Hybrid-search + reranking + fine-tuned LLM customer support API.",
    version="0.1.0",
)

_retriever, _reranker = build_default_pipeline()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        embedding_backend="dense" if _retriever.embedding_model.is_real_backend else "tfidf-fallback",
        reranker_backend="cross-encoder" if _reranker.is_real_backend else "lexical-fallback",
    )


@app.post("/v1/tickets", response_model=TicketResponse)
def resolve_ticket(request: TicketRequest) -> TicketResponse:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")

    response = answer_query(request.query, _retriever, _reranker, top_k=request.top_k)
    return TicketResponse(**asdict(response))
