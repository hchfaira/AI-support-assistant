"""Build a grounded context block from reranked documents for LLM generation."""
from __future__ import annotations

from src.reranker.inference import RerankedDocument


def build_context(documents: list[RerankedDocument], max_chars: int = 2000) -> str:
    """Concatenate top documents into a single context block, each tagged with
    its source id so the LLM (and the citation verifier) can reference them.
    """
    parts = []
    used_chars = 0
    for doc in documents:
        block = f"[Source: {doc.doc_id}]\n{doc.text}\n"
        if used_chars + len(block) > max_chars:
            break
        parts.append(block)
        used_chars += len(block)
    return "\n".join(parts)


def build_prompt(query: str, context: str) -> str:
    """Build the instruction-tuned prompt passed to the LLM."""
    return (
        "You are a customer support assistant. Answer the customer's question "
        "using ONLY the information in the context below. If the context does "
        "not contain the answer, say you don't have enough information and "
        "set escalation_required to true. Respond with a single JSON object "
        'matching this schema: {"intent": str, "priority": "low"|"medium"|"high", '
        '"answer": str, "escalation_required": bool}.\n\n'
        f"Context:\n{context}\n\n"
        f"Customer question: {query}\n\n"
        "JSON response:"
    )
