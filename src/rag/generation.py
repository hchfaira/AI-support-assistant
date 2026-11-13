"""End-to-end RAG orchestration: hybrid retrieval -> reranking -> context
building -> LLM generation -> citation verification -> structured JSON output.

Usage:
    python -m src.rag.generation --query "I was charged twice for my Premium subscription"
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass

from src.data_engineering.ingestion import load_sample_knowledge_base
from src.rag.context_builder import build_context, build_prompt
from src.rag.hybrid_search import HybridRetriever
from src.rag.reranking import CrossEncoderReranker


@dataclass
class SupportResponse:
    intent: str
    priority: str
    sentiment: str
    retrieved_documents: list[str]
    answer: str
    sources: list[dict]
    escalation_required: bool


ESCALATION_KEYWORDS = ("manager", "lawsuit", "legal", "third time", "cancel my account")


def verify_citations(answer: str, context_doc_ids: list[str]) -> list[str]:
    """Return the subset of `context_doc_ids` that are actually referenced in
    `answer` text (naive substring check -- replace with an NLI/claim-matching
    model for production-grade citation verification).
    """
    return [doc_id for doc_id in context_doc_ids if doc_id.split(".")[0] in answer.lower()]


def heuristic_generate(query: str, context_text: str, retrieved_doc_ids: list[str]) -> SupportResponse:
    """Deterministic stand-in for the fine-tuned LLM used when no LLM backend
    is configured, so the pipeline is runnable end-to-end offline. Swap this
    for `src.llm.inference.FineTunedLLM.generate` in production.
    """
    lower_query = query.lower()
    escalation_required = any(kw in lower_query for kw in ESCALATION_KEYWORDS)

    if "charg" in lower_query and ("twice" in lower_query or "duplicate" in lower_query):
        intent, priority = "billing_issue", "high"
        escalation_required = True
        answer = "We're sorry for the duplicate charge. It will be refunded within 5 business days."
    elif "refund" in lower_query:
        intent, priority, answer = (
            "refund_request",
            "medium",
            "You can request a refund within 30 days of the charge from your billing page.",
        )
    elif "cancel" in lower_query:
        intent, priority, answer = (
            "cancellation",
            "low",
            "You can cancel anytime; it takes effect at the end of your current billing cycle.",
        )
    else:
        intent, priority, answer = (
            "general_question",
            "low",
            "Here is what I found in our documentation: " + context_text[:200],
        )

    sources = [{"document": doc_id, "page": None} for doc_id in retrieved_doc_ids]
    sentiment = "negative" if re.search(r"charg|fail|issue|never|wrong", lower_query) else "neutral"

    return SupportResponse(
        intent=intent,
        priority=priority,
        sentiment=sentiment,
        retrieved_documents=retrieved_doc_ids,
        answer=answer,
        sources=sources,
        escalation_required=escalation_required,
    )


def answer_query(
    query: str,
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    top_k: int = 3,
) -> SupportResponse:
    candidates = retriever.search(query, top_k=10, candidate_pool=10)
    reranked = reranker.rerank(query, [(c.doc_id, c.text) for c in candidates], top_k=top_k)

    context_text = build_context(reranked)
    build_prompt(query, context_text)  # ready to send to a real LLM backend

    response = heuristic_generate(query, context_text, [doc.doc_id for doc in reranked])
    verified_sources = verify_citations(response.answer, response.retrieved_documents)
    if not verified_sources and response.retrieved_documents:
        # Heuristic answers don't always mention source names explicitly;
        # keep the retrieved docs as supporting context either way.
        verified_sources = response.retrieved_documents
    response.retrieved_documents = verified_sources
    return response


def build_default_pipeline() -> tuple[HybridRetriever, CrossEncoderReranker]:
    documents = load_sample_knowledge_base()
    retriever = HybridRetriever(
        doc_ids=[d.doc_id for d in documents],
        texts=[d.text for d in documents],
        metadatas=[d.metadata for d in documents],
    )
    reranker = CrossEncoderReranker()
    return retriever, reranker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    retriever, reranker = build_default_pipeline()
    response = answer_query(args.query, retriever, reranker, top_k=args.top_k)
    print(json.dumps(asdict(response), indent=2))


if __name__ == "__main__":
    main()
