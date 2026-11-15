from src.data_engineering.ingestion import load_sample_knowledge_base
from src.rag.hybrid_search import HybridRetriever


def _build_retriever() -> HybridRetriever:
    documents = load_sample_knowledge_base()
    return HybridRetriever(
        doc_ids=[d.doc_id for d in documents],
        texts=[d.text for d in documents],
        metadatas=[d.metadata for d in documents],
    )


def test_hybrid_search_finds_relevant_billing_document():
    retriever = _build_retriever()
    results = retriever.search("I was charged twice for my subscription", top_k=3)
    doc_ids = [r.doc_id for r in results]
    assert "billing_policy.pdf" in doc_ids or "duplicate_payment_procedure.pdf" in doc_ids


def test_hybrid_search_returns_at_most_top_k():
    retriever = _build_retriever()
    results = retriever.search("refund", top_k=2)
    assert len(results) <= 2


def test_vector_only_search_returns_scored_results():
    retriever = _build_retriever()
    results = retriever.vector_only_search("password reset", top_k=5)
    assert len(results) <= 5
    assert all(hasattr(r, "score") for r in results)
