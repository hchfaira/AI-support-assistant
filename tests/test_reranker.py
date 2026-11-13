from src.reranker.inference import CrossEncoderReranker


def test_reranker_reorders_candidates_by_relevance():
    reranker = CrossEncoderReranker()
    query = "How can I request a refund?"
    candidates = [
        ("account_creation.pdf", "To create an account, provide an email and password."),
        ("refund_policy.pdf", "Refund requests must be submitted within 30 days of the charge."),
    ]
    results = reranker.rerank(query, candidates)
    assert results[0].doc_id == "refund_policy.pdf"


def test_reranker_respects_top_k():
    reranker = CrossEncoderReranker()
    candidates = [(f"doc{i}", f"text {i}") for i in range(5)]
    results = reranker.rerank("some query", candidates, top_k=2)
    assert len(results) == 2


def test_reranker_handles_empty_candidates():
    reranker = CrossEncoderReranker()
    assert reranker.rerank("query", []) == []
