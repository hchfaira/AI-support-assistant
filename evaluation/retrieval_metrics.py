"""CLI entry point: run the four-experiment comparison table described in the
project README (Baseline / Hybrid / Hybrid+Reranker / Hybrid+Reranker+FT-LLM)
against the bundled sample knowledge base and evaluation tickets.

Usage:
    python -m evaluation.retrieval_metrics
"""
from __future__ import annotations

import json
from pathlib import Path

from src.data_engineering.ingestion import load_sample_knowledge_base
from src.embeddings.evaluation import evaluate_embedding_model
from src.rag.hybrid_search import HybridRetriever
from src.reranker.evaluation import evaluate_with_without_reranker
from src.reranker.inference import CrossEncoderReranker


def load_eval_tickets(path: str | Path = "data/sample/eval_tickets.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_retrieval_only_vs_hybrid(
    documents: list | None = None, tickets: list[dict] | None = None
) -> dict:
    """Compare vector-only retrieval (Baseline) vs hybrid BM25+vector (Experiment 1).

    Defaults to the bundled sample knowledge base/eval tickets; pass
    `documents`/`tickets` (e.g. loaded from `data/processed/` via
    `src.data_engineering.support_dataset_builder`) to run this comparison
    against a different corpus.
    """
    documents = documents if documents is not None else load_sample_knowledge_base()
    tickets = tickets if tickets is not None else load_eval_tickets()
    queries = [t["query"] for t in tickets]
    relevant = [t["relevant_docs"] for t in tickets]

    retriever = HybridRetriever(
        doc_ids=[d.doc_id for d in documents],
        texts=[d.text for d in documents],
    )

    def _metrics(rankings: list[list[str]]) -> dict:
        from src.reranker.evaluation import (
            _average_precision,
            _ndcg_at_k,
            _reciprocal_rank,
            _recall_at_k,
        )

        recalls5, recalls10, mrrs, ndcgs = [], [], [], []
        for ranking, rel in zip(rankings, relevant):
            rel_set = set(rel)
            recalls5.append(_recall_at_k(ranking, rel_set, 5))
            recalls10.append(_recall_at_k(ranking, rel_set, 10))
            mrrs.append(_reciprocal_rank(ranking, rel_set))
            ndcgs.append(_ndcg_at_k(ranking, rel_set))
        n = len(rankings) or 1
        return {
            "recall@5": sum(recalls5) / n,
            "recall@10": sum(recalls10) / n,
            "mrr@10": sum(mrrs) / n,
            "ndcg@10": sum(ndcgs) / n,
        }

    vector_only_rankings = [
        [d.doc_id for d in retriever.vector_only_search(q, top_k=10)] for q in queries
    ]
    hybrid_rankings = [
        [d.doc_id for d in retriever.search(q, top_k=10, candidate_pool=10)] for q in queries
    ]

    return {
        "baseline_vector_only": _metrics(vector_only_rankings),
        "experiment1_hybrid": _metrics(hybrid_rankings),
    }


def run_full_comparison() -> dict:
    """Run all four experiments and return a single comparison table."""
    documents = load_sample_knowledge_base()
    tickets = load_eval_tickets()
    queries = [t["query"] for t in tickets]
    relevant = [t["relevant_docs"] for t in tickets]

    retriever = HybridRetriever(
        doc_ids=[d.doc_id for d in documents],
        texts=[d.text for d in documents],
    )
    reranker = CrossEncoderReranker()

    retrieval_only, with_reranker = evaluate_with_without_reranker(
        retriever, reranker, queries, relevant
    )

    retrieval_comparison = run_retrieval_only_vs_hybrid()
    retrieval_comparison["experiment2_hybrid_plus_reranker"] = {
        "recall@5": with_reranker.recall_at_5,
        "recall@10": with_reranker.recall_at_10,
        "mrr@10": with_reranker.mrr_at_10,
        "ndcg@10": with_reranker.ndcg_at_10,
        "map": with_reranker.map_score,
    }
    retrieval_comparison["note_experiment3"] = (
        "Experiment 3 (hybrid + reranker + fine-tuned LLM) is evaluated at the "
        "generation layer -- see src/llm/evaluation.py (Exact Match, ROUGE-L, "
        "BERTScore, format compliance) rather than at the retrieval layer."
    )
    return retrieval_comparison


if __name__ == "__main__":
    import pprint

    pprint.pprint(run_full_comparison())
