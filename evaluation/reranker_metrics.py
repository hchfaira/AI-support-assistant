"""Reranker-focused evaluation CLI: quantify the lift a cross-encoder reranker
provides over the hybrid retriever's raw ranking (Recall@K, MRR@10, NDCG@10,
MAP), printed as a before/after comparison table.

Usage:
    python -m evaluation.reranker_metrics
"""
from __future__ import annotations

import json
from pathlib import Path

from src.data_engineering.ingestion import load_sample_knowledge_base
from src.rag.hybrid_search import HybridRetriever
from src.reranker.evaluation import evaluate_with_without_reranker
from src.reranker.inference import CrossEncoderReranker


def main() -> None:
    documents = load_sample_knowledge_base()
    tickets = json.loads(Path("data/sample/eval_tickets.json").read_text(encoding="utf-8"))
    queries = [t["query"] for t in tickets]
    relevant = [t["relevant_docs"] for t in tickets]

    retriever = HybridRetriever(
        doc_ids=[d.doc_id for d in documents], texts=[d.text for d in documents]
    )
    reranker = CrossEncoderReranker()

    retrieval_only, with_reranker = evaluate_with_without_reranker(
        retriever, reranker, queries, relevant
    )

    print(f"{'metric':<15}{'retrieval_only':<18}{'with_reranker':<18}delta")
    for field in ("recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10", "map_score"):
        before, after = getattr(retrieval_only, field), getattr(with_reranker, field)
        print(f"{field:<15}{before:<18.3f}{after:<18.3f}{after - before:+.3f}")


if __name__ == "__main__":
    main()
