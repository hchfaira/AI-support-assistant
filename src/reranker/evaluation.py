"""Reranker evaluation: compare retrieval quality with and without reranking.

Reuses the IR metrics from `src.embeddings.evaluation` (Recall@K, MRR@10,
NDCG@10, MAP) to quantify the lift a cross-encoder reranker provides over the
hybrid retriever's initial ranking.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.reranker.inference import CrossEncoderReranker
from src.rag.hybrid_search import HybridRetriever


@dataclass
class RerankerEvalResult:
    stage: str  # "retrieval_only" | "with_reranker"
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    map_score: float


def _recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def _reciprocal_rank(ranking: list[str], relevant: set[str], k: int = 10) -> float:
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(ranking: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(
        (1.0 if doc_id in relevant else 0.0) / np.log2(i + 2)
        for i, doc_id in enumerate(ranking[:k])
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def _average_precision(ranking: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits, precisions = 0, []
    for i, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0


def evaluate_with_without_reranker(
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    queries: list[str],
    relevant_doc_ids: list[list[str]],
    candidate_pool: int = 20,
    top_k: int = 10,
) -> tuple[RerankerEvalResult, RerankerEvalResult]:
    """Run hybrid retrieval for every query, then rerank, and report metrics
    for both stages so the reranker's contribution can be measured directly.
    """
    retrieval_rankings: list[list[str]] = []
    reranked_rankings: list[list[str]] = []

    for query in queries:
        candidates = retriever.search(query, top_k=candidate_pool, candidate_pool=candidate_pool)
        retrieval_rankings.append([c.doc_id for c in candidates])

        reranked = reranker.rerank(
            query, [(c.doc_id, c.text) for c in candidates], top_k=top_k
        )
        reranked_rankings.append([r.doc_id for r in reranked])

    def _aggregate(stage: str, rankings: list[list[str]]) -> RerankerEvalResult:
        recalls_5, recalls_10, mrrs, ndcgs, maps = [], [], [], [], []
        for ranking, relevant in zip(rankings, relevant_doc_ids):
            relevant_set = set(relevant)
            recalls_5.append(_recall_at_k(ranking, relevant_set, 5))
            recalls_10.append(_recall_at_k(ranking, relevant_set, top_k))
            mrrs.append(_reciprocal_rank(ranking, relevant_set, top_k))
            ndcgs.append(_ndcg_at_k(ranking, relevant_set, top_k))
            maps.append(_average_precision(ranking, relevant_set))
        return RerankerEvalResult(
            stage=stage,
            recall_at_5=float(np.mean(recalls_5)),
            recall_at_10=float(np.mean(recalls_10)),
            mrr_at_10=float(np.mean(mrrs)),
            ndcg_at_10=float(np.mean(ndcgs)),
            map_score=float(np.mean(maps)),
        )

    return (
        _aggregate("retrieval_only", retrieval_rankings),
        _aggregate("with_reranker", reranked_rankings),
    )
