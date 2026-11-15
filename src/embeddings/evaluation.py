"""Evaluate embedding models for retrieval quality using standard IR metrics:
Recall@K, MRR@10 and NDCG@10, plus mean encoding latency.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.embeddings.inference import EmbeddingModel, cosine_similarity_matrix


@dataclass
class RetrievalEvalResult:
    model_name: str
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    mean_encode_latency_ms: float


def _recall_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    return len(top_k & relevant) / len(relevant)


def _reciprocal_rank(ranked_doc_ids: list[str], relevant: set[str], k: int = 10) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += rel / np.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_embedding_model(
    model_name: str,
    corpus_doc_ids: list[str],
    corpus_texts: list[str],
    queries: list[str],
    relevant_doc_ids: list[list[str]],
    k: int = 10,
) -> RetrievalEvalResult:
    """Encode `corpus_texts` and `queries`, rank by cosine similarity, and
    compute Recall@5, Recall@10, MRR@10 and NDCG@10 against `relevant_doc_ids`.
    """
    model = EmbeddingModel(model_name)

    start = time.perf_counter()
    doc_vectors = model.encode(corpus_texts)
    query_vectors = model.encode(queries)
    elapsed_ms = (time.perf_counter() - start) * 1000
    mean_latency = elapsed_ms / max(len(queries) + len(corpus_texts), 1)

    similarities = cosine_similarity_matrix(query_vectors, doc_vectors)

    recalls_5, recalls_10, mrrs, ndcgs = [], [], [], []
    for row, relevant in zip(similarities, relevant_doc_ids):
        ranking = [corpus_doc_ids[i] for i in np.argsort(-row)]
        relevant_set = set(relevant)
        recalls_5.append(_recall_at_k(ranking, relevant_set, 5))
        recalls_10.append(_recall_at_k(ranking, relevant_set, k))
        mrrs.append(_reciprocal_rank(ranking, relevant_set, k))
        ndcgs.append(_ndcg_at_k(ranking, relevant_set, k))

    return RetrievalEvalResult(
        model_name=model_name,
        recall_at_5=float(np.mean(recalls_5)),
        recall_at_10=float(np.mean(recalls_10)),
        mrr_at_10=float(np.mean(mrrs)),
        ndcg_at_10=float(np.mean(ndcgs)),
        mean_encode_latency_ms=float(mean_latency),
    )
