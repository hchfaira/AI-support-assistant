"""Data/embedding drift detection for the retrieval corpus and incoming
queries: flags when the live query distribution diverges from the
distribution the retriever/reranker were evaluated on, so the team knows when
to refresh embeddings, retrain the reranker, or re-run evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DriftReport:
    mean_cosine_shift: float
    is_drifting: bool
    threshold: float


def embedding_centroid_drift(
    reference_vectors: np.ndarray,
    current_vectors: np.ndarray,
    threshold: float = 0.15,
) -> DriftReport:
    """Compare the centroid of `current_vectors` (e.g. this week's incoming
    query embeddings) against `reference_vectors` (the embeddings used during
    the last evaluation run). A large cosine distance between centroids
    suggests the query distribution has shifted and retrieval/reranker
    evaluation should be re-run.
    """

    def _centroid(vectors: np.ndarray) -> np.ndarray:
        centroid = vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    ref_centroid = _centroid(reference_vectors)
    cur_centroid = _centroid(current_vectors)
    cosine_similarity = float(np.dot(ref_centroid, cur_centroid))
    cosine_shift = 1.0 - cosine_similarity

    return DriftReport(
        mean_cosine_shift=cosine_shift,
        is_drifting=cosine_shift > threshold,
        threshold=threshold,
    )


def retrieval_score_drift(
    reference_scores: list[float],
    current_scores: list[float],
    threshold: float = 0.2,
) -> DriftReport:
    """Compare mean top-1 retrieval similarity scores between a reference
    window and the current window; a drop suggests the knowledge base no
    longer covers incoming queries well (new product areas, new ticket types).
    """
    ref_mean = float(np.mean(reference_scores)) if reference_scores else 0.0
    cur_mean = float(np.mean(current_scores)) if current_scores else 0.0
    shift = ref_mean - cur_mean

    return DriftReport(
        mean_cosine_shift=shift,
        is_drifting=shift > threshold,
        threshold=threshold,
    )
