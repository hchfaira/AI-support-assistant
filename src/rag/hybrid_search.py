"""Hybrid retrieval: fuse BM25 lexical search with dense vector search.

Fusion strategy: Reciprocal Rank Fusion (RRF), a simple and robust method for
combining ranked lists from heterogeneous retrievers without needing to
calibrate score scales between BM25 and cosine similarity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.embeddings.inference import EmbeddingModel, cosine_similarity_matrix
from src.rag.retriever import BM25Index


@dataclass
class RetrievedDocument:
    doc_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class HybridRetriever:
    """Combines BM25 and dense vector search with Reciprocal Rank Fusion."""

    def __init__(
        self,
        doc_ids: list[str],
        texts: list[str],
        metadatas: list[dict] | None = None,
        embedding_model: EmbeddingModel | None = None,
        rrf_k: int = 60,
    ):
        self.doc_ids = doc_ids
        self.texts = texts
        self.metadatas = metadatas or [{} for _ in doc_ids]
        self.rrf_k = rrf_k

        self.bm25_index = BM25Index.build(doc_ids, texts)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.doc_vectors = self.embedding_model.encode(texts)
        self._doc_lookup = dict(zip(doc_ids, texts))
        self._meta_lookup = dict(zip(doc_ids, self.metadatas))

    def _vector_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_vector = self.embedding_model.encode([query])
        similarities = cosine_similarity_matrix(query_vector, self.doc_vectors)[0]
        ranked_idx = np.argsort(-similarities)[:top_k]
        return [(self.doc_ids[i], float(similarities[i])) for i in ranked_idx]

    def search(self, query: str, top_k: int = 10, candidate_pool: int = 20) -> list[RetrievedDocument]:
        bm25_ranked = self.bm25_index.search(query, top_k=candidate_pool)
        vector_ranked = self._vector_search(query, top_k=candidate_pool)

        rrf_scores: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(bm25_ranked, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (doc_id, _) in enumerate(vector_ranked, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked_ids = sorted(rrf_scores, key=lambda d: -rrf_scores[d])[:top_k]
        return [
            RetrievedDocument(
                doc_id=doc_id,
                text=self._doc_lookup[doc_id],
                score=rrf_scores[doc_id],
                metadata=self._meta_lookup.get(doc_id, {}),
            )
            for doc_id in ranked_ids
        ]

    def vector_only_search(self, query: str, top_k: int = 10) -> list[RetrievedDocument]:
        """Baseline: dense vector search only (no BM25, no fusion)."""
        ranked = self._vector_search(query, top_k=top_k)
        return [
            RetrievedDocument(
                doc_id=doc_id,
                text=self._doc_lookup[doc_id],
                score=score,
                metadata=self._meta_lookup.get(doc_id, {}),
            )
            for doc_id, score in ranked
        ]
