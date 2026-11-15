"""Embedding model wrapper used by retrieval, chunking and evaluation.

Defaults to a small CPU-friendly Sentence-Transformers model
(`sentence-transformers/all-MiniLM-L6-v2`) so the rest of the pipeline is
runnable without a GPU. Swap `model_name` for a production model such as
`BAAI/bge-m3` or `intfloat/e5-base-v2` when running on GPU infrastructure.

If `sentence-transformers`/`torch` are not installed, falls back to a
deterministic TF-IDF based embedding so the pipeline still runs end-to-end in
constrained environments (CI, low-disk sandboxes). The fallback is clearly
not a substitute for a real dense encoder and should only be used for
smoke-testing the pipeline wiring.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


class EmbeddingModel:
    """Thin wrapper around a sentence embedding backend."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._backend = None
        self._tfidf_vectorizer = None
        try:
            from sentence_transformers import SentenceTransformer

            self._backend = SentenceTransformer(model_name)
        except Exception:
            # No internet / no torch available: fall back to TF-IDF so the
            # pipeline is still exercisable end-to-end.
            self._backend = None

    def encode(self, texts: Iterable[str], normalize: bool = True) -> np.ndarray:
        texts = list(texts)
        if self._backend is not None:
            vectors = self._backend.encode(
                texts, normalize_embeddings=normalize, show_progress_bar=False
            )
            return np.asarray(vectors)
        return self._tfidf_encode(texts, normalize=normalize)

    def _tfidf_encode(self, texts: list[str], normalize: bool) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer(max_features=512)
            matrix = self._tfidf_vectorizer.fit_transform(texts)
        else:
            matrix = self._tfidf_vectorizer.transform(texts)
        vectors = matrix.toarray()
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1e-8
            vectors = vectors / norms
        return vectors

    @property
    def is_real_backend(self) -> bool:
        """Whether a real dense encoder (not the TF-IDF fallback) is active."""
        return self._backend is not None


def cosine_similarity_matrix(query_vectors: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity between every query vector and every doc vector."""
    q = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-8)
    d = doc_vectors / (np.linalg.norm(doc_vectors, axis=1, keepdims=True) + 1e-8)
    return q @ d.T
