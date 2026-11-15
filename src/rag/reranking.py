"""Thin re-export so `src.rag` exposes a `reranking` module matching the
documented architecture, while the implementation lives in `src.reranker`
(kept there so it can also be trained/evaluated independently).
"""
from __future__ import annotations

from src.reranker.inference import CrossEncoderReranker, RerankedDocument

__all__ = ["CrossEncoderReranker", "RerankedDocument"]
