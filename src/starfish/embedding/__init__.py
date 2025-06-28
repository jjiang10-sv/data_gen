"""Starfish Embedding Module

This module provides embedding functionality using FAISS and SentenceTransformers
for semantic similarity and data deduplication during dataset generation.
"""

from .embedding_manager import EmbeddingManager
from .similarity_checker import SimilarityChecker
from .deduplicator import DataDeduplicator

__all__ = ["EmbeddingManager", "SimilarityChecker", "DataDeduplicator"]
