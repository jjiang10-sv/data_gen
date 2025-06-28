"""Embedding Manager for Starfish

This module provides embedding functionality using FAISS and SentenceTransformers
for semantic similarity search and data deduplication.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
from pathlib import Path
from starfish.common.logger import get_logger

logger = get_logger(__name__)


class EmbeddingManager:
    """
    Manages embeddings using SentenceTransformers and FAISS for efficient similarity search.

    Features:
    - Text embedding using pre-trained SentenceTransformers models
    - Fast similarity search using FAISS indexing
    - Persistent storage and loading of embeddings
    - Configurable similarity thresholds
    - Support for both exact and approximate nearest neighbor search
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",
        similarity_threshold: float = 0.85,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the EmbeddingManager.

        Args:
            model_name: SentenceTransformers model name or path
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            similarity_threshold: Threshold for determining similar items (0-1)
            cache_dir: Directory to cache embeddings and models
            device: Device to run model on ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.index_type = index_type
        self.similarity_threshold = similarity_threshold
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".starfish" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SentenceTransformer model
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = None
        self.metadata = []  # Store original texts and metadata
        self.id_to_index = {}  # Map custom IDs to FAISS indices

        logger.info(f"EmbeddingManager initialized with {model_name}, dim={self.embedding_dim}")

    def _create_index(self, dimension: int) -> faiss.Index:
        """Create a FAISS index based on the specified type."""
        if self.index_type == "flat":
            # L2 distance (Euclidean)
            index = faiss.IndexFlatL2(dimension)
        elif self.index_type == "ivf":
            # Inverted file index for faster approximate search
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World for very fast approximate search
            index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        return index

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Embed a list of texts using SentenceTransformers.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Normalize for cosine similarity
        )

        return embeddings.astype(np.float32)

    def add_texts(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> List[int]:
        """
        Add texts to the embedding index.

        Args:
            texts: List of texts to add
            metadata: Optional metadata for each text
            ids: Optional custom IDs for each text

        Returns:
            List of internal indices assigned to the texts
        """
        if not texts:
            return []

        # Generate embeddings
        embeddings = self.embed_texts(texts)

        # Initialize index if needed
        if self.index is None:
            self.index = self._create_index(self.embedding_dim)
            if self.index_type == "ivf":
                # Train the IVF index
                if len(embeddings) >= 100:  # Need at least as many points as clusters
                    self.index.train(embeddings)
                else:
                    logger.warning("Not enough data to train IVF index, using flat index instead")
                    self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings)

        # Store metadata
        if metadata is None:
            metadata = [{"text": text} for text in texts]
        else:
            # Ensure metadata includes the original text
            for i, meta in enumerate(metadata):
                if "text" not in meta:
                    meta["text"] = texts[i]

        self.metadata.extend(metadata)

        # Handle custom IDs
        indices = list(range(start_idx, start_idx + len(texts)))
        if ids:
            for i, custom_id in enumerate(ids):
                self.id_to_index[custom_id] = indices[i]

        logger.info(f"Added {len(texts)} texts to index. Total: {self.index.ntotal}")
        return indices

    def search_similar(self, query_text: str, k: int = 5, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for similar texts in the index.

        Args:
            query_text: Text to search for
            k: Number of similar items to return
            threshold: Similarity threshold (overrides default)

        Returns:
            List of dictionaries containing similar items with scores and metadata
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty or not initialized")
            return []

        # Embed query
        query_embedding = self.embed_texts([query_text], show_progress=False)

        # Search
        if self.index_type == "ivf" and hasattr(self.index, "nprobe"):
            self.index.nprobe = min(10, self.index.nlist)  # Search in 10 clusters

        scores, indices = self.index.search(query_embedding, k)

        # Convert L2 distances to cosine similarities
        # Since embeddings are normalized, L2 distance relates to cosine similarity
        similarities = 1 - (scores[0] / 2)  # Convert L2 to cosine similarity

        # Filter by threshold
        threshold = threshold or self.similarity_threshold
        results = []

        for idx, similarity in zip(indices[0], similarities):
            if idx != -1 and similarity >= threshold:  # -1 indicates no match found
                result = {
                    "index": int(idx),
                    "similarity": float(similarity),
                    "metadata": self.metadata[idx].copy() if idx < len(self.metadata) else {},
                    "text": self.metadata[idx].get("text", "") if idx < len(self.metadata) else "",
                }
                results.append(result)

        logger.debug(f"Found {len(results)} similar items for query (threshold={threshold})")
        return results

    def find_duplicates(self, texts: List[str], threshold: Optional[float] = None) -> List[List[int]]:
        """
        Find groups of duplicate/similar texts.

        Args:
            texts: List of texts to check for duplicates
            threshold: Similarity threshold for considering items duplicates

        Returns:
            List of lists, where each inner list contains indices of similar texts
        """
        threshold = threshold or self.similarity_threshold

        if not texts:
            return []

        # Embed all texts
        embeddings = self.embed_texts(texts, show_progress=True)

        # Create temporary index for comparison
        temp_index = faiss.IndexFlatL2(self.embedding_dim)
        temp_index.add(embeddings)

        # Find similar items
        duplicate_groups = []
        processed = set()

        for i, embedding in enumerate(embeddings):
            if i in processed:
                continue

            # Search for similar items
            query_embedding = embedding.reshape(1, -1)
            scores, indices = temp_index.search(query_embedding, len(texts))

            # Convert to similarities and filter
            similarities = 1 - (scores[0] / 2)
            similar_indices = []

            for idx, similarity in zip(indices[0], similarities):
                if similarity >= threshold and idx not in processed:
                    similar_indices.append(idx)
                    processed.add(idx)

            if len(similar_indices) > 1:
                duplicate_groups.append(similar_indices)

        logger.info(f"Found {len(duplicate_groups)} groups of duplicates")
        return duplicate_groups

    def save_index(self, filepath: str) -> None:
        """Save the FAISS index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(filepath.with_suffix(".faiss")))

        # Save metadata and configuration
        metadata_file = filepath.with_suffix(".pkl")
        with open(metadata_file, "wb") as f:
            pickle.dump(
                {
                    "metadata": self.metadata,
                    "id_to_index": self.id_to_index,
                    "model_name": self.model_name,
                    "index_type": self.index_type,
                    "similarity_threshold": self.similarity_threshold,
                    "embedding_dim": self.embedding_dim,
                },
                f,
            )

        logger.info(f"Saved index to {filepath}")

    def load_index(self, filepath: str) -> None:
        """Load a FAISS index and metadata from disk."""
        filepath = Path(filepath)

        # Load FAISS index
        index_file = filepath.with_suffix(".faiss")
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        self.index = faiss.read_index(str(index_file))

        # Load metadata and configuration
        metadata_file = filepath.with_suffix(".pkl")
        if metadata_file.exists():
            with open(metadata_file, "rb") as f:
                data = pickle.load(f)
                self.metadata = data.get("metadata", [])
                self.id_to_index = data.get("id_to_index", {})
                # Verify model compatibility
                saved_model = data.get("model_name", self.model_name)
                if saved_model != self.model_name:
                    logger.warning(f"Model mismatch: saved={saved_model}, current={self.model_name}")

        logger.info(f"Loaded index from {filepath} ({self.index.ntotal} items)")

    def get_embedding_by_id(self, custom_id: str) -> Optional[np.ndarray]:
        """Get embedding vector by custom ID."""
        if custom_id not in self.id_to_index:
            return None

        idx = self.id_to_index[custom_id]
        if self.index is None or idx >= self.index.ntotal:
            return None

        return self.index.reconstruct(idx)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        return {
            "model_name": self.model_name,
            "index_type": self.index_type,
            "embedding_dimension": self.embedding_dim,
            "total_items": self.index.ntotal if self.index else 0,
            "similarity_threshold": self.similarity_threshold,
            "metadata_count": len(self.metadata),
            "custom_ids_count": len(self.id_to_index),
        }
