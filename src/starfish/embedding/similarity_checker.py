"""Similarity Checker for Data Generation

This module provides utilities for checking similarity between generated data points
to ensure diversity and quality in synthetic datasets.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from .embedding_manager import EmbeddingManager
from starfish.common.logger import get_logger

logger = get_logger(__name__)


class SimilarityChecker:
    """
    Checks similarity between data points during generation to ensure diversity.

    Features:
    - Real-time similarity checking during data generation
    - Configurable similarity thresholds
    - Support for different text fields in data structures
    - Batch processing for efficiency
    """

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        similarity_threshold: float = 0.85,
        text_fields: Optional[List[str]] = None,
        combine_fields: bool = True,
    ):
        """
        Initialize the SimilarityChecker.

        Args:
            embedding_manager: Pre-configured EmbeddingManager instance
            similarity_threshold: Threshold for considering items similar (0-1)
            text_fields: List of field names to extract text from data items
            combine_fields: Whether to combine multiple text fields into one
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.similarity_threshold = similarity_threshold
        self.text_fields = text_fields or ["text", "query", "question", "content", "prompt"]
        self.combine_fields = combine_fields

        logger.info(f"SimilarityChecker initialized with threshold={similarity_threshold}")

    def extract_text(self, data_item: Union[str, Dict[str, Any]]) -> str:
        """
        Extract text from a data item.

        Args:
            data_item: String or dictionary containing text data

        Returns:
            Extracted text string
        """
        if isinstance(data_item, str):
            return data_item

        if isinstance(data_item, dict):
            texts = []
            for field in self.text_fields:
                if field in data_item and data_item[field]:
                    texts.append(str(data_item[field]))

            if not texts:
                # Fallback: concatenate all string values
                texts = [str(v) for v in data_item.values() if isinstance(v, (str, int, float))]

            if self.combine_fields:
                return " ".join(texts)
            else:
                return texts[0] if texts else ""

        return str(data_item)

    def is_similar_to_existing(
        self, new_item: Union[str, Dict[str, Any]], existing_items: List[Union[str, Dict[str, Any]]], threshold: Optional[float] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if a new item is similar to any existing items.

        Args:
            new_item: New data item to check
            existing_items: List of existing items to compare against
            threshold: Custom similarity threshold

        Returns:
            Tuple of (is_similar: bool, most_similar_item: Dict or None)
        """
        if not existing_items:
            return False, None

        threshold = threshold or self.similarity_threshold
        new_text = self.extract_text(new_item)

        # Extract texts from existing items
        existing_texts = [self.extract_text(item) for item in existing_items]

        # Add existing texts to embedding manager if not already added
        # Note: This creates a temporary index for this comparison
        temp_indices = self.embedding_manager.add_texts(existing_texts)

        try:
            # Search for similar items
            similar_items = self.embedding_manager.search_similar(
                new_text,
                k=1,  # Just find the most similar
                threshold=threshold,
            )

            if similar_items:
                most_similar = similar_items[0]
                original_item = existing_items[most_similar["index"] - temp_indices[0]]
                return True, {"item": original_item, "similarity": most_similar["similarity"], "text": most_similar["text"]}

            return False, None

        finally:
            # Clean up temporary embeddings
            # Note: This is a simplified cleanup - in production you might want
            # to maintain separate indices or use a more sophisticated approach
            pass

    def filter_similar_items(
        self, items: List[Union[str, Dict[str, Any]]], threshold: Optional[float] = None, keep_first: bool = True
    ) -> Tuple[List[Union[str, Dict[str, Any]]], List[List[int]]]:
        """
        Filter out similar items from a list.

        Args:
            items: List of items to filter
            threshold: Custom similarity threshold
            keep_first: Whether to keep the first item in each similar group

        Returns:
            Tuple of (filtered_items, duplicate_groups)
        """
        if not items:
            return [], []

        threshold = threshold or self.similarity_threshold

        # Extract texts
        texts = [self.extract_text(item) for item in items]

        # Find duplicate groups
        duplicate_groups = self.embedding_manager.find_duplicates(texts, threshold)

        # Create set of indices to remove
        indices_to_remove = set()
        for group in duplicate_groups:
            if keep_first:
                # Remove all but the first item in each group
                indices_to_remove.update(group[1:])
            else:
                # Remove all items in the group
                indices_to_remove.update(group)

        # Filter items
        filtered_items = [item for i, item in enumerate(items) if i not in indices_to_remove]

        logger.info(f"Filtered {len(items)} items to {len(filtered_items)} (removed {len(indices_to_remove)} duplicates)")
        return filtered_items, duplicate_groups

    def check_diversity_batch(self, items: List[Union[str, Dict[str, Any]]], min_distance: float = 0.3) -> Dict[str, Any]:
        """
        Check diversity metrics for a batch of items.

        Args:
            items: List of items to analyze
            min_distance: Minimum desired distance between items

        Returns:
            Dictionary with diversity metrics
        """
        if not items:
            return {"diversity_score": 0, "avg_similarity": 0, "min_similarity": 0, "max_similarity": 0}

        if len(items) == 1:
            return {"diversity_score": 1.0, "avg_similarity": 0, "min_similarity": 0, "max_similarity": 0}

        # Extract texts and embed
        texts = [self.extract_text(item) for item in items]
        embeddings = self.embedding_manager.embed_texts(texts, show_progress=False)

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Cosine similarity for normalized embeddings
                similarity = float(embeddings[i].dot(embeddings[j]))
                similarities.append(similarity)

        if not similarities:
            return {"diversity_score": 1.0, "avg_similarity": 0, "min_similarity": 0, "max_similarity": 0}

        avg_similarity = sum(similarities) / len(similarities)
        min_similarity = min(similarities)
        max_similarity = max(similarities)

        # Diversity score: higher when average similarity is lower
        diversity_score = max(0, 1 - avg_similarity)

        # Check if minimum distance requirement is met
        meets_min_distance = avg_similarity <= (1 - min_distance)

        return {
            "diversity_score": diversity_score,
            "avg_similarity": avg_similarity,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity,
            "meets_min_distance": meets_min_distance,
            "num_items": len(items),
            "num_comparisons": len(similarities),
        }

    def suggest_diverse_subset(
        self, items: List[Union[str, Dict[str, Any]]], target_size: int, diversity_weight: float = 0.7
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Select a diverse subset of items using a greedy approach.

        Args:
            items: List of items to select from
            target_size: Number of items to select
            diversity_weight: Weight for diversity vs. original order (0-1)

        Returns:
            List of selected diverse items
        """
        if not items or target_size <= 0:
            return []

        if len(items) <= target_size:
            return items.copy()

        # Extract texts and embed
        texts = [self.extract_text(item) for item in items]
        embeddings = self.embedding_manager.embed_texts(texts, show_progress=False)

        # Start with the first item
        selected_indices = [0]
        remaining_indices = list(range(1, len(items)))

        while len(selected_indices) < target_size and remaining_indices:
            best_idx = None
            best_score = -1

            for idx in remaining_indices:
                # Calculate minimum distance to all selected items
                min_distance = min(1 - float(embeddings[idx].dot(embeddings[selected_idx])) for selected_idx in selected_indices)

                # Score combines diversity and original order preference
                order_preference = 1 - (idx / len(items))  # Prefer earlier items
                score = diversity_weight * min_distance + (1 - diversity_weight) * order_preference

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        # Return items in original order
        selected_indices.sort()
        selected_items = [items[i] for i in selected_indices]

        logger.info(f"Selected {len(selected_items)} diverse items from {len(items)} total")
        return selected_items
