"""Data Deduplicator for Dataset Generation

This module provides advanced deduplication capabilities for generated datasets
using semantic embeddings to detect and remove similar content.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from .embedding_manager import EmbeddingManager
from .similarity_checker import SimilarityChecker
from starfish.common.logger import get_logger
import json
import hashlib

logger = get_logger(__name__)


class DataDeduplicator:
    """
    Advanced deduplication for generated datasets using semantic embeddings.

    Features:
    - Semantic deduplication using embeddings
    - Exact match deduplication using hashing
    - Field-specific deduplication strategies
    - Preservation of highest quality items
    - Detailed deduplication reports
    """

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        similarity_threshold: float = 0.9,
        exact_match_fields: Optional[List[str]] = None,
        semantic_fields: Optional[List[str]] = None,
        quality_scorer: Optional[Callable] = None,
    ):
        """
        Initialize the DataDeduplicator.

        Args:
            embedding_manager: Pre-configured EmbeddingManager instance
            similarity_threshold: Threshold for semantic similarity (0-1)
            exact_match_fields: Fields to check for exact matches
            semantic_fields: Fields to check for semantic similarity
            quality_scorer: Function to score item quality for keeping best duplicates
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.similarity_checker = SimilarityChecker(embedding_manager=self.embedding_manager, similarity_threshold=similarity_threshold)

        self.similarity_threshold = similarity_threshold
        self.exact_match_fields = exact_match_fields or ["id", "uuid"]
        self.semantic_fields = semantic_fields or ["text", "query", "question", "content", "prompt", "response", "answer"]
        self.quality_scorer = quality_scorer or self._default_quality_scorer

        logger.info(f"DataDeduplicator initialized with threshold={similarity_threshold}")

    def _default_quality_scorer(self, item: Dict[str, Any]) -> float:
        """
        Default quality scoring function.

        Args:
            item: Data item to score

        Returns:
            Quality score (higher is better)
        """
        score = 0.0

        # Length bonus for longer content
        for field in self.semantic_fields:
            if field in item and isinstance(item[field], str):
                score += len(item[field]) * 0.001

        # Completeness bonus
        non_empty_fields = sum(1 for v in item.values() if v is not None and str(v).strip())
        score += non_empty_fields * 0.1

        # Specific quality indicators
        if "score" in item:
            score += float(item.get("score", 0))

        if "confidence" in item:
            score += float(item.get("confidence", 0))

        return score

    def _extract_exact_match_signature(self, item: Dict[str, Any]) -> str:
        """
        Create a signature for exact match detection.

        Args:
            item: Data item

        Returns:
            Hash signature for exact matching
        """
        signature_parts = []

        for field in self.exact_match_fields:
            if field in item and item[field] is not None:
                signature_parts.append(f"{field}:{item[field]}")

        # Fallback to content hash if no exact match fields
        if not signature_parts:
            content = json.dumps(item, sort_keys=True, ensure_ascii=False)
            return hashlib.md5(content.encode()).hexdigest()

        signature = "|".join(signature_parts)
        return hashlib.md5(signature.encode()).hexdigest()

    def _extract_semantic_content(self, item: Dict[str, Any]) -> str:
        """
        Extract semantic content for similarity comparison.

        Args:
            item: Data item

        Returns:
            Combined semantic content
        """
        return self.similarity_checker.extract_text(item)

    def deduplicate_exact(self, items: List[Dict[str, Any]], keep_best: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Remove exact duplicates based on specified fields.

        Args:
            items: List of data items
            keep_best: Whether to keep the highest quality item in each duplicate group

        Returns:
            Tuple of (deduplicated_items, report)
        """
        if not items:
            return [], {"exact_duplicates_removed": 0, "groups": []}

        # Group items by signature
        signature_groups = {}
        for i, item in enumerate(items):
            signature = self._extract_exact_match_signature(item)
            if signature not in signature_groups:
                signature_groups[signature] = []
            signature_groups[signature].append((i, item))

        # Process groups
        deduplicated_items = []
        duplicate_groups = []
        total_removed = 0

        for signature, group in signature_groups.items():
            if len(group) == 1:
                # No duplicates
                deduplicated_items.append(group[0][1])
            else:
                # Duplicates found
                duplicate_groups.append([idx for idx, _ in group])
                total_removed += len(group) - 1

                if keep_best:
                    # Keep the highest quality item
                    best_item = max(group, key=lambda x: self.quality_scorer(x[1]))[1]
                    deduplicated_items.append(best_item)
                else:
                    # Keep the first item
                    deduplicated_items.append(group[0][1])

        report = {"exact_duplicates_removed": total_removed, "groups": duplicate_groups, "original_count": len(items), "final_count": len(deduplicated_items)}

        logger.info(f"Exact deduplication: {len(items)} -> {len(deduplicated_items)} ({total_removed} removed)")
        return deduplicated_items, report

    def deduplicate_semantic(
        self, items: List[Dict[str, Any]], threshold: Optional[float] = None, keep_best: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Remove semantic duplicates using embedding similarity.

        Args:
            items: List of data items
            threshold: Custom similarity threshold
            keep_best: Whether to keep the highest quality item in each duplicate group

        Returns:
            Tuple of (deduplicated_items, report)
        """
        if not items:
            return [], {"semantic_duplicates_removed": 0, "groups": []}

        threshold = threshold or self.similarity_threshold

        # Extract semantic content
        semantic_contents = [self._extract_semantic_content(item) for item in items]

        # Find duplicate groups using embedding similarity
        duplicate_groups = self.embedding_manager.find_duplicates(semantic_contents, threshold)

        # Track which items to keep
        items_to_remove = set()
        processed_groups = []

        for group in duplicate_groups:
            if len(group) > 1:
                processed_groups.append(group)

                if keep_best:
                    # Find the best item in the group
                    group_items = [(idx, items[idx]) for idx in group]
                    best_idx = max(group_items, key=lambda x: self.quality_scorer(x[1]))[0]

                    # Remove all except the best
                    for idx in group:
                        if idx != best_idx:
                            items_to_remove.add(idx)
                else:
                    # Remove all except the first
                    for idx in group[1:]:
                        items_to_remove.add(idx)

        # Create deduplicated list
        deduplicated_items = [item for i, item in enumerate(items) if i not in items_to_remove]

        report = {
            "semantic_duplicates_removed": len(items_to_remove),
            "groups": processed_groups,
            "similarity_threshold": threshold,
            "original_count": len(items),
            "final_count": len(deduplicated_items),
        }

        logger.info(f"Semantic deduplication: {len(items)} -> {len(deduplicated_items)} ({len(items_to_remove)} removed)")
        return deduplicated_items, report

    def deduplicate_comprehensive(
        self, items: List[Dict[str, Any]], exact_first: bool = True, semantic_threshold: Optional[float] = None, keep_best: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Perform comprehensive deduplication using both exact and semantic methods.

        Args:
            items: List of data items
            exact_first: Whether to perform exact deduplication first
            semantic_threshold: Custom similarity threshold for semantic deduplication
            keep_best: Whether to keep the highest quality item in each duplicate group

        Returns:
            Tuple of (deduplicated_items, comprehensive_report)
        """
        if not items:
            return [], {"total_removed": 0, "stages": []}

        current_items = items.copy()
        reports = []

        if exact_first:
            # Stage 1: Exact deduplication
            current_items, exact_report = self.deduplicate_exact(current_items, keep_best)
            exact_report["stage"] = "exact"
            reports.append(exact_report)

            # Stage 2: Semantic deduplication
            current_items, semantic_report = self.deduplicate_semantic(current_items, semantic_threshold, keep_best)
            semantic_report["stage"] = "semantic"
            reports.append(semantic_report)
        else:
            # Stage 1: Semantic deduplication
            current_items, semantic_report = self.deduplicate_semantic(current_items, semantic_threshold, keep_best)
            semantic_report["stage"] = "semantic"
            reports.append(semantic_report)

            # Stage 2: Exact deduplication
            current_items, exact_report = self.deduplicate_exact(current_items, keep_best)
            exact_report["stage"] = "exact"
            reports.append(exact_report)

        total_removed = len(items) - len(current_items)

        comprehensive_report = {
            "total_removed": total_removed,
            "original_count": len(items),
            "final_count": len(current_items),
            "reduction_percentage": (total_removed / len(items)) * 100 if items else 0,
            "stages": reports,
            "processing_order": ["exact", "semantic"] if exact_first else ["semantic", "exact"],
        }

        logger.info(
            f"Comprehensive deduplication: {len(items)} -> {len(current_items)} "
            f"({total_removed} removed, {comprehensive_report['reduction_percentage']:.1f}% reduction)"
        )

        return current_items, comprehensive_report

    def analyze_duplicates(self, items: List[Dict[str, Any]], semantic_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze duplicate patterns without removing items.

        Args:
            items: List of data items to analyze
            semantic_threshold: Custom similarity threshold

        Returns:
            Analysis report with duplicate statistics
        """
        if not items:
            return {"total_items": 0, "analysis": "No items to analyze"}

        # Exact duplicate analysis
        signature_counts = {}
        for item in items:
            signature = self._extract_exact_match_signature(item)
            signature_counts[signature] = signature_counts.get(signature, 0) + 1

        exact_duplicates = sum(count - 1 for count in signature_counts.values() if count > 1)
        exact_groups = sum(1 for count in signature_counts.values() if count > 1)

        # Semantic duplicate analysis
        semantic_contents = [self._extract_semantic_content(item) for item in items]
        threshold = semantic_threshold or self.similarity_threshold
        duplicate_groups = self.embedding_manager.find_duplicates(semantic_contents, threshold)

        semantic_duplicates = sum(len(group) - 1 for group in duplicate_groups if len(group) > 1)
        semantic_groups = len([group for group in duplicate_groups if len(group) > 1])

        # Diversity analysis
        diversity_metrics = self.similarity_checker.check_diversity_batch(items)

        analysis = {
            "total_items": len(items),
            "exact_duplicates": {"count": exact_duplicates, "groups": exact_groups, "percentage": (exact_duplicates / len(items)) * 100 if items else 0},
            "semantic_duplicates": {
                "count": semantic_duplicates,
                "groups": semantic_groups,
                "percentage": (semantic_duplicates / len(items)) * 100 if items else 0,
                "threshold": threshold,
            },
            "diversity_metrics": diversity_metrics,
            "quality_scores": {
                "min": min(self.quality_scorer(item) for item in items),
                "max": max(self.quality_scorer(item) for item in items),
                "avg": sum(self.quality_scorer(item) for item in items) / len(items),
            },
        }

        logger.info(f"Duplicate analysis: {exact_duplicates} exact, {semantic_duplicates} semantic duplicates found")
        return analysis
