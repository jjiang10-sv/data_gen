"""Index Manager for Starfish Embeddings

This module provides high-level index management functionality for the embedding system.
Deprecated in favor of EmbeddingManager, but maintained for backward compatibility.
"""

import warnings
from typing import Optional
from .embedding_manager import EmbeddingManager

# Show deprecation warning
warnings.warn("index_manager.py is deprecated. Use EmbeddingManager from starfish.embedding instead.", DeprecationWarning, stacklevel=2)


class IndexManager(EmbeddingManager):
    """
    Deprecated: Use EmbeddingManager instead.

    This class is maintained for backward compatibility only.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("IndexManager is deprecated. Use EmbeddingManager instead.", DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
