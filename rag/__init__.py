"""
RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

This package implements the RAG model described in:
"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
Lewis et al., 2020 (arXiv:2005.11401v4)
"""

from .models import (
    RAGSequenceForGeneration,
    RAGTokenForGeneration,
    RAGConfig,
)
from .retrieval import (
    DPRRetriever,
    BM25Retriever,
    FAISSIndex,
)
from .generation import (
    BARTGenerator,
)
from .training import (
    RAGTrainer,
)

__version__ = "1.0.0"
__author__ = "RAG Implementation Team"

__all__ = [
    # Models
    "RAGSequenceForGeneration",
    "RAGTokenForGeneration",
    "RAGConfig",
    # Retrieval
    "DPRRetriever",
    "BM25Retriever",
    "FAISSIndex",
    # Generation
    "BARTGenerator",
    # Training
    "RAGTrainer",
]
