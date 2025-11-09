"""
RAG model implementations.

Implements:
- RAG-Sequence: Marginalizes over same documents for entire sequence
- RAG-Token: Marginalizes per token, can use different docs per token
"""

from .base import RAGModel
from .rag_sequence import RAGSequenceForGeneration
from .rag_token import RAGTokenForGeneration
from ..config import RAGConfig

__all__ = [
    "RAGModel",
    "RAGSequenceForGeneration",
    "RAGTokenForGeneration",
    "RAGConfig",
]
