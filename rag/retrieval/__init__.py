"""
Retrieval components for RAG.

Implements:
- DPR (Dense Passage Retrieval) with BERT bi-encoders
- BM25 baseline for ablation studies
- FAISS indexing for efficient MIPS (Maximum Inner Product Search)
"""

from .base import BaseRetriever, MockRetriever
from .dpr import DPRRetriever
from .bm25 import BM25Retriever
from .faiss_index import FAISSIndex, build_index

__all__ = [
    "BaseRetriever",
    "MockRetriever",
    "DPRRetriever",
    "BM25Retriever",
    "FAISSIndex",
    "build_index",
]
