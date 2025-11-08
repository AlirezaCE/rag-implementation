"""
Base retriever class for RAG.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import torch
from dataclasses import dataclass


@dataclass
class RetrievedDocument:
    """Container for a retrieved document."""

    doc_id: str
    text: str
    title: Optional[str] = None
    score: float = 0.0
    embeddings: Optional[torch.Tensor] = None
    metadata: Optional[Dict] = None

    def __repr__(self):
        return f"RetrievedDocument(id={self.doc_id}, score={self.score:.4f}, title={self.title})"


class BaseRetriever(ABC):
    """
    Base class for all retrievers.

    Retrievers take a query and return top-K relevant documents.
    This matches the p_η(z|x) component in the RAG paper.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    @abstractmethod
    def retrieve(
        self,
        queries: List[str],
        k: int = 5,
        **kwargs
    ) -> List[List[RetrievedDocument]]:
        """
        Retrieve top-K documents for each query.

        Args:
            queries: List of query strings
            k: Number of documents to retrieve (default: 5 as in paper)
            **kwargs: Additional retriever-specific arguments

        Returns:
            List of lists of RetrievedDocument objects, one list per query
        """
        pass

    @abstractmethod
    def encode_queries(
        self,
        queries: List[str],
        **kwargs
    ) -> torch.Tensor:
        """
        Encode queries into dense vectors.

        This implements the q(x) = BERT_q(x) function from the paper.

        Args:
            queries: List of query strings

        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        pass

    def encode_documents(
        self,
        documents: List[str],
        **kwargs
    ) -> torch.Tensor:
        """
        Encode documents into dense vectors.

        This implements the d(z) = BERT_d(z) function from the paper.

        Args:
            documents: List of document strings

        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        pass

    def get_retrieval_scores(
        self,
        query_embeds: torch.Tensor,
        doc_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute retrieval scores between queries and documents.

        For DPR, this is: p_η(z|x) ∝ exp(d(z)^T q(x))

        Args:
            query_embeds: Query embeddings (batch_size, embed_dim)
            doc_embeds: Document embeddings (num_docs, embed_dim)

        Returns:
            Scores of shape (batch_size, num_docs)
        """
        # Inner product for dense retrieval
        return torch.matmul(query_embeds, doc_embeds.transpose(0, 1))

    def __call__(
        self,
        queries: List[str],
        k: int = 5,
        **kwargs
    ) -> List[List[RetrievedDocument]]:
        """Allow calling retriever as a function."""
        return self.retrieve(queries, k=k, **kwargs)

    def save(self, path: str):
        """Save retriever state."""
        raise NotImplementedError("Subclass must implement save method")

    @classmethod
    def load(cls, path: str):
        """Load retriever from saved state."""
        raise NotImplementedError("Subclass must implement load method")


class MockRetriever(BaseRetriever):
    """
    Mock retriever for testing purposes.
    Returns dummy documents.
    """

    def __init__(self, num_docs: int = 100, embed_dim: int = 768):
        super().__init__()
        self.num_docs = num_docs
        self.embed_dim = embed_dim
        self.documents = [
            f"This is document {i} with some content." for i in range(num_docs)
        ]

    def retrieve(
        self,
        queries: List[str],
        k: int = 5,
        **kwargs
    ) -> List[List[RetrievedDocument]]:
        """Return k dummy documents per query."""
        results = []
        for query in queries:
            docs = []
            for i in range(min(k, self.num_docs)):
                docs.append(
                    RetrievedDocument(
                        doc_id=f"doc_{i}",
                        text=self.documents[i],
                        title=f"Document {i}",
                        score=1.0 / (i + 1),  # Decreasing scores
                    )
                )
            results.append(docs)
        return results

    def encode_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        """Return random query embeddings."""
        return torch.randn(len(queries), self.embed_dim)

    def encode_documents(self, documents: List[str], **kwargs) -> torch.Tensor:
        """Return random document embeddings."""
        return torch.randn(len(documents), self.embed_dim)
