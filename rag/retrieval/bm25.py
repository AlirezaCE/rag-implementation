"""
BM25 retriever for ablation studies.

As mentioned in RAG paper Table 6:
"We compare RAG's dense retriever to a word overlap-based BM25 retriever.
For FEVER, BM25 performs best, perhaps since FEVER claims are heavily
entity-centric and thus well-suited for word overlap-based retrieval."
"""

import numpy as np
from typing import List, Optional, Dict
from rank_bm25 import BM25Okapi
import pickle
from .base import BaseRetriever, RetrievedDocument


class BM25Retriever(BaseRetriever):
    """
    BM25 retriever using traditional sparse retrieval.

    BM25 is a bag-of-words retrieval function based on term frequency
    and inverse document frequency, with tunable parameters k1 and b.
    """

    def __init__(
        self,
        passages: Optional[List[Dict]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[callable] = None,
    ):
        """
        Initialize BM25 retriever.

        Args:
            passages: List of passage dictionaries
            k1: BM25 k1 parameter (term saturation)
            b: BM25 b parameter (length normalization)
            tokenizer: Custom tokenizer function
        """
        super().__init__()

        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or self._default_tokenizer
        self.passages = passages or []
        self.bm25 = None

        if self.passages:
            self._build_index()

    def _default_tokenizer(self, text: str) -> List[str]:
        """Simple whitespace tokenizer."""
        return text.lower().split()

    def _build_index(self):
        """Build BM25 index from passages."""
        if not self.passages:
            raise ValueError("No passages provided")

        print(f"Building BM25 index for {len(self.passages)} passages...")

        # Tokenize all passages
        tokenized_corpus = [
            self.tokenizer(passage.get("text", passage.get("passage_text", "")))
            for passage in self.passages
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

        print("BM25 index built successfully!")

    def add_passages(self, passages: List[Dict]):
        """
        Add passages and rebuild index.

        Args:
            passages: List of passage dictionaries
        """
        self.passages.extend(passages)
        self._build_index()

    def retrieve(
        self,
        queries: List[str],
        k: int = 5,
        **kwargs
    ) -> List[List[RetrievedDocument]]:
        """
        Retrieve top-K documents for each query using BM25.

        Args:
            queries: List of query strings
            k: Number of documents to retrieve

        Returns:
            List of lists of RetrievedDocument objects
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Please add passages first.")

        if isinstance(queries, str):
            queries = [queries]

        results = []
        for query in queries:
            # Tokenize query
            tokenized_query = self.tokenizer(query)

            # Get BM25 scores for all documents
            scores = self.bm25.get_scores(tokenized_query)

            # Get top-K indices
            top_k_indices = np.argsort(scores)[::-1][:k]

            # Build results
            query_results = []
            for idx in top_k_indices:
                passage = self.passages[idx]
                doc = RetrievedDocument(
                    doc_id=passage.get("id", str(idx)),
                    text=passage.get("text", passage.get("passage_text", "")),
                    title=passage.get("title", ""),
                    score=float(scores[idx]),
                    metadata=passage.get("metadata"),
                )
                query_results.append(doc)

            results.append(query_results)

        return results

    def encode_queries(self, queries: List[str], **kwargs):
        """BM25 doesn't use dense encodings."""
        raise NotImplementedError("BM25 doesn't support dense query encoding")

    def encode_documents(self, documents: List[str], **kwargs):
        """BM25 doesn't use dense encodings."""
        raise NotImplementedError("BM25 doesn't support dense document encoding")

    def get_scores(self, query: str) -> np.ndarray:
        """
        Get BM25 scores for a query against all documents.

        Args:
            query: Query string

        Returns:
            Array of BM25 scores for all documents
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built")

        tokenized_query = self.tokenizer(query)
        return self.bm25.get_scores(tokenized_query)

    def save(self, path: str):
        """
        Save BM25 index to disk.

        Args:
            path: Path to save index
        """
        state = {
            "passages": self.passages,
            "k1": self.k1,
            "b": self.b,
            "bm25": self.bm25,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"BM25 index saved to {path}")

    @classmethod
    def load(cls, path: str, tokenizer: Optional[callable] = None) -> "BM25Retriever":
        """
        Load BM25 index from disk.

        Args:
            path: Path to load from
            tokenizer: Optional custom tokenizer

        Returns:
            Loaded BM25Retriever instance
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls(
            passages=state["passages"],
            k1=state["k1"],
            b=state["b"],
            tokenizer=tokenizer,
        )
        instance.bm25 = state["bm25"]

        print(f"Loaded BM25 index with {len(instance.passages)} passages")
        return instance

    @property
    def ntotal(self) -> int:
        """Get total number of indexed passages."""
        return len(self.passages)


def create_bm25_index(
    passages: List[Dict],
    k1: float = 1.5,
    b: float = 0.75,
    tokenizer: Optional[callable] = None,
    save_path: Optional[str] = None,
) -> BM25Retriever:
    """
    Create BM25 index from passages.

    Args:
        passages: List of passage dictionaries
        k1: BM25 k1 parameter
        b: BM25 b parameter
        tokenizer: Optional tokenizer function
        save_path: Optional path to save index

    Returns:
        BM25Retriever instance
    """
    retriever = BM25Retriever(
        passages=passages,
        k1=k1,
        b=b,
        tokenizer=tokenizer,
    )

    if save_path:
        retriever.save(save_path)

    return retriever
