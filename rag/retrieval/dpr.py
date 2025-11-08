"""
Dense Passage Retrieval (DPR) implementation.

Implements the retriever p_η(z|x) from the RAG paper using DPR's bi-encoder architecture:
    p_η(z|x) ∝ exp(d(z)^T q(x))
    d(z) = BERT_d(z)  # Document encoder
    q(x) = BERT_q(x)  # Query encoder

Reference:
    Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering", 2020
"""

import torch
import torch.nn as nn
from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoderTokenizer,
    AutoTokenizer,
)
from typing import List, Optional, Union, Dict
import numpy as np
from tqdm import tqdm

from .base import BaseRetriever, RetrievedDocument
from .faiss_index import FAISSIndex


class DPRRetriever(BaseRetriever):
    """
    Dense Passage Retrieval using BERT bi-encoders.

    As described in the RAG paper (Section 2.2):
    - Uses pre-trained DPR encoders from Karpukhin et al.
    - Query encoder: BERT_q(x) produces query embeddings
    - Document encoder: BERT_d(z) produces document embeddings
    - Retrieval via Maximum Inner Product Search (MIPS) with FAISS
    """

    def __init__(
        self,
        question_encoder: str = "facebook/dpr-question_encoder-single-nq-base",
        ctx_encoder: str = "facebook/dpr-ctx_encoder-single-nq-base",
        index: Optional[FAISSIndex] = None,
        index_path: Optional[str] = None,
        passages: Optional[List[Dict]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_query_length: int = 256,
        max_doc_length: int = 256,
        freeze_doc_encoder: bool = True,
    ):
        """
        Initialize DPR retriever.

        Args:
            question_encoder: HuggingFace model name for query encoder
            ctx_encoder: HuggingFace model name for document encoder
            index: Pre-built FAISS index
            index_path: Path to saved FAISS index
            passages: List of passages/documents
            device: Device to run models on
            max_query_length: Maximum query token length
            max_doc_length: Maximum document token length
            freeze_doc_encoder: Whether to freeze document encoder (as in paper)
        """
        super().__init__()

        self.device = device
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

        # Load question (query) encoder
        self.question_encoder = DPRQuestionEncoder.from_pretrained(question_encoder)
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_encoder)

        # Load context (document) encoder
        self.ctx_encoder = DPRContextEncoder.from_pretrained(ctx_encoder)
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_encoder)

        # Move to device
        self.question_encoder.to(self.device)
        self.ctx_encoder.to(self.device)

        # Freeze document encoder if specified (as in paper)
        if freeze_doc_encoder:
            for param in self.ctx_encoder.parameters():
                param.requires_grad = False
            self.ctx_encoder.eval()

        # Initialize or load index
        self.index = index
        if index_path is not None:
            self.load_index(index_path)

        self.passages = passages or []

    def encode_queries(
        self,
        queries: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Encode queries using BERT_q(x).

        Args:
            queries: Single query string or list of queries
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Query embeddings of shape (num_queries, hidden_size)
        """
        if isinstance(queries, str):
            queries = [queries]

        self.question_encoder.eval()

        all_embeddings = []
        num_batches = (len(queries) + batch_size - 1) // batch_size

        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding queries")

        with torch.no_grad():
            for i in iterator:
                batch_queries = queries[i * batch_size : (i + 1) * batch_size]

                # Tokenize
                inputs = self.question_tokenizer(
                    batch_queries,
                    padding=True,
                    truncation=True,
                    max_length=self.max_query_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Encode
                outputs = self.question_encoder(**inputs)
                embeddings = outputs.pooler_output  # (batch_size, hidden_size)

                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def encode_documents(
        self,
        documents: Union[str, List[str]],
        titles: Optional[List[str]] = None,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Encode documents using BERT_d(z).

        Args:
            documents: Single document or list of documents
            titles: Optional document titles
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Document embeddings of shape (num_docs, hidden_size)
        """
        if isinstance(documents, str):
            documents = [documents]

        if titles is None:
            titles = [""] * len(documents)

        self.ctx_encoder.eval()

        all_embeddings = []
        num_batches = (len(documents) + batch_size - 1) // batch_size

        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding documents")

        with torch.no_grad():
            for i in iterator:
                batch_docs = documents[i * batch_size : (i + 1) * batch_size]
                batch_titles = titles[i * batch_size : (i + 1) * batch_size]

                # Tokenize (DPR uses title + SEP + text format)
                inputs = self.ctx_tokenizer(
                    batch_titles,
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_doc_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Encode
                outputs = self.ctx_encoder(**inputs)
                embeddings = outputs.pooler_output  # (batch_size, hidden_size)

                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def retrieve(
        self,
        queries: Union[str, List[str]],
        k: int = 5,
        return_embeddings: bool = False,
    ) -> List[List[RetrievedDocument]]:
        """
        Retrieve top-K documents for each query.

        Implements top-k(p_η(·|x)) using MIPS (Maximum Inner Product Search).

        Args:
            queries: Query string or list of queries
            k: Number of documents to retrieve
            return_embeddings: Whether to return document embeddings

        Returns:
            List of lists of RetrievedDocument objects
        """
        if isinstance(queries, str):
            queries = [queries]

        if self.index is None:
            raise ValueError("No index loaded. Please load or build an index first.")

        # Encode queries
        query_embeds = self.encode_queries(queries)

        # Search index
        scores, indices = self.index.search(query_embeds.numpy(), k)

        # Build results
        results = []
        for i, query in enumerate(queries):
            query_results = []
            for j in range(k):
                doc_idx = indices[i, j]
                score = scores[i, j]

                if doc_idx < len(self.passages):
                    passage = self.passages[doc_idx]
                    doc = RetrievedDocument(
                        doc_id=passage.get("id", str(doc_idx)),
                        text=passage.get("text", passage.get("passage_text", "")),
                        title=passage.get("title", ""),
                        score=float(score),
                    )

                    if return_embeddings and hasattr(self.index, "reconstruct"):
                        doc.embeddings = torch.from_numpy(
                            self.index.reconstruct(int(doc_idx))
                        )

                    query_results.append(doc)

            results.append(query_results)

        return results

    def build_index(
        self,
        documents: List[str],
        titles: Optional[List[str]] = None,
        index_type: str = "IndexHNSWFlat",
        use_gpu: bool = False,
        batch_size: int = 32,
        save_path: Optional[str] = None,
    ):
        """
        Build FAISS index from documents.

        Args:
            documents: List of document texts
            titles: Optional document titles
            index_type: FAISS index type (default: IndexHNSWFlat as in paper)
            use_gpu: Whether to use GPU for indexing
            batch_size: Batch size for encoding
            save_path: Optional path to save index
        """
        print(f"Encoding {len(documents)} documents...")

        # Encode all documents
        doc_embeds = self.encode_documents(
            documents,
            titles=titles,
            batch_size=batch_size,
            show_progress=True,
        )

        print(f"Building {index_type} index...")

        # Build FAISS index
        self.index = FAISSIndex(
            embedding_dim=doc_embeds.shape[1],
            index_type=index_type,
            use_gpu=use_gpu,
        )
        self.index.add(doc_embeds.numpy())

        # Store passages for later retrieval
        self.passages = [
            {
                "id": str(i),
                "text": doc,
                "title": titles[i] if titles else "",
            }
            for i, doc in enumerate(documents)
        ]

        if save_path:
            self.save_index(save_path)

        print("Index built successfully!")

    def load_index(self, index_path: str, passages_path: Optional[str] = None):
        """
        Load pre-built FAISS index.

        Args:
            index_path: Path to saved FAISS index
            passages_path: Path to passages JSON file
        """
        import os
        import json

        print(f"Loading index from {index_path}...")
        self.index = FAISSIndex.load(index_path)

        # Load passages if path provided
        if passages_path and os.path.exists(passages_path):
            with open(passages_path, "r") as f:
                self.passages = json.load(f)
            print(f"Loaded {len(self.passages)} passages")

    def save_index(self, index_path: str, passages_path: Optional[str] = None):
        """
        Save FAISS index and passages.

        Args:
            index_path: Path to save FAISS index
            passages_path: Optional path to save passages
        """
        import json

        if self.index is None:
            raise ValueError("No index to save")

        print(f"Saving index to {index_path}...")
        self.index.save(index_path)

        # Save passages
        if passages_path and self.passages:
            with open(passages_path, "w") as f:
                json.dump(self.passages, f)
            print(f"Saved {len(self.passages)} passages to {passages_path}")

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.question_encoder.config.hidden_size

    def train_mode(self):
        """Set query encoder to training mode."""
        self.question_encoder.train()

    def eval_mode(self):
        """Set to evaluation mode."""
        self.question_encoder.eval()
        self.ctx_encoder.eval()

    def get_trainable_parameters(self):
        """Get trainable parameters (only query encoder if doc encoder is frozen)."""
        return self.question_encoder.parameters()
