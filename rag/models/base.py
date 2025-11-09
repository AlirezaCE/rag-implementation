"""
Base RAG model class.

Implements shared functionality for RAG-Sequence and RAG-Token.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Union, Tuple
from abc import ABC, abstractmethod

from ..retrieval import BaseRetriever, DPRRetriever
from ..generation import BARTGenerator
from ..config import RAGConfig


class RAGModel(nn.Module, ABC):
    """
    Base class for RAG models.

    RAG models combine:
    1. Retriever p_η(z|x): Retrieves top-K documents given query
    2. Generator p_θ(y_i|x,z,y_{1:i-1}): Generates output given input and documents

    The key difference between RAG-Sequence and RAG-Token is in how they
    marginalize over the latent documents z.
    """

    def __init__(
        self,
        config: RAGConfig,
        retriever: Optional[BaseRetriever] = None,
        generator: Optional[BARTGenerator] = None,
    ):
        """
        Initialize RAG model.

        Args:
            config: RAG configuration
            retriever: Optional retriever instance
            generator: Optional generator instance
        """
        super().__init__()

        self.config = config

        # Initialize retriever
        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = DPRRetriever(
                question_encoder=config.retriever_name_or_path,
                ctx_encoder=config.doc_encoder_name_or_path,
                index_path=config.index_path,
                freeze_doc_encoder=config.freeze_retriever,
            )

        # Initialize generator
        if generator is not None:
            self.generator = generator
        else:
            self.generator = BARTGenerator(
                model_name=config.generator_name_or_path,
                freeze=config.freeze_generator,
            )

        self.num_docs = config.num_retrieved_docs

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        n_docs: Optional[int] = None,
        **kwargs
    ):
        """
        Forward pass (to be implemented by subclasses).

        Args:
            input_ids: Input query IDs
            attention_mask: Attention mask
            labels: Target labels (for training)
            n_docs: Number of documents to retrieve

        Returns:
            Model outputs
        """
        pass

    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_docs: Optional[int] = None,
        max_length: int = 128,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate outputs (to be implemented by subclasses).

        Args:
            input_ids: Input query IDs
            attention_mask: Attention mask
            n_docs: Number of documents to retrieve
            max_length: Maximum generation length
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        pass

    def retrieve_documents(
        self,
        queries: List[str],
        n_docs: Optional[int] = None,
    ) -> Tuple[List[List[str]], torch.Tensor]:
        """
        Retrieve documents for queries.

        Args:
            queries: List of query strings
            n_docs: Number of documents to retrieve

        Returns:
            Tuple of (retrieved_docs, retrieval_scores)
            - retrieved_docs: List of lists of document texts
            - retrieval_scores: Tensor of shape (batch_size, n_docs)
        """
        if n_docs is None:
            n_docs = self.num_docs

        # Retrieve documents
        results = self.retriever.retrieve(queries, k=n_docs)

        # Extract texts and scores
        retrieved_docs = []
        retrieval_scores = []

        for query_results in results:
            docs = [doc.text for doc in query_results]
            scores = [doc.score for doc in query_results]

            retrieved_docs.append(docs)
            retrieval_scores.append(scores)

        retrieval_scores = torch.tensor(retrieval_scores)

        return retrieved_docs, retrieval_scores

    def compute_retrieval_distribution(
        self,
        retrieval_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute p_η(z|x) from retrieval scores.

        In DPR: p_η(z|x) ∝ exp(d(z)^T q(x))

        Args:
            retrieval_scores: Raw retrieval scores of shape (batch_size, n_docs)

        Returns:
            Retrieval probabilities (normalized)
        """
        # Scores are already inner products, so they're in log space
        # Normalize with softmax to get probabilities
        return torch.softmax(retrieval_scores, dim=-1)

    def prepare_generator_inputs(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        doc_texts: List[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for generator by concatenating query and documents.

        Args:
            query_ids: Query token IDs (batch_size, query_len)
            query_mask: Query attention mask
            doc_texts: Retrieved document texts

        Returns:
            Tuple of (combined_input_ids, combined_attention_mask)
        """
        batch_size = len(doc_texts)
        n_docs = len(doc_texts[0]) if doc_texts else 0

        # Decode queries
        queries = self.generator.decode(query_ids, skip_special_tokens=True)

        # Prepare inputs for each (query, doc) pair
        all_inputs = []

        for i in range(batch_size):
            for j in range(n_docs):
                # Concatenate query and document
                inputs = self.generator.prepare_inputs(
                    query=queries[i],
                    context=doc_texts[i][j],
                    max_length=self.config.generator_max_length,
                )
                all_inputs.append(inputs)

        # Stack all inputs
        if all_inputs:
            combined_input_ids = torch.cat(
                [inp["input_ids"] for inp in all_inputs], dim=0
            )
            combined_attention_mask = torch.cat(
                [inp["attention_mask"] for inp in all_inputs], dim=0
            )
        else:
            combined_input_ids = None
            combined_attention_mask = None

        return combined_input_ids, combined_attention_mask

    def set_retriever(self, retriever: BaseRetriever):
        """Set retriever component."""
        self.retriever = retriever

    def set_generator(self, generator: BARTGenerator):
        """Set generator component."""
        self.generator = generator

    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.retriever.question_encoder.to(device)
        self.generator.to(device)
        return self

    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        if hasattr(self.retriever, "train_mode"):
            self.retriever.train_mode() if mode else self.retriever.eval_mode()
        self.generator.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        super().eval()
        if hasattr(self.retriever, "eval_mode"):
            self.retriever.eval_mode()
        self.generator.eval()
        return self

    def save_pretrained(self, save_directory: str):
        """Save model components."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        self.config.save(os.path.join(save_directory, "config.json"))

        # Save generator
        self.generator.save_pretrained(
            os.path.join(save_directory, "generator")
        )

        # Save retriever (if possible)
        if hasattr(self.retriever, "save"):
            self.retriever.save_index(
                os.path.join(save_directory, "index.faiss")
            )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[RAGConfig] = None,
        **kwargs
    ):
        """Load pretrained model."""
        import os
        import json

        if config is None:
            config_path = os.path.join(model_name_or_path, "config.json")
            if os.path.exists(config_path):
                config = RAGConfig.load(config_path)
            else:
                config = RAGConfig()

        # Load generator
        generator_path = os.path.join(model_name_or_path, "generator")
        if os.path.exists(generator_path):
            generator = BARTGenerator.from_pretrained(generator_path)
        else:
            generator = None

        # Load retriever
        retriever = None  # Would need to be loaded separately

        return cls(config=config, retriever=retriever, generator=generator)
