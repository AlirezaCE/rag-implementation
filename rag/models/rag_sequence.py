"""
RAG-Sequence model implementation.

From paper Section 2.1:
"The RAG-Sequence model uses the same retrieved document to generate the
complete sequence. Technically, it treats the retrieved document as a single
latent variable that is marginalized to get the seq2seq probability p(y|x) via
a top-K approximation."

Formula:
    p_RAG-Sequence(y|x) ≈ Σ_{z ∈ top-k(p(·|x))} p_η(z|x) * p_θ(y|x,z)
                         = Σ_{z ∈ top-k(p(·|x))} p_η(z|x) * Π_i p_θ(y_i|x,z,y_{1:i-1})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import numpy as np

from .base import RAGModel
from ..config import RAGConfig
from ..retrieval import BaseRetriever
from ..generation import BARTGenerator, ThoroughDecoding, FastDecoding


class RAGSequenceForGeneration(RAGModel):
    """
    RAG-Sequence model for generation tasks.

    Marginalizes over documents for the entire generated sequence.
    """

    def __init__(
        self,
        config: RAGConfig,
        retriever: Optional[BaseRetriever] = None,
        generator: Optional[BARTGenerator] = None,
    ):
        """
        Initialize RAG-Sequence model.

        Args:
            config: RAG configuration
            retriever: Optional retriever instance
            generator: Optional generator instance
        """
        super().__init__(config, retriever, generator)

        self.use_thorough_decoding = config.use_thorough_decoding

        # Initialize decoders
        if self.use_thorough_decoding:
            self.decoder = ThoroughDecoding(num_beams=config.generator_num_beams)
        else:
            self.decoder = FastDecoding(num_beams=config.generator_num_beams)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        n_docs: Optional[int] = None,
        context_input_ids: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        doc_scores: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass for RAG-Sequence.

        Args:
            input_ids: Query input IDs (batch_size, query_len)
            attention_mask: Query attention mask
            labels: Target labels (batch_size, target_len)
            n_docs: Number of documents to retrieve
            context_input_ids: Pre-computed context inputs (batch_size * n_docs, seq_len)
            context_attention_mask: Context attention masks
            doc_scores: Pre-computed document scores (batch_size, n_docs)
            **kwargs: Additional arguments

        Returns:
            Dictionary with loss and logits
        """
        if n_docs is None:
            n_docs = self.num_docs

        batch_size = input_ids.shape[0]

        # Retrieve documents if not provided
        if context_input_ids is None or doc_scores is None:
            # Decode queries
            queries = self.generator.decode(input_ids, skip_special_tokens=True)

            # Retrieve documents
            doc_texts, retrieval_scores = self.retrieve_documents(queries, n_docs)

            # Move to device
            if retrieval_scores.device != input_ids.device:
                retrieval_scores = retrieval_scores.to(input_ids.device)

            # Compute retrieval distribution p_η(z|x)
            doc_scores = self.compute_retrieval_distribution(retrieval_scores)

            # Prepare generator inputs
            context_input_ids, context_attention_mask = self.prepare_generator_inputs(
                input_ids, attention_mask, doc_texts
            )

            # Move to device
            if context_input_ids is not None:
                context_input_ids = context_input_ids.to(input_ids.device)
                context_attention_mask = context_attention_mask.to(input_ids.device)

        # Reshape for batched processing
        # context_input_ids: (batch_size * n_docs, seq_len)

        # Compute p_θ(y|x,z) for each document
        if labels is not None:
            # Training: compute loss
            # Expand labels for each document
            labels_expanded = labels.unsqueeze(1).repeat(1, n_docs, 1)  # (batch, n_docs, target_len)
            labels_expanded = labels_expanded.view(batch_size * n_docs, -1)

            # Forward through generator
            gen_outputs = self.generator(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask,
                labels=labels_expanded,
            )

            # Get per-document losses (negative log-likelihoods)
            # gen_outputs.loss is averaged, we need per-example losses
            # Recompute to get per-example losses
            logits = gen_outputs.logits  # (batch * n_docs, target_len, vocab_size)

            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Gather log probs for target tokens
            target_log_probs = torch.gather(
                log_probs[:, :-1, :],  # Shift by 1
                2,
                labels_expanded[:, 1:].unsqueeze(-1)  # Target tokens
            ).squeeze(-1)

            # Mask padding tokens
            label_mask = (labels_expanded[:, 1:] != -100).float()
            target_log_probs = target_log_probs * label_mask

            # Sum over sequence length: log p_θ(y|x,z)
            seq_log_probs = target_log_probs.sum(dim=-1)  # (batch * n_docs,)

            # Reshape back to (batch_size, n_docs)
            seq_log_probs = seq_log_probs.view(batch_size, n_docs)

            # Marginalize over documents (RAG-Sequence formula)
            # p(y|x) = Σ_z p_η(z|x) * p_θ(y|x,z)
            # In log space: log p(y|x) = log Σ_z exp(log p_η(z|x) + log p_θ(y|x,z))

            # doc_scores are already probabilities, convert to log
            log_doc_scores = torch.log(doc_scores + 1e-10)

            # Combine: log p_η(z|x) + log p_θ(y|x,z)
            combined_log_probs = log_doc_scores + seq_log_probs

            # Marginalize using logsumexp
            marginal_log_prob = torch.logsumexp(combined_log_probs, dim=1)  # (batch_size,)

            # Loss is negative marginal log-likelihood
            loss = -marginal_log_prob.mean()

            return {
                "loss": loss,
                "logits": logits,
                "doc_scores": doc_scores,
                "marginal_log_prob": marginal_log_prob,
            }

        else:
            # Inference: return logits for generation
            gen_outputs = self.generator(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask,
            )

            return {
                "logits": gen_outputs.logits,
                "doc_scores": doc_scores,
            }

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_docs: Optional[int] = None,
        max_length: int = 128,
        num_beams: int = 4,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate output sequences using RAG-Sequence.

        Uses Thorough or Fast decoding as specified in config.

        Args:
            input_ids: Input query IDs
            attention_mask: Attention mask
            n_docs: Number of documents to retrieve
            max_length: Maximum generation length
            num_beams: Number of beams
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        if n_docs is None:
            n_docs = self.num_docs

        batch_size = input_ids.shape[0]

        # Decode queries
        queries = self.generator.decode(input_ids, skip_special_tokens=True)

        # Retrieve documents
        doc_texts, retrieval_scores = self.retrieve_documents(queries, n_docs)

        # Move to device
        if retrieval_scores.device != input_ids.device:
            retrieval_scores = retrieval_scores.to(input_ids.device)

        # Compute retrieval distribution
        doc_scores = self.compute_retrieval_distribution(retrieval_scores)

        # Prepare inputs for each document
        all_generated = []

        for i in range(batch_size):
            query = queries[i]
            docs = doc_texts[i]
            scores = doc_scores[i]

            # Prepare inputs for each document
            input_ids_list = []
            attention_mask_list = []

            for doc in docs:
                inputs = self.generator.prepare_inputs(
                    query=query,
                    context=doc,
                    max_length=self.config.generator_max_length,
                )
                input_ids_list.append(inputs["input_ids"].to(self.device))
                attention_mask_list.append(inputs["attention_mask"].to(self.device))

            # Decode using configured strategy
            if self.use_thorough_decoding:
                generated, score = self.decoder.decode(
                    self.generator,
                    input_ids_list,
                    attention_mask_list,
                    scores,
                    max_length=max_length,
                )
            else:
                generated, score = self.decoder.decode(
                    self.generator,
                    input_ids_list,
                    attention_mask_list,
                    scores,
                    max_length=max_length,
                )

            all_generated.append(generated)

        # Stack results
        if len(all_generated) > 0:
            # Pad sequences to same length
            max_len = max(g.shape[0] for g in all_generated)
            padded = []
            for g in all_generated:
                if len(g.shape) == 0:
                    g = g.unsqueeze(0)
                pad_len = max_len - g.shape[0]
                if pad_len > 0:
                    g = F.pad(g, (0, pad_len), value=self.generator.tokenizer.pad_token_id)
                padded.append(g)

            result = torch.stack(padded, dim=0)
        else:
            result = torch.tensor([])

        return result

    def generate_from_query(
        self,
        query: str,
        max_length: int = 128,
        num_beams: int = 4,
        **kwargs
    ) -> str:
        """
        Convenience method to generate from a single query string.

        Args:
            query: Input query string
            max_length: Maximum generation length
            num_beams: Number of beams
            **kwargs: Additional arguments

        Returns:
            Generated text
        """
        # Tokenize query
        inputs = self.generator.tokenizer(
            query,
            return_tensors="pt",
            max_length=256,
            truncation=True,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Generate
        output_ids = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )

        # Decode
        output_text = self.generator.decode(output_ids, skip_special_tokens=True)

        return output_text[0] if isinstance(output_text, list) else output_text

    @property
    def device(self):
        """Get device of model."""
        return next(self.generator.model.parameters()).device
