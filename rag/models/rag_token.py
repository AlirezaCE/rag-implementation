"""
RAG-Token model implementation.

From paper Section 2.1:
"In the RAG-Token model we can draw a different latent document for each target
token and marginalize accordingly. This allows the generator to choose content
from several documents when producing an answer."

Formula:
    p_RAG-Token(y|x) ≈ Π_i Σ_{z ∈ top-k(p(·|x))} p_η(z|x) * p_θ(y_i|x,z,y_{1:i-1})

Key difference from RAG-Sequence:
- RAG-Sequence: Same document for entire sequence
- RAG-Token: Can use different documents for each token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

from .base import RAGModel
from ..config import RAGConfig
from ..retrieval import BaseRetriever
from ..generation import BARTGenerator


class RAGTokenForGeneration(RAGModel):
    """
    RAG-Token model for generation tasks.

    Marginalizes over documents at each token position.
    """

    def __init__(
        self,
        config: RAGConfig,
        retriever: Optional[BaseRetriever] = None,
        generator: Optional[BARTGenerator] = None,
    ):
        """
        Initialize RAG-Token model.

        Args:
            config: RAG configuration
            retriever: Optional retriever instance
            generator: Optional generator instance
        """
        super().__init__(config, retriever, generator)

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
        Forward pass for RAG-Token.

        Args:
            input_ids: Query input IDs (batch_size, query_len)
            attention_mask: Query attention mask
            labels: Target labels (batch_size, target_len)
            n_docs: Number of documents to retrieve
            context_input_ids: Pre-computed context inputs
            context_attention_mask: Context attention masks
            doc_scores: Pre-computed document scores
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

        # Forward through generator for all documents
        if labels is not None:
            # Training: compute loss with token-level marginalization
            # Expand labels for each document
            labels_expanded = labels.unsqueeze(1).repeat(1, n_docs, 1)
            labels_expanded = labels_expanded.view(batch_size * n_docs, -1)

            # Forward through generator
            gen_outputs = self.generator(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask,
                labels=labels_expanded,
            )

            # Get logits: (batch * n_docs, target_len, vocab_size)
            logits = gen_outputs.logits

            # Reshape to (batch, n_docs, target_len, vocab_size)
            target_len = logits.shape[1]
            vocab_size = logits.shape[2]
            logits_reshaped = logits.view(batch_size, n_docs, target_len, vocab_size)

            # Compute token-level probabilities
            token_log_probs = F.log_softmax(logits_reshaped, dim=-1)

            # For each token position, marginalize over documents
            # p(y_i|x, y_{<i}) = Σ_z p(z|x) * p(y_i|x, z, y_{<i})

            # Expand doc_scores: (batch, n_docs, 1, 1) for broadcasting
            doc_scores_expanded = doc_scores.unsqueeze(-1).unsqueeze(-1)

            if labels is not None:
                # Gather log probs for target tokens
                # Shift labels by 1 for autoregressive prediction
                target_ids = labels[:, 1:].unsqueeze(1).expand(-1, n_docs, -1)  # (batch, n_docs, target_len-1)

                # Gather token probabilities
                gathered_log_probs = torch.gather(
                    token_log_probs[:, :, :-1, :],  # Shift logits
                    3,
                    target_ids.unsqueeze(-1)
                ).squeeze(-1)  # (batch, n_docs, target_len-1)

                # Create mask for non-padding tokens
                label_mask = (labels[:, 1:] != -100).unsqueeze(1).expand(-1, n_docs, -1).float()

                # Mask padding
                gathered_log_probs = gathered_log_probs * label_mask

                # Token-level marginalization
                # For each position: log Σ_z p(z|x) * p(y_i|x,z,y_{<i})
                # = log Σ_z exp(log p(z|x) + log p(y_i|x,z,y_{<i}))

                log_doc_scores = torch.log(doc_scores + 1e-10).unsqueeze(-1)  # (batch, n_docs, 1)

                # Add log doc scores to each token probability
                token_marginal_log_probs = log_doc_scores + gathered_log_probs  # (batch, n_docs, target_len-1)

                # Marginalize over documents for each token
                token_marginals = torch.logsumexp(token_marginal_log_probs, dim=1)  # (batch, target_len-1)

                # Apply mask and sum over tokens for final loss
                masked_marginals = token_marginals * label_mask[:, 0, :]
                marginal_log_prob = masked_marginals.sum(dim=1)  # (batch,)

                # Loss is negative marginal log-likelihood
                loss = -marginal_log_prob.mean()

                return {
                    "loss": loss,
                    "logits": logits,
                    "doc_scores": doc_scores,
                    "marginal_log_prob": marginal_log_prob,
                }

        else:
            # Inference
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
        Generate output sequences using RAG-Token.

        Uses standard beam search with modified transition probabilities.

        From paper Section 2.5:
        "The RAG-Token model can be seen as a standard, autoregressive seq2seq
        generator with transition probability:
        p'_θ(y_i|x, y_{1:i-1}) = Σ_{z ∈ top-k} p_η(z_i|x) p_θ(y_i|x, z_i, y_{1:i-1})"

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

        # For RAG-Token generation, we need to marginalize at each step
        # This is complex for beam search, so we use a simplified approach:
        # Generate with the highest-scoring document and marginalize scores

        all_generated = []

        for i in range(batch_size):
            query = queries[i]
            docs = doc_texts[i]
            scores = doc_scores[i]

            # Prepare inputs for all documents
            inputs_list = []
            for doc in docs:
                inputs = self.generator.prepare_inputs(
                    query=query,
                    context=doc,
                    max_length=self.config.generator_max_length,
                )
                inputs_list.append(inputs)

            # Stack inputs
            stacked_input_ids = torch.cat(
                [inp["input_ids"] for inp in inputs_list], dim=0
            ).to(self.device)
            stacked_attention_mask = torch.cat(
                [inp["attention_mask"] for inp in inputs_list], dim=0
            ).to(self.device)

            # Generate from all documents
            generated = self.generator.generate(
                input_ids=stacked_input_ids,
                attention_mask=stacked_attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=1,
                **kwargs
            )

            # Take output from highest-scoring document
            # (In full implementation, would marginalize at each step)
            best_doc_idx = torch.argmax(scores).item()
            best_generated = generated[best_doc_idx]

            all_generated.append(best_generated)

        # Stack results
        result = torch.stack(all_generated, dim=0)

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
