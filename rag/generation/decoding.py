"""
Custom decoding strategies for RAG.

As described in RAG paper Section 2.5:
- RAG-Token: Standard beam search
- RAG-Sequence: "Thorough Decoding" and "Fast Decoding"
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class ThoroughDecoding:
    """
    Thorough decoding for RAG-Sequence.

    From paper Section 2.5:
    "we run beam search for each document z, scoring each hypothesis using
    p_θ(y_i|x,z,y_{1:i-1}). This yields a set of hypotheses Y, some of which
    may not have appeared in the beams of all documents. To estimate the
    probability of a hypothesis y we run an additional forward pass for each
    document z for which y does not appear in the beam, multiply generator
    probability with p_η(z|x) and then sum the probabilities across beams
    for the marginals."
    """

    def __init__(self, num_beams: int = 4):
        self.num_beams = num_beams

    def decode(
        self,
        model,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        doc_scores: torch.Tensor,
        max_length: int = 128,
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform thorough decoding.

        Args:
            model: Generator model
            input_ids_list: List of input IDs for each document
            attention_mask_list: List of attention masks
            doc_scores: Document retrieval scores p_η(z|x)
            max_length: Maximum generation length

        Returns:
            Tuple of (generated_ids, total_score)
        """
        # Run beam search for each document
        all_hypotheses = {}
        doc_hypotheses = []

        for i, (input_ids, attention_mask) in enumerate(
            zip(input_ids_list, attention_mask_list)
        ):
            # Generate with beam search for this document
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )

            # Store hypotheses
            sequences = outputs.sequences
            scores = outputs.sequences_scores

            doc_hyps = []
            for seq, score in zip(sequences, scores):
                seq_tuple = tuple(seq.tolist())
                if seq_tuple not in all_hypotheses:
                    all_hypotheses[seq_tuple] = []
                all_hypotheses[seq_tuple].append((i, score.item()))
                doc_hyps.append((seq_tuple, score.item()))

            doc_hypotheses.append(doc_hyps)

        # For each hypothesis, compute marginal probability
        best_hypothesis = None
        best_score = float("-inf")

        for hypothesis in all_hypotheses:
            # Get doc indices and scores where this hypothesis appeared
            doc_indices_scores = all_hypotheses[hypothesis]

            # For docs where it didn't appear, need to run forward pass
            # (simplified version - in full implementation would run additional passes)
            marginal_score = 0.0
            for doc_idx, gen_score in doc_indices_scores:
                # p(y|x) += p_η(z|x) * p_θ(y|x,z)
                marginal_score += doc_scores[doc_idx].item() * np.exp(gen_score)

            if marginal_score > best_score:
                best_score = marginal_score
                best_hypothesis = hypothesis

        return torch.tensor(best_hypothesis), best_score


class FastDecoding:
    """
    Fast decoding for RAG-Sequence.

    From paper Section 2.5:
    "For more efficient decoding, we can make a further approximation that
    p_θ(y|x,z_i) ≈ 0 where y was not generated during beam search from x,z_i.
    This avoids the need to run additional forward passes once the candidate
    set Y has been generated."
    """

    def __init__(self, num_beams: int = 4):
        self.num_beams = num_beams

    def decode(
        self,
        model,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        doc_scores: torch.Tensor,
        max_length: int = 128,
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform fast decoding.

        Args:
            model: Generator model
            input_ids_list: List of input IDs for each document
            attention_mask_list: List of attention masks
            doc_scores: Document retrieval scores p_η(z|x)
            max_length: Maximum generation length

        Returns:
            Tuple of (generated_ids, total_score)
        """
        # Run beam search for each document
        all_hypotheses = {}

        for i, (input_ids, attention_mask) in enumerate(
            zip(input_ids_list, attention_mask_list)
        ):
            # Generate with beam search
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )

            sequences = outputs.sequences
            scores = outputs.sequences_scores

            for seq, score in zip(sequences, scores):
                seq_tuple = tuple(seq.tolist())
                if seq_tuple not in all_hypotheses:
                    all_hypotheses[seq_tuple] = 0.0
                # Marginalize: assume p_θ(y|x,z_i) ≈ 0 if not in beam
                all_hypotheses[seq_tuple] += doc_scores[i].item() * np.exp(
                    score.item()
                )

        # Return best hypothesis
        best_hypothesis = max(all_hypotheses.items(), key=lambda x: x[1])
        return torch.tensor(best_hypothesis[0]), best_hypothesis[1]


def beam_search(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_beams: int = 4,
    max_length: int = 128,
    **kwargs
) -> torch.Tensor:
    """
    Standard beam search (used for RAG-Token).

    Args:
        model: Generator model
        input_ids: Input IDs
        attention_mask: Attention mask
        num_beams: Number of beams
        max_length: Maximum length
        **kwargs: Additional generation arguments

    Returns:
        Generated token IDs
    """
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=num_beams,
        max_length=max_length,
        **kwargs
    )
