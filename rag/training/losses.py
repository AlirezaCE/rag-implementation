"""
Loss functions for RAG training.

Implements the loss functions described in the RAG paper:
- RAG-Sequence: Marginalizes over documents for the entire sequence
- RAG-Token: Marginalizes per token
"""

import torch
import torch.nn.functional as F


def rag_sequence_loss(logits, doc_scores, target_ids, pad_token_id=1):
    """
    Compute RAG-Sequence loss.

    For RAG-Sequence, we marginalize over documents for the entire sequence:
    p(y|x) = sum_z p(y|x,z) * p(z|x)

    where z is a retrieved document.

    Args:
        logits: Generator logits [batch_size, num_docs, seq_len, vocab_size]
        doc_scores: Document retrieval scores [batch_size, num_docs]
        target_ids: Target token IDs [batch_size, seq_len]
        pad_token_id: ID of padding token

    Returns:
        loss: Scalar loss value
    """
    batch_size, num_docs, seq_len, vocab_size = logits.shape

    # Compute log probabilities for each document
    # [batch_size, num_docs, seq_len, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log probabilities of target tokens
    # [batch_size, seq_len] -> [batch_size, 1, seq_len, 1]
    target_ids_expanded = target_ids.unsqueeze(1).unsqueeze(-1).expand(
        batch_size, num_docs, seq_len, 1
    )

    # [batch_size, num_docs, seq_len]
    target_log_probs = log_probs.gather(dim=-1, index=target_ids_expanded).squeeze(-1)

    # Create mask for non-padding tokens
    # [batch_size, seq_len] -> [batch_size, 1, seq_len]
    mask = (target_ids != pad_token_id).unsqueeze(1).float()

    # Compute sequence log probability for each document
    # Sum over sequence length: [batch_size, num_docs]
    seq_log_probs = (target_log_probs * mask).sum(dim=-1)

    # Add document scores (in log space)
    # [batch_size, num_docs]
    doc_log_probs = F.log_softmax(doc_scores, dim=-1)

    # Marginalize over documents: log(sum(exp(log_p)))
    # [batch_size, num_docs]
    combined_log_probs = seq_log_probs + doc_log_probs

    # [batch_size]
    marginal_log_probs = torch.logsumexp(combined_log_probs, dim=-1)

    # Compute negative log likelihood
    loss = -marginal_log_probs.mean()

    return loss


def rag_token_loss(logits, doc_scores, target_ids, pad_token_id=1):
    """
    Compute RAG-Token loss.

    For RAG-Token, we marginalize over documents for each token:
    p(y_i|x,y_{<i}) = sum_z p(y_i|x,z,y_{<i}) * p(z|x,y_{<i})

    Args:
        logits: Generator logits [batch_size, num_docs, seq_len, vocab_size]
        doc_scores: Document retrieval scores [batch_size, num_docs]
        target_ids: Target token IDs [batch_size, seq_len]
        pad_token_id: ID of padding token

    Returns:
        loss: Scalar loss value
    """
    batch_size, num_docs, seq_len, vocab_size = logits.shape

    # Compute log probabilities for each document
    # [batch_size, num_docs, seq_len, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log probabilities of target tokens
    # [batch_size, seq_len] -> [batch_size, 1, seq_len, 1]
    target_ids_expanded = target_ids.unsqueeze(1).unsqueeze(-1).expand(
        batch_size, num_docs, seq_len, 1
    )

    # [batch_size, num_docs, seq_len]
    target_log_probs = log_probs.gather(dim=-1, index=target_ids_expanded).squeeze(-1)

    # Add document scores (in log space)
    # [batch_size, num_docs] -> [batch_size, num_docs, 1]
    doc_log_probs = F.log_softmax(doc_scores, dim=-1).unsqueeze(-1)

    # Combine token and document probabilities
    # [batch_size, num_docs, seq_len]
    combined_log_probs = target_log_probs + doc_log_probs

    # Marginalize over documents per token
    # [batch_size, seq_len]
    marginal_log_probs = torch.logsumexp(combined_log_probs, dim=1)

    # Create mask for non-padding tokens
    # [batch_size, seq_len]
    mask = (target_ids != pad_token_id).float()

    # Compute negative log likelihood
    # Sum over valid tokens and average over batch
    loss = -(marginal_log_probs * mask).sum() / mask.sum()

    return loss
