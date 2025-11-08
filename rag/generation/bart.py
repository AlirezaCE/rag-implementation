"""
BART Generator for RAG.

As described in RAG paper Section 2.3:
"The generator component p_θ(y_i|x,z,y_{1:i-1}) could be modelled using
any encoder-decoder. We use BART-large, a pre-trained seq2seq transformer
with 400M parameters."

"To combine the input x with the retrieved content z when generating from BART,
we simply concatenate them."
"""

import torch
import torch.nn as nn
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    BartConfig,
)
from typing import Optional, List, Dict, Union, Tuple


class BARTGenerator(nn.Module):
    """
    BART generator for RAG.

    This implements the parametric memory p_θ(y_i|x,z,y_{1:i-1}).
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large",
        config: Optional[BartConfig] = None,
        freeze: bool = False,
    ):
        """
        Initialize BART generator.

        Args:
            model_name: HuggingFace model name or path
            config: Optional BART config
            freeze: Whether to freeze generator parameters
        """
        super().__init__()

        if config is not None:
            self.model = BartForConditionalGeneration(config)
        else:
            self.model = BartForConditionalGeneration.from_pretrained(model_name)

        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.config = self.model.config

        # Freeze if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def prepare_inputs(
        self,
        query: str,
        context: Union[str, List[str]],
        max_length: int = 256,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs by concatenating query and context.

        As in paper: "We simply concatenate [the input x] with the retrieved
        content z when generating from BART."

        Args:
            query: Input query/question
            context: Retrieved context (single string or list)
            max_length: Maximum input length

        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Concatenate query and context
        if isinstance(context, list):
            # Multiple contexts - concatenate with separator
            combined = query + " " + " ".join(context)
        else:
            combined = query + " " + context

        # Tokenize
        inputs = self.tokenizer(
            combined,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        return inputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass through BART.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            decoder_input_ids: Decoder input IDs (for teacher forcing)
            labels: Labels for computing loss
            **kwargs: Additional arguments

        Returns:
            BART model outputs
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **kwargs
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        min_length: int = 1,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 3,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate output sequences.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum generation length
            min_length: Minimum generation length
            num_beams: Number of beams for beam search
            length_penalty: Length penalty factor
            early_stopping: Whether to stop early
            no_repeat_ngram_size: Size of no-repeat n-grams

        Returns:
            Generated token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            no_repeat_ngram_size=no_repeat_ngram_size,
            **kwargs
        )

    def compute_doc_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute p_θ(y|x,z) for RAG.

        This computes the seq2seq probability for the entire sequence
        given input and retrieved document.

        Args:
            input_ids: Input IDs (query + context)
            attention_mask: Attention mask
            target_ids: Target sequence IDs

        Returns:
            Log probabilities of shape (batch_size,)
        """
        if target_ids is None:
            raise ValueError("target_ids required for computing scores")

        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=target_ids[:, :-1],
            labels=target_ids,
            return_dict=True,
        )

        # Get log probabilities
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Gather probabilities for target tokens
        target_log_probs = torch.gather(
            log_probs,
            2,
            target_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        # Sum over sequence (log probabilities)
        seq_log_probs = target_log_probs.sum(dim=1)

        return seq_log_probs

    def get_encoder_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Get encoder outputs for caching.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Encoder outputs
        """
        return self.model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded strings
        """
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def encode(
        self,
        texts: Union[str, List[str]],
        max_length: int = 256,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs.

        Args:
            texts: Text or list of texts
            max_length: Maximum length
            **kwargs: Additional tokenizer arguments

        Returns:
            Dictionary with input_ids and attention_mask
        """
        if isinstance(texts, str):
            texts = [texts]

        return self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            **kwargs
        )

    @property
    def device(self):
        """Get device of model."""
        return next(self.model.parameters()).device

    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self

    def train(self, mode: bool = True):
        """Set training mode."""
        self.model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self

    def get_output_embeddings(self):
        """Get output embeddings layer."""
        return self.model.get_output_embeddings()

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings."""
        return self.model.resize_token_embeddings(new_num_tokens)

    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load pretrained model."""
        return cls(model_name=model_name_or_path, **kwargs)


class BARTGeneratorWithRetrieval(BARTGenerator):
    """
    BART generator with integrated retrieval for convenience.

    This wraps both encoding and generation with retrieved contexts.
    """

    def generate_with_context(
        self,
        query: str,
        contexts: List[str],
        max_input_length: int = 256,
        max_output_length: int = 128,
        num_beams: int = 4,
        **kwargs
    ) -> str:
        """
        Generate output given query and retrieved contexts.

        Args:
            query: Input query
            contexts: List of retrieved context documents
            max_input_length: Maximum input length
            max_output_length: Maximum output length
            num_beams: Number of beams for beam search
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        # Prepare inputs with concatenated contexts
        inputs = self.prepare_inputs(
            query=query,
            context=contexts,
            max_length=max_input_length,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        output_ids = self.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_output_length,
            num_beams=num_beams,
            **kwargs
        )

        # Decode
        output_text = self.decode(output_ids)[0]

        return output_text

    def score_generation(
        self,
        query: str,
        context: str,
        target: str,
        max_input_length: int = 256,
    ) -> float:
        """
        Compute generation probability p_θ(y|x,z).

        Args:
            query: Input query
            context: Retrieved context
            target: Target output
            max_input_length: Maximum input length

        Returns:
            Log probability of generating target
        """
        # Prepare inputs
        inputs = self.prepare_inputs(
            query=query,
            context=context,
            max_length=max_input_length,
        )

        # Encode target
        target_inputs = self.tokenizer(
            target,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        target_ids = target_inputs["input_ids"].to(self.device)

        # Compute log probability
        log_prob = self.compute_doc_scores(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            target_ids=target_ids,
        )

        return log_prob.item()
