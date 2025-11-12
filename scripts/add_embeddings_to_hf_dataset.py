#!/usr/bin/env python3
"""
Add embeddings to HuggingFace dataset for RAG.

This script loads a HF dataset without embeddings and adds them using DPR context encoder.
"""

import argparse
import torch
import logging
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_embeddings(
    dataset_path: str,
    output_path: str,
    ctx_encoder_name: str = "facebook/dpr-ctx_encoder-single-nq-base",
    batch_size: int = 32,
    device: str = None,
):
    """
    Add embeddings to HuggingFace dataset.

    Args:
        dataset_path: Path to HF dataset directory (without embeddings)
        output_path: Path to save dataset with embeddings
        ctx_encoder_name: DPR context encoder model name
        batch_size: Batch size for encoding
        device: Device to use (cuda/cpu)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    logger.info(f"Dataset loaded: {len(dataset)} passages")
    logger.info(f"Columns: {dataset.column_names}")

    # Check if embeddings already exist
    if "embeddings" in dataset.column_names:
        logger.warning("Dataset already has embeddings column. Overwriting...")
        dataset = dataset.remove_columns(["embeddings"])

    # Load context encoder
    logger.info(f"Loading context encoder: {ctx_encoder_name}")
    ctx_encoder = DPRContextEncoder.from_pretrained(ctx_encoder_name)
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_encoder_name)
    ctx_encoder.to(device)
    ctx_encoder.eval()

    # Get embedding dimension
    with torch.no_grad():
        sample_input = ctx_tokenizer(
            ["sample"], ["sample text"],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        sample_embedding = ctx_encoder(**sample_input).pooler_output
        embedding_dim = sample_embedding.shape[-1]

    logger.info(f"Embedding dimension: {embedding_dim}")

    # Encode all passages
    logger.info("Encoding passages...")
    all_embeddings = []

    num_batches = (len(dataset) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Encoding"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(dataset))

            batch = dataset[start_idx:end_idx]
            titles = batch.get("title", [""] * len(batch["text"]))
            texts = batch["text"]

            # Tokenize
            inputs = ctx_tokenizer(
                titles,
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)

            # Encode
            embeddings = ctx_encoder(**inputs).pooler_output
            embeddings = embeddings.cpu().numpy()

            all_embeddings.append(embeddings)

    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Embeddings shape: {all_embeddings.shape}")

    # Add embeddings to dataset
    logger.info("Adding embeddings to dataset...")
    dataset = dataset.add_column("embeddings", list(all_embeddings))

    # Verify
    logger.info(f"New columns: {dataset.column_names}")
    logger.info(f"Sample entry keys: {list(dataset[0].keys())}")
    logger.info(f"Embeddings shape for first entry: {dataset[0]['embeddings'].shape}")

    # Save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving dataset with embeddings to {output_path}")
    dataset.save_to_disk(str(output_path))

    logger.info("✓ Complete!")
    logger.info(f"Dataset with embeddings saved to: {output_path}")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Add embeddings to HuggingFace dataset for RAG"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to HF dataset directory (without embeddings)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save dataset with embeddings",
    )
    parser.add_argument(
        "--ctx_encoder",
        type=str,
        default="facebook/dpr-ctx_encoder-single-nq-base",
        help="DPR context encoder model name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)",
    )

    args = parser.parse_args()

    add_embeddings(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        ctx_encoder_name=args.ctx_encoder,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
