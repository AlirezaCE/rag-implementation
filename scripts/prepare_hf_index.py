#!/usr/bin/env python3
"""
Prepare passages with embeddings for Hugging Face RAG.

This script:
1. Loads passages from JSONL
2. Computes embeddings using DPR context encoder
3. Saves as HF Dataset with embeddings
4. Creates FAISS index
"""

import argparse
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import faiss
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_hf_rag_index(
    passages_jsonl: str,
    output_dir: str,
    ctx_encoder: str = "facebook/dpr-ctx_encoder-single-nq-base",
    batch_size: int = 64,
    max_length: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Prepare passages with embeddings for HF RAG.

    Args:
        passages_jsonl: Path to passages.jsonl
        output_dir: Output directory for HF dataset
        ctx_encoder: DPR context encoder model
        batch_size: Batch size for embedding computation
        max_length: Max token length
        device: Device to use
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("Preparing HF RAG Index with Embeddings")
    logger.info("="*70)
    logger.info(f"Input: {passages_jsonl}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Context encoder: {ctx_encoder}")

    # Load passages
    logger.info("\n[1/5] Loading passages...")
    passages = []
    with open(passages_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            passages.append({
                'text': data.get('text', ''),
                'title': data.get('title', ''),
            })

    logger.info(f"Loaded {len(passages)} passages")

    # Load DPR context encoder
    logger.info("\n[2/5] Loading DPR context encoder...")
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_encoder)
    encoder = DPRContextEncoder.from_pretrained(ctx_encoder)
    encoder = encoder.to(device)
    encoder.eval()

    # Compute embeddings
    logger.info("\n[3/5] Computing embeddings...")
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(passages), batch_size), desc="Computing embeddings"):
            batch_passages = passages[i:i+batch_size]

            # Combine title and text
            texts = [
                f"{p['title']} {p['text']}" if p['title'] else p['text']
                for p in batch_passages
            ]

            # Tokenize
            inputs = tokenizer(
                texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)

            # Get embeddings
            outputs = encoder(**inputs)
            embeddings = outputs.pooler_output.cpu().numpy()
            all_embeddings.append(embeddings)

    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Computed embeddings shape: {all_embeddings.shape}")

    # Create dataset with embeddings
    logger.info("\n[4/5] Creating HF Dataset...")
    dataset_dict = {
        'title': [p['title'] for p in passages],
        'text': [p['text'] for p in passages],
        'embeddings': all_embeddings
    }
    dataset = Dataset.from_dict(dataset_dict)

    # Add FAISS index
    logger.info("Adding FAISS index to dataset...")
    dataset.add_faiss_index(column='embeddings')

    # Save dataset
    logger.info(f"\n[5/5] Saving to {output_dir}...")
    dataset.save_to_disk(output_dir)

    # Also save the FAISS index separately for reference
    logger.info("Saving FAISS index separately...")
    index = dataset.get_index('embeddings').faiss_index
    faiss.write_index(index, str(output_path / "hf_dataset_index.faiss"))

    logger.info("\n" + "="*70)
    logger.info("✓ Preparation complete!")
    logger.info("="*70)
    logger.info(f"Dataset saved to: {output_dir}")
    logger.info(f"Number of passages: {len(dataset)}")
    logger.info(f"Embedding dimension: {all_embeddings.shape[1]}")
    logger.info(f"Dataset columns: {dataset.column_names}")
    logger.info("\nSample entry:")
    logger.info(f"  Title: {dataset[0]['title'][:50]}...")
    logger.info(f"  Text: {dataset[0]['text'][:100]}...")
    logger.info(f"  Embedding shape: {dataset[0]['embeddings'].shape}")
    logger.info("="*70)

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Prepare passages with embeddings for Hugging Face RAG"
    )
    parser.add_argument("--passages", type=str, required=True,
                       help="Path to passages.jsonl file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for HF dataset")
    parser.add_argument("--ctx_encoder", type=str,
                       default="facebook/dpr-ctx_encoder-single-nq-base",
                       help="DPR context encoder model")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for embedding computation")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Max token length")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")

    args = parser.parse_args()

    prepare_hf_rag_index(
        passages_jsonl=args.passages,
        output_dir=args.output_dir,
        ctx_encoder=args.ctx_encoder,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )


if __name__ == "__main__":
    main()
