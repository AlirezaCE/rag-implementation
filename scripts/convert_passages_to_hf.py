#!/usr/bin/env python3
"""
Convert passages from JSONL format to Hugging Face Dataset format.

This is required for using HF's RagRetriever with custom passages.
"""

import argparse
import json
from pathlib import Path
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_passages(input_jsonl: str, output_dir: str):
    """
    Convert passages.jsonl to Hugging Face Dataset format.

    Args:
        input_jsonl: Path to passages.jsonl file
        output_dir: Directory to save HF dataset
    """
    logger.info(f"Reading passages from {input_jsonl}")

    # Read passages from JSONL
    passages = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)

            # HF RAG expects: title, text, embeddings (optional)
            passage = {
                'id': str(i),
                'text': data.get('text', ''),
                'title': data.get('title', ''),
            }
            passages.append(passage)

            if (i + 1) % 10000 == 0:
                logger.info(f"Loaded {i + 1} passages...")

    logger.info(f"Total passages loaded: {len(passages)}")

    # Convert to HF Dataset
    logger.info("Converting to Hugging Face Dataset...")
    dataset = Dataset.from_list(passages)

    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving dataset to {output_dir}")
    dataset.save_to_disk(output_dir)

    logger.info("✓ Conversion complete!")
    logger.info(f"Dataset schema: {dataset}")
    logger.info(f"Sample entry: {dataset[0]}")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert passages.jsonl to Hugging Face Dataset format"
    )
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input passages.jsonl file")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output HF dataset directory")

    args = parser.parse_args()

    convert_passages(args.input, args.output)


if __name__ == "__main__":
    main()
