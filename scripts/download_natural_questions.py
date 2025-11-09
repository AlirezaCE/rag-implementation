#!/usr/bin/env python3
"""
Download Natural Questions dataset for RAG fine-tuning.

Natural Questions is the dataset used in the RAG paper for training.
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_natural_questions(output_dir: str, split: str = "train", max_examples: int = None):
    """
    Download Natural Questions dataset.

    Args:
        output_dir: Directory to save processed data
        split: Dataset split (train/validation)
        max_examples: Maximum examples to download (None = all)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not found. Install with: pip install datasets")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading Natural Questions ({split} split)...")

    try:
        # Load Natural Questions dataset
        # Using the NQ-open variant which is preprocessed for open-domain QA
        dataset = load_dataset("nq_open", split=split)

        logger.info(f"Loaded {len(dataset)} examples")

        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
            logger.info(f"Limited to {len(dataset)} examples")

        # Process and save
        examples = []
        logger.info("Processing examples...")

        for idx, example in enumerate(tqdm(dataset)):
            # NQ-open format:
            # - question: the question text
            # - answer: list of acceptable answers
            processed = {
                "id": f"nq_{split}_{idx}",
                "question": example["question"],
                "answers": example["answer"],  # List of valid answers
            }
            examples.append(processed)

        # Save as JSONL
        output_file = output_path / f"nq_{split}.jsonl"
        logger.info(f"Saving {len(examples)} examples to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in tqdm(examples, desc="Writing"):
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        # Save metadata
        metadata = {
            "dataset": "natural_questions_open",
            "split": split,
            "total_examples": len(examples),
            "output_file": str(output_file)
        }
        metadata_file = output_path / f"nq_{split}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\n✓ Download complete!")
        logger.info(f"Examples: {len(examples)}")
        logger.info(f"Saved to: {output_file}")
        logger.info(f"Metadata: {metadata_file}")

        # Show example
        logger.info(f"\nExample question:")
        logger.info(f"Q: {examples[0]['question']}")
        logger.info(f"A: {examples[0]['answers']}")

    except Exception as e:
        logger.error(f"Error downloading Natural Questions: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download Natural Questions dataset for RAG fine-tuning"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/natural_questions",
        help="Directory to save processed data (default: ./data/natural_questions)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to download (default: train)"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to download (default: all)"
    )

    args = parser.parse_args()

    download_natural_questions(
        output_dir=args.output_dir,
        split=args.split,
        max_examples=args.max_examples
    )


if __name__ == "__main__":
    main()
