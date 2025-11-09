#!/usr/bin/env python3
"""
Download a small test dataset for RAG.

This script downloads a small, manageable dataset for testing the RAG pipeline
without needing to download the full Wikipedia dump.
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


def download_simple_wikipedia(output_dir: str, max_articles: int = 1000):
    """
    Download Simple Wikipedia dataset (smaller and easier to work with).

    Args:
        output_dir: Directory to save articles
        max_articles: Maximum number of articles to download
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not found. Install with: pip install datasets")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading Simple Wikipedia dataset (max {max_articles} articles)")
    logger.info(f"Output directory: {output_path.absolute()}")

    try:
        # Load a smaller Wikipedia dataset
        logger.info("Loading dataset...")
        dataset = load_dataset(
            "wikipedia",
            "20220301.simple",  # Simple English Wikipedia (much smaller)
            split="train",
            trust_remote_code=True
        )

        logger.info(f"Dataset loaded with {len(dataset)} total articles")

        # Limit to max_articles
        if len(dataset) > max_articles:
            dataset = dataset.select(range(max_articles))
            logger.info(f"Limited to {max_articles} articles")

        # Process and save
        documents = []
        logger.info("Processing articles...")

        for idx, article in enumerate(tqdm(dataset)):
            doc = {
                "id": f"wiki_{idx}",
                "title": article.get("title", ""),
                "text": article.get("text", ""),
                "url": article.get("url", "")
            }
            documents.append(doc)

        # Save as single JSONL file
        output_file = output_path / "wikipedia_batch_1.jsonl"
        logger.info(f"Saving {len(documents)} articles to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in tqdm(documents, desc="Writing"):
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        # Save metadata
        metadata = {
            "dataset": "simple_wikipedia",
            "date": "20220301",
            "total_articles": len(documents),
            "output_dir": str(output_path.absolute())
        }
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\n✓ Download complete!")
        logger.info(f"Downloaded {len(documents)} articles")
        logger.info(f"Saved to: {output_file}")
        logger.info(f"Metadata: {metadata_file}")

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.info("\nTrying alternative dataset...")
        download_squad_context(output_dir, max_articles)


def download_squad_context(output_dir: str, max_articles: int = 1000):
    """
    Download SQuAD dataset contexts as an alternative smaller dataset.

    Args:
        output_dir: Directory to save articles
        max_articles: Maximum number of articles to download
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not found")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SQuAD dataset...")

    try:
        dataset = load_dataset("squad", split="train")
        logger.info(f"Loaded {len(dataset)} SQuAD examples")

        # Extract unique contexts as "documents"
        contexts_set = set()
        documents = []

        logger.info("Extracting unique contexts...")
        for idx, example in enumerate(tqdm(dataset)):
            if len(documents) >= max_articles:
                break

            context = example["context"]
            title = example["title"]

            # Only add unique contexts
            if context not in contexts_set:
                contexts_set.add(context)
                doc = {
                    "id": f"squad_{len(documents)}",
                    "title": title,
                    "text": context,
                    "url": ""
                }
                documents.append(doc)

        # Save as single JSONL file
        output_file = output_path / "wikipedia_batch_1.jsonl"
        logger.info(f"Saving {len(documents)} documents to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in tqdm(documents, desc="Writing"):
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        # Save metadata
        metadata = {
            "dataset": "squad_contexts",
            "total_articles": len(documents),
            "output_dir": str(output_path.absolute())
        }
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\n✓ Download complete!")
        logger.info(f"Downloaded {len(documents)} documents")
        logger.info(f"Saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error downloading SQuAD: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download small test dataset for RAG"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/test_wikipedia",
        help="Directory to save articles (default: ./data/test_wikipedia)"
    )
    parser.add_argument(
        "--max_articles",
        type=int,
        default=1000,
        help="Maximum number of articles to download (default: 1000)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="simple",
        choices=["simple", "squad"],
        help="Dataset to use (simple=Simple Wikipedia, squad=SQuAD contexts)"
    )

    args = parser.parse_args()

    if args.dataset == "simple":
        download_simple_wikipedia(args.output_dir, args.max_articles)
    else:
        download_squad_context(args.output_dir, args.max_articles)


if __name__ == "__main__":
    main()
