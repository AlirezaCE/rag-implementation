#!/usr/bin/env python3
"""
Download and process Wikipedia dump for RAG.

This script downloads a Wikipedia dump from the specified date and
processes it into a format suitable for RAG retrieval.
"""

import argparse
import os
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


def download_wikipedia(output_dir: str, date: str = "2018-12", language: str = "en"):
    """
    Download Wikipedia dump using HuggingFace datasets.

    Args:
        output_dir: Directory to save processed Wikipedia articles
        date: Wikipedia dump date (format: YYYY-MM)
        language: Wikipedia language code (default: en)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not found. Install with: pip install datasets")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading Wikipedia dump (date: {date}, language: {language})")
    logger.info(f"Output directory: {output_path.absolute()}")

    try:
        # Load Wikipedia dataset using the new Wikimedia dumps format
        # The new recommended way is to use wikimedia/wikipedia dataset
        logger.info("Loading Wikipedia dataset from Wikimedia...")
        dataset = load_dataset(
            "wikimedia/wikipedia",
            f"{date}.{language}",
            split="train"
        )

        logger.info(f"Loaded {len(dataset)} Wikipedia articles")

        # Process and save articles
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

            # Save in batches to avoid memory issues
            if (idx + 1) % 10000 == 0:
                batch_file = output_path / f"wikipedia_batch_{(idx + 1) // 10000}.jsonl"
                with open(batch_file, 'w', encoding='utf-8') as f:
                    for doc in documents:
                        f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                logger.info(f"Saved batch to {batch_file}")
                documents = []

        # Save remaining documents
        if documents:
            batch_num = (len(dataset) // 10000) + 1
            batch_file = output_path / f"wikipedia_batch_{batch_num}.jsonl"
            with open(batch_file, 'w', encoding='utf-8') as f:
                for doc in documents:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            logger.info(f"Saved final batch to {batch_file}")

        # Save metadata
        metadata = {
            "date": date,
            "language": language,
            "total_articles": len(dataset),
            "output_dir": str(output_path.absolute())
        }
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Download complete! Processed {len(dataset)} articles")
        logger.info(f"Metadata saved to {metadata_file}")

    except Exception as e:
        logger.error(f"Error downloading Wikipedia: {e}")
        logger.error("Note: Wikipedia dumps might not be available for all dates.")
        logger.error("Try using a different date or use the alternative method below.")
        logger.info("\nAlternative: You can also use a simpler dataset for testing:")
        logger.info("  from datasets import load_dataset")
        logger.info("  dataset = load_dataset('wiki_snippets', split='train')")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download Wikipedia dump for RAG"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed Wikipedia articles"
    )
    parser.add_argument(
        "--date",
        type=str,
        default="20220301",
        help="Wikipedia dump date (format: YYYYMMDD, default: 20220301)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Wikipedia language code (default: en)"
    )

    args = parser.parse_args()

    download_wikipedia(
        output_dir=args.output_dir,
        date=args.date,
        language=args.language
    )


if __name__ == "__main__":
    main()
