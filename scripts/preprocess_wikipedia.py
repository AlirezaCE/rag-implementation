#!/usr/bin/env python3
"""
Preprocess Wikipedia articles for RAG.

This script processes raw Wikipedia articles into passages suitable for retrieval:
- Splits long articles into passages (100 words as per RAG paper)
- Cleans and normalizes text
- Creates passage database
"""

import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Args:
        text: Raw text string

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)

    # Remove very short lines (likely formatting artifacts)
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
    text = ' '.join(lines)

    return text.strip()


def split_into_passages(text: str, title: str, passage_length: int = 100) -> List[str]:
    """
    Split text into passages of approximately passage_length words.

    Following RAG paper: "We split documents into disjoint 100-word passages"

    Args:
        text: Article text
        title: Article title (prepended to each passage)
        passage_length: Target number of words per passage

    Returns:
        List of passage strings
    """
    words = text.split()
    passages = []

    # Split into chunks
    for i in range(0, len(words), passage_length):
        passage_words = words[i:i + passage_length]
        if len(passage_words) >= 20:  # Only keep passages with at least 20 words
            # Prepend title to passage as per RAG paper
            passage = f"{title}. {' '.join(passage_words)}"
            passages.append(passage)

    return passages


def preprocess_wikipedia(
    input_dir: str,
    output_file: str,
    passage_length: int = 100,
    max_articles: int = None
):
    """
    Preprocess Wikipedia articles into passages.

    Args:
        input_dir: Directory containing raw Wikipedia JSONL files
        output_file: Output file path for processed passages
        passage_length: Number of words per passage
        max_articles: Maximum number of articles to process (for testing)
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing Wikipedia articles from {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Passage length: {passage_length} words")

    # Find all JSONL files
    jsonl_files = sorted(input_path.glob("wikipedia_batch_*.jsonl"))

    if not jsonl_files:
        logger.error(f"No Wikipedia batch files found in {input_path}")
        logger.info("Expected files like: wikipedia_batch_1.jsonl, wikipedia_batch_2.jsonl, etc.")
        return

    logger.info(f"Found {len(jsonl_files)} batch files")

    all_passages = []
    article_count = 0
    passage_count = 0

    # Process each batch file
    for batch_file in jsonl_files:
        logger.info(f"Processing {batch_file.name}...")

        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing {batch_file.name}"):
                if max_articles and article_count >= max_articles:
                    break

                try:
                    article = json.loads(line)
                    title = article.get('title', '')
                    text = article.get('text', '')

                    if not title or not text:
                        continue

                    # Clean text
                    text = clean_text(text)

                    if len(text) < 50:  # Skip very short articles
                        continue

                    # Split into passages
                    passages = split_into_passages(text, title, passage_length)

                    # Add passages with metadata
                    for idx, passage in enumerate(passages):
                        passage_data = {
                            'id': f"{article['id']}_passage_{idx}",
                            'title': title,
                            'text': passage,
                            'article_id': article['id']
                        }
                        all_passages.append(passage_data)
                        passage_count += 1

                    article_count += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing JSON: {e}")
                    continue

        if max_articles and article_count >= max_articles:
            break

    # Save processed passages
    logger.info(f"Saving {len(all_passages)} passages to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        for passage in tqdm(all_passages, desc="Writing passages"):
            f.write(json.dumps(passage, ensure_ascii=False) + '\n')

    # Save statistics
    stats = {
        'total_articles': article_count,
        'total_passages': passage_count,
        'avg_passages_per_article': passage_count / article_count if article_count > 0 else 0,
        'passage_length_words': passage_length
    }

    stats_file = output_path.parent / f"{output_path.stem}_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nPreprocessing complete!")
    logger.info(f"Processed {article_count} articles into {passage_count} passages")
    logger.info(f"Average passages per article: {stats['avg_passages_per_article']:.2f}")
    logger.info(f"Output saved to: {output_path}")
    logger.info(f"Statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Wikipedia articles into passages for RAG"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw Wikipedia JSONL files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file path for processed passages (JSONL format)"
    )
    parser.add_argument(
        "--passage_length",
        type=int,
        default=100,
        help="Number of words per passage (default: 100, as per RAG paper)"
    )
    parser.add_argument(
        "--max_articles",
        type=int,
        default=None,
        help="Maximum number of articles to process (for testing)"
    )

    args = parser.parse_args()

    preprocess_wikipedia(
        input_dir=args.input_dir,
        output_file=args.output_file,
        passage_length=args.passage_length,
        max_articles=args.max_articles
    )


if __name__ == "__main__":
    main()
