#!/usr/bin/env python3
"""
Merge multiple Wikipedia batch files into a single file for indexing.
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_batches(input_dir: str, output_file: str, max_passages: int = 200000):
    """
    Merge Wikipedia batch files.

    Args:
        input_dir: Directory containing wikipedia_batch_*.jsonl files
        output_file: Output merged file path
        max_passages: Maximum number of passages to include
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all batch files
    batch_files = sorted(input_path.glob("wikipedia_batch_*.jsonl"),
                         key=lambda x: int(x.stem.split('_')[-1]))

    logger.info(f"Found {len(batch_files)} batch files")
    logger.info(f"Will merge up to {max_passages} passages")

    passages_written = 0

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for batch_file in tqdm(batch_files, desc="Processing batches"):
            if passages_written >= max_passages:
                break

            with open(batch_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if passages_written >= max_passages:
                        break

                    out_f.write(line)
                    passages_written += 1

    logger.info(f"✓ Merged {passages_written} passages to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / (1024**2):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Merge Wikipedia batch files")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing batch files")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output merged file path")
    parser.add_argument("--max_passages", type=int, default=200000,
                       help="Maximum passages to include (default: 200000)")

    args = parser.parse_args()

    merge_batches(args.input_dir, args.output_file, args.max_passages)


if __name__ == "__main__":
    main()
