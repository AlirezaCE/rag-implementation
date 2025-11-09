#!/usr/bin/env python3
"""
Build FAISS index for RAG retrieval.

This script builds a FAISS index from preprocessed passages using DPR embeddings.
Following the RAG paper:
- Uses DPR (Dense Passage Retrieval) for encoding passages
- Builds FAISS index for efficient Maximum Inner Product Search (MIPS)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_passages(passages_file: str, max_passages: int = None):
    """
    Load passages from JSONL file.

    Args:
        passages_file: Path to passages JSONL file
        max_passages: Maximum number of passages to load (for testing)

    Returns:
        List of passage dictionaries
    """
    passages = []
    with open(passages_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, desc="Loading passages")):
            if max_passages and idx >= max_passages:
                break
            passage = json.loads(line)
            passages.append(passage)

    logger.info(f"Loaded {len(passages)} passages")
    return passages


def encode_passages(passages, model_name: str = "facebook/dpr-ctx_encoder-single-nq-base", batch_size: int = 32):
    """
    Encode passages using DPR context encoder.

    Args:
        passages: List of passage dictionaries
        model_name: HuggingFace model name for passage encoder
        batch_size: Batch size for encoding

    Returns:
        numpy array of embeddings [num_passages, embedding_dim]
    """
    try:
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
    except ImportError:
        logger.error("transformers library not found")
        raise

    logger.info(f"Loading DPR encoder: {model_name}")

    # Load model and tokenizer
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
    model = DPRContextEncoder.from_pretrained(model_name)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    logger.info(f"Using device: {device}")
    logger.info(f"Encoding {len(passages)} passages...")

    all_embeddings = []

    # Encode in batches
    for i in tqdm(range(0, len(passages), batch_size), desc="Encoding"):
        batch_passages = passages[i:i + batch_size]
        batch_texts = [p['text'] for p in batch_passages]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            embeddings = model(**inputs).pooler_output

        all_embeddings.append(embeddings.cpu().numpy())

    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)

    logger.info(f"Encoded passages shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings, index_type: str = "Flat"):
    """
    Build FAISS index from embeddings.

    Args:
        embeddings: numpy array of embeddings [num_passages, embedding_dim]
        index_type: Type of FAISS index ("Flat", "IVF", "HNSW")

    Returns:
        FAISS index
    """
    try:
        import faiss
    except ImportError:
        logger.error("faiss library not found. Install with: pip install faiss-gpu or faiss-cpu")
        raise

    dimension = embeddings.shape[1]
    num_passages = embeddings.shape[0]

    logger.info(f"Building FAISS index (type: {index_type})")
    logger.info(f"Dimension: {dimension}, Num passages: {num_passages}")

    # Normalize embeddings for cosine similarity (MIPS)
    faiss.normalize_L2(embeddings)

    if index_type == "Flat":
        # Exact search (slower but accurate)
        index = faiss.IndexFlatIP(dimension)

    elif index_type == "IVF":
        # Inverted file index (faster, approximate)
        nlist = min(4096, num_passages // 39)  # Number of clusters
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

        # Train the index
        logger.info("Training IVF index...")
        index.train(embeddings)

    elif index_type == "HNSW":
        # Hierarchical Navigable Small World (fast, approximate)
        M = 32  # Number of connections per layer
        index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)

    else:
        raise ValueError(f"Unknown index type: {index_type}")

    # Add vectors to index
    logger.info("Adding vectors to index...")
    index.add(embeddings)

    logger.info(f"Index built successfully! Total vectors: {index.ntotal}")
    return index


def save_index(index, passages, output_dir: str):
    """
    Save FAISS index and passage metadata.

    Args:
        index: FAISS index
        passages: List of passage dictionaries
        output_dir: Output directory
    """
    import faiss

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    index_file = output_path / "index.faiss"
    logger.info(f"Saving FAISS index to {index_file}")
    faiss.write_index(index, str(index_file))

    # Save passages metadata
    passages_file = output_path / "passages.jsonl"
    logger.info(f"Saving passages metadata to {passages_file}")
    with open(passages_file, 'w', encoding='utf-8') as f:
        for passage in tqdm(passages, desc="Saving passages"):
            f.write(json.dumps(passage, ensure_ascii=False) + '\n')

    # Save index metadata
    metadata = {
        'num_passages': len(passages),
        'embedding_dim': index.d,
        'index_type': type(index).__name__,
        'total_vectors': index.ntotal
    }
    metadata_file = output_path / "index_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Index saved to {output_path}")
    logger.info(f"Metadata: {metadata}")


def build_index(
    passages_file: str,
    output_dir: str,
    model_name: str = "facebook/dpr-ctx_encoder-single-nq-base",
    index_type: str = "Flat",
    batch_size: int = 32,
    max_passages: int = None
):
    """
    Build complete FAISS index from passages.

    Args:
        passages_file: Path to preprocessed passages JSONL file
        output_dir: Directory to save index and metadata
        model_name: HuggingFace model name for encoding
        index_type: Type of FAISS index
        batch_size: Batch size for encoding
        max_passages: Maximum passages to index (for testing)
    """
    # Load passages
    passages = load_passages(passages_file, max_passages)

    # Encode passages
    embeddings = encode_passages(passages, model_name, batch_size)

    # Build index
    index = build_faiss_index(embeddings, index_type)

    # Save everything
    save_index(index, passages, output_dir)

    logger.info("\n✓ Index building complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index for RAG retrieval"
    )
    parser.add_argument(
        "--passages_file",
        type=str,
        required=True,
        help="Path to preprocessed passages JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save index and metadata"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/dpr-ctx_encoder-single-nq-base",
        help="HuggingFace model name for passage encoding"
    )
    parser.add_argument(
        "--index_type",
        type=str,
        default="Flat",
        choices=["Flat", "IVF", "HNSW"],
        help="Type of FAISS index (Flat=exact, IVF/HNSW=approximate)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--max_passages",
        type=int,
        default=None,
        help="Maximum number of passages to index (for testing)"
    )

    args = parser.parse_args()

    build_index(
        passages_file=args.passages_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        index_type=args.index_type,
        batch_size=args.batch_size,
        max_passages=args.max_passages
    )


if __name__ == "__main__":
    main()
