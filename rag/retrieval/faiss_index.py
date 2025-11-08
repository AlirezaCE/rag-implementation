"""
FAISS index for efficient Maximum Inner Product Search (MIPS).

As described in RAG paper Section 2.2:
"Calculating top-k(pη(·|x)), the list of k documents z with highest prior
probability pη(z|x), is a Maximum Inner Product Search (MIPS) problem,
which can be approximately solved in sub-linear time."

Uses Hierarchical Navigable Small World (HNSW) approximation as in the paper.
"""

import numpy as np
import torch
from typing import Tuple, Optional, List
import os


class FAISSIndex:
    """
    Wrapper for FAISS index operations.

    Supports various index types including:
    - IndexFlatIP: Exact inner product search
    - IndexHNSWFlat: HNSW approximation (used in paper)
    - IndexIVFFlat: Inverted file index
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "IndexHNSWFlat",
        metric: str = "inner_product",
        use_gpu: bool = False,
        nprobe: int = 128,
        **kwargs
    ):
        """
        Initialize FAISS index.

        Args:
            embedding_dim: Dimension of embeddings (768 for BERT-base)
            index_type: Type of FAISS index
            metric: Distance metric ('inner_product' or 'l2')
            use_gpu: Whether to use GPU for search
            nprobe: Number of cells to visit for IVF indices
            **kwargs: Additional index-specific parameters
        """
        import faiss

        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.use_gpu = use_gpu
        self.nprobe = nprobe

        # Create index
        self.index = self._create_index(index_type, **kwargs)

        # Move to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            print(f"Using GPU for FAISS index")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def _create_index(self, index_type: str, **kwargs):
        """Create FAISS index of specified type."""
        import faiss

        if index_type == "IndexFlatIP":
            # Exact inner product search
            index = faiss.IndexFlatIP(self.embedding_dim)

        elif index_type == "IndexHNSWFlat":
            # Hierarchical Navigable Small World (as in paper)
            # Good balance between speed and accuracy
            M = kwargs.get("M", 128)  # Number of connections per layer
            index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            index.hnsw.efConstruction = kwargs.get("efConstruction", 200)
            index.hnsw.efSearch = kwargs.get("efSearch", 128)

        elif index_type == "IndexIVFFlat":
            # Inverted file index
            nlist = kwargs.get("nlist", 100)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            index.nprobe = self.nprobe

        elif index_type == "IndexIVFPQ":
            # Inverted file with product quantization (memory efficient)
            nlist = kwargs.get("nlist", 100)
            m = kwargs.get("m", 8)  # Number of subquantizers
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, 8)
            index.nprobe = self.nprobe

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        return index

    def add(self, embeddings: np.ndarray):
        """
        Add embeddings to index.

        Args:
            embeddings: Numpy array of shape (num_docs, embedding_dim)
        """
        import faiss

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Normalize for inner product search (converts to cosine similarity)
        if self.metric == "inner_product":
            faiss.normalize_L2(embeddings)

        # Train index if needed (for IVF indices)
        if not self.index.is_trained:
            print(f"Training index on {len(embeddings)} vectors...")
            self.index.train(embeddings)

        # Add vectors
        print(f"Adding {len(embeddings)} vectors to index...")
        self.index.add(embeddings)
        print(f"Index now contains {self.index.ntotal} vectors")

    def search(
        self,
        queries: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-K nearest neighbors.

        Implements Maximum Inner Product Search (MIPS).

        Args:
            queries: Query embeddings of shape (num_queries, embedding_dim)
            k: Number of neighbors to return

        Returns:
            scores: Array of shape (num_queries, k) with similarity scores
            indices: Array of shape (num_queries, k) with document indices
        """
        import faiss

        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)

        # Normalize queries
        if self.metric == "inner_product":
            faiss.normalize_L2(queries)

        # Search
        scores, indices = self.index.search(queries, k)

        return scores, indices

    def reconstruct(self, idx: int) -> np.ndarray:
        """
        Reconstruct vector at given index.

        Args:
            idx: Index of vector to reconstruct

        Returns:
            Reconstructed vector
        """
        return self.index.reconstruct(int(idx))

    def reconstruct_batch(self, indices: List[int]) -> np.ndarray:
        """
        Reconstruct multiple vectors.

        Args:
            indices: List of indices

        Returns:
            Array of reconstructed vectors
        """
        vectors = []
        for idx in indices:
            vectors.append(self.reconstruct(idx))
        return np.array(vectors)

    def save(self, path: str):
        """
        Save index to disk.

        Args:
            path: Path to save index
        """
        import faiss

        # Move to CPU if on GPU
        index_to_save = self.index
        if self.use_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self.index)

        # Save
        faiss.write_index(index_to_save, path)
        print(f"Index saved to {path}")

    @classmethod
    def load(cls, path: str, use_gpu: bool = False) -> "FAISSIndex":
        """
        Load index from disk.

        Args:
            path: Path to load index from
            use_gpu: Whether to move index to GPU

        Returns:
            Loaded FAISSIndex instance
        """
        import faiss

        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")

        # Load index
        index = faiss.read_index(path)
        print(f"Loaded index with {index.ntotal} vectors")

        # Create wrapper instance
        instance = cls.__new__(cls)
        instance.embedding_dim = index.d
        instance.index = index
        instance.use_gpu = use_gpu

        # Move to GPU if requested
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            instance.index = faiss.index_cpu_to_gpu(res, 0, index)

        return instance

    @property
    def ntotal(self) -> int:
        """Get total number of indexed vectors."""
        return self.index.ntotal


def build_index(
    embeddings: np.ndarray,
    index_type: str = "IndexHNSWFlat",
    use_gpu: bool = False,
    save_path: Optional[str] = None,
    **kwargs
) -> FAISSIndex:
    """
    Build FAISS index from embeddings.

    Args:
        embeddings: Numpy array of shape (num_docs, embedding_dim)
        index_type: Type of FAISS index
        use_gpu: Whether to use GPU
        save_path: Optional path to save index
        **kwargs: Additional index parameters

    Returns:
        Built FAISSIndex instance
    """
    print(f"Building {index_type} index for {len(embeddings)} vectors...")

    # Create index
    index = FAISSIndex(
        embedding_dim=embeddings.shape[1],
        index_type=index_type,
        use_gpu=use_gpu,
        **kwargs
    )

    # Add embeddings
    index.add(embeddings)

    # Save if requested
    if save_path:
        index.save(save_path)

    return index


def compress_index(
    index: FAISSIndex,
    output_path: str,
    compression_type: str = "PQ",
    **kwargs
):
    """
    Compress FAISS index to reduce memory usage.

    As mentioned in paper appendix:
    "We also compress the document index using FAISS's compression tools,
    reducing the CPU memory requirement to 36GB."

    Args:
        index: Input FAISSIndex
        output_path: Path to save compressed index
        compression_type: Type of compression ('PQ', 'SQ', etc.)
        **kwargs: Compression parameters
    """
    import faiss

    print(f"Compressing index with {compression_type}...")

    original_index = index.index
    if index.use_gpu:
        original_index = faiss.index_gpu_to_cpu(original_index)

    if compression_type == "PQ":
        # Product Quantization
        m = kwargs.get("m", 8)
        compressed = faiss.IndexIVFPQ(
            faiss.IndexFlatIP(index.embedding_dim),
            index.embedding_dim,
            kwargs.get("nlist", 100),
            m,
            8
        )
        compressed.train(kwargs.get("training_vectors"))
        compressed.add(kwargs.get("training_vectors"))

    elif compression_type == "SQ":
        # Scalar Quantization
        compressed = faiss.IndexScalarQuantizer(
            index.embedding_dim,
            faiss.ScalarQuantizer.QT_8bit
        )
        compressed.train(kwargs.get("training_vectors"))
        compressed.add(kwargs.get("training_vectors"))

    else:
        raise ValueError(f"Unknown compression type: {compression_type}")

    # Save compressed index
    faiss.write_index(compressed, output_path)
    print(f"Compressed index saved to {output_path}")
