"""
Configuration classes for RAG models.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json


@dataclass
class RAGConfig:
    """Configuration class for RAG models."""

    # Model architecture
    model_type: str = "rag_sequence"  # or "rag_token"

    # Generator config
    generator_name_or_path: str = "facebook/bart-large"
    generator_max_length: int = 128
    generator_num_beams: int = 4

    # Retriever config
    retriever_name_or_path: str = "facebook/dpr-question_encoder-single-nq-base"
    doc_encoder_name_or_path: str = "facebook/dpr-ctx_encoder-single-nq-base"

    # Retrieval parameters
    num_retrieved_docs: int = 5  # k in paper (can be 5, 10, 50)
    max_doc_length: int = 100  # 100-word chunks as in paper
    index_name: str = "exact"  # FAISS index type

    # Index paths
    index_path: Optional[str] = None
    passages_path: Optional[str] = None

    # Training config
    freeze_retriever: bool = True  # Freeze document encoder (as in paper)
    freeze_generator: bool = False

    # Decoding strategy for RAG-Sequence
    use_thorough_decoding: bool = False  # Use thorough vs fast decoding

    # Mixed precision
    use_fp16: bool = True

    # Additional config
    n_docs: int = 21015324  # Total number of documents (21M Wikipedia passages)
    retrieval_vector_size: int = 768  # BERT-base hidden size
    retrieval_batch_size: int = 8

    # Device
    device: str = "cuda"  # or "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_str: str) -> "RAGConfig":
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, path: str):
        """Save config to file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "RAGConfig":
        """Load config from file."""
        with open(path, "r") as f:
            return cls.from_json(f.read())


@dataclass
class RetrievalConfig:
    """Configuration for retrieval component."""

    # Retriever type
    retriever_type: str = "dpr"  # "dpr" or "bm25"

    # DPR config
    query_encoder: str = "facebook/dpr-question_encoder-single-nq-base"
    doc_encoder: str = "facebook/dpr-ctx_encoder-single-nq-base"

    # FAISS config
    index_type: str = "IndexHNSWFlat"  # Hierarchical Navigable Small World (as in paper)
    index_path: Optional[str] = None
    use_gpu: bool = True

    # Retrieval parameters
    top_k: int = 5
    batch_size: int = 32
    max_length: int = 256

    # BM25 config (for ablation)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass
class GeneratorConfig:
    """Configuration for generator component."""

    # Model
    model_name: str = "facebook/bart-large"

    # Generation parameters
    max_length: int = 128
    min_length: int = 1
    num_beams: int = 4
    length_penalty: float = 1.0
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3

    # Special tokens
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None


@dataclass
class TrainingConfig:
    """Training configuration for RAG models."""

    # Model paths
    model_name_or_path: Optional[str] = None
    output_dir: str = "./outputs"

    # Training hyperparameters
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 2  # Small due to memory constraints
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = 8
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 500

    # Mixed precision (as in paper)
    fp16: bool = True
    fp16_opt_level: str = "O1"

    # Retriever training
    freeze_retriever: bool = True  # Keep document encoder frozen (as in paper)
    update_retriever: bool = True  # Update query encoder

    # Logging and checkpointing
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3

    # Evaluation
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Data
    max_source_length: int = 256
    max_target_length: int = 128
    num_workers: int = 4

    # Distributed training
    local_rank: int = -1

    # Seed
    seed: int = 42

    # Wandb/TensorBoard
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None


@dataclass
class TaskConfig:
    """Task-specific configuration."""

    # Task type
    task: str = "open_qa"  # open_qa, abstractive_qa, question_generation, fact_verification

    # Dataset
    dataset_name: str = "natural_questions"
    dataset_config: Optional[str] = None

    # Task-specific parameters
    num_labels: Optional[int] = None  # For classification tasks
    metric_names: List[str] = field(default_factory=lambda: ["exact_match"])

    # Data processing
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False

    # Answer extraction (for QA)
    n_best_size: int = 20
    max_answer_length: int = 30

    # For FEVER
    fever_mode: str = "3-way"  # "2-way" or "3-way" classification


def get_default_config(task: str) -> Dict[str, Any]:
    """Get default configuration for a specific task."""

    configs = {
        "open_qa": {
            "rag": RAGConfig(
                model_type="rag_sequence",
                num_retrieved_docs=5,
                generator_max_length=50,
                generator_num_beams=1,  # Greedy decoding for QA (as in paper)
                use_thorough_decoding=True,
            ),
            "training": TrainingConfig(
                num_train_epochs=10,
                learning_rate=3e-5,
                per_device_train_batch_size=2,
            ),
            "task": TaskConfig(
                task="open_qa",
                metric_names=["exact_match"],
            ),
        },
        "abstractive_qa": {
            "rag": RAGConfig(
                model_type="rag_sequence",
                num_retrieved_docs=10,
                generator_max_length=128,
                generator_num_beams=4,
                use_thorough_decoding=False,  # Fast decoding
            ),
            "training": TrainingConfig(
                num_train_epochs=10,
                learning_rate=3e-5,
            ),
            "task": TaskConfig(
                task="abstractive_qa",
                metric_names=["bleu", "rouge"],
            ),
        },
        "question_generation": {
            "rag": RAGConfig(
                model_type="rag_token",  # RAG-Token performs better
                num_retrieved_docs=10,
                generator_max_length=128,
                generator_num_beams=4,
            ),
            "training": TrainingConfig(
                num_train_epochs=10,
                learning_rate=3e-5,
            ),
            "task": TaskConfig(
                task="question_generation",
                metric_names=["bleu", "qbleu"],
            ),
        },
        "fact_verification": {
            "rag": RAGConfig(
                model_type="rag_token",  # Either works for classification
                num_retrieved_docs=10,
                generator_max_length=1,  # Single token for class
            ),
            "training": TrainingConfig(
                num_train_epochs=10,
                learning_rate=3e-5,
            ),
            "task": TaskConfig(
                task="fact_verification",
                metric_names=["accuracy"],
                num_labels=3,  # SUPPORTS, REFUTES, NOT ENOUGH INFO
                fever_mode="3-way",
            ),
        },
    }

    return configs.get(task, configs["open_qa"])
