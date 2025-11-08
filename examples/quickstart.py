"""
Quick Start Example for RAG System

This script demonstrates how to use the RAG implementation for basic question answering.
"""

import sys
sys.path.insert(0, "../")

import torch
from rag import RAGSequenceForGeneration, RAGTokenForGeneration, RAGConfig
from rag.retrieval import DPRRetriever, MockRetriever
from rag.generation import BARTGenerator

print("="*80)
print("RAG System Quick Start Example")
print("="*80)

# ============================================================================
# Example 1: Using RAG with Mock Retriever (for testing without Wikipedia index)
# ============================================================================

print("\n[Example 1] RAG with Mock Retriever")
print("-" * 80)

# Create configuration
config = RAGConfig(
    model_type="rag_sequence",
    num_retrieved_docs=5,
    generator_name_or_path="facebook/bart-base",  # Using base for faster loading
    use_thorough_decoding=False,  # Fast decoding
)

# Create mock retriever (returns dummy documents)
mock_retriever = MockRetriever(num_docs=100, embed_dim=768)

# Initialize RAG-Sequence model
print("Loading RAG-Sequence model...")
rag_model = RAGSequenceForGeneration(
    config=config,
    retriever=mock_retriever,
)

# Move to device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
rag_model = rag_model.to(device)
rag_model.eval()

# Generate answer to a question
question = "What is the capital of France?"
print(f"\nQuestion: {question}")
print("Generating answer...")

try:
    answer = rag_model.generate_from_query(
        query=question,
        max_length=50,
        num_beams=2,
    )
    print(f"Answer: {answer}")
except Exception as e:
    print(f"Error during generation: {e}")
    print("Note: This is expected with mock retriever - it's for testing structure only")

# ============================================================================
# Example 2: RAG-Token vs RAG-Sequence Comparison
# ============================================================================

print("\n[Example 2] Comparing RAG-Token and RAG-Sequence")
print("-" * 80)

# RAG-Sequence config
seq_config = RAGConfig(
    model_type="rag_sequence",
    num_retrieved_docs=5,
    generator_name_or_path="facebook/bart-base",
)

# RAG-Token config
token_config = RAGConfig(
    model_type="rag_token",
    num_retrieved_docs=5,
    generator_name_or_path="facebook/bart-base",
)

print("Model configurations:")
print(f"  RAG-Sequence: k={seq_config.num_retrieved_docs}, thorough_decoding={seq_config.use_thorough_decoding}")
print(f"  RAG-Token: k={token_config.num_retrieved_docs}")

# ============================================================================
# Example 3: Building a Simple Index (demonstrative)
# ============================================================================

print("\n[Example 3] Building a Simple Document Index")
print("-" * 80)

# Sample documents (in real use, this would be Wikipedia)
sample_documents = [
    "Paris is the capital and most populous city of France.",
    "France is a country in Western Europe.",
    "The Eiffel Tower is located in Paris, France.",
    "London is the capital of the United Kingdom.",
    "Berlin is the capital of Germany.",
]

print(f"Sample documents ({len(sample_documents)} docs):")
for i, doc in enumerate(sample_documents, 1):
    print(f"  {i}. {doc}")

# In a real scenario, you would:
# 1. Load Wikipedia dump
# 2. Split into 100-word chunks
# 3. Encode with DPR document encoder
# 4. Build FAISS index

print("\nTo build a real index:")
print("  1. Download Wikipedia dump (Dec 2018)")
print("  2. Run: python scripts/build_index.py --input wikipedia.txt --output ./data/index")
print("  3. This creates a FAISS index with 21M document vectors")

# ============================================================================
# Example 4: Configuration for Different Tasks
# ============================================================================

print("\n[Example 4] Task-Specific Configurations")
print("-" * 80)

from rag.config import get_default_config

tasks = ["open_qa", "abstractive_qa", "question_generation", "fact_verification"]

for task in tasks:
    task_config = get_default_config(task)
    print(f"\n{task.upper()}:")
    print(f"  Model: {task_config['rag'].model_type}")
    print(f"  Num docs: {task_config['rag'].num_retrieved_docs}")
    print(f"  Max length: {task_config['rag'].generator_max_length}")
    print(f"  Metrics: {', '.join(task_config['task'].metric_names)}")

# ============================================================================
# Example 5: Loading Pre-trained RAG Model (when available)
# ============================================================================

print("\n[Example 5] Loading Pre-trained RAG Model")
print("-" * 80)

print("To load a pre-trained RAG model:")
print("""
from rag import RAGSequenceForGeneration

# Load from HuggingFace
model = RAGSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

# Or load your own trained model
model = RAGSequenceForGeneration.from_pretrained("./outputs/my_rag_model")

# Generate
answer = model.generate_from_query("What is RAG?")
""")

# ============================================================================
# Example 6: Training Loop Sketch
# ============================================================================

print("\n[Example 6] Training Loop (Sketch)")
print("-" * 80)

print("""
# Pseudocode for training RAG

from rag import RAGSequenceForGeneration, RAGConfig
from rag.training import RAGTrainer
from datasets import load_dataset

# Load model
config = RAGConfig(num_retrieved_docs=5)
model = RAGSequenceForGeneration(config)

# Load dataset
dataset = load_dataset("natural_questions")

# Create trainer
trainer = RAGTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    learning_rate=3e-5,
    num_epochs=10,
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
print(f"Exact Match: {results['exact_match']}")
""")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("Quick Start Summary")
print("="*80)

print("""
✅ What's Implemented:
  - RAG-Sequence and RAG-Token models
  - DPR retrieval with FAISS indexing
  - BM25 baseline retriever
  - BART generator wrapper
  - Configuration system
  - Model saving/loading

🚧 Next Steps:
  1. Build Wikipedia index (21M passages)
  2. Implement training pipeline
  3. Add task-specific datasets
  4. Add evaluation metrics
  5. Create complete examples

📚 For More Information:
  - README.md: Full documentation
  - IMPLEMENTATION_STATUS.md: Current progress
  - Paper: arxiv.org/abs/2005.11401

🎯 To Run with Real Data:
  1. Build index: python scripts/build_index.py
  2. Train model: python scripts/train.py --task open_qa
  3. Evaluate: python scripts/evaluate.py --model ./outputs/model

""")

print("="*80)
print("End of Quick Start Example")
print("="*80)
