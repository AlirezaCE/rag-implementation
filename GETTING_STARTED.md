# Getting Started with RAG Implementation

This guide will help you get started with the RAG (Retrieval-Augmented Generation) implementation based on the paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020).

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Building Document Index](#building-document-index)
4. [Training a Model](#training-a-model)
5. [Running Inference](#running-inference)
6. [Evaluation](#evaluation)
7. [Advanced Usage](#advanced-usage)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, especially for training)
- 64GB+ RAM (for full Wikipedia index)

### Basic Installation

```bash
# Clone repository
cd /path/to/RAG_KI_NLP2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Optional: Install with GPU Support

```bash
# For FAISS with GPU
conda install -c conda-forge faiss-gpu

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## Quick Start

### Option 1: Using Mock Retriever (for testing)

```python
from rag import RAGSequenceForGeneration, RAGConfig
from rag.retrieval import MockRetriever

# Create config
config = RAGConfig(
    model_type="rag_sequence",
    num_retrieved_docs=5,
    generator_name_or_path="facebook/bart-base",
)

# Create model with mock retriever
retriever = MockRetriever(num_docs=100)
model = RAGSequenceForGeneration(config=config, retriever=retriever)

# Generate
question = "What is the capital of France?"
answer = model.generate_from_query(question, max_length=50)
print(answer)
```

### Option 2: Using Pre-trained Models (when available)

```python
from rag import RAGSequenceForGeneration

# Load pre-trained model
model = RAGSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

# Generate
answer = model.generate_from_query("What is RAG?")
print(answer)
```

## Building Document Index

The RAG system requires a document index for retrieval. Here's how to build one:

### Step 1: Download Wikipedia

```bash
# Download December 2018 Wikipedia dump (as used in paper)
# This will download ~20GB of data
python scripts/download_wikipedia.py \
    --output_dir ./data/wikipedia \
    --date 2018-12
```

### Step 2: Preprocess Documents

```bash
# Split articles into 100-word chunks (as in paper)
python scripts/preprocess_wikipedia.py \
    --input_dir ./data/wikipedia \
    --output_file ./data/passages.jsonl \
    --chunk_size 100 \
    --max_passages 21000000
```

### Step 3: Build FAISS Index

```bash
# Build index with DPR document encoder
python scripts/build_index.py \
    --passages_file ./data/passages.jsonl \
    --output_dir ./data/index \
    --doc_encoder facebook/dpr-ctx_encoder-single-nq-base \
    --index_type IndexHNSWFlat \
    --batch_size 32 \
    --use_gpu
```

This will create:
- `./data/index/index.faiss` - FAISS index (~100GB, or ~36GB compressed)
- `./data/index/passages.json` - Passage metadata

### Step 4: Load Index

```python
from rag import RAGSequenceForGeneration, RAGConfig

config = RAGConfig(
    index_path="./data/index/index.faiss",
    passages_path="./data/index/passages.json",
)

model = RAGSequenceForGeneration(config)
```

## Training a Model

### Prepare Dataset

```python
from datasets import load_dataset

# Load Natural Questions dataset
dataset = load_dataset("natural_questions")

# Or custom dataset
from rag.data import prepare_qa_dataset

train_data = prepare_qa_dataset(
    questions=["What is...?", "Who was...?"],
    answers=["Paris", "Einstein"],
)
```

### Train RAG-Sequence

```bash
python scripts/train.py \
    --task open_qa \
    --dataset natural_questions \
    --model_type rag_sequence \
    --output_dir ./outputs/nq_rag_sequence \
    --num_epochs 10 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --num_retrieved_docs 5 \
    --fp16 \
    --freeze_retriever
```

### Train RAG-Token

```bash
python scripts/train.py \
    --task open_qa \
    --dataset natural_questions \
    --model_type rag_token \
    --output_dir ./outputs/nq_rag_token \
    --num_epochs 10 \
    --batch_size 2 \
    --learning_rate 3e-5 \
    --num_retrieved_docs 5
```

### Training from Python

```python
from rag import RAGSequenceForGeneration, RAGConfig
from rag.training import RAGTrainer, TrainingConfig
from datasets import load_dataset

# Load model
model_config = RAGConfig(num_retrieved_docs=5)
model = RAGSequenceForGeneration(model_config)

# Load data
dataset = load_dataset("natural_questions")

# Create trainer
train_config = TrainingConfig(
    output_dir="./outputs",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    learning_rate=3e-5,
    fp16=True,
)

trainer = RAGTrainer(
    model=model,
    config=train_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# Train
trainer.train()
```

## Running Inference

### Basic Inference

```python
from rag import RAGSequenceForGeneration

# Load trained model
model = RAGSequenceForGeneration.from_pretrained("./outputs/my_model")

# Single question
answer = model.generate_from_query(
    query="What is the capital of France?",
    max_length=50,
    num_beams=4,
)
print(answer)
```

### Batch Inference

```python
import torch
from rag import RAGSequenceForGeneration

model = RAGSequenceForGeneration.from_pretrained("./outputs/my_model")

questions = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "When was Python created?",
]

# Tokenize
inputs = model.generator.tokenizer(
    questions,
    return_tensors="pt",
    padding=True,
    truncation=True,
)

# Generate
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=50,
)

# Decode
answers = model.generator.decode(outputs, skip_special_tokens=True)

for q, a in zip(questions, answers):
    print(f"Q: {q}")
    print(f"A: {a}\n")
```

### Interactive Mode

```bash
python scripts/inference.py \
    --model_path ./outputs/my_model \
    --index_path ./data/index \
    --interactive
```

## Evaluation

### Evaluate on Benchmark

```bash
python scripts/evaluate.py \
    --model_path ./outputs/nq_rag_sequence \
    --dataset natural_questions \
    --split test \
    --metric exact_match
```

### Custom Evaluation

```python
from rag import RAGSequenceForGeneration
from rag.utils.metrics import compute_exact_match

model = RAGSequenceForGeneration.from_pretrained("./outputs/my_model")

# Your test data
test_questions = ["What is...?", "Who was...?"]
test_answers = ["Paris", "Einstein"]

predictions = []
for q in test_questions:
    pred = model.generate_from_query(q)
    predictions.append(pred)

# Compute metrics
em_score = compute_exact_match(predictions, test_answers)
print(f"Exact Match: {em_score:.2f}")
```

## Advanced Usage

### Using BM25 Retriever (Ablation)

```python
from rag import RAGSequenceForGeneration, RAGConfig
from rag.retrieval import BM25Retriever

# Create BM25 retriever
passages = [{"id": "1", "text": "...", "title": "..."}]
retriever = BM25Retriever(passages=passages)

# Use with RAG
config = RAGConfig()
model = RAGSequenceForGeneration(config=config, retriever=retriever)
```

### Index Hot-Swapping

```python
# Load model with old index
model = RAGSequenceForGeneration.from_pretrained("./outputs/model")
model.retriever.load_index("./data/index_2016")

answer_2016 = model.generate_from_query("Who is the president of USA?")

# Swap to new index (no retraining needed!)
model.retriever.load_index("./data/index_2020")

answer_2020 = model.generate_from_query("Who is the president of USA?")
```

### Custom Decoding Strategies

```python
from rag import RAGSequenceForGeneration, RAGConfig

# RAG-Sequence with thorough decoding
config = RAGConfig(use_thorough_decoding=True)
model = RAGSequenceForGeneration(config)

# Generate with more documents at test time
answer = model.generate(
    input_ids=input_ids,
    n_docs=50,  # More than training (5-10)
    max_length=128,
)
```

### Multi-GPU Training

```bash
# Using PyTorch DistributedDataParallel
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --task open_qa \
    --dataset natural_questions \
    --output_dir ./outputs/nq_multi_gpu
```

## Task-Specific Guides

### Open-Domain QA

```bash
# Natural Questions
python scripts/train.py --task open_qa --dataset natural_questions

# TriviaQA
python scripts/train.py --task open_qa --dataset triviaqa

# WebQuestions
python scripts/train.py --task open_qa --dataset web_questions
```

### Question Generation

```bash
python scripts/train.py \
    --task question_generation \
    --dataset jeopardy \
    --model_type rag_token \
    --num_retrieved_docs 10
```

### Fact Verification

```bash
python scripts/train.py \
    --task fact_verification \
    --dataset fever \
    --fever_mode 3-way
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
config.per_device_train_batch_size = 1
config.gradient_accumulation_steps = 8

# Use fewer documents
config.num_retrieved_docs = 3

# Use BART-base instead of BART-large
config.generator_name_or_path = "facebook/bart-base"
```

### Slow Retrieval

```bash
# Use GPU for FAISS
python scripts/build_index.py --use_gpu

# Use compressed index
python scripts/compress_index.py \
    --input ./data/index/index.faiss \
    --output ./data/index/index_compressed.faiss
```

### Index Not Found

```python
# Check paths
config.index_path = "./data/index/index.faiss"
config.passages_path = "./data/index/passages.json"

# Verify files exist
import os
assert os.path.exists(config.index_path)
assert os.path.exists(config.passages_path)
```

## Next Steps

1. ✅ Complete Quick Start example
2. ✅ Build Wikipedia index
3. ✅ Train your first model
4. ✅ Evaluate on benchmarks
5. ⬜ Experiment with hyperparameters
6. ⬜ Try different tasks
7. ⬜ Contribute improvements

## Resources

- **Paper**: https://arxiv.org/abs/2005.11401
- **HuggingFace Models**: https://huggingface.co/models?search=rag
- **Demo**: https://huggingface.co/rag/
- **Issues**: Open an issue on GitHub

## Citation

```bibtex
@article{lewis2020retrieval,
  title={Retrieval-augmented generation for knowledge-intensive nlp tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and others},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={9459--9474},
  year={2020}
}
```
