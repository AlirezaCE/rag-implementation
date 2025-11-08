# RAG Implementation Status

This document tracks the implementation progress of the complete RAG system based on the paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (arXiv:2005.11401v4).

## ✅ Completed Components

### 1. Project Structure
- [x] `requirements.txt` - All dependencies including HuggingFace Transformers, FAISS, PyTorch
- [x] `setup.py` - Package setup with entry points
- [x] `README.md` - Comprehensive documentation
- [x] Configuration system (`rag/config.py`)

### 2. Retrieval Components (`rag/retrieval/`)
- [x] **Base Retriever** (`base.py`)
  - Abstract base class for all retrievers
  - `RetrievedDocument` dataclass
  - Mock retriever for testing

- [x] **DPR Retriever** (`dpr.py`)
  - BERT bi-encoder implementation
  - Query encoder: `BERT_q(x)`
  - Document encoder: `BERT_d(z)`
  - Maximum Inner Product Search (MIPS)
  - Integration with FAISS
  - Document encoding frozen during training (as in paper)

- [x] **FAISS Index** (`faiss_index.py`)
  - Hierarchical Navigable Small World (HNSW) support
  - GPU acceleration
  - Index compression (reduces memory to 36GB as mentioned in paper)
  - Multiple index types (Flat, HNSW, IVF, PQ)

- [x] **BM25 Retriever** (`bm25.py`)
  - Traditional sparse retrieval baseline
  - For ablation studies (Table 6 in paper)

### 3. Generation Components (`rag/generation/`)
- [x] **BART Generator** (`bart.py`)
  - BART-large (400M parameters) wrapper
  - Input concatenation: query + retrieved documents
  - `p_θ(y_i|x,z,y_{1:i-1})` implementation
  - Sequence scoring for marginalization

- [x] **Decoding Strategies** (`decoding.py`)
  - Thorough Decoding for RAG-Sequence
  - Fast Decoding for RAG-Sequence
  - Standard beam search for RAG-Token

### 4. Core RAG Models (`rag/models/`)
- [x] **Base RAG Model** (`base.py`)
  - Shared functionality
  - Document retrieval
  - Input preparation
  - Model saving/loading

## 🚧 In Progress

### 5. RAG Model Implementations
- [ ] **RAG-Sequence** (`rag_sequence.py`)
  - Marginalization: `p(y|x) = Σ_z p_η(z|x) * p_θ(y|x,z)`
  - Uses same documents for entire sequence
  - Thorough/Fast decoding options

- [ ] **RAG-Token** (`rag_token.py`)
  - Marginalization: `p(y|x) = Π_i Σ_z p_η(z|x) * p_θ(y_i|x,z,y_{1:i-1})`
  - Can use different documents per token
  - Standard beam search

## 📋 TODO

### 6. Training Pipeline (`rag/training/`)
- [ ] **Trainer** (`trainer.py`)
  - End-to-end joint training
  - Freeze document encoder, update query encoder + generator
  - Mixed precision (FP16) training
  - Adam optimizer
  - Negative log-likelihood loss

- [ ] **Loss Functions** (`losses.py`)
  - Marginal log-likelihood for RAG-Sequence
  - Marginal log-likelihood for RAG-Token

- [ ] **Optimizers** (`optimizers.py`)
  - Adam with learning rate scheduling
  - Gradient clipping

### 7. Task-Specific Components (`tasks/`)
- [ ] **Open-Domain QA** (`open_qa.py`)
  - Natural Questions, TriviaQA, WebQuestions, CuratedTrec
  - Exact Match evaluation

- [ ] **Abstractive QA** (`abstractive_qa.py`)
  - MS-MARCO dataset
  - BLEU/ROUGE metrics

- [ ] **Question Generation** (`question_generation.py`)
  - Jeopardy dataset
  - Q-BLEU metric
  - Human evaluation framework

- [ ] **Fact Verification** (`fact_verification.py`)
  - FEVER dataset (2-way and 3-way)
  - Label accuracy

### 8. Data Processing (`data/`)
- [ ] **Dataset Loaders** (`datasets.py`)
  - HuggingFace datasets integration
  - Custom dataset classes

- [ ] **Wikipedia Processing** (`wikipedia.py`)
  - Download December 2018 dump
  - Split into 100-word chunks
  - Create 21M document corpus

- [ ] **Preprocessing** (`preprocessing.py`)
  - Tokenization
  - Answer extraction for QA

### 9. Evaluation (`rag/utils/`)
- [ ] **Metrics** (`metrics.py`)
  - Exact Match (EM)
  - BLEU, ROUGE, Q-BLEU
  - Accuracy
  - F1 score

- [ ] **Logging** (`logging.py`)
  - WandB integration
  - TensorBoard support

### 10. Scripts (`scripts/`)
- [ ] **Training Scripts**
  - `train.py` - Main training script
  - Multi-GPU support
  - Checkpoint management

- [ ] **Evaluation Scripts**
  - `evaluate.py` - Evaluation on test sets
  - Per-task evaluation

- [ ] **Index Building**
  - `build_index.py` - Build FAISS index from Wikipedia
  - `download_wikipedia.py` - Download and process Wikipedia

- [ ] **Inference**
  - `inference.py` - Run inference with trained model
  - Interactive demo

### 11. Examples (`examples/`)
- [ ] Jupyter notebooks
  - Basic Q&A example
  - Question generation example
  - Fact verification example
  - Custom dataset example

### 12. Tests (`tests/`)
- [ ] Unit tests for all components
- [ ] Integration tests
- [ ] End-to-end tests

## Key Implementation Details (from Paper)

### Model Architecture
- **Generator**: BART-large (400M params)
- **Retriever**: DPR with BERT-base encoders (110M params each)
- **Total trainable params**: ~626M (516M with frozen doc encoder)

### Training Details
- **Optimizer**: Adam
- **Learning rate**: 3e-5
- **Batch size**: Effective 8 (2 per device × 4 accumulation steps)
- **Mixed precision**: FP16
- **Epochs**: 10 for most tasks
- **Document encoder**: Frozen (index not updated during training)
- **Query encoder**: Updated during training

### Retrieval Details
- **Index**: 21M Wikipedia passages (100-word chunks, Dec 2018 dump)
- **FAISS**: Hierarchical Navigable Small World (HNSW)
- **k**: 5-10 documents for training, up to 50 for test
- **Compressed index**: 36GB RAM (from ~100GB)

### Task-Specific Settings

#### Open-Domain QA
- k=5 for RAG-Token, k=50 for RAG-Sequence
- Greedy decoding (num_beams=1)
- Thorough decoding for RAG-Sequence

#### Abstractive QA (MS-MARCO)
- k=10 for both models
- Beam size: 4
- Fast decoding for RAG-Sequence

#### Question Generation (Jeopardy)
- k=10
- Beam size: 4
- RAG-Token performs better

#### Fact Verification (FEVER)
- k=10
- Output length: 1 (single classification token)
- Both models equivalent for classification

## Next Steps

1. Complete RAG-Sequence and RAG-Token implementations
2. Implement training pipeline
3. Add task-specific components
4. Create Wikipedia processing scripts
5. Add evaluation metrics
6. Create example notebooks
7. Add comprehensive tests

## Usage Example (Once Complete)

```python
from rag import RAGSequenceForGeneration, RAGConfig

# Load model
config = RAGConfig(
    num_retrieved_docs=5,
    generator_name_or_path="facebook/bart-large",
    retriever_name_or_path="facebook/dpr-question_encoder-single-nq-base",
)

model = RAGSequenceForGeneration(config)

# Load index
model.retriever.load_index("./data/index")

# Generate answer
question = "What is the capital of France?"
answer = model.generate_from_query(question)
print(answer)  # "Paris"
```

## Performance Targets (from Paper)

| Task | Dataset | Target (EM/BLEU/Acc) |
|------|---------|---------------------|
| Open QA | Natural Questions | 44.5 |
| Open QA | TriviaQA | 56.8 |
| Open QA | WebQuestions | 45.2 |
| Open QA | CuratedTrec | 52.2 |
| Abs. QA | MS-MARCO | 44.2 (BLEU-1) |
| QGen | Jeopardy | 22.2 (Q-BLEU-1) |
| Fact Ver | FEVER-3 | 72.5 (Acc) |

## References

- Original Paper: https://arxiv.org/abs/2005.11401
- HuggingFace Implementation: https://github.com/huggingface/transformers/tree/master/examples/research_projects/rag
- Demo: https://huggingface.co/rag/
