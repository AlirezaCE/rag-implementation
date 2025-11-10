# RAG Model Testing Guide

After training your RAG model with `train_rag_fixed.py`, use this guide to test and evaluate it.

## Quick Overview

Your trained model will be saved in `./models/rag_finetuned/` with:
- `best_model/` - Fine-tuned BART generator
- `question_encoder/` - Fine-tuned DPR question encoder
- `training_info.json` - Training metadata

---

## 1. Interactive Testing (Recommended First Step)

**Test your model interactively by asking questions:**

```bash
python scripts/interactive_test.py \
    --model_dir ./models/rag_finetuned \
    --index_dir ./data/index_200k
```

**Usage:**
- Type any question and press Enter
- Type `nodocs` to toggle document display on/off
- Type `quit` or `exit` to stop

**Example Session:**
```
Your question: What is the capital of France?

[1/3] Retrieving top 5 documents...
Retrieved Documents:
  1. [Score: 0.8523]
     Paris is the capital and largest city of France...

[2/3] Generating answer...

[3/3] Result:
ANSWER: Paris
```

---

## 2. Formal Evaluation on Test Set

**Evaluate on Natural Questions test/validation set:**

```bash
python scripts/evaluate_rag.py \
    --model_dir ./models/rag_finetuned \
    --test_data ./data/natural_questions/nq_validation.jsonl \
    --index_dir ./data/index_200k \
    --output_file ./results/evaluation_results.json
```

**Quick test on 100 samples:**
```bash
python scripts/evaluate_rag.py \
    --model_dir ./models/rag_finetuned \
    --test_data ./data/natural_questions/nq_validation.jsonl \
    --index_dir ./data/index_200k \
    --num_samples 100 \
    --output_file ./results/eval_100.json
```

**Output:**
- Exact Match (EM) score
- F1 score
- Per-question predictions
- Sample outputs

**Expected Results (after proper training):**
- Exact Match: ~30-45% (competitive with paper)
- F1 Score: ~40-55%

---

## 3. Batch Question Testing

**Test on custom questions from a file:**

Create a file `my_questions.txt`:
```
What is the capital of France?
Who invented the telephone?
When did World War II end?
What is the largest planet?
```

Run batch testing:
```bash
python scripts/interactive_test.py \
    --model_dir ./models/rag_finetuned \
    --index_dir ./data/index_200k \
    --questions_file my_questions.txt \
    --output_file ./results/my_answers.json
```

---

## 4. Compare with Baseline

**Test baseline (pre-trained) model:**

```python
# Create baseline_test.py
from transformers import BartTokenizer, BartForConditionalGeneration

# Load pre-trained BART (no RAG)
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

question = "What is the capital of France?"
inputs = tokenizer(question, return_tensors="pt")
outputs = model.generate(**inputs)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Baseline answer: {answer}")
```

**Compare:**
- Baseline BART (no retrieval)
- RAG with pre-trained encoders
- RAG with fine-tuned encoders (your model)

---

## 5. Inspect Model Components

**Check what was saved:**

```bash
ls -lah ./models/rag_finetuned/
```

**Load and inspect:**

```python
import json

# Check training info
with open('./models/rag_finetuned/training_info.json', 'r') as f:
    info = json.load(f)
    print(f"Trained for {info['global_step']} steps")
    print(f"Best val loss: {info['best_val_loss']}")

# Load evaluation results
with open('./results/evaluation_results.json', 'r') as f:
    results = json.load(f)
    print(f"Exact Match: {results['metrics']['exact_match']:.2f}%")
    print(f"F1 Score: {results['metrics']['f1_score']:.2f}%")

    # Show failed predictions
    failures = [p for p in results['predictions'] if p['exact_match'] == 0]
    print(f"\nFailed: {len(failures)}/{len(results['predictions'])}")
```

---

## 6. Advanced Testing

### Test Retrieval Quality

```python
from rag.retrieval import DPRRetriever, FAISSIndex
import json

# Load retriever
index = FAISSIndex.load("./data/index_200k/index.faiss")
with open("./data/index_200k/passages.jsonl", 'r') as f:
    passages = [json.loads(line) for line in f]

retriever = DPRRetriever(
    question_encoder="./models/rag_finetuned/question_encoder",
    ctx_encoder="facebook/dpr-ctx_encoder-single-nq-base",
    index=index,
    passages=passages
)

# Test retrieval
question = "What is the capital of France?"
results = retriever.retrieve([question], k=10)

print(f"Top 10 retrieved passages for: {question}\n")
for i, doc in enumerate(results[0], 1):
    print(f"{i}. [Score: {doc.score:.4f}] {doc.text[:100]}...")
```

### Test Generation Quality

```python
from transformers import BartForConditionalGeneration, BartTokenizer

generator = BartForConditionalGeneration.from_pretrained(
    "./models/rag_finetuned/best_model"
)
tokenizer = BartTokenizer.from_pretrained(
    "./models/rag_finetuned/best_model"
)

# Test with different contexts
question = "What is the capital of France?"
contexts = [
    "Paris is the capital of France.",
    "London is the capital of England.",
    "The capital of France is a major European city."
]

for ctx in contexts:
    input_text = f"{question} {ctx}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = generator.generate(**inputs, max_length=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Context: {ctx[:50]}...")
    print(f"Answer: {answer}\n")
```

---

## 7. Performance Benchmarks

**Expected performance on Natural Questions:**

| Metric | Baseline BART | RAG (pretrained) | RAG (fine-tuned) | Paper |
|--------|---------------|------------------|------------------|-------|
| EM     | ~15%          | ~25-30%          | ~35-45%          | 44.5% |
| F1     | ~20%          | ~35-40%          | ~45-55%          | 51.8% |

**Your training used:**
- 100 steps (very short - increase for better results)
- Batch size 2 × 4 accumulation = effective batch 8
- Only 200k passages (paper uses 21M)

**To improve:**
1. Train longer (1000+ steps)
2. Use more passages (1-21M)
3. Increase batch size
4. Tune hyperparameters

---

## 8. Debugging Issues

### Model doesn't load
```bash
# Check if files exist
ls ./models/rag_finetuned/best_model/
ls ./models/rag_finetuned/question_encoder/

# Try loading manually
python -c "from transformers import BartForConditionalGeneration; \
           model = BartForConditionalGeneration.from_pretrained('./models/rag_finetuned/best_model')"
```

### Poor performance
1. **Check training loss:** Look at `training_info.json` - loss should decrease
2. **Check retrieval:** Use retrieval test above - top docs should be relevant
3. **Increase training:** 100 steps is very short, try 1000+
4. **Check data:** Verify training data format matches expected schema

### Out of memory
```bash
# Reduce batch size during evaluation
python scripts/evaluate_rag.py \
    --model_dir ./models/rag_finetuned \
    --test_data ./data/natural_questions/nq_validation.jsonl \
    --index_dir ./data/index_200k \
    --batch_size 1 \
    --num_samples 50
```

---

## 9. Typical Workflow

```bash
# 1. Quick interactive test
python scripts/interactive_test.py \
    --model_dir ./models/rag_finetuned \
    --index_dir ./data/index_200k

# Ask a few questions to verify it works

# 2. Evaluate on small sample (100 examples)
python scripts/evaluate_rag.py \
    --model_dir ./models/rag_finetuned \
    --test_data ./data/natural_questions/nq_validation.jsonl \
    --index_dir ./data/index_200k \
    --num_samples 100 \
    --output_file ./results/eval_100.json

# 3. Check results
cat ./results/eval_100.json | grep -A5 "metrics"

# 4. If good, run full evaluation
python scripts/evaluate_rag.py \
    --model_dir ./models/rag_finetuned \
    --test_data ./data/natural_questions/nq_validation.jsonl \
    --index_dir ./data/index_200k \
    --output_file ./results/final_eval.json
```

---

## 10. Export Results for Presentation

```python
# Create a nice report
import json

with open('./results/evaluation_results.json', 'r') as f:
    results = json.load(f)

# Summary
print("="*60)
print("RAG MODEL EVALUATION REPORT")
print("="*60)
print(f"\nMetrics:")
print(f"  Exact Match: {results['metrics']['exact_match']:.2f}%")
print(f"  F1 Score: {results['metrics']['f1_score']:.2f}%")
print(f"  Samples: {results['metrics']['num_examples']}")

print(f"\n\nBest Predictions (F1 > 0.9):")
best = sorted(results['predictions'], key=lambda x: x['f1'], reverse=True)[:5]
for i, pred in enumerate(best, 1):
    print(f"\n{i}. {pred['question']}")
    print(f"   Predicted: {pred['prediction']}")
    print(f"   Actual: {pred['ground_truth']}")
    print(f"   F1: {pred['f1']:.2f}")

print(f"\n\nWorst Predictions (F1 < 0.2):")
worst = sorted(results['predictions'], key=lambda x: x['f1'])[:5]
for i, pred in enumerate(worst, 1):
    print(f"\n{i}. {pred['question']}")
    print(f"   Predicted: {pred['prediction']}")
    print(f"   Actual: {pred['ground_truth']}")
    print(f"   F1: {pred['f1']:.2f}")
```

---

## Questions?

Check:
- `README.md` - Overall project documentation
- `GETTING_STARTED.md` - Setup instructions
- `IMPLEMENTATION_STATUS.md` - What's implemented
- RAG paper: https://arxiv.org/abs/2005.11401
