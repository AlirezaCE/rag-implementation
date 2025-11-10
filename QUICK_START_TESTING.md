# Quick Start: Testing Your RAG Model

## After Training Completes

Once your training finishes (after running `train_rag_fixed.py`), follow these steps:

---

## Step 1: Quick Sanity Check (2 minutes)

```bash
python scripts/quick_test.py \
    --model_dir ./models/rag_finetuned \
    --index_dir ./data/index_200k
```

**What it does:**
- ✓ Verifies model files exist
- ✓ Tests loading model
- ✓ Runs 5 sample questions
- ✓ Shows if everything works

**Expected output:**
```
SUMMARY: 5/5 questions answered successfully
✓ ALL TESTS PASSED! Your model is working correctly.
```

---

## Step 2: Interactive Testing (5-10 minutes)

```bash
python scripts/interactive_test.py \
    --model_dir ./models/rag_finetuned \
    --index_dir ./data/index_200k
```

**What to do:**
1. Ask your own questions
2. See how it retrieves documents
3. Check answer quality
4. Type `quit` when done

**Example:**
```
Your question: What is the capital of France?

Retrieved Documents:
  1. [Score: 0.8523]
     Paris is the capital and largest city of France...

ANSWER: Paris
```

---

## Step 3: Formal Evaluation (10-30 minutes)

### Small test (100 examples):
```bash
python scripts/evaluate_rag.py \
    --model_dir ./models/rag_finetuned \
    --test_data ./data/natural_questions/nq_validation.jsonl \
    --index_dir ./data/index_200k \
    --num_samples 100 \
    --output_file ./results/eval_100.json
```

### Full evaluation (all examples):
```bash
python scripts/evaluate_rag.py \
    --model_dir ./models/rag_finetuned \
    --test_data ./data/natural_questions/nq_validation.jsonl \
    --index_dir ./data/index_200k \
    --output_file ./results/final_eval.json
```

**What you get:**
- Exact Match (EM) percentage
- F1 Score
- Detailed predictions file (JSON)
- Sample correct/incorrect predictions

**Expected metrics (with 100 training steps):**
- EM: 20-35%
- F1: 30-45%

*(Note: Paper reports 44.5% EM with full training)*

---

## Step 4: Analyze Results

```bash
# View metrics
cat ./results/eval_100.json | grep -A10 "metrics"

# Or in Python:
python -c "
import json
with open('./results/eval_100.json', 'r') as f:
    results = json.load(f)
print(f\"Exact Match: {results['metrics']['exact_match']:.2f}%\")
print(f\"F1 Score: {results['metrics']['f1_score']:.2f}%\")
"
```

---

## Interpreting Results

### Good Signs ✓
- EM > 25%
- F1 > 35%
- Answers look reasonable in interactive mode
- Model retrieves relevant documents

### Warning Signs ⚠
- EM < 15%
- F1 < 25%
- Answers are gibberish
- Retrieved docs seem random

### If Results Are Poor:
1. **Training too short:** You only ran 100 steps (try 1000+)
2. **Index too small:** Only 200k passages (paper uses 21M)
3. **Learning rate:** Try 1e-5 or 5e-5
4. **Data issues:** Check training data format

---

## What Each Metric Means

**Exact Match (EM):**
- % of questions where prediction EXACTLY matches ground truth
- Strict metric (case-insensitive, ignores articles)
- Example: Predicted "Paris" vs. "Paris, France" = NO match

**F1 Score:**
- Measures word overlap between prediction and truth
- More lenient than EM
- Example: Predicted "Paris, France" vs. "Paris" = high F1

---

## Comparison Baselines

| Approach | Expected EM | Expected F1 |
|----------|-------------|-------------|
| Random answers | ~0% | ~5% |
| BART (no retrieval) | ~15% | ~25% |
| RAG (100 steps) | ~25% | ~35% |
| RAG (1000 steps) | ~35% | ~45% |
| RAG (paper, full) | 44.5% | 51.8% |

---

## Next Steps

### If satisfied with results:
1. ✓ Document your findings
2. ✓ Save the model
3. ✓ Try different hyperparameters
4. ✓ Test on other datasets

### If want to improve:
1. Train longer (increase `--max_steps`)
2. Use more data (larger index)
3. Tune hyperparameters
4. Try different models (BART-large vs BART-base)

---

## Common Issues

**"Model not found":**
```bash
ls ./models/rag_finetuned/
# Should show: best_model/, question_encoder/, training_info.json
```

**"CUDA out of memory":**
```bash
# Reduce batch size in evaluation
python scripts/evaluate_rag.py ... --batch_size 1 --num_samples 50
```

**"Index file not found":**
```bash
ls ./data/index_200k/
# Should show: index.faiss, passages.jsonl
```

**Poor performance:**
- Check `training_info.json` for training loss
- Try interactive mode to see actual outputs
- Increase training steps significantly

---

## Full Documentation

For more detailed testing options, see:
- **TESTING_GUIDE.md** - Complete testing documentation
- **README.md** - Project overview
- **GETTING_STARTED.md** - Setup instructions

---

## Quick Reference Commands

```bash
# Quick test (2 min)
python scripts/quick_test.py --model_dir ./models/rag_finetuned --index_dir ./data/index_200k

# Interactive (manual)
python scripts/interactive_test.py --model_dir ./models/rag_finetuned --index_dir ./data/index_200k

# Evaluate 100 samples (5 min)
python scripts/evaluate_rag.py --model_dir ./models/rag_finetuned --test_data ./data/natural_questions/nq_validation.jsonl --index_dir ./data/index_200k --num_samples 100 --output_file ./results/eval.json

# View results
cat ./results/eval.json | grep -A5 "metrics"
```

---

**That's it! Start with quick_test.py and go from there.**
