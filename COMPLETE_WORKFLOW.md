# Complete RAG Training & Testing Workflow

This guide shows you **exactly** what to do from start to finish.

---

## Current Status ❌

You **cannot test yet** because:
- ❌ No training data downloaded
- ❌ No FAISS index built
- ❌ No model trained
- ❌ `./models/rag_finetuned/` doesn't exist

---

## Step-by-Step Workflow

### Phase 1: Setup & Data Preparation (First Time Only)

#### Step 1.1: Install Dependencies
```bash
pip install datasets faiss-cpu numpy tqdm
```

*(Use `faiss-gpu` instead of `faiss-cpu` if you have GPU)*

---

#### Step 1.2: Download Natural Questions Dataset
```bash
python scripts/download_natural_questions.py
```

**What this does:**
- Downloads NQ training and validation data
- Saves to `./data/natural_questions/`
- Creates `nq_train.jsonl` and `nq_validation.jsonl`

**Time:** 10-30 minutes (depending on internet speed)

---

#### Step 1.3: Download Wikipedia Passages
```bash
python scripts/download_wikipedia.py
```

**What this does:**
- Downloads Wikipedia dump (Dec 2018 version used in paper)
- Processes into passages
- Saves to `./data/wikipedia/passages.jsonl`

**Time:** 1-2 hours (large download)

**Alternative (Faster):** Use smaller test data:
```bash
python scripts/download_test_data.py
```
This creates a smaller dataset for testing (~10 minutes)

---

#### Step 1.4: Build FAISS Index
```bash
python scripts/build_index.py \
    --passages_file ./data/wikipedia/passages.jsonl \
    --output_dir ./data/index_200k \
    --num_passages 200000
```

**What this does:**
- Encodes passages using DPR
- Builds FAISS index
- Saves `index.faiss` and `passages.jsonl` to `./data/index_200k/`

**Time:** 30 minutes - 2 hours (depends on GPU)

---

### Phase 2: Training

#### Step 2.1: Train RAG Model
```bash
python scripts/train_rag_fixed.py \
    --train_data ./data/natural_questions/nq_train.jsonl \
    --val_data ./data/natural_questions/nq_validation.jsonl \
    --index_dir ./data/index_200k \
    --output_dir ./models/rag_finetuned \
    --max_steps 100 \
    --batch_size 2 \
    --gradient_accumulation_steps 4
```

**What this does:**
- Fine-tunes RAG model on Natural Questions
- Saves best model to `./models/rag_finetuned/`
- Creates:
  - `best_model/` - Generator (BART)
  - `question_encoder/` - Question encoder (DPR)
  - `training_info.json` - Training metadata

**Time:** 30 minutes - 2 hours (with 100 steps)

**For Quick Test (10 steps):**
```bash
python scripts/train_rag_fixed.py \
    --train_data ./data/natural_questions/nq_train.jsonl \
    --val_data ./data/natural_questions/nq_validation.jsonl \
    --index_dir ./data/index_200k \
    --output_dir ./models/rag_finetuned \
    --max_steps 10 \
    --batch_size 2 \
    --gradient_accumulation_steps 4
```

---

### Phase 3: Testing (AFTER Training Completes)

#### Step 3.1: Quick Sanity Check
```bash
python scripts/quick_test.py \
    --model_dir ./models/rag_finetuned \
    --index_dir ./data/index_200k
```

**Expected output:**
```
✓ ALL TESTS PASSED! Your model is working correctly.
```

---

#### Step 3.2: Interactive Testing
```bash
python scripts/interactive_test.py \
    --model_dir ./models/rag_finetuned \
    --index_dir ./data/index_200k
```

Ask your own questions!

---

#### Step 3.3: Formal Evaluation
```bash
python scripts/evaluate_rag.py \
    --model_dir ./models/rag_finetuned \
    --test_data ./data/natural_questions/nq_validation.jsonl \
    --index_dir ./data/index_200k \
    --num_samples 100 \
    --output_file ./results/eval_100.json
```

---

## Quick Start (Minimal Version)

If you just want to test the system works:

```bash
# 1. Install dependencies
pip install datasets faiss-cpu numpy tqdm

# 2. Download small test data
python scripts/download_test_data.py

# 3. Build small index (using test data)
python scripts/build_index.py \
    --passages_file ./data/test/passages.jsonl \
    --output_dir ./data/test_index \
    --num_passages 10000

# 4. Quick training (10 steps)
python scripts/train_rag_fixed.py \
    --train_data ./data/test/nq_train.jsonl \
    --val_data ./data/test/nq_val.jsonl \
    --index_dir ./data/test_index \
    --output_dir ./models/rag_test \
    --max_steps 10 \
    --batch_size 2 \
    --gradient_accumulation_steps 2

# 5. Test it
python scripts/quick_test.py \
    --model_dir ./models/rag_test \
    --index_dir ./data/test_index
```

---

## Current Next Steps for You

Based on your situation, here's what you should do:

### Option A: Full Training (Recommended for Real Results)

1. **Check if data exists:**
   ```bash
   ls ./data/natural_questions/
   ls ./data/index_200k/
   ```

2. **If missing, download:**
   ```bash
   # Install datasets library first
   pip install datasets

   # Download data
   python scripts/download_natural_questions.py
   python scripts/download_wikipedia.py

   # Build index
   python scripts/build_index.py \
       --passages_file ./data/wikipedia/passages.jsonl \
       --output_dir ./data/index_200k \
       --num_passages 200000
   ```

3. **Then train:**
   ```bash
   python scripts/train_rag_fixed.py \
       --train_data ./data/natural_questions/nq_train.jsonl \
       --val_data ./data/natural_questions/nq_validation.jsonl \
       --index_dir ./data/index_200k \
       --output_dir ./models/rag_finetuned \
       --max_steps 100 \
       --batch_size 2 \
       --gradient_accumulation_steps 4
   ```

4. **Finally test:**
   ```bash
   python scripts/quick_test.py \
       --model_dir ./models/rag_finetuned \
       --index_dir ./data/index_200k
   ```

---

### Option B: Quick Test (Just to See It Work)

1. **Check if test data script exists:**
   ```bash
   python scripts/download_test_data.py --help
   ```

2. **If it exists, use it for quick testing**

3. **If not, I can create it for you**

---

## Troubleshooting

### "Can't find training data"
- You need to run data download scripts first
- Or create sample data manually

### "Index not found"
- You need to build the FAISS index first
- Run `build_index.py`

### "Model not found"
- You need to train the model first
- Run `train_rag_fixed.py`

### "Out of memory"
- Reduce batch size: `--batch_size 1`
- Reduce num passages in index
- Use CPU instead of GPU

---

## Which Option Should You Choose?

**If you have:**
- ✅ GPU available
- ✅ Fast internet
- ✅ Several hours
- ✅ Want real results

→ **Choose Option A (Full Training)**

**If you have:**
- ❌ Limited time
- ❌ Slow internet
- ✅ Just want to see it work

→ **Choose Option B (Quick Test)** - I can help set this up

---

## What Do You Want to Do?

1. **Proceed with full training?** - I'll help you check if data exists and guide through download
2. **Quick test only?** - I'll create a minimal test dataset for you
3. **Something else?** - Let me know!
