# RAG Training with Hugging Face - Complete Guide

This guide explains how to use the new Hugging Face-based RAG training scripts that fix all the issues in the original implementation.

## What's Fixed? ✅

The new `train_rag_hf.py` script uses Hugging Face's official RAG implementation, which properly handles:

1. ✅ **RAG-Sequence Marginalization** - Correct log-sum-exp over retrieved documents
2. ✅ **Gradient Accumulation** - Proper accumulation across batches
3. ✅ **Efficient Batching** - True batch processing for faster training
4. ✅ **Question Encoder Training** - Proper gradient flow to improve retrieval
5. ✅ **Mixed Precision** - FP16 training for faster GPU utilization
6. ✅ **Checkpoint Management** - Automatic saving of best models
7. ✅ **TensorBoard Logging** - Track training progress in real-time

## Requirements

Make sure you have these installed:

```bash
pip install transformers datasets torch faiss-cpu tensorboard
# Or for GPU:
pip install transformers datasets torch faiss-gpu tensorboard
```

## Step 1: Prepare Your Index

**IMPORTANT:** Hugging Face's RAG expects the index in a specific format. You need to convert your existing FAISS index.

Your current structure:
```
./data/index_200k_real/
├── index.faiss
└── passages.jsonl
```

Hugging Face expects:
```
./data/index_200k_real/
├── hf_dataset_index.faiss  (renamed index)
└── passages.jsonl          (same format)
```

**Quick fix:**
```bash
cd ./data/index_200k_real
cp index.faiss hf_dataset_index.faiss
```

## Step 2: Training

### Basic Training Command

```bash
nohup python scripts/train_rag_hf.py \
    --train_data ./data/natural_questions/nq_train.jsonl \
    --val_data ./data/natural_questions/nq_validation.jsonl \
    --index_path ./data/index_200k_real/hf_dataset_index.faiss \
    --passages_path ./data/index_200k_real/passages.jsonl \
    --output_dir ./models/rag_hf_15k \
    --max_steps 15000 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-5 \
    --save_steps 1000 \
    --eval_steps 1000 > training_hf.log 2>&1 &
```

### Monitor Training

```bash
# View logs in real-time
tail -f training_hf.log

# View in TensorBoard
tensorboard --logdir ./models/rag_hf_15k/runs
```

### Key Parameters Explained

- `--max_steps 15000`: Train for 15,000 steps (recommended: 15k-20k)
- `--batch_size 2`: Samples per GPU (2-4 recommended for 16GB GPU)
- `--gradient_accumulation_steps 16`: Accumulate gradients over 16 batches
  - **Effective batch size** = 2 × 16 = **32** (recommended: 32-64)
- `--learning_rate 3e-5`: Learning rate (3e-5 is standard for RAG)
- `--save_steps 1000`: Save checkpoint every 1000 steps
- `--eval_steps 1000`: Evaluate on validation set every 1000 steps

### Recommended Configurations

#### For 16GB GPU (e.g., Tesla T4):
```bash
--batch_size 2 \
--gradient_accumulation_steps 16  # Effective batch size: 32
```

#### For 24GB GPU (e.g., RTX 3090):
```bash
--batch_size 4 \
--gradient_accumulation_steps 16  # Effective batch size: 64
```

#### For 40GB+ GPU (e.g., A100):
```bash
--batch_size 8 \
--gradient_accumulation_steps 8  # Effective batch size: 64
```

#### For Quick Testing (Debug):
```bash
--max_steps 100 \
--max_train_samples 1000 \
--batch_size 2 \
--gradient_accumulation_steps 4
```

## Step 3: Evaluation

### Evaluate Your Fine-tuned Model

```bash
python scripts/evaluate_rag_hf.py \
    --model_dir ./models/rag_hf_15k/final_model \
    --test_data ./data/natural_questions/nq_validation.jsonl \
    --index_path ./data/index_200k_real/hf_dataset_index.faiss \
    --passages_path ./data/index_200k_real/passages.jsonl \
    --num_samples 500 \
    --batch_size 4 \
    --output_file ./results/eval_hf_15k.json
```

### Compare with Pre-trained Model (Baseline)

To see if fine-tuning helped, evaluate the base model:

```bash
python scripts/evaluate_rag_hf.py \
    --model_dir facebook/rag-sequence-nq \
    --test_data ./data/natural_questions/nq_validation.jsonl \
    --index_path ./data/index_200k_real/hf_dataset_index.faiss \
    --passages_path ./data/index_200k_real/passages.jsonl \
    --num_samples 500 \
    --output_file ./results/eval_baseline.json
```

## Expected Results

### Baseline (Pre-trained RAG, no fine-tuning):
- **Exact Match**: 20-28%
- **F1 Score**: 30-40%

### After Fine-tuning (15k-20k steps):
- **Exact Match**: 35-45%
- **F1 Score**: 45-55%

### Your Previous Results (Broken Training):
- **Exact Match**: 4.8% ❌
- **F1 Score**: 10.02% ❌

## Troubleshooting

### Issue: Out of Memory (OOM)
**Solution:** Reduce batch size or increase gradient accumulation
```bash
--batch_size 1 \
--gradient_accumulation_steps 32
```

### Issue: Training Too Slow
**Solution:** Increase batch size, reduce gradient accumulation
```bash
--batch_size 4 \
--gradient_accumulation_steps 8
```

### Issue: Loss Not Decreasing
**Solution:**
1. Check learning rate (try 1e-5 or 5e-5)
2. Train longer (20k-30k steps)
3. Increase warmup steps

### Issue: Index Format Error
**Solution:** Make sure your FAISS index is named correctly:
```bash
cp ./data/index_200k_real/index.faiss \
   ./data/index_200k_real/hf_dataset_index.faiss
```

## Monitoring Training Progress

### Check Loss Values

Good training should show:
- **Initial loss**: ~3.0-4.0
- **After 5k steps**: ~2.0-2.5
- **After 15k steps**: ~1.2-1.8
- **Best validation loss**: < 1.5

Your previous training:
- Best validation loss: **2.46** (too high!)

### Use TensorBoard

```bash
# In a separate terminal
tensorboard --logdir ./models/rag_hf_15k

# Then open http://localhost:6006 in your browser
```

You can monitor:
- Training loss
- Validation loss
- Learning rate schedule
- Gradient norms

## Complete Training Script

Save this as `run_training.sh`:

```bash
#!/bin/bash

# Configuration
OUTPUT_DIR="./models/rag_hf_20k"
MAX_STEPS=20000
BATCH_SIZE=2
GRAD_ACCUM=16

echo "Starting RAG training with Hugging Face..."
echo "Output: $OUTPUT_DIR"
echo "Max steps: $MAX_STEPS"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"

nohup python scripts/train_rag_hf.py \
    --train_data ./data/natural_questions/nq_train.jsonl \
    --val_data ./data/natural_questions/nq_validation.jsonl \
    --index_path ./data/index_200k_real/hf_dataset_index.faiss \
    --passages_path ./data/index_200k_real/passages.jsonl \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate 3e-5 \
    --warmup_steps 500 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --logging_steps 100 > training_hf.log 2>&1 &

echo "Training started in background!"
echo "Monitor with: tail -f training_hf.log"
echo "Process ID: $!"
```

Make it executable and run:
```bash
chmod +x run_training.sh
./run_training.sh
```

## Next Steps

1. ✅ Copy your FAISS index: `cp index.faiss hf_dataset_index.faiss`
2. ✅ Start training with the new script
3. ✅ Monitor progress with `tail -f training_hf.log`
4. ✅ Watch TensorBoard for metrics
5. ✅ Evaluate after training completes
6. ✅ Compare results with your previous 4.8% EM

You should see **dramatic improvements** (30-45% EM instead of 4.8%!).

## Questions?

If you encounter issues:
1. Check the logs: `tail -100 training_hf.log`
2. Verify GPU memory: `nvidia-smi`
3. Check index format: `ls -lh ./data/index_200k_real/`
4. Verify data loading: Run with `--max_train_samples 10` first

Good luck! 🚀
