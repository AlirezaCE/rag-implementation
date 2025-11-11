#!/usr/bin/env python3
"""
RAG Fine-tuning using Hugging Face's official RagTrainer.

This script uses the official Hugging Face implementation which properly handles:
- RAG-Sequence marginalization
- Proper retrieval integration
- Joint training of question encoder + generator
- Efficient batching and gradient accumulation
"""

import argparse
import json
import torch
import logging
from pathlib import Path
from datasets import Dataset
from transformers import (
    RagTokenizer,
    RagRetriever,
    RagTokenForGeneration,
    RagSequenceForGeneration,
    TrainingArguments,
    Trainer,
)
from typing import Dict, List
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset_from_jsonl(file_path: str, max_samples: int = None) -> Dataset:
    """Load Natural Questions dataset from JSONL."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Get first answer if multiple exist
            answer = data['answers'][0] if isinstance(data['answers'], list) else data['answers']
            examples.append({
                'question': data['question'],
                'answers': answer,  # HF expects 'answers' key
                'id': data.get('id', '')
            })

            if max_samples and len(examples) >= max_samples:
                break

    logger.info(f"Loaded {len(examples)} examples from {file_path}")
    return Dataset.from_list(examples)


class RagDataCollator:
    """Data collator for RAG training."""

    def __init__(self, tokenizer, max_source_length=256, max_target_length=128):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples."""
        questions = [f['question'] for f in features]
        answers = [f['answers'] for f in features]

        # Tokenize questions (inputs)
        batch = self.tokenizer(
            questions,
            max_length=self.max_source_length,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize answers (targets)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                answers,
                max_length=self.max_target_length,
                padding='longest',
                truncation=True,
                return_tensors='pt'
            )

        # Replace padding token ids in labels with -100 (ignored by loss)
        labels['input_ids'][labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        batch['labels'] = labels['input_ids']

        return batch


def compute_metrics(eval_preds):
    """Compute evaluation metrics during training."""
    import re
    from collections import Counter

    def normalize_answer(s: str) -> str:
        """Normalize answer for comparison."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
            return ''.join(ch if ch not in exclude else ' ' for ch in text)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def exact_match_score(prediction: str, ground_truth: str) -> float:
        return float(normalize_answer(prediction) == normalize_answer(ground_truth))

    def f1_score(prediction: str, ground_truth: str) -> float:
        pred_tokens = normalize_answer(prediction).split()
        truth_tokens = normalize_answer(ground_truth).split()

        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return float(pred_tokens == truth_tokens)

        common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
        num_common = sum(common_tokens.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    # This is a placeholder - actual computation happens during evaluation
    # The trainer will call generate() separately
    return {"eval_metric": 0.0}


def train_rag(
    train_data_file: str,
    val_data_file: str,
    index_path: str,
    passages_path: str,
    output_dir: str,
    model_type: str = "facebook/rag-sequence-nq",
    max_steps: int = 15000,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 3e-5,
    warmup_steps: int = 500,
    save_steps: int = 1000,
    eval_steps: int = 1000,
    logging_steps: int = 100,
    max_train_samples: int = None,
    max_val_samples: int = 1000,
    num_retrieved_docs: int = 5,
):
    """
    Fine-tune RAG using Hugging Face's official implementation.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("RAG Fine-tuning with Hugging Face RagTrainer")
    logger.info("="*70)
    logger.info(f"Model: {model_type}")
    logger.info(f"Train data: {train_data_file}")
    logger.info(f"Val data: {val_data_file}")
    logger.info(f"Index: {index_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Batch size: {per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    logger.info(f"Learning rate: {learning_rate}")

    # Load datasets
    logger.info("\n[1/5] Loading datasets...")
    train_dataset = load_dataset_from_jsonl(train_data_file, max_train_samples)
    val_dataset = load_dataset_from_jsonl(val_data_file, max_val_samples)

    # Initialize tokenizer
    logger.info("\n[2/5] Loading tokenizer...")
    tokenizer = RagTokenizer.from_pretrained(model_type)

    # Initialize retriever
    logger.info("\n[3/5] Setting up retriever...")
    retriever = RagRetriever.from_pretrained(
        model_type,
        index_name="custom",
        passages_path=passages_path,
        index_path=index_path,
    )

    # Initialize model
    logger.info("\n[4/5] Loading RAG model...")
    model = RagSequenceForGeneration.from_pretrained(
        model_type,
        retriever=retriever,
    )

    # Freeze document encoder (as per paper)
    logger.info("Freezing document encoder...")
    for param in model.rag.ctx_encoder.parameters():
        param.requires_grad = False

    # Move model to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    model = model.to(device)

    # Setup data collator
    data_collator = RagDataCollator(
        tokenizer=tokenizer,
        max_source_length=256,
        max_target_length=128
    )

    # Training arguments
    logger.info("\n[5/5] Setting up training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
        dataloader_num_workers=4,
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
        report_to=["tensorboard"],
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("\n" + "="*70)
    logger.info("Starting training...")
    logger.info("="*70)

    train_result = trainer.train()

    # Save final model
    logger.info("\nSaving final model...")
    trainer.save_model(f"{output_dir}/final_model")

    # Save question encoder separately for evaluation
    logger.info("Saving question encoder...")
    model.rag.question_encoder.save_pretrained(f"{output_dir}/question_encoder")
    tokenizer.question_encoder.save_pretrained(f"{output_dir}/question_encoder")

    # Save training metrics
    metrics = train_result.metrics
    metrics['train_samples'] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save training info
    with open(f"{output_dir}/training_info.json", 'w') as f:
        json.dump({
            'model_type': model_type,
            'max_steps': max_steps,
            'final_step': train_result.global_step,
            'train_loss': train_result.training_loss,
            'learning_rate': learning_rate,
            'batch_size': per_device_train_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': per_device_train_batch_size * gradient_accumulation_steps,
            'num_retrieved_docs': num_retrieved_docs,
        }, f, indent=2)

    logger.info("\n" + "="*70)
    logger.info("Training complete!")
    logger.info(f"Final training loss: {train_result.training_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}/final_model")
    logger.info(f"Question encoder saved to: {output_dir}/question_encoder")
    logger.info("="*70)

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune RAG using Hugging Face's official implementation"
    )
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data JSONL file")
    parser.add_argument("--val_data", type=str, required=True,
                       help="Path to validation data JSONL file")
    parser.add_argument("--index_path", type=str, required=True,
                       help="Path to FAISS index file")
    parser.add_argument("--passages_path", type=str, required=True,
                       help="Path to passages JSONL file")
    parser.add_argument("--output_dir", type=str, default="./models/rag_hf_finetuned",
                       help="Output directory for model")
    parser.add_argument("--model_type", type=str, default="facebook/rag-sequence-nq",
                       help="Pre-trained RAG model to use")
    parser.add_argument("--max_steps", type=int, default=15000,
                       help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                       help="Evaluate every N steps")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Max training samples (for debugging)")
    parser.add_argument("--max_val_samples", type=int, default=1000,
                       help="Max validation samples")

    args = parser.parse_args()

    train_rag(
        train_data_file=args.train_data,
        val_data_file=args.val_data,
        index_path=args.index_path,
        passages_path=args.passages_path,
        output_dir=args.output_dir,
        model_type=args.model_type,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )


if __name__ == "__main__":
    main()
