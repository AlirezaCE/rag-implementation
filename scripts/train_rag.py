#!/usr/bin/env python3
"""
Fine-tune RAG model on Natural Questions.

Implements the training procedure from the RAG paper:
- Joint training of query encoder + generator
- Document encoder frozen
- Marginalization over retrieved documents
"""

import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Dict

from rag import RAGSequenceForGeneration, RAGConfig
from rag.retrieval import DPRRetriever, FAISSIndex
from rag.training import RAGTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NaturalQuestionsDataset(Dataset):
    """Dataset for Natural Questions."""

    def __init__(self, data_file: str):
        """
        Load Natural Questions data.

        Args:
            data_file: Path to JSONL file with questions and answers
        """
        self.examples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))

        logger.info(f"Loaded {len(self.examples)} examples from {data_file}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'question': example['question'],
            'answers': example['answers'],  # List of valid answers
            'id': example.get('id', f'example_{idx}')
        }


def collate_fn(batch):
    """Collate batch of examples."""
    return {
        'questions': [ex['question'] for ex in batch],
        'answers': [ex['answers'] for ex in batch],
        'ids': [ex['id'] for ex in batch]
    }


def train_rag(
    train_data_file: str,
    val_data_file: str,
    index_dir: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    max_steps: int = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Fine-tune RAG model on Natural Questions.

    Following the RAG paper approach:
    - Train query encoder + generator jointly
    - Keep document encoder frozen
    - Use retrieval-augmented generation

    Args:
        train_data_file: Path to training data JSONL
        val_data_file: Path to validation data JSONL
        index_dir: Directory containing FAISS index and passages
        output_dir: Directory to save fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate (3e-5 as per paper)
        max_steps: Maximum training steps (None = full epochs)
        device: Device to train on
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("RAG Fine-tuning (Following Paper Approach)")
    logger.info("="*70)
    logger.info(f"Train data: {train_data_file}")
    logger.info(f"Val data: {val_data_file}")
    logger.info(f"Index: {index_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {num_epochs}")

    # Load datasets
    logger.info("\n[1/5] Loading datasets...")
    train_dataset = NaturalQuestionsDataset(train_data_file)
    val_dataset = NaturalQuestionsDataset(val_data_file)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Load FAISS index and passages
    logger.info("\n[2/5] Loading FAISS index...")
    index = FAISSIndex.load(f"{index_dir}/index.faiss")

    passages = []
    with open(f"{index_dir}/passages.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            passages.append(json.loads(line))
    logger.info(f"Loaded {len(passages)} passages")

    # Create retriever
    logger.info("\n[3/5] Creating DPR retriever...")
    retriever = DPRRetriever(
        question_encoder="facebook/dpr-question_encoder-single-nq-base",
        ctx_encoder="facebook/dpr-ctx_encoder-single-nq-base",
        index=index,
        passages=passages,
        device=device,
        freeze_doc_encoder=True  # Important: freeze as per paper
    )

    # Create RAG model
    logger.info("\n[4/5] Creating RAG model...")
    config = RAGConfig(
        model_type="rag_sequence",
        num_retrieved_docs=5,
        generator_name_or_path="facebook/bart-base",
    )

    model = RAGSequenceForGeneration(config=config, retriever=retriever)
    model = model.to(device)

    # Create trainer
    logger.info("\n[5/5] Setting up trainer...")
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    # Optimizer - only train query encoder and generator
    optimizer = AdamW([
        {'params': retriever.question_encoder.parameters()},
        {'params': model.generator.parameters()}
    ], lr=learning_rate)

    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    if max_steps:
        total_steps = min(total_steps, max_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    logger.info(f"Total training steps: {total_steps}")

    # Training loop
    logger.info("\n" + "="*70)
    logger.info("Starting training...")
    logger.info("="*70 + "\n")

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 70)

        # Training
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            questions = batch['questions']
            answers = batch['answers']

            # Forward pass with retrieval
            # The model will retrieve documents and generate
            outputs = model(
                input_ids=questions,
                labels=answers,
                return_dict=True
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })

            if max_steps and global_step >= max_steps:
                break

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")

        # Validation
        logger.info("\nValidating...")
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                questions = batch['questions']
                answers = batch['answers']

                outputs = model(
                    input_ids=questions,
                    labels=answers,
                    return_dict=True
                )

                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Validation loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"New best model! Saving to {output_dir}/best_model")

            # Save model
            model.save_pretrained(f"{output_dir}/best_model")
            retriever.question_encoder.save_pretrained(f"{output_dir}/question_encoder")

            # Save config
            with open(f"{output_dir}/training_config.json", 'w') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'best_val_loss': best_val_loss,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size
                }, f, indent=2)

        if max_steps and global_step >= max_steps:
            break

    logger.info("\n" + "="*70)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}/best_model")
    logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune RAG on Natural Questions (following paper approach)"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to validation data JSONL file"
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Directory containing FAISS index and passages"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/rag_finetuned",
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size (default: 8)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate (default: 3e-5, as per paper)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (default: None = full epochs)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: auto-detect)"
    )

    args = parser.parse_args()

    train_rag(
        train_data_file=args.train_data,
        val_data_file=args.val_data,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        device=args.device
    )


if __name__ == "__main__":
    main()
