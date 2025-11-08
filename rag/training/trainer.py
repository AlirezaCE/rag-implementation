"""
RAG Trainer implementation.

Implements end-to-end training for RAG models as described in paper Section 2.4:
"We jointly train the retriever and generator components without any direct
supervision on what document should be retrieved."

Key points from paper:
- Train query encoder + generator
- Freeze document encoder (keep index fixed)
- Use mixed precision (FP16)
- Adam optimizer with learning rate 3e-5
- Minimize negative marginal log-likelihood
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import Optional, Dict, List
from tqdm import tqdm
import os

from ..models import RAGModel
from ..config import TrainingConfig


class RAGTrainer:
    """
    Trainer for RAG models.

    Handles:
    - End-to-end training loop
    - Mixed precision training
    - Checkpoint management
    - Evaluation
    - Logging
    """

    def __init__(
        self,
        model: RAGModel,
        config: TrainingConfig,
        train_dataset: Optional[DataLoader] = None,
        eval_dataset: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Initialize RAG trainer.

        Args:
            model: RAG model to train
            config: Training configuration
            train_dataset: Training data loader
            eval_dataset: Evaluation data loader
            optimizer: Optional optimizer (created if None)
            scheduler: Optional LR scheduler (created if None)
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer

        # LR Scheduler
        if scheduler is None and train_dataset is not None:
            self.scheduler = self._create_scheduler()
        else:
            self.scheduler = scheduler

        # Mixed precision
        self.scaler = None
        if config.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

        # Tracking
        self.global_step = 0
        self.epoch = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        # Get trainable parameters
        # Query encoder + generator (document encoder is frozen)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        num_training_steps = (
            len(self.train_dataset) * self.config.num_train_epochs
        ) // self.config.gradient_accumulation_steps

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )

        return scheduler

    def train(self):
        """
        Main training loop.

        Implements Algorithm 1 from paper (implicit):
        For each batch:
        1. Retrieve documents
        2. Compute p_η(z|x) and p_θ(y|x,z)
        3. Marginalize to get p(y|x)
        4. Compute -log p(y|x)
        5. Backpropagate through query encoder and generator
        """
        print(f"Starting training for {self.config.num_train_epochs} epochs")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.config.per_device_train_batch_size}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Mixed precision: {self.config.fp16}")

        self.model.train()

        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.num_train_epochs}")

            epoch_loss = 0.0
            progress_bar = tqdm(self.train_dataset, desc=f"Training")

            for step, batch in enumerate(progress_bar):
                loss = self.training_step(batch)
                epoch_loss += loss

                # Update progress bar
                progress_bar.set_postfix({"loss": loss})

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self.log_metrics({"train_loss": loss})

                # Evaluation
                if (
                    self.config.evaluation_strategy == "steps"
                    and self.global_step % self.config.eval_steps == 0
                    and self.eval_dataset is not None
                ):
                    eval_metrics = self.evaluate()
                    self.log_metrics(eval_metrics)

                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                self.global_step += 1

            # Epoch-level evaluation
            if (
                self.config.evaluation_strategy == "epoch"
                and self.eval_dataset is not None
            ):
                eval_metrics = self.evaluate()
                self.log_metrics(eval_metrics)

            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint-epoch-{epoch}")

            print(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(self.train_dataset):.4f}")

        print("\nTraining completed!")

    def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.

        Args:
            batch: Batch of training data

        Returns:
            Loss value
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Mixed precision context
        if self.config.fp16:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs["loss"]
        else:
            outputs = self.model(**batch)
            loss = outputs["loss"]

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.config.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.fp16:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            # Optimizer step
            if self.config.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad()

        return loss.item() * self.config.gradient_accumulation_steps

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model.

        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating...")
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataset, desc="Evaluation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs["loss"]

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        self.model.train()

        return {
            "eval_loss": avg_loss,
        }

    def save_checkpoint(self, checkpoint_name: Optional[str] = None):
        """
        Save model checkpoint.

        Args:
            checkpoint_name: Optional checkpoint name
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint-{self.global_step}"

        output_dir = os.path.join(self.config.output_dir, checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Saving checkpoint to {output_dir}")

        # Save model
        self.model.save_pretrained(output_dir)

        # Save training state
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }, os.path.join(output_dir, "trainer_state.pt"))

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        print(f"Loading checkpoint from {checkpoint_path}")

        # Load training state
        state_path = os.path.join(checkpoint_path, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path)
            self.epoch = state["epoch"]
            self.global_step = state["global_step"]
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            if self.scheduler and state["scheduler_state_dict"]:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])

    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
        """
        # Print metrics
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Step {self.global_step}: {metrics_str}")

        # TODO: Add WandB/TensorBoard logging
        # if "wandb" in self.config.report_to:
        #     import wandb
        #     wandb.log(metrics, step=self.global_step)
