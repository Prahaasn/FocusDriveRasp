"""
Training loop for FocusDrive LFM2-VL model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import json
from typing import Optional, Dict, Any
import time

from ..utils.metrics import MetricsCalculator, AverageMeter, EarlyStopping


class Trainer:
    """Trainer for FocusDrive driver distraction classifier."""

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: torch.device = None,
        save_dir: Path = None,
        class_names: list = None,
        class_weights: Optional[torch.Tensor] = None,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        """
        Initialize trainer.

        Args:
            model: The model to train
            processor: HuggingFace processor
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            save_dir: Directory to save checkpoints
            class_names: List of class names
            class_weights: Weights for imbalanced classes
            mixed_precision: Use mixed precision training
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir) if save_dir else Path("models/checkpoints")
        self.class_names = class_names or ['Attentive', 'Distracted']
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Loss function with class weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Mixed precision scaler (device-aware)
        if mixed_precision:
            if self.device.type == 'cuda':
                self.scaler = torch.amp.GradScaler('cuda')
            else:
                # MPS and CPU don't support mixed precision properly
                self.scaler = None
                self.mixed_precision = False
        else:
            self.scaler = None

        # History tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_metrics': [],
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_recall = 0.0

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {mixed_precision}")
        print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  Save directory: {self.save_dir}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        loss_meter = AverageMeter()
        metrics_calc = MetricsCalculator(
            num_classes=len(self.class_names),
            class_names=self.class_names
        )

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            total=len(self.train_loader)
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['image']  # List of PIL images
            labels = batch['label'].to(self.device)

            # Process each image separately, then batch manually
            all_inputs = []
            for img in images:
                inp = self.processor(
                    text="<image> Classify: attentive or distracted?",
                    images=img,
                    return_tensors="pt"
                )
                all_inputs.append(inp)

            # Manually batch the inputs
            inputs = {
                'input_ids': torch.cat([inp['input_ids'] for inp in all_inputs], dim=0),
                'attention_mask': torch.cat([inp['attention_mask'] for inp in all_inputs], dim=0),
                'pixel_values': torch.cat([inp['pixel_values'] for inp in all_inputs], dim=0),
            }
            
            # Add optional fields if they exist
            if 'pixel_attention_mask' in all_inputs[0]:
                inputs['pixel_attention_mask'] = torch.cat([inp['pixel_attention_mask'] for inp in all_inputs], dim=0)
            if 'spatial_shapes' in all_inputs[0]:
                inputs['spatial_shapes'] = torch.cat([inp['spatial_shapes'] for inp in all_inputs], dim=0)

            # Move to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            # Forward pass with mixed precision
            if self.mixed_precision and self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs['loss'] if outputs['loss'] is not None else \
                           self.criterion(outputs['logits'], labels)

                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps

                # Backward pass with scaler
                self.scaler.scale(loss).backward()

                # Update weights
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    if self.scheduler is not None:
                        self.scheduler.step()

            else:
                outputs = self.model(**inputs, labels=labels)
                loss = outputs['loss'] if outputs['loss'] is not None else \
                       self.criterion(outputs['logits'], labels)

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Update weights
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.scheduler is not None:
                        self.scheduler.step()

            # Get predictions
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)

            # Update metrics
            loss_meter.update(loss.item() * self.gradient_accumulation_steps, labels.size(0))
            metrics_calc.update(predictions, labels, probabilities)

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'lr': f"{current_lr:.2e}"
            })

        # Compute final metrics
        metrics = metrics_calc.compute()
        metrics['loss'] = loss_meter.avg
        metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

        return metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        loss_meter = AverageMeter()
        metrics_calc = MetricsCalculator(
            num_classes=len(self.class_names),
            class_names=self.class_names
        )

        # Progress bar
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch} [Val]  ",
            total=len(self.val_loader)
        )

        for batch in pbar:
            # Move batch to device
            images = batch['image']
            labels = batch['label'].to(self.device)

            # Process each image separately, then batch manually
            all_inputs = []
            for img in images:
                inp = self.processor(
                    text="<image> Classify: attentive or distracted?",
                    images=img,
                    return_tensors="pt"
                )
                all_inputs.append(inp)

            # Manually batch the inputs
            inputs = {
                'input_ids': torch.cat([inp['input_ids'] for inp in all_inputs], dim=0),
                'attention_mask': torch.cat([inp['attention_mask'] for inp in all_inputs], dim=0),
                'pixel_values': torch.cat([inp['pixel_values'] for inp in all_inputs], dim=0),
            }
            
            # Add optional fields if they exist
            if 'pixel_attention_mask' in all_inputs[0]:
                inputs['pixel_attention_mask'] = torch.cat([inp['pixel_attention_mask'] for inp in all_inputs], dim=0)
            if 'spatial_shapes' in all_inputs[0]:
                inputs['spatial_shapes'] = torch.cat([inp['spatial_shapes'] for inp in all_inputs], dim=0)
                
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            # Forward pass
            outputs = self.model(**inputs, labels=labels)
            loss = outputs['loss'] if outputs['loss'] is not None else \
                   self.criterion(outputs['logits'], labels)

            # Get predictions
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)

            # Update metrics
            loss_meter.update(loss.item(), labels.size(0))
            metrics_calc.update(predictions, labels, probabilities)

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}"})

        # Compute final metrics
        metrics = metrics_calc.compute()
        metrics['loss'] = loss_meter.avg

        # Print summary
        print(f"\nValidation Metrics:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        if 'distraction_recall' in metrics:
            print(f"  Distraction Recall: {metrics['distraction_recall']:.4f} " +
                  f"({metrics['distraction_recall']*100:.2f}%)")

        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_val_recall': self.best_val_recall
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best model saved! (Accuracy: {metrics['accuracy']:.4f})")

        # Save latest
        latest_path = self.save_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)

    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 5,
        save_every: int = 1
    ):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_every: Save checkpoint every N epochs
        """
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='max'
        )

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(train_metrics['learning_rate'])

            # Check if best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_recall = val_metrics.get('distraction_recall', 0.0)

            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping
            if early_stopping(val_metrics['accuracy']):
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                print(f"  Best validation accuracy: {self.best_val_acc:.4f}")
                break

        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best distraction recall: {self.best_val_recall:.4f}")

        # Save final history
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining history saved to: {history_path}")
