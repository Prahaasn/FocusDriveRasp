"""
Metrics and evaluation utilities for FocusDrive.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MetricsCalculator:
    """Calculate and track metrics during training."""

    def __init__(self, num_classes: int = 2, class_names: List[str] = None):
        """
        Initialize metrics calculator.

        Args:
            num_classes: Number of classes
            class_names: List of class names for display
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []

    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        probabilities: torch.Tensor = None
    ):
        """
        Update metrics with new batch.

        Args:
            predictions: Predicted class indices
            labels: Ground truth labels
            probabilities: Class probabilities (optional)
        """
        # Convert to numpy (detach first if it requires grad)
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()

        self.all_predictions.extend(predictions.tolist())
        self.all_labels.extend(labels.tolist())

        if probabilities is not None:
            self.all_probabilities.extend(probabilities.tolist())

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metrics
        """
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        # Overall accuracy
        accuracy = accuracy_score(labels, predictions)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            average=None,
            zero_division=0
        )

        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average='weighted',
            zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)

        # Compile metrics
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
        }

        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision[i]
            metrics[f'recall_{class_name}'] = recall[i]
            metrics[f'f1_{class_name}'] = f1[i]
            metrics[f'support_{class_name}'] = support[i]

        # Special attention to distraction recall (class 1)
        if len(self.class_names) > 1:
            metrics['distraction_recall'] = recall[1]  # Critical metric!

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(self.all_labels, self.all_predictions)

    def print_summary(self):
        """Print a summary of all metrics."""
        metrics = self.compute()

        print("\n" + "=" * 60)
        print("METRICS SUMMARY")
        print("=" * 60)

        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
        print(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
        print(f"Weighted F1: {metrics['f1_weighted']:.4f}")

        print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
        print("-" * 60)

        for class_name in self.class_names:
            precision = metrics[f'precision_{class_name}']
            recall = metrics[f'recall_{class_name}']
            f1 = metrics[f'f1_{class_name}']
            support = metrics[f'support_{class_name}']

            print(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<12.0f}")

        # Highlight critical metric
        if 'distraction_recall' in metrics:
            print(f"\n⚠️  CRITICAL: Distraction Recall = {metrics['distraction_recall']:.4f} " +
                  f"({metrics['distraction_recall']*100:.2f}%)")
            if metrics['distraction_recall'] >= 0.95:
                print("   ✓ Target achieved (>95%)")
            else:
                print(f"   ✗ Below target (need >95%, currently {metrics['distraction_recall']*100:.2f}%)")

        print("=" * 60)

    def plot_confusion_matrix(self, save_path: Path = None, normalize: bool = False):
        """
        Plot confusion matrix.

        Args:
            save_path: Path to save the plot (optional)
            normalize: Whether to normalize the confusion matrix
        """
        cm = self.get_confusion_matrix()

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar=True
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def get_classification_report(self) -> str:
        """Get sklearn classification report as string."""
        return classification_report(
            self.all_labels,
            self.all_predictions,
            target_names=self.class_names,
            digits=4
        )


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'max':
            self.is_better = lambda score, best: score > best + min_delta
        else:
            self.is_better = lambda score, best: score < best - min_delta

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """Reset early stopping."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


if __name__ == "__main__":
    """Test metrics calculator."""
    print("Testing MetricsCalculator...\n")

    # Create dummy data
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    predictions = torch.tensor([0, 0, 1, 0, 0, 1, 1, 1, 1, 0])
    probabilities = torch.softmax(torch.randn(10, 2), dim=1)

    # Initialize calculator
    calculator = MetricsCalculator(
        num_classes=2,
        class_names=['Attentive', 'Distracted']
    )

    # Update with data
    calculator.update(predictions, labels, probabilities)

    # Compute and print metrics
    metrics = calculator.compute()
    calculator.print_summary()

    print("\n✓ Metrics calculator test passed!")
