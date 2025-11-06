"""
Evaluate trained MobileNetV3 model on test set.

This script:
1. Loads the best trained model
2. Runs inference on test set (3,364 images)
3. Computes detailed metrics
4. Generates confusion matrix visualization
5. Shows per-class performance

Usage:
    python evaluate_mobilenet.py
"""

import torch
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.mobilenet_classifier import MobileNetDriverClassifier
from src.data.mobilenet_dataset import get_dataloaders
from src.utils.metrics import MetricsCalculator


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Dictionary of metrics
    """
    model.eval()

    metrics_calc = MetricsCalculator(
        num_classes=2,
        class_names=['Attentive', 'Distracted']
    )

    print("\n" + "=" * 60)
    print("Running inference on test set...")
    print("=" * 60)

    pbar = tqdm(test_loader, desc="Evaluating", total=len(test_loader))

    with torch.no_grad():
        for batch in pbar:
            # Move batch to device
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(images)
            logits = outputs['logits']

            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)

            # Update metrics
            metrics_calc.update(predictions, labels, probabilities)

    # Compute final metrics
    metrics = metrics_calc.compute()

    return metrics, metrics_calc


def main():
    """Main evaluation function."""
    print("=" * 80)
    print("MobileNetV3 Model Evaluation on Test Set")
    print("=" * 80)

    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "processed"
    model_dir = project_root / "models" / "mobilenet_checkpoints" / "best_model_pretrained"

    # Check if model exists
    if not model_dir.exists():
        print(f"\n❌ Error: Model not found at {model_dir}")
        print("\nPlease train the model first:")
        print("  python train_mobilenet.py --epochs 20 --batch-size 32")
        sys.exit(1)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"\n🖥️  Device: {device}")

    # Load test data
    print(f"\n📊 Loading test data...")
    _, _, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4
    )

    # Load model
    print(f"\n🔧 Loading trained model from {model_dir}...")
    model = MobileNetDriverClassifier.load_pretrained(model_dir, device=str(device))

    # Evaluate
    metrics, metrics_calc = evaluate_model(model, test_loader, device)

    # Print detailed results
    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)

    metrics_calc.print_summary()

    # Print classification report
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print(metrics_calc.get_classification_report())

    # Save confusion matrix
    save_dir = project_root / "models" / "mobilenet_checkpoints"
    confusion_matrix_path = save_dir / "confusion_matrix.png"

    print(f"\n📊 Generating confusion matrix...")
    metrics_calc.plot_confusion_matrix(
        save_path=confusion_matrix_path,
        normalize=False
    )

    # Also save normalized version
    confusion_matrix_norm_path = save_dir / "confusion_matrix_normalized.png"
    metrics_calc.plot_confusion_matrix(
        save_path=confusion_matrix_norm_path,
        normalize=True
    )

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"✓ Distraction Recall: {metrics['distraction_recall']:.4f} ({metrics['distraction_recall']*100:.2f}%)")

    if metrics['accuracy'] >= 0.92:
        print("\n✅ Model meets accuracy target (>92%)")
    else:
        print(f"\n⚠️  Model below accuracy target ({metrics['accuracy']*100:.2f}% < 92%)")

    if metrics['distraction_recall'] >= 0.95:
        print("✅ Model meets distraction recall target (>95%)")
    else:
        print(f"⚠️  Model below distraction recall target ({metrics['distraction_recall']*100:.2f}% < 95%)")

    print("\n" + "=" * 80)
    print("Evaluation complete! Model is ready for deployment.")
    print("=" * 80)
    print(f"\nConfusion matrices saved to:")
    print(f"  - {confusion_matrix_path}")
    print(f"  - {confusion_matrix_norm_path}")
    print(f"\nNext steps:")
    print("  1. Test with webcam: python demo_mobilenet.py")
    print("  2. Quantize for edge: python quantize_model.py")
    print("  3. Deploy to Raspberry Pi")


if __name__ == "__main__":
    main()
