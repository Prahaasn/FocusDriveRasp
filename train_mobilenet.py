"""
Training script for MobileNetV3 driver distraction classifier.

Simplified and optimized for:
- Fast training (1-2 hours on Mac)
- Edge deployment (Raspberry Pi)
- High accuracy (>92%)

Usage:
    python train_mobilenet.py --epochs 20 --batch-size 32 --lr 0.001
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pathlib import Path
import json
import sys
from tqdm import tqdm
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.mobilenet_classifier import MobileNetDriverClassifier
from src.data.mobilenet_dataset import get_dataloaders
from src.utils.metrics import MetricsCalculator, AverageMeter, EarlyStopping


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MobileNetV3 driver distraction classifier'
    )

    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory with processed data CSVs'
    )

    # Model arguments
    parser.add_argument(
        '--num-classes',
        type=int,
        default=2,
        help='Number of output classes'
    )
    parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='Freeze backbone (only train classifier head)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'sgd'],
        help='Optimizer to use'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='cosine',
        choices=['cosine', 'plateau', 'none'],
        help='Learning rate scheduler'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0001,
        help='Weight decay'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=7,
        help='Early stopping patience'
    )

    # System arguments
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to train on'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )

    # Checkpoint arguments
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models/mobilenet_checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    return parser.parse_args()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter()
    metrics_calc = MetricsCalculator(
        num_classes=2,
        class_names=['Attentive', 'Distracted']
    )

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch} [Train]",
        total=len(train_loader)
    )

    for batch in pbar:
        # Move batch to device
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, labels)
        loss = outputs['loss']

        # Backward pass
        loss.backward()
        optimizer.step()

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

    return metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch):
    """Validate the model."""
    model.eval()

    loss_meter = AverageMeter()
    metrics_calc = MetricsCalculator(
        num_classes=2,
        class_names=['Attentive', 'Distracted']
    )

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch} [Val]  ",
        total=len(val_loader)
    )

    for batch in pbar:
        # Move batch to device
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(images, labels)
        loss = outputs['loss']

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


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_dir, is_best=False):
    """Save model checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Save regular checkpoint
    checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = save_dir / "best_model.pt"
        torch.save(checkpoint, best_path)

        # Also save in MobileNetDriverClassifier format
        model.save_pretrained(save_dir / "best_model_pretrained")
        print(f"  ✓ New best model saved! (Accuracy: {metrics['accuracy']:.4f})")

    # Save latest
    latest_path = save_dir / "latest_model.pt"
    torch.save(checkpoint, latest_path)


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 80)
    print("MobileNetV3 - Driver Distraction Detection Training")
    print("=" * 80)

    # Set up paths
    project_root = Path(__file__).parent
    data_dir = project_root / args.data_dir
    save_dir = project_root / args.save_dir

    # Check data exists
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        print("❌ Error: Data CSVs not found!")
        print(f"Expected: {train_csv}")
        print(f"Expected: {val_csv}")
        print("\nPlease run:")
        print("  1. python src/data/download_dataset.py")
        print("  2. python src/data/preprocess.py")
        sys.exit(1)

    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"\n🖥️  Device: {device}")

    # Create data loaders
    print(f"\n📊 Creating data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Load class weights
    config_path = data_dir / "config.json"
    class_weights = None
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        if 'class_weights' in config['metadata']:
            weights = config['metadata']['class_weights']
            class_weights = torch.tensor([weights[str(i)] for i in range(len(weights))])
            class_weights = class_weights.to(device)
            print(f"Class weights loaded: {class_weights.tolist()}")

    # Initialize model
    print(f"\n🔧 Initializing MobileNetV3 model...")
    model = MobileNetDriverClassifier(
        num_classes=args.num_classes,
        freeze_backbone=args.freeze_backbone,
        pretrained=True,
        dropout=args.dropout,
        device=device
    )

    model.print_parameter_stats()

    # Initialize optimizer
    print(f"\n⚙️  Setting up optimizer and scheduler...")

    if args.optimizer == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:  # sgd
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )

    # Initialize scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
    else:
        scheduler = None

    print(f"  Optimizer: {args.optimizer.upper()} (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  Scheduler: {args.scheduler}")

    # Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_acc = 0.0
    if args.resume:
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['metrics'].get('accuracy', 0.0)
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best accuracy so far: {best_val_acc:.4f}")

    # Print training configuration
    print("\n" + "=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print(f"Save directory: {save_dir}")
    print("=" * 80)

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        mode='max'
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_metrics': [],
        'learning_rates': []
    }

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['accuracy'])
            else:
                scheduler.step()

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_metrics'].append(val_metrics)
        history['learning_rates'].append(current_lr)

        # Check if best model
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
            best_val_recall = val_metrics.get('distraction_recall', 0.0)

        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, save_dir, is_best)

        # Early stopping
        if early_stopping(val_metrics['accuracy']):
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
            print(f"  Best validation accuracy: {best_val_acc:.4f}")
            break

    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {elapsed_time/60:.2f} minutes ({elapsed_time/3600:.2f} hours)")
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Best distraction recall: {best_val_recall:.4f} ({best_val_recall*100:.2f}%)")

    # Check targets
    if best_val_acc >= 0.92:
        print("\n✓ Target accuracy achieved (>92%)!")
    else:
        print(f"\n⚠️  Target accuracy not reached (current: {best_val_acc*100:.2f}%, target: 92%)")

    if best_val_recall >= 0.95:
        print("✓ Target distraction recall achieved (>95%)!")
    else:
        print(f"⚠️  Target distraction recall not reached (current: {best_val_recall*100:.2f}%, target: 95%)")

    # Save final history
    # Convert numpy types to Python types for JSON serialization
    history_serializable = {
        'train_loss': [float(x) for x in history['train_loss']],
        'train_acc': [float(x) for x in history['train_acc']],
        'val_loss': [float(x) for x in history['val_loss']],
        'val_acc': [float(x) for x in history['val_acc']],
        'learning_rates': [float(x) for x in history['learning_rates']],
        'val_metrics': [
            {k: float(v) if isinstance(v, (int, float)) else v
             for k, v in metrics.items()}
            for metrics in history['val_metrics']
        ]
    }

    history_path = save_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    print(f"Best model saved to: {save_dir / 'best_model_pretrained'}")


if __name__ == "__main__":
    main()
