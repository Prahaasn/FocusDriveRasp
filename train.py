"""
Main training script for FocusDrive driver distraction detection.

Usage:
    python train.py --epochs 30 --batch-size 4 --lr 1e-5

This script will:
1. Load preprocessed data
2. Initialize LFM2-VL-1.6B model
3. Train with AdamW optimizer and cosine scheduler
4. Save checkpoints and best model
5. Generate training metrics and plots
"""

import os
# Enable MPS fallback for unsupported operations (MUST BE BEFORE TORCH)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from pathlib import Path
import json
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.lfm_classifier import LFMDriverClassifier
from src.data.dataset import DriverDistractionDataset, get_dataloader
from src.training.trainer import Trainer
from src.utils.metrics import MetricsCalculator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train FocusDrive driver distraction classifier'
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
        '--model-name',
        type=str,
        default='LiquidAI/LFM2-VL-1.6B',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=2,
        help='Number of output classes'
    )
    parser.add_argument(
        '--freeze-vision',
        action='store_true',
        help='Freeze vision encoder'
    )
    parser.add_argument(
        '--freeze-language',
        action='store_true',
        help='Freeze language model'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay for AdamW'
    )
    parser.add_argument(
        '--warmup-ratio',
        type=float,
        default=0.1,
        help='Warmup ratio for scheduler'
    )
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=5,
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
    parser.add_argument(
        '--mixed-precision',
        action='store_true',
        default=True,
        help='Use mixed precision training'
    )
    parser.add_argument(
        '--no-mixed-precision',
        action='store_false',
        dest='mixed_precision',
        help='Disable mixed precision training'
    )

    # Checkpoint arguments
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models/checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    # Data augmentation
    parser.add_argument(
        '--augment',
        action='store_true',
        default=True,
        help='Use data augmentation'
    )
    parser.add_argument(
        '--no-augment',
        action='store_false',
        dest='augment',
        help='Disable data augmentation'
    )

    return parser.parse_args()


def load_class_weights(data_dir: Path, device: torch.device):
    """Load class weights from config.json."""
    config_path = data_dir / "config.json"

    if not config_path.exists():
        print("  Warning: config.json not found, using no class weights")
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)

    if 'class_weights' not in config['metadata']:
        return None

    weights = config['metadata']['class_weights']
    weight_tensor = torch.tensor([weights[str(i)] for i in range(len(weights))])

    print(f"Class weights loaded: {weight_tensor.tolist()}")

    return weight_tensor.to(device)


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 80)
    print("FocusDrive - Driver Distraction Detection Training")
    print("=" * 80)

    # Set up paths
    project_root = Path(__file__).parent
    data_dir = project_root / args.data_dir
    save_dir = project_root / args.save_dir

    # Check data exists
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        print("L Error: Data CSVs not found!")
        print(f"Expected: {train_csv}")
        print(f"Expected: {val_csv}")
        print("\nPlease run:")
        print("  1. python src/data/download_dataset.py")
        print("  2. python src/data/preprocess.py")
        sys.exit(1)

    # Determine device
    # MPS is now compatible thanks to fallback mode + float32
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("✓ Using MPS (Mac GPU) with float32 for compatibility")
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"\n=  Device: {device}")

    # Load processor
    print(f"\n= Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Create data loaders
    print(f"\n= Creating data loaders...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Augmentation: {args.augment}")
    print(f"  Num workers: {args.num_workers}")

    train_dataset = DriverDistractionDataset(
        csv_path=train_csv,
        processor=processor,
        augment=args.augment
    )

    val_dataset = DriverDistractionDataset(
        csv_path=val_csv,
        processor=processor,
        augment=False  # No augmentation for validation
    )

    # Custom collate function to handle PIL images
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])
        paths = [item['image_path'] for item in batch]
        class_names = [item['class_name'] for item in batch]

        return {
            'image': images,  # List of PIL images
            'label': labels,
            'image_path': paths,
            'class_name': class_names
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with PIL
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with PIL
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Load class weights
    class_weights = load_class_weights(data_dir, device)

    # Initialize model
    print(f"\n> Initializing model: {args.model_name}")
    model = LFMDriverClassifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        freeze_vision_encoder=args.freeze_vision,
        freeze_language_model=args.freeze_language,
        device=device
    )

    model.print_parameter_stats()

    # Initialize optimizer
    print(f"\n  Setting up optimizer and scheduler...")
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Calculate total training steps
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  Scheduler: Cosine with warmup")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f"\n= Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"  Resuming from epoch {start_epoch}")

    # Initialize trainer
    print(f"\n<  Initializing trainer...")
    trainer = Trainer(
        model=model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        class_names=['Attentive', 'Distracted'],
        class_weights=class_weights,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Print training configuration
    print("\n" + "=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print(f"Save directory: {save_dir}")
    print("=" * 80)

    # Train
    try:
        trainer.train(
            num_epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            save_every=1
        )

        print("\n Training completed successfully!")
        print(f"\nBest model saved to: {save_dir / 'best_model.pt'}")
        print(f"Final accuracy: {trainer.best_val_acc:.4f} ({trainer.best_val_acc*100:.2f}%)")
        print(f"Distraction recall: {trainer.best_val_recall:.4f} ({trainer.best_val_recall*100:.2f}%)")

        if trainer.best_val_acc >= 0.92:
            print("\n< Target accuracy achieved (>92%)!")
        else:
            print(f"\n  Target accuracy not reached (current: {trainer.best_val_acc*100:.2f}%, target: 92%)")

        if trainer.best_val_recall >= 0.95:
            print("< Target distraction recall achieved (>95%)!")
        else:
            print(f"  Target distraction recall not reached (current: {trainer.best_val_recall*100:.2f}%, target: 95%)")

    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
        print("Saving current checkpoint...")
        trainer.save_checkpoint(
            epoch=len(trainer.history['train_loss']),
            metrics={'accuracy': trainer.best_val_acc},
            is_best=False
        )
        print("Checkpoint saved.")

    except Exception as e:
        print(f"\nL Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
