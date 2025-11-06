"""
PyTorch Dataset for MobileNetV3 driver distraction detection.

Optimized for:
- 224×224 input size
- ImageNet normalization
- Fast data loading
- Standard CNN preprocessing
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import torchvision.transforms as T


class MobileNetDataset(Dataset):
    """
    Dataset for driver distraction detection with MobileNetV3.

    Args:
        csv_path: Path to CSV file with columns: image_path, focusdrive_class
        augment: Whether to apply data augmentation (default: False)
        image_size: Target image size (height, width). Default: (224, 224)
    """

    def __init__(
        self,
        csv_path: Path,
        augment: bool = False,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.df = pd.read_csv(csv_path)
        self.augment = augment
        self.image_size = image_size

        # ImageNet normalization (MobileNetV3 is pretrained on ImageNet)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Define transforms
        if self.augment:
            # Training augmentation
            self.transform = T.Compose([
                T.Resize(self.image_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05
                ),
                T.RandomAffine(
                    degrees=5,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                ),
                T.RandomGrayscale(p=0.05),
                T.ToTensor(),
                self.normalize
            ])
        else:
            # Validation/test transform (no augmentation)
            self.transform = T.Compose([
                T.Resize(self.image_size),
                T.ToTensor(),
                self.normalize
            ])

        print(f"MobileNetDataset loaded: {len(self.df)} samples")
        print(f"  Image size: {self.image_size}")
        print(f"  Augmentation: {'ENABLED' if self.augment else 'DISABLED'}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - image: Preprocessed image tensor [3, 224, 224]
            - label: Class label (0=Attentive, 1=Distracted)
            - image_path: Path to original image
            - class_name: Human-readable class name
        """
        row = self.df.iloc[idx]

        # Load image
        image_path = Path(row['image_path'])
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        image_tensor = self.transform(image)

        # Get label
        label = int(row['focusdrive_class'])

        # Get class name
        class_name = 'Attentive' if label == 0 else 'Distracted'

        return {
            'image': image_tensor,
            'label': label,
            'image_path': str(image_path),
            'class_name': class_name
        }


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        image_size: Target image size (height, width)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    # Create datasets
    train_dataset = MobileNetDataset(
        csv_path=data_dir / "train.csv",
        augment=True,
        image_size=image_size
    )

    val_dataset = MobileNetDataset(
        csv_path=data_dir / "val.csv",
        augment=False,
        image_size=image_size
    )

    test_dataset = MobileNetDataset(
        csv_path=data_dir / "test.csv",
        augment=False,
        image_size=image_size
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Test the dataset."""
    print("Testing MobileNetDataset...\n")

    # Test with sample data
    data_dir = Path("data/processed")

    if not (data_dir / "train.csv").exists():
        print("Error: Data not found. Please run:")
        print("  1. python src/data/download_dataset.py")
        print("  2. python src/data/preprocess.py")
    else:
        # Create a single dataset
        dataset = MobileNetDataset(
            csv_path=data_dir / "train.csv",
            augment=True
        )

        # Test __getitem__
        sample = dataset[0]
        print(f"\nSample data:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Image dtype: {sample['image'].dtype}")
        print(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        print(f"  Label: {sample['label']} ({sample['class_name']})")
        print(f"  Path: {sample['image_path']}")

        # Test dataloader
        print("\nTesting DataLoader...")
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=8,
            num_workers=0  # 0 for testing
        )

        # Get one batch
        batch = next(iter(train_loader))
        print(f"\nBatch data:")
        print(f"  Images shape: {batch['image'].shape}")
        print(f"  Labels shape: {batch['label'].shape}")
        print(f"  Labels: {batch['label'].tolist()}")
        print(f"  Class names: {batch['class_name'][:3]}...")  # First 3

        print("\n✓ Dataset test passed!")
