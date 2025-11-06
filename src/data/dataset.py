"""
PyTorch Dataset for FocusDrive driver distraction detection.

Supports:
- Single-frame classification
- Image augmentation
- LFM2-VL preprocessing
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Tuple
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class DriverDistractionDataset(Dataset):
    """
    Dataset for driver distraction detection.

    Args:
        csv_path: Path to CSV file with columns: image_path, focusdrive_class
        processor: HuggingFace processor for LFM2-VL (optional)
        transform: Custom transform function (optional)
        augment: Whether to apply data augmentation (default: False)
        image_size: Target image size (height, width). Default: (512, 512) for LFM2-VL
    """

    def __init__(
        self,
        csv_path: Path,
        processor: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        augment: bool = False,
        image_size: Tuple[int, int] = (512, 512)
    ):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.transform = transform
        self.augment = augment
        self.image_size = image_size

        # Define augmentation pipeline if enabled
        if self.augment:
            self.augmentation = self._get_augmentation_transform()
        else:
            self.augmentation = None

        # Basic preprocessing (resize + normalize)
        self.basic_transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"Dataset loaded: {len(self.df)} samples")
        if self.augment:
            print("  Augmentation: ENABLED")

    def _get_augmentation_transform(self):
        """Define data augmentation pipeline."""
        return T.Compose([
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
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - image: Processed image (if using processor) or tensor
            - label: Class label (0=Attentive, 1=Distracted)
            - image_path: Path to original image
            - class_name: Human-readable class name
        """
        row = self.df.iloc[idx]

        # Load image
        image_path = Path(row['image_path'])
        image = Image.open(image_path).convert('RGB')

        # Get label
        label = int(row['focusdrive_class'])
        class_name = row['class_name']

        # Apply transforms
        if self.processor is not None:
            # Use HuggingFace processor for LFM2-VL
            # The processor expects PIL images
            if self.augment and self.augmentation:
                # Apply augmentation before processor
                image = self.augmentation(image)

            # Process for LFM2-VL
            # Note: We'll handle the actual processing in the training loop
            # since processor might need text prompts too
            processed = {
                'image': image,  # Keep as PIL for now
                'label': label,
                'image_path': str(image_path),
                'class_name': class_name
            }

        elif self.transform is not None:
            # Use custom transform
            image = self.transform(image)
            processed = {
                'image': image,
                'label': label,
                'image_path': str(image_path),
                'class_name': class_name
            }

        else:
            # Use default transform pipeline
            if self.augment and self.augmentation:
                image = self.augmentation(image)
            else:
                image = self.basic_transform(image)

            processed = {
                'image': image,
                'label': label,
                'image_path': str(image_path),
                'class_name': class_name
            }

        return processed

    def get_class_distribution(self):
        """Get the distribution of classes in the dataset."""
        return self.df['focusdrive_class'].value_counts().sort_index().to_dict()


def collate_fn_for_lfm(batch, processor, prompt_template: str = "Classify the driver's behavior as 'attentive' or 'distracted'."):
    """
    Custom collate function for LFM2-VL model.

    Args:
        batch: List of samples from DriverDistractionDataset
        processor: HuggingFace processor for LFM2-VL
        prompt_template: Prompt to use for classification

    Returns:
        Dictionary with processed inputs ready for LFM2-VL
    """
    images = [sample['image'] for sample in batch]
    labels = torch.tensor([sample['label'] for sample in batch])

    # Create conversations for each image
    conversations = []
    for image in images:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_template},
                ],
            },
        ]
        conversations.append(conversation)

    # Process all conversations
    # Note: This is a simplified version. Actual implementation may vary
    # based on how we structure the fine-tuning
    processed_inputs = processor(
        conversations,
        return_tensors="pt",
        padding=True
    )

    # Add labels
    processed_inputs['labels'] = labels

    # Add metadata
    processed_inputs['metadata'] = {
        'image_paths': [sample['image_path'] for sample in batch],
        'class_names': [sample['class_name'] for sample in batch]
    }

    return processed_inputs


def get_dataloader(
    csv_path: Path,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    processor: Optional[Callable] = None,
    augment: bool = False,
    image_size: Tuple[int, int] = (512, 512)
):
    """
    Create a DataLoader for the dataset.

    Args:
        csv_path: Path to CSV file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        processor: HuggingFace processor (optional)
        augment: Whether to apply augmentation
        image_size: Target image size

    Returns:
        PyTorch DataLoader
    """
    dataset = DriverDistractionDataset(
        csv_path=csv_path,
        processor=processor,
        augment=augment,
        image_size=image_size
    )

    # Use custom collate function if processor is provided
    if processor is not None:
        collate_fn = lambda batch: collate_fn_for_lfm(batch, processor)
    else:
        collate_fn = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"DataLoader created:")
    print(f"  Samples: {len(dataset)}")
    print(f"  Batches: {len(dataloader)}")
    print(f"  Batch size: {batch_size}")

    return dataloader


if __name__ == "__main__":
    """Test the dataset."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / "data" / "processed" / "train.csv"

    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        print("Please run: python src/data/preprocess.py")
        sys.exit(1)

    # Test dataset without processor
    print("Testing dataset without processor...")
    dataset = DriverDistractionDataset(
        csv_path=csv_path,
        augment=True
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")

    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Class name: {sample['class_name']}")
    print(f"  Path: {sample['image_path']}")

    print("\n✓ Dataset test passed!")
