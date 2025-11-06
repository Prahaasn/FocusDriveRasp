"""
Data preprocessing pipeline for FocusDrive.

Maps State Farm's 10 classes to 3 classes:
- c0 (safe driving) → Attentive (0)
- c1, c2, c3, c4 (texting/phone) → Distracted (1)
- c5, c6, c7, c8, c9 (other distractions) → Distracted (1)

Note: For drowsiness detection, we'll need additional data or
modify the approach since State Farm doesn't have drowsy samples.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# Class mapping configuration
CLASS_MAPPING = {
    # Original State Farm classes to their descriptions
    'c0': 'safe driving',
    'c1': 'texting - right',
    'c2': 'talking on the phone - right',
    'c3': 'texting - left',
    'c4': 'talking on the phone - left',
    'c5': 'operating the radio',
    'c6': 'drinking',
    'c7': 'reaching behind',
    'c8': 'hair and makeup',
    'c9': 'talking to passenger'
}

# Map to 3 classes for FocusDrive
# Note: Since State Farm doesn't have drowsy samples, we use 2 classes for now
FOCUSDRIVE_MAPPING = {
    'c0': 0,  # Attentive
    'c1': 1,  # Distracted (texting - right)
    'c2': 1,  # Distracted (talking on phone - right)
    'c3': 1,  # Distracted (texting - left)
    'c4': 1,  # Distracted (talking on phone - left)
    'c5': 1,  # Distracted (operating radio)
    'c6': 1,  # Distracted (drinking)
    'c7': 1,  # Distracted (reaching behind)
    'c8': 1,  # Distracted (hair and makeup)
    'c9': 1,  # Distracted (talking to passenger)
}

CLASS_NAMES = {
    0: 'Attentive',
    1: 'Distracted',
    # 2: 'Drowsy'  # Not in State Farm dataset
}


def verify_image(image_path: Path) -> bool:
    """Verify that an image can be opened and is valid."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def create_metadata(raw_data_dir: Path, output_csv: Path) -> pd.DataFrame:
    """
    Create metadata CSV with all images and their labels.

    Returns:
        DataFrame with columns: image_path, original_class, focusdrive_class, class_name
    """
    print("\n📝 Creating metadata CSV...")

    train_dir = raw_data_dir / "imgs" / "train"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    data = []

    # Iterate through each class directory
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        original_class = class_dir.name
        if original_class not in FOCUSDRIVE_MAPPING:
            print(f"⚠️  Warning: Unknown class {original_class}, skipping...")
            continue

        focusdrive_class = FOCUSDRIVE_MAPPING[original_class]
        class_name = CLASS_NAMES[focusdrive_class]

        # Get all images in this class
        images = list(class_dir.glob("*.jpg"))

        print(f"  Processing {original_class} ({class_name}): {len(images)} images")

        for img_path in tqdm(images, desc=f"  {original_class}", leave=False):
            # Verify image is valid
            if not verify_image(img_path):
                print(f"    ⚠️  Skipping corrupted image: {img_path.name}")
                continue

            data.append({
                'image_path': str(img_path),
                'original_class': original_class,
                'focusdrive_class': focusdrive_class,
                'class_name': class_name,
                'image_name': img_path.name
            })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n✓ Metadata CSV created: {output_csv}")
    print(f"  Total images: {len(df)}")
    print(f"\n  Class distribution:")
    for class_id, class_name in CLASS_NAMES.items():
        count = len(df[df['focusdrive_class'] == class_id])
        percentage = (count / len(df)) * 100
        print(f"    {class_name}: {count} ({percentage:.1f}%)")

    return df


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        df: Metadata DataFrame
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for testing (default 0.15)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    print(f"\n📊 Splitting dataset: {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test")

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df['focusdrive_class'],
        random_state=random_state
    )

    # Second split: separate train and validation
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_df['focusdrive_class'],
        random_state=random_state
    )

    print(f"  Train: {len(train_df)} images")
    print(f"  Val:   {len(val_df)} images")
    print(f"  Test:  {len(test_df)} images")

    # Print class distribution for each split
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n  {split_name} distribution:")
        for class_id, class_name in CLASS_NAMES.items():
            count = len(split_df[split_df['focusdrive_class'] == class_id])
            percentage = (count / len(split_df)) * 100
            print(f"    {class_name}: {count} ({percentage:.1f}%)")

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path
):
    """Save train, val, test splits to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"\n✓ Split CSVs saved:")
    print(f"  {train_csv}")
    print(f"  {val_csv}")
    print(f"  {test_csv}")


def create_class_weights(df: pd.DataFrame) -> Dict[int, float]:
    """
    Calculate class weights for handling imbalanced dataset.

    Returns:
        Dictionary mapping class_id to weight
    """
    class_counts = df['focusdrive_class'].value_counts().to_dict()
    total_samples = len(df)

    # Calculate weights: inverse of class frequency
    weights = {}
    for class_id in CLASS_NAMES.keys():
        count = class_counts.get(class_id, 0)
        if count > 0:
            # Weight = total_samples / (num_classes * class_count)
            weights[class_id] = total_samples / (len(CLASS_NAMES) * count)
        else:
            weights[class_id] = 0.0

    print("\n⚖️  Class weights (for handling imbalance):")
    for class_id, weight in weights.items():
        print(f"  {CLASS_NAMES[class_id]}: {weight:.4f}")

    return weights


def create_config(output_path: Path, metadata: dict):
    """Save preprocessing configuration and metadata."""
    config = {
        'class_mapping': CLASS_MAPPING,
        'focusdrive_mapping': FOCUSDRIVE_MAPPING,
        'class_names': CLASS_NAMES,
        'metadata': metadata
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Configuration saved: {output_path}")


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("FocusDrive - Data Preprocessing Pipeline")
    print("=" * 60)

    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    raw_data_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    # Check if raw data exists
    if not (raw_data_dir / "imgs" / "train").exists():
        print("❌ Raw data not found!")
        print("Please run: python src/data/download_dataset.py")
        return

    # Step 1: Create metadata
    metadata_csv = processed_dir / "metadata.csv"
    df = create_metadata(raw_data_dir, metadata_csv)

    # Step 2: Split dataset
    train_df, val_df, test_df = split_dataset(df)

    # Step 3: Save splits
    save_splits(train_df, val_df, test_df, processed_dir)

    # Step 4: Calculate class weights
    class_weights = create_class_weights(train_df)

    # Step 5: Save configuration
    config_metadata = {
        'total_images': len(df),
        'train_images': len(train_df),
        'val_images': len(val_df),
        'test_images': len(test_df),
        'class_weights': class_weights,
        'splits': {
            'train': 0.70,
            'val': 0.15,
            'test': 0.15
        }
    }
    create_config(processed_dir / "config.json", config_metadata)

    print("\n" + "=" * 60)
    print("🎉 Preprocessing complete!")
    print("=" * 60)
    print(f"\nProcessed data location: {processed_dir}")
    print("\nNext steps:")
    print("1. Review the data splits in data/processed/")
    print("2. Create the PyTorch Dataset class")
    print("3. Start training!")


if __name__ == "__main__":
    main()
