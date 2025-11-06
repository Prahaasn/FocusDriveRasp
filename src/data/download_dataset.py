"""
Download State Farm Distracted Driver Detection dataset from Kaggle.

This script:
1. Checks for Kaggle API credentials
2. Downloads the State Farm dataset
3. Extracts it to data/raw/
4. Verifies the download
"""

import os
import sys
import zipfile
from pathlib import Path
import subprocess


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        print("❌ Kaggle API credentials not found!")
        print("\nTo set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Scroll to 'API' section and click 'Create New Token'")
        print("3. This downloads kaggle.json")
        print(f"4. Move it to {kaggle_dir}/")
        print(f"5. Run: chmod 600 {kaggle_json}")
        return False

    # Check permissions
    if os.name != 'nt':  # Not Windows
        stats = kaggle_json.stat()
        if oct(stats.st_mode)[-3:] != '600':
            print(f"⚠️  Warning: {kaggle_json} has incorrect permissions")
            print(f"Run: chmod 600 {kaggle_json}")
            return False

    print("✓ Kaggle API credentials found")
    return True


def download_dataset(output_dir: Path):
    """Download State Farm Distracted Driver Detection dataset."""
    dataset_name = "c/state-farm-distracted-driver-detection"

    print(f"\n📥 Downloading dataset: {dataset_name}")
    print(f"📁 Destination: {output_dir}")
    print("⏳ This may take a while (~2.5 GB)...\n")

    try:
        # Download using kaggle CLI
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", "state-farm-distracted-driver-detection", "-p", str(output_dir)],
            capture_output=True,
            text=True,
            check=True
        )

        print(result.stdout)
        print("✓ Download complete!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Kaggle CLI not found. Install with: pip install kaggle")
        return False


def extract_dataset(zip_path: Path, extract_to: Path):
    """Extract the downloaded zip file."""
    print(f"\n📦 Extracting {zip_path.name}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total number of files
            total_files = len(zip_ref.namelist())

            # Extract with progress
            for i, member in enumerate(zip_ref.namelist(), 1):
                zip_ref.extract(member, extract_to)
                if i % 1000 == 0 or i == total_files:
                    print(f"  Extracted {i}/{total_files} files...", end='\r')

            print(f"\n✓ Extraction complete! ({total_files} files)")

        return True

    except Exception as e:
        print(f"❌ Error extracting dataset: {e}")
        return False


def verify_dataset(data_dir: Path):
    """Verify the dataset structure."""
    print("\n🔍 Verifying dataset structure...")

    # Expected directories
    expected_dirs = ["train", "test"]
    expected_classes = [f"c{i}" for i in range(10)]

    # Check main directories
    for dir_name in expected_dirs:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            print(f"❌ Missing directory: {dir_name}")
            return False

    # Check train classes
    train_dir = data_dir / "train"
    for class_name in expected_classes:
        class_path = train_dir / class_name
        if not class_path.exists():
            print(f"❌ Missing class directory: train/{class_name}")
            return False

        # Count images
        images = list(class_path.glob("*.jpg"))
        print(f"  ✓ {class_name}: {len(images)} images")

    # Check test directory
    test_dir = data_dir / "test"
    test_images = list(test_dir.glob("*.jpg"))
    print(f"  ✓ test: {len(test_images)} images")

    print("\n✓ Dataset verification complete!")
    print(f"\nDataset location: {data_dir}")
    print(f"Total training images: {sum(len(list((train_dir / c).glob('*.jpg'))) for c in expected_classes)}")
    print(f"Total test images: {len(test_images)}")

    return True


def main():
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("State Farm Distracted Driver Detection - Dataset Downloader")
    print("=" * 60)

    # Step 1: Check credentials
    if not check_kaggle_credentials():
        sys.exit(1)

    # Step 2: Check if already downloaded
    expected_zip = raw_data_dir / "state-farm-distracted-driver-detection.zip"
    imgs_dir = raw_data_dir / "imgs"

    if imgs_dir.exists() and verify_dataset(imgs_dir):
        print("\n✓ Dataset already downloaded and verified!")
        print("To re-download, delete the data/raw/ directory first.")
        return

    # Step 3: Download dataset
    if not expected_zip.exists():
        if not download_dataset(raw_data_dir):
            sys.exit(1)
    else:
        print(f"\n✓ Zip file already exists: {expected_zip.name}")

    # Step 4: Extract dataset
    if not imgs_dir.exists():
        if not extract_dataset(expected_zip, raw_data_dir):
            sys.exit(1)

    # Step 5: Verify
    if not verify_dataset(imgs_dir):
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 Dataset setup complete! Ready for preprocessing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
