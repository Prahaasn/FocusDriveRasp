# FocusDrive - Quick Start Guide

Get FocusDrive up and running in 10 minutes!

## 🚀 Prerequisites

- **Mac M1/M2/M3** with 16GB+ RAM (for training)
- **Python 3.8+**
- **20-30GB** free disk space
- **Kaggle account** (for dataset download)

## 📋 Step-by-Step Setup

### Step 1: Install Dependencies (2 minutes)

```bash
cd "Focus Drive Lstm model"
pip install -r requirements.txt
```

This will install:
- PyTorch (with MPS support for Mac)
- Transformers (for LFM2-VL)
- OpenCV (for camera)
- And all other dependencies

### Step 2: Set Up Kaggle API (2 minutes)

1. **Get your Kaggle API credentials:**
   - Go to https://www.kaggle.com/settings/account
   - Scroll to "API" section
   - Click "Create New Token"
   - This downloads `kaggle.json`

2. **Install credentials:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Step 3: Download Dataset (5-10 minutes)

```bash
python src/data/download_dataset.py
```

This will:
- Download State Farm dataset (~2.5 GB)
- Extract all images
- Verify dataset structure

**Expected output:**
```
State Farm Distracted Driver Detection - Dataset Downloader
============================================================
✓ Kaggle API credentials found
📥 Downloading dataset...
✓ Download complete!
📦 Extracting...
✓ Extraction complete!
🔍 Verifying dataset structure...
  ✓ c0: 2489 images
  ✓ c1: 2267 images
  ...
🎉 Dataset setup complete!
```

### Step 4: Preprocess Data (2-3 minutes)

```bash
python src/data/preprocess.py
```

This will:
- Map 10 classes → 2 classes (Attentive vs Distracted)
- Create train/val/test splits (70/15/15)
- Calculate class weights
- Save metadata CSVs

**Expected output:**
```
FocusDrive - Data Preprocessing Pipeline
=========================================
📝 Creating metadata CSV...
  Total images: 22424
  Class distribution:
    Attentive: 2489 (11.1%)
    Distracted: 19935 (88.9%)

📊 Splitting dataset...
  Train: 15697 images
  Val:   3363 images
  Test:  3364 images

🎉 Preprocessing complete!
```

### Step 5: Train the Model (3-6 hours)

```bash
# Basic training command
python train.py --epochs 30 --batch-size 4 --lr 1e-5
```

**What happens:**
- Downloads LFM2-VL-1.6B (~3-4 GB) on first run
- Trains for 30 epochs with early stopping
- Saves checkpoints to `models/checkpoints/`
- Best model saved as `best_model.pt`

**Monitor progress:**
```
Epoch 1/30
----------------------------------------------------------
Epoch 1 [Train]: 100%|████████| 3925/3925 [12:34<00:00, 5.20it/s]

Validation Metrics:
  Loss: 0.2345
  Accuracy: 0.9234 (92.34%)
  Distraction Recall: 0.9567 (95.67%)

  ✓ New best model saved! (Accuracy: 0.9234)
```

**Tips for faster training:**
- Use `--freeze-vision` to freeze vision encoder (trains 2-3x faster)
- Lower `--batch-size 2` if you run out of memory
- Use `--gradient-accumulation-steps 2` for effective larger batch size

### Step 6: Test with Webcam (1 minute)

```bash
python demo.py --model models/checkpoints/best_model.pt
```

**What you'll see:**
- Live webcam feed
- Real-time classification (Attentive/Distracted)
- Confidence scores
- FPS counter
- Alert borders when distracted

**Controls:**
- Press `q` to quit

## 🎯 Quick Commands Reference

```bash
# Download dataset
python src/data/download_dataset.py

# Preprocess data
python src/data/preprocess.py

# Train (fast mode - frozen vision encoder)
python train.py --epochs 30 --batch-size 4 --freeze-vision

# Train (full fine-tuning)
python train.py --epochs 30 --batch-size 2

# Demo with webcam
python demo.py --model models/checkpoints/best_model.pt

# Demo with lower FPS (less CPU intensive)
python demo.py --model models/checkpoints/best_model.pt --fps-target 3

# Test individual components
python src/data/dataset.py
python src/models/lfm_classifier.py
```

## 📊 Expected Results

After training for 30 epochs on Mac M1/M2/M3:

| Metric | Target | Typical Result |
|--------|--------|---------------|
| Accuracy | >92% | 92-95% |
| Distraction Recall | >95% | 95-98% |
| Training Time | - | 3-6 hours |
| Model Size | - | ~3-4 GB |
| Inference Speed | - | 5-10 FPS on Mac |

## 🐛 Common Issues

### 1. "Out of memory" during training

```bash
# Solution: Reduce batch size
python train.py --batch-size 2 --gradient-accumulation-steps 2
```

### 2. "Kaggle API credentials not found"

```bash
# Check if kaggle.json exists
ls -la ~/.kaggle/

# Fix permissions
chmod 600 ~/.kaggle/kaggle.json
```

### 3. "Camera not found" in demo

```bash
# List available cameras
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# Use different camera
python demo.py --model models/checkpoints/best_model.pt --camera 1
```

### 4. Model download fails

```bash
# Check internet connection
ping huggingface.co

# Retry with manual download
# Visit: https://huggingface.co/LiquidAI/LFM2-VL-1.6B
```

### 5. MPS (Mac GPU) issues

```bash
# Fall back to CPU
python train.py --device cpu
```

## 📚 Next Steps

Once you have a working model:

1. **Optimize for deployment:**
   - Quantize model to INT8 (coming soon)
   - Convert to ONNX/TFLite (coming soon)

2. **Deploy to Raspberry Pi:**
   - Copy model to Pi
   - Install lightweight dependencies
   - Run real-time monitoring

3. **Improve accuracy:**
   - Fine-tune with unfrozen vision encoder
   - Add data augmentation
   - Collect more diverse data

4. **Add features:**
   - Audio alerts
   - Multi-frame temporal analysis
   - Drowsiness detection

## 💡 Tips for Best Results

1. **Training:**
   - Start with frozen vision encoder for faster iteration
   - Monitor distraction recall (more important than accuracy)
   - Use early stopping to prevent overfitting

2. **Demo:**
   - Good lighting is important
   - Position camera at driver's eye level
   - Adjust FPS based on hardware capabilities

3. **Deployment:**
   - Quantize model for edge devices
   - Use batch size 1 for real-time inference
   - Consider frame skipping for lower-end hardware

## 📞 Getting Help

- **Documentation:** See README.md
- **Issues:** Check troubleshooting section above
- **Model info:** https://huggingface.co/LiquidAI/LFM2-VL-1.6B

---

**Ready to make roads safer? Let's go! 🚗💨**
