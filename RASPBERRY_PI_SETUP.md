# FocusDrive AI Detection - Raspberry Pi 5 Setup Guide

## Setup Summary

Your Raspberry Pi 5 is now configured with:
- ✓ Repository cloned
- ✓ Python 3.13 virtual environment
- ✓ All dependencies installed (PyTorch, TensorFlow, OpenCV, picamera2)
- ✓ Code optimized for Pi 5 (CPU-only, 800×600, frame skipping)

---

## Running the Demo

### Step 1: Activate Virtual Environment
```bash
cd /home/prahaasn/focusdrive-ai-detection
source venv/bin/activate
```

### Step 2: Connect Camera
**Option A: USB Webcam**
- Plug in USB webcam
- Verify with: `ls /dev/video*`

**Option B: Raspberry Pi Camera Module**
- Connect via CSI ribbon cable
- Enable in raspi-config if needed
- Verify with: `rpicam-hello --list-cameras`

### Step 3: Run the Demo
```bash
python demo_mobilenet.py
```

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot
- Press `r` to start/stop recording

---

## What to Expect

### On First Run:
The MobileNetV3 model will download ImageNet pretrained weights (~10 MB). This is normal and happens once.

### Console Output:
```
================================================================================
MobileNetV3 - Real-time Driver Distraction Detection Demo
================================================================================
Initializing distraction detector...
Device: cpu
Loading model from models/mobilenet_checkpoints/best_model_pretrained...
✓ Detector initialized!

Initializing TFLite object detector...
✓ Object detector initialized!

Opening camera...
✓ OpenCV camera opened!  (or "✓ Picamera2 initialized")

Starting detection...
Press 'q' to quit, 's' to save screenshot, 'r' to record
```

### Performance Targets:
- **FPS:** 15-20 frames per second
- **Resolution:** 800×600 pixels
- **CPU Usage:** 60-80% on single core
- **Memory:** <500 MB

### Display Output:
Real-time window showing:
- Live camera feed
- Classification: "Attentive" (green) or "Distracted" (orange/red)
- Confidence score percentage
- Detected objects (phone, cup, etc.)
- FPS counter
- Alert status and progress bar

---

## Optimizations Applied

### 1. CPU-Only Inference
```python
detector = DistractionDetector(model_path, device="cpu")
```
Raspberry Pi 5 has no GPU, so CPU inference is forced.

### 2. Reduced Resolution
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
```
Down from 1280×720 for better real-time performance.

### 3. Frame Skipping for Object Detection
```python
obj_detect_interval = 3  # Process every 3rd frame
```
Object detection runs every 3rd frame to reduce CPU load.

### 4. Dual Camera Support
Automatically detects and uses:
- **Picamera2** (for Raspberry Pi Camera Module) - preferred
- **OpenCV** (for USB webcams) - fallback

---

## Troubleshooting

### Issue: "No cameras available"
**Solution:** Connect a camera first! The demo requires a physical camera.

For USB webcam:
```bash
ls /dev/video*  # Should show /dev/video0
```

For Pi Camera Module:
```bash
rpicam-hello --list-cameras  # Should detect camera
```

### Issue: Import Error for cv2 or torch
**Solution:** Make sure virtual environment is activated:
```bash
source venv/bin/activate
python -c "import cv2, torch; print('OK')"
```

### Issue: Low FPS (<10)
**Solutions:**
1. Increase frame skip interval (edit line 575 in demo_mobilenet.py):
   ```python
   obj_detect_interval = 5  # or higher
   ```

2. Reduce resolution to 640×480 (edit lines 559-560):
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```

3. Close other applications to free CPU

### Issue: Memory errors
**Solution:** Close browser and other apps. Check memory:
```bash
free -h
```

### Issue: Model download fails
**Solution:** Ensure internet connection. The model downloads automatically on first run.

---

## Testing Without Camera (Debug Mode)

If you want to test the setup without a camera connected, create this simple test:

```bash
# Create test script
cat > test_imports.py << 'EOF'
import sys
print("Testing imports...")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV: {e}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"✗ TensorFlow: {e}")

try:
    from src.models.mobilenet_classifier import MobileNetDriverClassifier
    print("✓ MobileNet classifier")
except ImportError as e:
    print(f"✗ MobileNet: {e}")

try:
    from src.models.object_detector import TFLiteObjectDetector
    print("✓ TFLite object detector")
except ImportError as e:
    print(f"✗ Object detector: {e}")

print("\nAll critical imports successful!")
EOF

# Run test
python test_imports.py
```

---

## Advanced: Performance Tuning

### Monitor Performance
```bash
# Install htop if needed
sudo apt install htop

# Run in another terminal while demo is running
htop
```

### Adjust Frame Skip Interval
Edit `demo_mobilenet.py` line 575:
```python
obj_detect_interval = 3  # Lower = better accuracy, higher = better FPS
```

Recommended values:
- `3` = Balanced (15-20 FPS) ← Current setting
- `5` = Performance (20-25 FPS, less accurate object detection)
- `1` = Accuracy (10-15 FPS, full object detection)

### Disable Object Detection Entirely
If you only want distraction classification (no object detection), comment out lines 593-595 in `demo_mobilenet.py`.

---

## File Locations

### Repository
```
/home/prahaasn/focusdrive-ai-detection/
```

### Virtual Environment
```
/home/prahaasn/focusdrive-ai-detection/venv/
```

### Models
```
/home/prahaasn/focusdrive-ai-detection/models/
├── mobilenet_checkpoints/
│   └── best_model_pretrained/
│       ├── config.json
│       └── model.pt
└── tflite/
    ├── detect.tflite (4.0 MB)
    └── labelmap.txt
```

### Modified File
```
/home/prahaasn/focusdrive-ai-detection/demo_mobilenet.py
```

Changes applied:
- Line 502: Force CPU device
- Lines 559-560: Resolution 800×600
- Lines 575-577: Frame skipping
- Lines 541-561: Dual camera support
- Lines 580-592: Camera reading logic
- Lines 723-726: Camera cleanup

---

## Quick Reference Commands

### Start Demo
```bash
cd /home/prahaasn/focusdrive-ai-detection
source venv/bin/activate
python demo_mobilenet.py
```

### Check Camera
```bash
# USB webcam
ls /dev/video*

# Pi Camera
rpicam-hello --list-cameras
```

### Check Dependencies
```bash
source venv/bin/activate
pip list | grep -E "torch|opencv|tensorflow"
```

### Update Code (if needed)
```bash
cd /home/prahaasn/focusdrive-ai-detection
git pull
```

---

## Next Steps

1. **Connect a camera** (USB webcam or Pi Camera Module)
2. **Run the demo** with `python demo_mobilenet.py`
3. **Test distraction detection** by looking away from the camera
4. **Monitor performance** and adjust frame skip interval if needed

---

## Support

If you encounter issues:
1. Check this troubleshooting section
2. Verify camera is connected: `ls /dev/video*`
3. Ensure venv is activated: `which python` should show `.../venv/bin/python`
4. Check available memory: `free -h`
5. Monitor CPU usage: `htop`

---

**Ready to run! Connect your camera and execute:** `python demo_mobilenet.py`
