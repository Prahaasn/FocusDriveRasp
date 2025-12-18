#!/usr/bin/env python3
"""
Quick setup verification script for Raspberry Pi 5.
Tests all critical dependencies and model files.
"""

import sys
from pathlib import Path

print("=" * 80)
print("FocusDrive Setup Verification for Raspberry Pi 5")
print("=" * 80)

# Test 1: Core Python imports
print("\n[1/7] Testing core imports...")
errors = []

try:
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"  ✗ OpenCV failed: {e}")
    errors.append("OpenCV")

try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"      Device: {torch.device('cpu')}")
except ImportError as e:
    print(f"  ✗ PyTorch failed: {e}")
    errors.append("PyTorch")

try:
    import torchvision
    print(f"  ✓ TorchVision {torchvision.__version__}")
except ImportError as e:
    print(f"  ✗ TorchVision failed: {e}")
    errors.append("TorchVision")

try:
    import tensorflow as tf
    print(f"  ✓ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"  ✗ TensorFlow failed: {e}")
    errors.append("TensorFlow")

try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"  ✗ NumPy failed: {e}")
    errors.append("NumPy")

# Test 2: Optional dependencies
print("\n[2/7] Testing optional dependencies...")
try:
    from picamera2 import Picamera2
    print("  ✓ Picamera2 (for Raspberry Pi Camera Module)")
except ImportError:
    print("  ⚠ Picamera2 not available (USB webcam will be used)")

# Test 3: Custom modules
print("\n[3/7] Testing custom modules...")
try:
    from src.models.mobilenet_classifier import MobileNetDriverClassifier
    print("  ✓ MobileNet classifier")
except ImportError as e:
    print(f"  ✗ MobileNet classifier failed: {e}")
    errors.append("MobileNet")

try:
    from src.models.object_detector import TFLiteObjectDetector
    print("  ✓ TFLite object detector")
except ImportError as e:
    print(f"  ✗ TFLite object detector failed: {e}")
    errors.append("ObjectDetector")

try:
    from src.logic.distraction_reasoning import DistractionReasoning
    print("  ✓ Distraction reasoning engine")
except ImportError as e:
    print(f"  ✗ Distraction reasoning failed: {e}")
    errors.append("DistractionReasoning")

try:
    from src.utils.speed_monitor import SpeedMonitor
    print("  ✓ Speed monitor")
except ImportError as e:
    print(f"  ✗ Speed monitor failed: {e}")
    errors.append("SpeedMonitor")

# Test 4: Model files
print("\n[4/7] Checking model files...")
project_root = Path(__file__).parent

mobilenet_path = project_root / "models" / "mobilenet_checkpoints" / "best_model_pretrained"
if mobilenet_path.exists():
    config_file = mobilenet_path / "config.json"
    model_file = mobilenet_path / "model.pt"
    if config_file.exists() and model_file.exists():
        print(f"  ✓ MobileNet checkpoint found")
        print(f"      Config: {config_file}")
        print(f"      Model: {model_file}")
    else:
        print(f"  ✗ Checkpoint files incomplete")
        errors.append("MobileNetCheckpoint")
else:
    print(f"  ✗ MobileNet checkpoint not found at {mobilenet_path}")
    errors.append("MobileNetCheckpoint")

tflite_model = project_root / "models" / "tflite" / "detect.tflite"
tflite_labels = project_root / "models" / "tflite" / "labelmap.txt"
if tflite_model.exists() and tflite_labels.exists():
    size_mb = tflite_model.stat().st_size / (1024 * 1024)
    print(f"  ✓ TFLite model found ({size_mb:.1f} MB)")
else:
    print(f"  ✗ TFLite model not found")
    errors.append("TFLiteModel")

# Test 5: Camera detection
print("\n[5/7] Checking camera availability...")
camera_found = False

# Check USB webcams
try:
    import os
    video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
    if video_devices:
        print(f"  ✓ USB camera(s) detected: {', '.join(video_devices)}")
        camera_found = True
except:
    pass

# Check Pi Camera
try:
    import subprocess
    result = subprocess.run(['rpicam-hello', '--list-cameras'],
                          capture_output=True, text=True, timeout=2)
    if 'No cameras available' not in result.stdout:
        print("  ✓ Raspberry Pi Camera Module detected")
        camera_found = True
except:
    pass

if not camera_found:
    print("  ⚠ No camera detected - please connect a camera before running demo")
    print("      USB webcam: Will appear as /dev/video0")
    print("      Pi Camera: Enable in raspi-config if needed")

# Test 6: System resources
print("\n[6/7] Checking system resources...")
try:
    import subprocess
    # Check available memory
    result = subprocess.run(['free', '-h'], capture_output=True, text=True)
    lines = result.stdout.split('\n')
    for line in lines:
        if 'Mem:' in line:
            parts = line.split()
            print(f"  ✓ Memory: {parts[1]} total, {parts[6]} available")
except:
    print("  ⚠ Could not check memory")

# Check CPU
try:
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()
        if 'Raspberry Pi 5' in cpuinfo or 'BCM2712' in cpuinfo:
            print("  ✓ Raspberry Pi 5 detected")
        else:
            print("  ⚠ Platform may not be Raspberry Pi 5")
except:
    print("  ⚠ Could not detect platform")

# Test 7: Demo file
print("\n[7/7] Checking demo file...")
demo_file = project_root / "demo_mobilenet.py"
if demo_file.exists():
    print(f"  ✓ Demo file found: {demo_file}")
    # Check for our modifications
    with open(demo_file, 'r') as f:
        content = f.read()
        if 'device="cpu"' in content:
            print("  ✓ CPU optimization applied")
        if 'obj_detect_interval' in content:
            print("  ✓ Frame skipping optimization applied")
        if 'using_picamera' in content:
            print("  ✓ Dual camera support added")
        if '800' in content and '600' in content:
            print("  ✓ Resolution optimized (800×600)")
else:
    print(f"  ✗ Demo file not found")
    errors.append("DemoFile")

# Summary
print("\n" + "=" * 80)
if errors:
    print("SETUP INCOMPLETE - Errors found:")
    for error in errors:
        print(f"  ✗ {error}")
    print("\nPlease fix the errors above before running the demo.")
    sys.exit(1)
else:
    print("✓ SETUP VERIFICATION COMPLETE!")
    print("\nAll dependencies installed and configured correctly.")
    print("\nNext steps:")
    if not camera_found:
        print("  1. Connect a camera (USB webcam or Pi Camera Module)")
        print("  2. Run: python demo_mobilenet.py")
    else:
        print("  1. Run: python demo_mobilenet.py")
    print("\nPress 'q' in the demo window to quit.")
print("=" * 80)
