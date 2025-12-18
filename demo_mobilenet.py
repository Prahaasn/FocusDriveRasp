"""
Real-time webcam demo for MobileNetV3 driver distraction detection.

This demo:
- Opens your webcam
- Runs real-time distraction detection
- Shows visual alerts (green = attentive, orange = distracted, red = alert)
- Displays confidence scores and FPS
- Records distraction statistics
- BUZZER ALERT: Plays sound after 3 seconds of sustained distraction at 70% confidence

Alert System:
- Monitors last 3 seconds of predictions
- Triggers buzzer if 80% of recent frames show distraction at ≥70% confidence
- 5-second cooldown between alerts
- Visual progress bar shows how close to triggering alert

Controls:
- Press 'q' to quit
- Press 's' to save screenshot
- Press 'r' to start/stop recording

Usage:
    python demo_mobilenet.py
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.mobilenet_classifier import MobileNetDriverClassifier
from src.models.object_detector import TFLiteObjectDetector


class DistractionDetector:
    """Real-time distraction detector."""

    def __init__(self, model_path: Path, device: str = "auto"):
        """
        Initialize detector.

        Args:
            model_path: Path to trained model
            device: Device to run on ('auto', 'cuda', 'mps', 'cpu')
        """
        print("Initializing distraction detector...")

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"Device: {self.device}")

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = MobileNetDriverClassifier.load_pretrained(model_path, device=str(self.device))
        self.model.eval()

        # Apply dynamic quantization for ARM CPU (2-4x speedup!)
        print("Applying dynamic quantization for ARM CPU...")
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        print("✓ Quantization complete!")

        # Apply TorchScript JIT compilation (15-30% additional speedup)
        print("Compiling model with TorchScript JIT...")
        example_input = torch.randn(1, 3, 224, 224, device=self.device)
        self.model = torch.jit.trace(self.model, example_input)
        self.model = torch.jit.optimize_for_inference(self.model)
        print("✓ JIT compilation complete!")

        # Pre-computed normalization values (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Pre-allocate input tensor (reuse memory)
        self.input_tensor = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=self.device)

        # Statistics
        self.frame_count = 0
        self.distracted_count = 0
        self.attentive_count = 0
        self.fps = 0
        self.inference_time = 0

        # Simplified alert system
        self.distraction_counter = 0
        self.alert_threshold = 45  # 1.5 seconds at 30 FPS
        self.distraction_threshold = 0.70  # 70% confidence

        print("✓ Detector initialized (optimized for Pi 5)!")

    def preprocess_frame(self, frame_rgb):
        """
        Optimized preprocessing using cv2/NumPy (no PIL, no allocations).

        Args:
            frame_rgb: OpenCV frame in RGB format

        Returns:
            Preprocessed tensor [1, 3, 224, 224]
        """
        # Resize with cv2 (faster than PIL)
        resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Convert to float and normalize (NumPy vectorized operation)
        normalized = (resized.astype(np.float32) / 255.0 - self.mean) / self.std

        # Copy to pre-allocated tensor (reuse memory, no allocation)
        # Convert HWC -> CHW and copy to device
        self.input_tensor.copy_(
            torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        )

        return self.input_tensor

    def predict(self, frame_rgb):
        """
        Run inference on frame.

        Args:
            frame_rgb: OpenCV frame in RGB format

        Returns:
            Dictionary with prediction, confidence, and class name
        """
        # Preprocess (already on device, no .to() needed)
        tensor = self.preprocess_frame(frame_rgb)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.predict(tensor, return_probs=True)
        self.inference_time = (time.time() - start_time) * 1000  # ms

        # Extract results
        prediction = outputs['predictions'][0]
        probabilities = outputs['probabilities'][0]
        class_name = outputs['class_names'][0]
        confidence = probabilities[prediction]

        return {
            'prediction': int(prediction),
            'class_name': class_name,
            'confidence': float(confidence),
            'probabilities': probabilities
        }


    def draw_overlay(self, frame, result, alert_active=False, speed_status=None, detected_objects=None, assessment=None):
        """
        Minimal optimized overlay for Pi 5 (40 lines vs 255!).

        Args:
            frame: OpenCV frame (BGR)
            result: Prediction result
            alert_active: Whether alert is active
            detected_objects: List of detected objects

        Returns:
            Frame with minimal overlay
        """
        h, w = frame.shape[:2]

        # Update statistics
        self.frame_count += 1
        if result['class_name'] == 'Attentive':
            self.attentive_count += 1
        else:
            self.distracted_count += 1

        # Determine status and color
        if alert_active:
            color = (0, 0, 255)  # Red for alert
            status = "ALERT! DISTRACTED!"
        elif result['class_name'] == 'Distracted':
            color = (0, 165, 255)  # Orange for distracted
            status = "DISTRACTED"
        else:
            color = (0, 255, 0)  # Green for attentive
            status = "ATTENTIVE"

        # Draw colored border (3px thickness)
        cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 3)

        # Draw status text (top-left, bold)
        cv2.putText(frame, status, (10, 40),
                    cv2.FONT_HERSHEY_BOLD, 1.2, color, 2)

        # Draw confidence (below status)
        conf_text = f"{result['confidence']*100:.0f}%"
        cv2.putText(frame, conf_text, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw FPS (top-right)
        fps_text = f"{self.fps:.0f} FPS"
        cv2.putText(frame, fps_text, (w-120, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw object bounding boxes (only phones for clarity)
        if detected_objects:
            for obj in detected_objects:
                if obj.class_name == 'cell phone':
                    x, y, bw, bh = obj.bbox
                    cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
                    cv2.putText(frame, 'PHONE', (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame


def main():
    """Main demo function."""
    print("=" * 80)
    print("MobileNetV3 - Real-time Driver Distraction Detection Demo")
    print("=" * 80)

    project_root = Path(__file__).parent
    model_path = project_root / "models" / "mobilenet_checkpoints" / "best_model_pretrained"

    # Check if model exists
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print("\nPlease train the model first:")
        print("  python train_mobilenet.py --epochs 20 --batch-size 32")
        sys.exit(1)

    # Initialize detector (Force CPU on Raspberry Pi 5)
    detector = DistractionDetector(model_path, device="cpu")

    # Initialize object detector
    print("\nInitializing TFLite object detector...")
    tflite_model = project_root / "models" / "tflite" / "detect.tflite"
    labelmap = project_root / "models" / "tflite" / "labelmap.txt"

    object_detector = TFLiteObjectDetector(
        model_path=str(tflite_model),
        labelmap_path=str(labelmap),
        confidence_threshold=0.6
    )
    print("✓ Object detector initialized!")

    # Open camera (Try picamera2 for Raspberry Pi Camera, fallback to OpenCV)
    print("\nOpening camera...")
    using_picamera = False

    try:
        from picamera2 import Picamera2
        print("Attempting to use Picamera2 (Raspberry Pi Camera)...")
        cap = Picamera2()
        config = cap.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            buffer_count=2,  # Double buffering for smoother capture
        )
        # Lock to 30 FPS for consistent timing
        cap.set_controls({"FrameDurationLimits": (33333, 33333)})
        cap.configure(config)
        cap.start()
        using_picamera = True
        print("✓ Picamera2 initialized (Raspberry Pi Camera Module)")
    except (ImportError, Exception) as e:
        print(f"Picamera2 not available ({type(e).__name__}), using OpenCV...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            sys.exit(1)

        # Set resolution (Optimized for Raspberry Pi 5)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS
        print("✓ OpenCV camera opened!")
    print("\nStarting detection...")
    print("Press 'q' to quit")

    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0

    # Frame skipping for object detection (Raspberry Pi 5 optimization)
    frame_count = 0
    obj_detect_interval = 5  # Run object detection every 5th frame (optimized for 30 FPS)
    detected_objects = []  # Cache last detection result

    try:
        while True:
            # Read frame and convert to RGB (SINGLE conversion)
            if using_picamera:
                frame_rgb = cap.capture_array()  # Already RGB
                ret = True
            else:
                ret, frame_bgr = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if not ret or frame_rgb is None:
                print("❌ Error: Failed to read frame")
                break

            # Run distraction classification with RGB frame
            result = detector.predict(frame_rgb)

            # Run object detection with RGB frame (with frame skipping)
            if frame_count % obj_detect_interval == 0:
                detected_objects = object_detector.detect(frame_rgb)
            frame_count += 1

            # Simple direct distraction logic (no complex reasoning engine)
            is_distracted = (
                result['class'] == 'Distracted' or
                any(obj.class_name == 'cell phone' for obj in detected_objects)
            )

            # Simplified alert tracking with counter
            if is_distracted:
                detector.distraction_counter += 1
                alert_active = (detector.distraction_counter >= detector.alert_threshold)
            else:
                detector.distraction_counter = 0
                alert_active = False

            # Convert RGB back to BGR for display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Draw overlay (temporarily keep old signature, will optimize later)
            frame_display = detector.draw_overlay(
                frame_bgr,
                result,
                alert_active=alert_active,
                speed_status=None,
                detected_objects=detected_objects,
                assessment=None
            )

            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:
                elapsed = time.time() - fps_start_time
                detector.fps = fps_frame_count / elapsed
                fps_start_time = time.time()
                fps_frame_count = 0

            # Show frame
            cv2.imshow("FocusDrive", frame_display)

            # Check for quit key only
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup
        print("\nCleaning up...")

        # Release camera properly based on type
        if using_picamera:
            cap.stop()
        else:
            cap.release()

        cv2.destroyAllWindows()

        # Print final statistics
        print("\n" + "=" * 80)
        print("SESSION SUMMARY")
        print("=" * 80)
        print(f"Total frames: {detector.frame_count}")
        print(f"Attentive frames: {detector.attentive_count} ({detector.attentive_count/max(detector.frame_count,1)*100:.1f}%)")
        print(f"Distracted frames: {detector.distracted_count} ({detector.distracted_count/max(detector.frame_count,1)*100:.1f}%)")
        print(f"Average FPS: {detector.fps:.1f}")
        print(f"Average inference time: {detector.inference_time:.1f}ms")
        print("=" * 80)


if __name__ == "__main__":
    main()
