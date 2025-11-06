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
from datetime import datetime
import torchvision.transforms as T
from collections import deque
import platform

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.mobilenet_classifier import MobileNetDriverClassifier
from src.utils.speed_monitor import SpeedMonitor

# Try to import audio library
try:
    if platform.system() == 'Darwin':  # macOS
        import subprocess
        AUDIO_AVAILABLE = True
    else:
        from playsound import playsound
        AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  Audio library not available. Install with: pip install playsound")


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

        # Define transforms (same as training)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Statistics
        self.frame_count = 0
        self.distracted_count = 0
        self.attentive_count = 0
        self.fps = 0
        self.inference_time = 0

        # Alert system
        self.distraction_history = deque(maxlen=90)  # 3 seconds at 30 FPS
        self.alert_triggered = False
        self.last_alert_time = 0
        self.alert_cooldown = 5.0  # seconds between alerts
        self.distraction_threshold = 0.70  # 70% confidence
        self.sustained_duration = 3.0  # 3 seconds

        print("✓ Detector initialized!")
        print(f"  Alert settings: {self.sustained_duration}s sustained distraction at {self.distraction_threshold*100:.0f}% confidence")

    def preprocess_frame(self, frame):
        """
        Preprocess webcam frame.

        Args:
            frame: OpenCV frame (BGR)

        Returns:
            Preprocessed tensor [1, 3, 224, 224]
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transforms
        tensor = self.transform(rgb_frame)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor

    def predict(self, frame):
        """
        Run inference on frame.

        Args:
            frame: OpenCV frame (BGR)

        Returns:
            Dictionary with prediction, confidence, and class name
        """
        # Preprocess
        tensor = self.preprocess_frame(frame).to(self.device)

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

    def check_sustained_distraction(self, result):
        """
        Check if driver has been distracted for sustained period.

        Args:
            result: Current prediction result

        Returns:
            True if alert should be triggered
        """
        current_time = time.time()

        # Add to history (1 if distracted above threshold, 0 otherwise)
        is_distracted = (result['class_name'] == 'Distracted' and
                        result['confidence'] >= self.distraction_threshold)
        self.distraction_history.append(1 if is_distracted else 0)

        # Need enough frames for sustained duration
        if len(self.distraction_history) < int(self.sustained_duration * 30):
            return False

        # Check if cooldown period has passed
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False

        # Calculate percentage of recent frames that are distracted
        distracted_frames = sum(self.distraction_history)
        total_frames = len(self.distraction_history)
        distracted_ratio = distracted_frames / total_frames

        # Trigger if >= 80% of recent frames show distraction
        if distracted_ratio >= 0.8:
            self.last_alert_time = current_time
            return True

        return False

    def play_alert_sound(self):
        """Play buzzer alert sound."""
        if not AUDIO_AVAILABLE:
            print("🔔 ALERT! (audio not available)")
            return

        try:
            if platform.system() == 'Darwin':  # macOS
                # Use afplay with system beep sound
                subprocess.Popen(['afplay', '/System/Library/Sounds/Funk.aiff'],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            else:
                # Use system beep on other platforms
                print('\a')  # Terminal bell
        except Exception as e:
            print(f"⚠️  Could not play sound: {e}")
            print("🔔 ALERT!")

    def draw_overlay(self, frame, result, alert_active=False, speed_status=None):
        """
        Draw overlay on frame.

        Args:
            frame: OpenCV frame
            result: Prediction result (None if detection inactive)
            alert_active: Whether alert is currently triggered
            speed_status: Speed monitor status dict

        Returns:
            Frame with overlay
        """
        h, w = frame.shape[:2]

        # If detection is inactive (speed too low), show gray overlay
        if result is None:
            color = (128, 128, 128)  # Gray
            status = "DETECTION INACTIVE"
            self.frame_count += 1
        # Determine color based on prediction and alert
        elif alert_active:
            color = (0, 0, 255)  # Red - flashing alert
            status = "⚠️ ALERT! DISTRACTED! ⚠️"
            self.frame_count += 1
        elif result['class_name'] == 'Attentive':
            color = (0, 255, 0)  # Green
            status = "ATTENTIVE"
            self.attentive_count += 1
            self.frame_count += 1
        else:
            color = (0, 165, 255)  # Orange
            status = "DISTRACTED!"
            self.distracted_count += 1
            self.frame_count += 1

        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), color, -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # Draw status text
        cv2.putText(
            frame,
            status,
            (20, 60),
            cv2.FONT_HERSHEY_DUPLEX,
            2.0,
            (255, 255, 255),
            3
        )

        # Draw confidence (only if detection is active)
        if result is not None:
            confidence_text = f"Confidence: {result['confidence']*100:.1f}%"
            cv2.putText(
                frame,
                confidence_text,
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # Draw speed information (if available)
        if speed_status is not None:
            speed = speed_status['speed']
            is_active = speed_status['is_active']

            # Speed display
            speed_text = f"Speed: {speed:.1f} mph"
            speed_color = (0, 255, 0) if speed >= 15.0 else (100, 100, 100)
            cv2.putText(
                frame,
                speed_text,
                (w - 250, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                speed_color,
                2
            )

            # Activation status
            if is_active:
                status_text = "✓ ACTIVE"
                status_color = (0, 255, 0)
            else:
                seconds_until = speed_status['seconds_until_active']
                if speed >= 15.0:
                    status_text = f"Activating in {seconds_until:.1f}s"
                    status_color = (0, 165, 255)
                else:
                    status_text = "Speed too low"
                    status_color = (100, 100, 100)

            cv2.putText(
                frame,
                status_text,
                (w - 250, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2
            )

        # Draw probabilities (only if detection is active)
        if result is not None:
            prob_attentive = result['probabilities'][0] * 100
            prob_distracted = result['probabilities'][1] * 100
            prob_text = f"Attentive: {prob_attentive:.1f}%  |  Distracted: {prob_distracted:.1f}%"
            cv2.putText(
                frame,
                prob_text,
                (20, h - 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Draw FPS and inference time
        fps_text = f"FPS: {self.fps:.1f}  |  Inference: {self.inference_time:.1f}ms"
        cv2.putText(
            frame,
            fps_text,
            (20, h - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Draw statistics
        distraction_rate = (self.distracted_count / max(self.frame_count, 1)) * 100
        stats_text = f"Frames: {self.frame_count}  |  Distraction Rate: {distraction_rate:.1f}%"
        cv2.putText(
            frame,
            stats_text,
            (20, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Draw alert indicator if active
        if alert_active:
            alert_text = "🔔 BUZZER ALERT ACTIVE!"
            cv2.putText(
                frame,
                alert_text,
                (w - 400, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Draw sustained distraction progress bar
        if len(self.distraction_history) > 0:
            distracted_ratio = sum(self.distraction_history) / len(self.distraction_history)
            bar_width = 300
            bar_height = 20
            bar_x = w - bar_width - 20
            bar_y = h - 40

            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)

            # Filled portion
            fill_width = int(bar_width * distracted_ratio)
            bar_color = (0, 0, 255) if distracted_ratio >= 0.8 else (0, 165, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), bar_color, -1)

            # Border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

            # Label
            bar_label = f"Alert Progress: {distracted_ratio*100:.0f}%"
            cv2.putText(frame, bar_label, (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw controls
        controls = "Controls: Q=Quit  S=Screenshot  R=Record"
        cv2.putText(
            frame,
            controls,
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

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

    # Initialize detector
    detector = DistractionDetector(model_path, device="auto")

    # Initialize speed monitor
    print("\nInitializing speed monitor...")
    speed_monitor = SpeedMonitor(
        method='simulated',
        speed_threshold=15.0,
        activation_duration=10.0
    )
    print("✓ Speed monitor initialized!")
    print(f"  Activation: Speed > {speed_monitor.speed_threshold} mph for > {speed_monitor.activation_duration}s")

    # Open webcam
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        sys.exit(1)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("✓ Webcam opened!")
    print("\nStarting detection...")
    print("Press 'q' to quit, 's' to save screenshot, 'r' to record")

    # Recording state
    is_recording = False
    video_writer = None

    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to read frame")
                break

            # Update speed monitor
            speed_status = speed_monitor.update()

            # Only run detection if speed conditions are met
            if speed_monitor.should_activate_detection():
                # Run detection
                result = detector.predict(frame)

                # Check for sustained distraction alert
                alert_triggered = detector.check_sustained_distraction(result)
                if alert_triggered:
                    detector.play_alert_sound()
                    print("🚨 ALERT: Sustained distraction detected!")
            else:
                # Detection inactive - create placeholder result
                result = None
                alert_triggered = False

            # Draw overlay
            frame = detector.draw_overlay(frame, result, alert_active=alert_triggered, speed_status=speed_status)

            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:
                elapsed = time.time() - fps_start_time
                detector.fps = fps_frame_count / elapsed
                fps_start_time = time.time()
                fps_frame_count = 0

            # Write frame if recording
            if is_recording and video_writer is not None:
                video_writer.write(frame)

            # Show frame
            cv2.imshow("FocusDrive - Distraction Detection", frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break

            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = project_root / f"screenshot_{timestamp}.png"
                cv2.imwrite(str(screenshot_path), frame)
                print(f"✓ Screenshot saved: {screenshot_path}")

            elif key == ord('r'):
                # Toggle recording
                if not is_recording:
                    # Start recording
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = project_root / f"recording_{timestamp}.mp4"

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 30
                    frame_size = (frame.shape[1], frame.shape[0])
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        fourcc,
                        fps,
                        frame_size
                    )

                    is_recording = True
                    print(f"🔴 Recording started: {video_path}")
                else:
                    # Stop recording
                    is_recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    print("⏹️  Recording stopped")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup
        print("\nCleaning up...")
        if video_writer is not None:
            video_writer.release()
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
