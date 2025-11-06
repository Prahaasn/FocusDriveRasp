"""
Real-time driver distraction detection demo using webcam.

Usage:
    python demo.py --model models/checkpoints/best_model.pt
    python demo.py --model models/checkpoints/best_model.pt --alert-sound

Features:
- Real-time webcam capture
- Driver distraction classification
- Visual alerts for distraction detection
- FPS counter and latency measurement
- Optional audio alerts
"""

import argparse
import cv2
import torch
import time
from PIL import Image
import numpy as np
from pathlib import Path
import sys
from collections import deque
import platform
import subprocess

sys.path.append(str(Path(__file__).parent))

from src.models.lfm_classifier import LFMDriverClassifier
from transformers import AutoProcessor


# Alert colors (BGR format for OpenCV)
COLOR_ATTENTIVE = (0, 255, 0)  # Green
COLOR_DISTRACTED = (0, 0, 255)  # Red
COLOR_WARNING = (0, 165, 255)  # Orange
COLOR_TEXT = (255, 255, 255)  # White
COLOR_BG = (0, 0, 0)  # Black


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='FocusDrive real-time demo'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device index'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.7,
        help='Confidence threshold for alerts'
    )
    parser.add_argument(
        '--alert-sound',
        action='store_true',
        help='Enable audio alerts (requires pygame)'
    )
    parser.add_argument(
        '--save-video',
        type=str,
        default=None,
        help='Path to save output video (optional)'
    )
    parser.add_argument(
        '--fps-target',
        type=int,
        default=5,
        help='Target FPS for inference (lower = less resource intensive)'
    )

    return parser.parse_args()


class DriverMonitor:
    """Real-time driver distraction monitor."""

    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        confidence_threshold: float = 0.7,
        enable_audio: bool = False
    ):
        """
        Initialize driver monitor.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on
            confidence_threshold: Confidence threshold for alerts
            enable_audio: Enable audio alerts
        """
        self.confidence_threshold = confidence_threshold
        self.enable_audio = enable_audio

        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"=�  Running on: {self.device}")

        # Load model
        print(f"=� Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Initialize model (you'll need to adjust this based on how you saved it)
        self.model = LFMDriverClassifier(
            model_name="LiquidAI/LFM2-VL-1.6B",
            num_classes=2,
            device=self.device
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained("LiquidAI/LFM2-VL-1.6B")

        # Class names
        self.class_names = ['Attentive', 'Distracted']

        # FPS tracking
        self.fps_history = []

        # Alert tracking
        self.distraction_count = 0
        self.total_frames = 0

        # Sustained distraction alert system (3 seconds at ~30 FPS)
        self.distraction_history = deque(maxlen=90)
        self.last_alert_time = 0
        self.alert_cooldown = 5.0  # seconds between alerts
        self.distraction_threshold = 0.70  # 70% confidence
        self.sustained_duration = 3.0  # 3 seconds

        # Audio alert (optional)
        if self.enable_audio:
            try:
                import pygame
                pygame.mixer.init()
                self.alert_sound = None  # You can load a .wav file here
                print("=
 Audio alerts enabled")
            except ImportError:
                print("�  pygame not installed, audio alerts disabled")
                self.enable_audio = False

        print(" Model loaded successfully\n")

    def preprocess_frame(self, frame: np.ndarray) -> Image.Image:
        """
        Preprocess camera frame.

        Args:
            frame: OpenCV frame (BGR format)

        Returns:
            PIL Image
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)

        return pil_image

    def predict(self, image: Image.Image) -> tuple:
        """
        Make prediction on image.

        Args:
            image: PIL Image

        Returns:
            Tuple of (class_id, class_name, confidence, probabilities)
        """
        start_time = time.time()

        # Prepare input
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Classify the driver's state."},
                ],
            },
        ]

        inputs = self.processor([conversation], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=1)[0]
        class_id = torch.argmax(probabilities).item()
        confidence = probabilities[class_id].item()
        class_name = self.class_names[class_id]

        inference_time = (time.time() - start_time) * 1000  # ms

        return class_id, class_name, confidence, probabilities.cpu().numpy(), inference_time

    def check_sustained_distraction(self, class_name: str, confidence: float) -> bool:
        """
        Check if driver has been distracted for sustained period.

        Args:
            class_name: Predicted class name
            confidence: Prediction confidence

        Returns:
            True if alert should be triggered
        """
        current_time = time.time()

        # Add to history (1 if distracted above threshold, 0 otherwise)
        is_distracted = (class_name == 'Distracted' and
                        confidence >= self.distraction_threshold)
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
        if not self.enable_audio or not self.audio_available:
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

    def draw_overlay(
        self,
        frame: np.ndarray,
        class_name: str,
        confidence: float,
        probabilities: np.ndarray,
        fps: float,
        inference_time: float
    ) -> np.ndarray:
        """
        Draw information overlay on frame.

        Args:
            frame: OpenCV frame
            class_name: Predicted class name
            confidence: Prediction confidence
            probabilities: Class probabilities
            fps: Current FPS
            inference_time: Inference time in ms

        Returns:
            Frame with overlay
        """
        h, w = frame.shape[:2]

        # Determine color based on prediction
        is_distracted = class_name == 'Distracted'
        alert_color = COLOR_DISTRACTED if is_distracted else COLOR_ATTENTIVE

        # Draw status bar at top
        bar_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), alert_color, -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # Draw status text
        status_text = f"STATUS: {class_name.upper()}"
        cv2.putText(
            frame,
            status_text,
            (20, 50),
            cv2.FONT_HERSHEY_BOLD,
            1.5,
            COLOR_TEXT if is_distracted else alert_color,
            3
        )

        # Draw confidence
        conf_text = f"Confidence: {confidence*100:.1f}%"
        cv2.putText(
            frame,
            conf_text,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_TEXT,
            2
        )

        # Draw probabilities
        y_offset = 140
        for i, (class_name_i, prob) in enumerate(zip(self.class_names, probabilities)):
            prob_text = f"{class_name_i}: {prob*100:.1f}%"
            color = COLOR_ATTENTIVE if i == 0 else COLOR_DISTRACTED

            cv2.putText(
                frame,
                prob_text,
                (20, y_offset + i*30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Draw performance metrics (bottom right)
        perf_y = h - 80
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (w - 200, perf_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TEXT,
            2
        )

        cv2.putText(
            frame,
            f"Latency: {inference_time:.1f}ms",
            (w - 200, perf_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TEXT,
            2
        )

        # Draw distraction counter (bottom left)
        distraction_pct = (self.distraction_count / max(self.total_frames, 1)) * 100
        cv2.putText(
            frame,
            f"Distraction: {distraction_pct:.1f}%",
            (20, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_WARNING,
            2
        )

        cv2.putText(
            frame,
            f"({self.distraction_count}/{self.total_frames} frames)",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_TEXT,
            1
        )

        # Draw alert border if distracted
        if is_distracted and confidence > self.confidence_threshold:
            cv2.rectangle(frame, (0, 0), (w-1, h-1), COLOR_DISTRACTED, 10)

            # Draw alert message
            alert_msg = "� DISTRACTION DETECTED �"
            text_size = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_BOLD, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2

            # Background for text
            cv2.rectangle(
                frame,
                (text_x - 10, text_y - 40),
                (text_x + text_size[0] + 10, text_y + 10),
                COLOR_BG,
                -1
            )

            cv2.putText(
                frame,
                alert_msg,
                (text_x, text_y),
                cv2.FONT_HERSHEY_BOLD,
                1.2,
                COLOR_DISTRACTED,
                3
            )

        # Draw sustained distraction progress bar
        if len(self.distraction_history) > 0:
            distracted_ratio = sum(self.distraction_history) / len(self.distraction_history)
            bar_width = 300
            bar_height = 20
            bar_x = w - bar_width - 20
            bar_y = h - 80

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

        return frame

    def run(
        self,
        camera_index: int = 0,
        fps_target: int = 5,
        save_video: str = None
    ):
        """
        Run real-time monitoring.

        Args:
            camera_index: Camera device index
            fps_target: Target FPS for inference
            save_video: Path to save output video (optional)
        """
        print("=" * 60)
        print("FocusDrive - Real-time Driver Monitoring")
        print("=" * 60)
        print(f"Camera: {camera_index}")
        print(f"Target FPS: {fps_target}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("\nPress 'q' to quit\n")

        # Open camera
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"L Error: Could not open camera {camera_index}")
            return

        # Get camera properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        camera_fps = int(cap.get(cv2.CAP_PROP_FPS))

        print(f" Camera opened: {frame_width}x{frame_height} @ {camera_fps}fps\n")

        # Video writer (optional)
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                save_video,
                fourcc,
                fps_target,
                (frame_width, frame_height)
            )
            print(f"=� Recording to: {save_video}\n")

        # Calculate frame skip for target FPS
        frame_interval = max(1, camera_fps // fps_target)

        frame_count = 0
        last_time = time.time()

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("L Error reading frame")
                    break

                frame_count += 1

                # Skip frames to match target FPS
                if frame_count % frame_interval != 0:
                    continue

                # Preprocess
                pil_image = self.preprocess_frame(frame)

                # Predict
                class_id, class_name, confidence, probabilities, inference_time = \
                    self.predict(pil_image)

                # Update stats
                self.total_frames += 1
                if class_name == 'Distracted':
                    self.distraction_count += 1

                # Check for sustained distraction alert
                alert_triggered = self.check_sustained_distraction(class_name, confidence)
                if alert_triggered:
                    self.play_alert_sound()
                    print("🚨 ALERT: Sustained distraction detected!")

                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - last_time)
                last_time = current_time
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = sum(self.fps_history) / len(self.fps_history)

                # Draw overlay
                display_frame = self.draw_overlay(
                    frame,
                    class_name,
                    confidence,
                    probabilities,
                    avg_fps,
                    inference_time
                )

                # Show frame
                cv2.imshow('FocusDrive - Driver Monitoring', display_frame)

                # Save frame if recording
                if video_writer:
                    video_writer.write(display_frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n�  Quit requested")
                    break

        except KeyboardInterrupt:
            print("\n�  Interrupted by user")

        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()

            # Print summary
            print("\n" + "=" * 60)
            print("Session Summary")
            print("=" * 60)
            print(f"Total frames processed: {self.total_frames}")
            print(f"Distracted frames: {self.distraction_count}")
            distraction_pct = (self.distraction_count / max(self.total_frames, 1)) * 100
            print(f"Distraction rate: {distraction_pct:.2f}%")
            if self.fps_history:
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                print(f"Average FPS: {avg_fps:.2f}")
            print("=" * 60)


def main():
    """Main function."""
    args = parse_args()

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"L Error: Model not found at {model_path}")
        print("\nPlease train a model first:")
        print("  python train.py --epochs 30")
        sys.exit(1)

    # Initialize monitor
    monitor = DriverMonitor(
        model_path=str(model_path),
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        enable_audio=args.alert_sound
    )

    # Run monitoring
    monitor.run(
        camera_index=args.camera,
        fps_target=args.fps_target,
        save_video=args.save_video
    )


if __name__ == "__main__":
    main()
