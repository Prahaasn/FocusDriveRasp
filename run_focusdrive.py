#!/usr/bin/env python3
"""
FocusDrive - Integrated AI Detection + BLE Broadcasting System

This is the MAIN script that runs everything:
- Real-time AI driver distraction detection
- BLE GATT server broadcasting results to iPhone
- Camera feed with visual alerts

Usage:
    sudo python3 run_focusdrive.py
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import time
import asyncio
from collections import deque

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.mobilenet_classifier import MobileNetDriverClassifier
from src.communication.ble_server import BLEServer
from src.communication.ble_config import STATE_ATTENTIVE, STATE_DISTRACTED, REASON_SAFE_DRIVING, REASON_PHONE

class FocusDriveSystem:
    """Integrated AI detection + BLE broadcasting system."""

    def __init__(self, model_path: Path, device: str = "auto"):
        """Initialize FocusDrive system."""
        print("="*60)
        print("  🚗 FocusDrive - AI Detection + BLE System")
        print("="*60)

        # Initialize AI detector
        print("\n[1/3] Loading AI detection model...")
        self.detector = MobileNetDriverClassifier(model_path, device=device)
        print(f"✓ Model loaded on {self.detector.device}")

        # Initialize BLE server
        print("\n[2/3] Starting BLE GATT server...")
        self.ble_server = None  # Will be initialized in async context

        # Initialize camera
        print("\n[3/3] Opening camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("✓ Camera ready (640x480)")

        # Stats tracking
        self.frame_count = 0
        self.distracted_count = 0
        self.attentive_count = 0
        self.fps = 0
        self.last_time = time.time()

        # Alert system (3-second window)
        self.alert_window = deque(maxlen=90)  # 3 seconds at 30 fps
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds

        print("\n" + "="*60)
        print("✓ FocusDrive System Ready!")
        print("="*60)
        print("\nBLE Device Name: FocusDrive")
        print("Service UUID: 9a1f0000-0000-1000-8000-00805f9b34fb")
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("\nConnect with nRF Connect app on iPhone to see live results!")
        print("="*60 + "\n")

    async def initialize_ble(self):
        """Initialize BLE server (must be called in async context)."""
        self.ble_server = BLEServer()
        await self.ble_server.start()
        print("✓ BLE server started and advertising as 'FocusDrive'\n")

    def predict(self, frame):
        """Run AI prediction on frame."""
        result = self.detector.predict(frame)
        return result

    async def send_to_ble(self, prediction: int, confidence: float):
        """Send prediction to BLE server."""
        if self.ble_server is None:
            return

        # Determine state and reason
        if prediction == 1:  # Distracted
            state = STATE_DISTRACTED
            reason = REASON_PHONE  # TODO: Detect specific distraction type
        else:  # Attentive
            state = STATE_ATTENTIVE
            reason = REASON_SAFE_DRIVING

        # Send via BLE
        await self.ble_server.send_driver_state(state, reason, confidence)

    def draw_ui(self, frame, result):
        """Draw UI overlay on frame."""
        h, w = frame.shape[:2]
        prediction = result['prediction']
        confidence = result['confidence']

        # Choose color based on prediction
        if prediction == 1:  # Distracted
            color = (0, 140, 255)  # Orange
            status = "DISTRACTED"
            self.distracted_count += 1
        else:  # Attentive
            color = (0, 255, 0)  # Green
            status = "ATTENTIVE"
            self.attentive_count += 1

        # Calculate FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = time.time()
            self.fps = 30 / (current_time - self.last_time)
            self.last_time = current_time

        # Draw status banner
        cv2.rectangle(frame, (0, 0), (w, 60), color, -1)
        cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_BOLD, 1.2, (0, 0, 0), 3)

        # Draw confidence
        conf_text = f"Confidence: {confidence:.1%}"
        cv2.putText(frame, conf_text, (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw stats
        total = self.distracted_count + self.attentive_count
        if total > 0:
            dist_pct = (self.distracted_count / total) * 100
            stats_text = f"Distracted: {dist_pct:.1f}% ({self.distracted_count}/{total} frames)"
            cv2.putText(frame, stats_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Alert system
        self.alert_window.append((prediction == 1, confidence >= 0.70))
        if len(self.alert_window) >= 60:  # After 2 seconds
            recent_distracted = sum(1 for dist, conf in self.alert_window if dist and conf)
            alert_percentage = recent_distracted / len(self.alert_window)

            # Progress bar showing how close to alert
            bar_width = 200
            bar_x = w - bar_width - 10
            bar_y = h - 40

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (60, 60, 60), -1)
            fill_width = int(bar_width * min(alert_percentage / 0.8, 1.0))

            bar_color = (0, 0, 255) if alert_percentage >= 0.8 else (0, 140, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + 20), bar_color, -1)

            cv2.putText(frame, "Alert", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Trigger alert
            if alert_percentage >= 0.8:
                current_time = time.time()
                if current_time - self.last_alert_time > self.alert_cooldown:
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
                    cv2.putText(frame, "!!! ALERT !!!", (w//2 - 150, h//2),
                              cv2.FONT_HERSHEY_BOLD, 2, (0, 0, 255), 4)
                    self.last_alert_time = current_time

        return frame

    async def run(self):
        """Main run loop."""
        # Initialize BLE
        await self.initialize_ble()

        # Start BLE notification loop in background
        ble_task = asyncio.create_task(self.ble_loop())

        # Main detection loop
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                # Run AI prediction
                result = self.predict(frame)

                # Draw UI
                frame = self.draw_ui(frame, result)

                # Show frame
                cv2.imshow('FocusDrive - AI Detection', frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nShutting down...")
                    break
                elif key == ord('s'):
                    filename = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")

                # Small delay to allow other async tasks
                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\nShutting down...")

        finally:
            # Cleanup
            ble_task.cancel()
            self.cap.release()
            cv2.destroyAllWindows()
            if self.ble_server:
                self.ble_server.stop()

            # Print final stats
            print("\n" + "="*60)
            print("  Session Summary")
            print("="*60)
            print(f"Total frames: {self.frame_count}")
            print(f"Attentive: {self.attentive_count} ({self.attentive_count/max(self.frame_count,1)*100:.1f}%)")
            print(f"Distracted: {self.distracted_count} ({self.distracted_count/max(self.frame_count,1)*100:.1f}%)")
            print("="*60)

    async def ble_loop(self):
        """Background loop to send BLE notifications."""
        try:
            last_prediction = None
            last_confidence = None

            while True:
                # Get current prediction from stats
                if self.frame_count > 0:
                    # Use the most recent prediction
                    prediction = 1 if self.alert_window and self.alert_window[-1][0] else 0
                    confidence = 0.75  # Approximate

                    # Only send if changed or every 1 second
                    if prediction != last_prediction or last_confidence != confidence:
                        await self.send_to_ble(prediction, confidence)
                        last_prediction = prediction
                        last_confidence = confidence

                # Send update every 1 second
                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            pass


async def main():
    """Main entry point."""
    import os

    # Check if running as root
    if os.geteuid() != 0:
        print("ERROR: This script must be run with sudo (for Bluetooth)")
        print("Usage: sudo python3 run_focusdrive.py")
        sys.exit(1)

    # Model path
    model_path = Path(__file__).parent / "models" / "mobilenet_driver_best.pth"

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first or check the path.")
        sys.exit(1)

    # Create and run system
    system = FocusDriveSystem(model_path, device="cpu")
    await system.run()


if __name__ == "__main__":
    asyncio.run(main())
