"""
Speed Monitor Module for FocusDrive

Monitors vehicle speed and determines when distraction detection should be active.
Supports simulated speed data for testing (can be extended for GPS/OBD-II).

Author: FocusDrive Team
Date: November 2024
"""

import time
import random
from collections import deque
from typing import Optional, Literal


class SpeedMonitor:
    """
    Monitors vehicle speed and manages detection activation based on speed thresholds.

    Detection activates when vehicle speed exceeds threshold for sustained duration.
    Includes simulated speed generation for testing without hardware.
    """

    def __init__(
        self,
        method: Literal['simulated', 'gps', 'obd'] = 'simulated',
        speed_threshold: float = 15.0,
        activation_duration: float = 10.0,
        history_size: int = 300  # 10 seconds at 30 FPS
    ):
        """
        Initialize speed monitor.

        Args:
            method: Speed data source ('simulated', 'gps', 'obd')
            speed_threshold: Minimum speed (mph) to activate detection
            activation_duration: Time (seconds) above threshold before activating
            history_size: Number of speed readings to track
        """
        self.method = method
        self.speed_threshold = speed_threshold
        self.activation_duration = activation_duration

        # Speed history tracking
        self.speed_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)

        # Activation state
        self.is_active = False
        self.time_above_threshold = 0.0
        self.time_below_threshold = 0.0

        # Simulated speed generator
        if method == 'simulated':
            self._init_simulator()

        # Last update time
        self.last_update_time = time.time()

    def _init_simulator(self):
        """Initialize simulated speed generator with realistic driving patterns."""
        self.sim_current_speed = 0.0
        self.sim_target_speed = 0.0
        self.sim_mode = 'stopped'  # stopped, accelerating, cruising, decelerating
        self.sim_mode_start_time = time.time()
        self.sim_mode_duration = 0.0

        # Driving pattern configuration
        self.sim_patterns = {
            'stopped': {'duration': (5, 15), 'next': 'accelerating'},
            'accelerating': {'duration': (3, 8), 'target_speed': (20, 60), 'next': 'cruising'},
            'cruising': {'duration': (10, 30), 'next': 'decelerating'},
            'decelerating': {'duration': (3, 8), 'next': 'stopped'}
        }

    def _update_simulated_speed(self) -> float:
        """
        Generate realistic simulated speed based on driving patterns.

        Returns:
            Current simulated speed in mph
        """
        current_time = time.time()
        time_in_mode = current_time - self.sim_mode_start_time

        # Check if it's time to change driving mode
        if time_in_mode >= self.sim_mode_duration:
            pattern = self.sim_patterns[self.sim_mode]
            next_mode = pattern['next']

            # Set target speed for new mode
            if next_mode == 'accelerating':
                self.sim_target_speed = random.uniform(*pattern['target_speed'])
            elif next_mode == 'stopped' or next_mode == 'decelerating':
                self.sim_target_speed = 0.0
            # cruising keeps current speed

            # Switch to new mode
            self.sim_mode = next_mode
            self.sim_mode_start_time = current_time
            pattern = self.sim_patterns[self.sim_mode]
            self.sim_mode_duration = random.uniform(*pattern['duration'])

        # Smoothly transition to target speed
        speed_diff = self.sim_target_speed - self.sim_current_speed

        if self.sim_mode == 'accelerating':
            # Accelerate smoothly (0.5-2 mph per update)
            acceleration = random.uniform(0.5, 2.0)
            if abs(speed_diff) > 0.5:
                self.sim_current_speed += acceleration if speed_diff > 0 else -acceleration
        elif self.sim_mode == 'decelerating':
            # Decelerate smoothly (0.5-2 mph per update)
            deceleration = random.uniform(0.5, 2.0)
            if self.sim_current_speed > 0:
                self.sim_current_speed = max(0, self.sim_current_speed - deceleration)
        elif self.sim_mode == 'cruising':
            # Add small random variations (±2 mph)
            variation = random.uniform(-0.5, 0.5)
            self.sim_current_speed = max(0, self.sim_current_speed + variation)
        # stopped mode: speed stays at 0

        # Ensure speed is non-negative
        self.sim_current_speed = max(0.0, self.sim_current_speed)

        return self.sim_current_speed

    def get_current_speed(self) -> float:
        """
        Get current vehicle speed.

        Returns:
            Current speed in mph
        """
        if self.method == 'simulated':
            return self._update_simulated_speed()
        elif self.method == 'gps':
            # Placeholder for GPS implementation
            # TODO: Integrate gpsd-py3 library
            raise NotImplementedError("GPS speed monitoring not yet implemented")
        elif self.method == 'obd':
            # Placeholder for OBD-II implementation
            # TODO: Integrate python-OBD library
            raise NotImplementedError("OBD-II speed monitoring not yet implemented")
        else:
            raise ValueError(f"Unknown speed method: {self.method}")

    def update(self) -> dict:
        """
        Update speed history and check activation status.

        Returns:
            Dictionary with current speed, activation status, and timing info
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Get current speed
        speed = self.get_current_speed()

        # Update history
        self.speed_history.append(speed)
        self.timestamp_history.append(current_time)

        # Update time above/below threshold
        if speed >= self.speed_threshold:
            self.time_above_threshold += dt
            self.time_below_threshold = 0.0
        else:
            self.time_below_threshold += dt
            self.time_above_threshold = 0.0

        # Update activation status
        if self.time_above_threshold >= self.activation_duration:
            self.is_active = True
        elif self.time_below_threshold >= 3.0:  # Deactivate after 3 seconds below threshold
            self.is_active = False

        return {
            'speed': speed,
            'is_active': self.is_active,
            'time_above_threshold': self.time_above_threshold,
            'time_below_threshold': self.time_below_threshold,
            'seconds_until_active': max(0, self.activation_duration - self.time_above_threshold),
            'mode': self.sim_mode if self.method == 'simulated' else None
        }

    def should_activate_detection(self) -> bool:
        """
        Check if distraction detection should be active.

        Returns:
            True if detection should run, False otherwise
        """
        return self.is_active

    def get_average_speed(self, duration: float = 5.0) -> float:
        """
        Get average speed over recent period.

        Args:
            duration: Time period in seconds to average over

        Returns:
            Average speed in mph over the period
        """
        if not self.speed_history or not self.timestamp_history:
            return 0.0

        current_time = time.time()
        cutoff_time = current_time - duration

        # Find speeds within duration window
        recent_speeds = []
        for speed, timestamp in zip(self.speed_history, self.timestamp_history):
            if timestamp >= cutoff_time:
                recent_speeds.append(speed)

        if not recent_speeds:
            return 0.0

        return sum(recent_speeds) / len(recent_speeds)

    def reset(self):
        """Reset speed monitor state."""
        self.speed_history.clear()
        self.timestamp_history.clear()
        self.is_active = False
        self.time_above_threshold = 0.0
        self.time_below_threshold = 0.0
        if self.method == 'simulated':
            self._init_simulator()


# Example usage and testing
if __name__ == "__main__":
    print("=== Speed Monitor Test ===\n")

    # Create speed monitor with simulated data
    monitor = SpeedMonitor(
        method='simulated',
        speed_threshold=15.0,
        activation_duration=10.0
    )

    print("Simulating 60 seconds of driving...")
    print("Threshold: 15 mph | Activation delay: 10 seconds\n")

    start_time = time.time()
    last_print = start_time

    try:
        while time.time() - start_time < 60:
            # Update speed monitor
            status = monitor.update()

            # Print status every 2 seconds
            if time.time() - last_print >= 2.0:
                elapsed = time.time() - start_time
                print(f"[{elapsed:05.1f}s] Speed: {status['speed']:5.1f} mph | "
                      f"Mode: {status['mode']:13s} | "
                      f"Active: {status['is_active']} | "
                      f"Time above: {status['time_above_threshold']:4.1f}s")
                last_print = time.time()

            # Simulate real-time updates (30 FPS)
            time.sleep(1/30)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")

    print("\n=== Test Complete ===")
    print(f"Final speed: {monitor.get_current_speed():.1f} mph")
    print(f"Average speed (last 5s): {monitor.get_average_speed(5.0):.1f} mph")
    print(f"Detection active: {monitor.is_active}")
