"""
Accelerometer Speed Estimator for FocusDrive

Estimates vehicle speed by integrating acceleration over time.
Implements Zero Velocity Update (ZUPT) drift correction to handle
accelerometer integration errors.

Author: FocusDrive Team
Date: December 2024
"""

import time
import numpy as np
from collections import deque
from typing import Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccelerometerSpeedEstimator:
    """
    Estimates vehicle speed from MPU6050 accelerometer with drift correction.

    Uses Zero Velocity Update (ZUPT) to detect stationary periods and
    reset velocity to zero, preventing unbounded integration drift.

    Example:
        from src.utils.mpu6050_driver import MPU6050Sensor

        sensor = MPU6050Sensor()
        sensor.initialize()
        sensor.calibrate()

        estimator = AccelerometerSpeedEstimator(sensor)

        while True:
            result = estimator.update(dt=0.033)  # 30 Hz
            print(f"Speed: {result['speed_mph']:.1f} mph")
    """

    def __init__(
        self,
        sensor,
        stationary_threshold: float = 0.05,
        stationary_duration: float = 0.5,
        alpha_bias: float = 0.98,
        forward_axis: int = 0
    ):
        """
        Initialize accelerometer speed estimator.

        Args:
            sensor: MPU6050Sensor instance
            stationary_threshold: Acceleration variance threshold for ZUPT (m/s²)
            stationary_duration: Time threshold to trigger ZUPT (seconds)
            alpha_bias: High-pass filter coefficient for bias removal (0.9-0.99)
            forward_axis: Vehicle forward axis (0=X, 1=Y, 2=Z)
        """
        self.sensor = sensor
        self.stationary_threshold = stationary_threshold
        self.stationary_duration = stationary_duration
        self.alpha_bias = alpha_bias
        self.forward_axis = forward_axis

        # Speed estimation state
        self.velocity_mps = 0.0  # Current velocity estimate (m/s)
        self.accel_bias = 0.0    # Estimated acceleration bias (m/s²)

        # Stationary detection state
        self.stationary_time = 0.0  # Time vehicle has been stationary
        self.is_stationary = False

        # Acceleration history for variance calculation (30 samples = 1 sec @ 30Hz)
        self.accel_history = deque(maxlen=30)

        # Physical constraints
        self.max_acceleration = 0.5 * 9.81  # 0.5g (~5 m/s²)
        self.max_speed_mps = 44.7  # 100 mph in m/s

        # Last update time
        self.last_update_time = time.time()

        # Last result (for error fallback)
        self._last_result = {
            'speed_mps': 0.0,
            'speed_mph': 0.0,
            'is_stationary': True,
            'accel_forward': 0.0,
            'drift_corrected': False
        }

        # Statistics for debugging
        self.zupt_count = 0  # Number of ZUPT corrections applied
        self.last_zupt_time = time.time()

    def update(self, dt: float) -> Dict[str, Any]:
        """
        Update speed estimate based on latest sensor reading.

        Args:
            dt: Time since last update (seconds)

        Returns:
            Dictionary with:
                - speed_mps: Speed in m/s
                - speed_mph: Speed in mph
                - is_stationary: True if vehicle detected as stopped
                - accel_forward: Forward acceleration (m/s²)
                - drift_corrected: True if ZUPT was applied this update

        Raises:
            Exception: If sensor reading fails (returns last known result)
        """
        try:
            # Read acceleration from sensor
            accel_vector = self.sensor.get_acceleration()

            # Extract forward axis acceleration
            accel_raw = accel_vector[self.forward_axis]

            # Apply high-pass filter to remove bias
            accel_forward = self._high_pass_filter(accel_raw)

            # Add to history for variance calculation
            self.accel_history.append(abs(accel_forward))

            # Validate acceleration is physically realistic
            if abs(accel_forward) > self.max_acceleration:
                logger.warning(
                    f"Unrealistic acceleration: {accel_forward:.2f} m/s² "
                    f"(max: {self.max_acceleration:.2f}). Using last result."
                )
                return self._last_result

            # Detect if vehicle is stationary
            drift_corrected = False
            if self._detect_stationary():
                self.stationary_time += dt
                self.is_stationary = True

                # Apply Zero Velocity Update if stationary long enough
                if self.stationary_time >= self.stationary_duration:
                    if abs(self.velocity_mps) > 0.1:  # Only log if resetting non-zero velocity
                        logger.debug(
                            f"ZUPT: Resetting velocity {self.velocity_mps:.2f} → 0.0 m/s"
                        )
                    self._apply_zero_velocity_update()
                    drift_corrected = True
            else:
                self.stationary_time = 0.0
                self.is_stationary = False

            # Integrate acceleration to get velocity (if not stationary)
            if not self.is_stationary:
                self.velocity_mps += accel_forward * dt

            # Apply velocity sanity check
            if abs(self.velocity_mps) > self.max_speed_mps:
                logger.warning(
                    f"Unrealistic velocity: {self.velocity_mps:.2f} m/s, resetting to 0"
                )
                self.velocity_mps = 0.0

            # Ensure velocity is non-negative (can't go backward)
            self.velocity_mps = max(0.0, self.velocity_mps)

            # Convert to mph
            speed_mph = self.velocity_mps * 2.23694  # m/s to mph

            # Store result
            self._last_result = {
                'speed_mps': self.velocity_mps,
                'speed_mph': speed_mph,
                'is_stationary': self.is_stationary,
                'accel_forward': accel_forward,
                'drift_corrected': drift_corrected
            }

            return self._last_result

        except Exception as e:
            logger.error(f"Speed estimation error: {e}")
            # Return last known good result
            return self._last_result

    def _detect_stationary(self) -> bool:
        """
        Detect if vehicle is stationary using acceleration variance.

        Uses acceleration history to calculate variance. Low variance
        indicates the vehicle is not moving.

        Returns:
            True if vehicle appears stationary, False otherwise
        """
        if len(self.accel_history) < 10:  # Need sufficient samples
            return False

        # Calculate variance of recent acceleration magnitudes
        variance = np.var(self.accel_history)

        # Low variance indicates stationary
        return variance < self.stationary_threshold ** 2

    def _apply_zero_velocity_update(self):
        """
        Apply Zero Velocity Update (ZUPT) drift correction.

        Resets velocity to zero when vehicle is detected as stationary.
        This prevents unbounded drift from accelerometer integration errors.
        """
        self.velocity_mps = 0.0
        self.zupt_count += 1
        self.last_zupt_time = time.time()

    def _high_pass_filter(self, raw_accel: float) -> float:
        """
        Apply high-pass filter to remove DC bias from acceleration.

        Uses exponential moving average to estimate and remove bias:
            bias = alpha * bias + (1-alpha) * raw_accel
            filtered = raw_accel - bias

        Args:
            raw_accel: Raw acceleration measurement (m/s²)

        Returns:
            Filtered acceleration with bias removed (m/s²)
        """
        # Update bias estimate
        self.accel_bias = self.alpha_bias * self.accel_bias + \
                         (1 - self.alpha_bias) * raw_accel

        # Return bias-corrected acceleration
        return raw_accel - self.accel_bias

    def get_speed_mph(self) -> float:
        """
        Get current speed estimate in mph.

        Returns:
            Speed in mph
        """
        return self.velocity_mps * 2.23694

    def get_speed_mps(self) -> float:
        """
        Get current speed estimate in m/s.

        Returns:
            Speed in m/s
        """
        return self.velocity_mps

    def reset(self):
        """Reset estimator state to initial conditions."""
        self.velocity_mps = 0.0
        self.accel_bias = 0.0
        self.stationary_time = 0.0
        self.is_stationary = False
        self.accel_history.clear()
        self.zupt_count = 0
        self.last_zupt_time = time.time()
        self._last_result = {
            'speed_mps': 0.0,
            'speed_mph': 0.0,
            'is_stationary': True,
            'accel_forward': 0.0,
            'drift_corrected': False
        }

    def calibrate(self):
        """
        Calibrate sensor (must be stationary).

        Wrapper for sensor.calibrate() for convenience.
        """
        return self.sensor.calibrate()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get estimator statistics for debugging.

        Returns:
            Dictionary with:
                - zupt_count: Number of ZUPT corrections applied
                - time_since_last_zupt: Seconds since last ZUPT
                - current_bias: Current acceleration bias estimate
                - accel_variance: Current acceleration variance
        """
        return {
            'zupt_count': self.zupt_count,
            'time_since_last_zupt': time.time() - self.last_zupt_time,
            'current_bias': self.accel_bias,
            'accel_variance': np.var(self.accel_history) if len(self.accel_history) > 0 else 0.0
        }


# Example usage and testing
if __name__ == "__main__":
    print("=== Accelerometer Speed Estimator Test ===\n")
    print("This test requires actual MPU6050 hardware.\n")

    try:
        from src.utils.mpu6050_driver import MPU6050Sensor

        # Initialize sensor
        print("Initializing MPU6050...")
        sensor = MPU6050Sensor(bus_number=1, address=0x68)

        if not sensor.initialize():
            print("Failed to initialize sensor")
            exit(1)

        print("Sensor initialized!\n")

        # Calibrate
        print("IMPORTANT: Keep sensor stationary and level!")
        input("Press Enter when ready to calibrate...")

        calibration = sensor.calibrate(samples=1000)
        print()

        # Initialize speed estimator
        estimator = AccelerometerSpeedEstimator(
            sensor=sensor,
            stationary_threshold=0.05,
            stationary_duration=0.5,
            forward_axis=0  # Use X-axis as forward
        )

        print("Speed estimator ready!")
        print("Reading speed for 30 seconds...")
        print("(Move the sensor forward to simulate acceleration)\n")

        start_time = time.time()
        last_print = start_time

        try:
            while time.time() - start_time < 30:
                # Calculate dt
                current_time = time.time()
                dt = current_time - last_print if last_print > 0 else 0.033
                last_print = current_time

                # Update speed
                result = estimator.update(dt=dt)

                # Print status
                print(
                    f"Speed: {result['speed_mph']:6.2f} mph | "
                    f"Accel: {result['accel_forward']:6.3f} m/s² | "
                    f"Stationary: {result['is_stationary']} | "
                    f"ZUPT: {result['drift_corrected']}",
                    end='\r'
                )

                # Print statistics every 5 seconds
                if int(current_time - start_time) % 5 == 0 and \
                   abs(current_time - start_time - int(current_time - start_time)) < 0.1:
                    stats = estimator.get_statistics()
                    print()  # New line
                    print(f"  [Stats] ZUPT count: {stats['zupt_count']}, "
                          f"Bias: {stats['current_bias']:.3f} m/s², "
                          f"Variance: {stats['accel_variance']:.6f}")

                time.sleep(0.033)  # ~30 Hz

        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")

        print("\n\n=== Test Complete ===")
        stats = estimator.get_statistics()
        print(f"Final speed: {estimator.get_speed_mph():.1f} mph")
        print(f"ZUPT corrections: {stats['zupt_count']}")
        print(f"Time since last ZUPT: {stats['time_since_last_zupt']:.1f}s")

        # Cleanup
        sensor.close()

    except ImportError:
        print("Error: MPU6050 driver not found or hardware not available")
        print("This test requires actual hardware and dependencies installed")
    except Exception as e:
        print(f"Error: {e}")
