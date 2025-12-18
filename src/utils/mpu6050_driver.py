"""
MPU6050 Hardware Driver for FocusDrive

Low-level I2C communication with MPU6050 6-axis accelerometer + gyroscope sensor.
Provides calibration, raw data reading, and unit conversion.

Author: FocusDrive Team
Date: December 2024
"""

import time
import numpy as np
from typing import Tuple, Dict, Optional

try:
    from smbus2 import SMBus
except ImportError:
    raise ImportError(
        "smbus2 library not found. Install with: pip install smbus2"
    )


# MPU6050 Register Addresses
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_CONFIG = 0x1C
GYRO_CONFIG = 0x1B
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H = 0x43
GYRO_YOUT_H = 0x45
GYRO_ZOUT_H = 0x47
WHO_AM_I = 0x75

# Configuration Values
ACCEL_RANGE_2G = 0x00
ACCEL_RANGE_4G = 0x08
ACCEL_RANGE_8G = 0x10
ACCEL_RANGE_16G = 0x18

GYRO_RANGE_250DEG = 0x00
GYRO_RANGE_500DEG = 0x08
GYRO_RANGE_1000DEG = 0x10
GYRO_RANGE_2000DEG = 0x18


class MPU6050Error(Exception):
    """Base exception for MPU6050 errors"""
    pass


class MPU6050ConnectionError(MPU6050Error):
    """Raised when I2C communication fails"""
    pass


class MPU6050CalibrationError(MPU6050Error):
    """Raised when calibration fails"""
    pass


class MPU6050Sensor:
    """
    MPU6050 6-axis IMU driver for Raspberry Pi.

    Provides low-level I2C communication, sensor initialization,
    calibration, and data reading for accelerometer and gyroscope.

    Example:
        sensor = MPU6050Sensor(bus_number=1, address=0x68)
        if sensor.initialize():
            calibration = sensor.calibrate(samples=1000)
            accel = sensor.get_acceleration()
            print(f"Acceleration: {accel} m/s²")
    """

    def __init__(self, bus_number: int = 1, address: int = MPU6050_ADDR):
        """
        Initialize MPU6050 sensor.

        Args:
            bus_number: I2C bus number (1 for Raspberry Pi)
            address: I2C address (0x68 or 0x69)
        """
        self.bus_number = bus_number
        self.address = address
        self.bus: Optional[SMBus] = None

        # Sensor configuration
        self.accel_range = ACCEL_RANGE_2G  # ±2g range
        self.gyro_range = GYRO_RANGE_250DEG  # ±250°/s range

        # Scale factors for converting raw values to physical units
        self.accel_scale = 16384.0  # LSB/g for ±2g range
        self.gyro_scale = 131.0     # LSB/(°/s) for ±250°/s range

        # Calibration offsets (measured during calibration)
        self.accel_offset = np.array([0.0, 0.0, 0.0])
        self.gyro_offset = np.array([0.0, 0.0, 0.0])

        # Gravity constant (m/s²)
        self.gravity = 9.81

    def initialize(self) -> bool:
        """
        Initialize I2C connection and wake up MPU6050 sensor.

        Returns:
            True if initialization successful, False otherwise

        Raises:
            MPU6050ConnectionError: If I2C connection fails
        """
        try:
            # Open I2C bus
            self.bus = SMBus(self.bus_number)

            # Verify sensor is present
            who_am_i = self.bus.read_byte_data(self.address, WHO_AM_I)
            if who_am_i != 0x68:
                raise MPU6050ConnectionError(
                    f"MPU6050 not found at address 0x{self.address:02x}. "
                    f"WHO_AM_I returned 0x{who_am_i:02x} (expected 0x68)"
                )

            # Wake up sensor (exit sleep mode)
            self.bus.write_byte_data(self.address, PWR_MGMT_1, 0x00)
            time.sleep(0.1)

            # Configure accelerometer range (±2g)
            self.bus.write_byte_data(self.address, ACCEL_CONFIG, self.accel_range)

            # Configure gyroscope range (±250°/s)
            self.bus.write_byte_data(self.address, GYRO_CONFIG, self.gyro_range)

            time.sleep(0.1)

            return True

        except OSError as e:
            raise MPU6050ConnectionError(
                f"Failed to initialize MPU6050: {e}. "
                "Check wiring and ensure I2C is enabled (sudo raspi-config)."
            )

    def _read_word_2c(self, register: int) -> int:
        """
        Read a 16-bit signed value from two consecutive registers.

        Args:
            register: High byte register address

        Returns:
            16-bit signed integer
        """
        high = self.bus.read_byte_data(self.address, register)
        low = self.bus.read_byte_data(self.address, register + 1)
        value = (high << 8) + low

        # Convert to signed value
        if value >= 0x8000:
            return -((65535 - value) + 1)
        else:
            return value

    def read_accelerometer_raw(self) -> Tuple[int, int, int]:
        """
        Read raw 16-bit accelerometer values.

        Returns:
            Tuple of (accel_x, accel_y, accel_z) raw values

        Raises:
            MPU6050ConnectionError: If I2C read fails
        """
        try:
            accel_x = self._read_word_2c(ACCEL_XOUT_H)
            accel_y = self._read_word_2c(ACCEL_YOUT_H)
            accel_z = self._read_word_2c(ACCEL_ZOUT_H)
            return (accel_x, accel_y, accel_z)
        except OSError as e:
            raise MPU6050ConnectionError(f"Failed to read accelerometer: {e}")

    def read_gyroscope_raw(self) -> Tuple[int, int, int]:
        """
        Read raw 16-bit gyroscope values.

        Returns:
            Tuple of (gyro_x, gyro_y, gyro_z) raw values

        Raises:
            MPU6050ConnectionError: If I2C read fails
        """
        try:
            gyro_x = self._read_word_2c(GYRO_XOUT_H)
            gyro_y = self._read_word_2c(GYRO_YOUT_H)
            gyro_z = self._read_word_2c(GYRO_ZOUT_H)
            return (gyro_x, gyro_y, gyro_z)
        except OSError as e:
            raise MPU6050ConnectionError(f"Failed to read gyroscope: {e}")

    def convert_to_g(self, raw_value: int) -> float:
        """
        Convert raw accelerometer value to g-force units.

        Args:
            raw_value: Raw 16-bit accelerometer reading

        Returns:
            Acceleration in g (Earth gravity units)
        """
        return raw_value / self.accel_scale

    def convert_to_mps2(self, raw_value: int) -> float:
        """
        Convert raw accelerometer value to m/s².

        Args:
            raw_value: Raw 16-bit accelerometer reading

        Returns:
            Acceleration in m/s²
        """
        return self.convert_to_g(raw_value) * self.gravity

    def convert_to_deg_per_sec(self, raw_value: int) -> float:
        """
        Convert raw gyroscope value to degrees/second.

        Args:
            raw_value: Raw 16-bit gyroscope reading

        Returns:
            Angular velocity in degrees/second
        """
        return raw_value / self.gyro_scale

    def calibrate(self, samples: int = 1000) -> Dict[str, float]:
        """
        Calibrate sensor by measuring bias when stationary.

        IMPORTANT: Sensor must be stationary and level during calibration!

        Args:
            samples: Number of samples to average for calibration

        Returns:
            Dictionary with calibration offsets

        Raises:
            MPU6050CalibrationError: If calibration validation fails
        """
        if self.bus is None:
            raise MPU6050Error("Sensor not initialized. Call initialize() first.")

        print(f"Collecting {samples} samples for calibration...")

        accel_x_samples = []
        accel_y_samples = []
        accel_z_samples = []
        gyro_x_samples = []
        gyro_y_samples = []
        gyro_z_samples = []

        # Collect samples
        for i in range(samples):
            # Read raw values
            accel_x, accel_y, accel_z = self.read_accelerometer_raw()
            gyro_x, gyro_y, gyro_z = self.read_gyroscope_raw()

            # Store samples
            accel_x_samples.append(self.convert_to_g(accel_x))
            accel_y_samples.append(self.convert_to_g(accel_y))
            accel_z_samples.append(self.convert_to_g(accel_z))
            gyro_x_samples.append(self.convert_to_deg_per_sec(gyro_x))
            gyro_y_samples.append(self.convert_to_deg_per_sec(gyro_y))
            gyro_z_samples.append(self.convert_to_deg_per_sec(gyro_z))

            # Small delay between readings
            if i % 100 == 0:
                print(f"  Progress: {i}/{samples}")
            time.sleep(0.001)

        # Calculate means (offsets)
        accel_x_mean = np.mean(accel_x_samples)
        accel_y_mean = np.mean(accel_y_samples)
        accel_z_mean = np.mean(accel_z_samples)
        gyro_x_mean = np.mean(gyro_x_samples)
        gyro_y_mean = np.mean(gyro_y_samples)
        gyro_z_mean = np.mean(gyro_z_samples)

        # Calculate standard deviations (for validation)
        accel_z_std = np.std(accel_z_samples)

        # Validate calibration
        # Z-axis should measure ~1g when sensor is level
        if abs(accel_z_mean - 1.0) > 0.3:
            raise MPU6050CalibrationError(
                f"Invalid calibration: Z-axis = {accel_z_mean:.2f}g (expected ~1.0g). "
                "Ensure sensor is level and stationary on a flat surface."
            )

        # Check variance is low (sensor stationary)
        if accel_z_std > 0.1:
            raise MPU6050CalibrationError(
                f"Excessive motion detected during calibration (std={accel_z_std:.3f}). "
                "Keep sensor completely stationary."
            )

        # Store offsets
        # X and Y should be ~0g, Z should be ~1g when level
        self.accel_offset = np.array([
            accel_x_mean,
            accel_y_mean,
            accel_z_mean - 1.0  # Subtract 1g from Z-axis
        ])

        self.gyro_offset = np.array([
            gyro_x_mean,
            gyro_y_mean,
            gyro_z_mean
        ])

        calibration = {
            'accel_x_offset': accel_x_mean,
            'accel_y_offset': accel_y_mean,
            'accel_z_offset': accel_z_mean,
            'gyro_x_offset': gyro_x_mean,
            'gyro_y_offset': gyro_y_mean,
            'gyro_z_offset': gyro_z_mean,
            'accel_z_std': accel_z_std
        }

        print(f"Calibration complete!")
        print(f"  Accel offsets: X={accel_x_mean:.3f}g, Y={accel_y_mean:.3f}g, Z={accel_z_mean:.3f}g")
        print(f"  Gyro offsets: X={gyro_x_mean:.3f}°/s, Y={gyro_y_mean:.3f}°/s, Z={gyro_z_mean:.3f}°/s")

        return calibration

    def get_acceleration(self) -> np.ndarray:
        """
        Get calibrated acceleration in m/s².

        Returns:
            NumPy array [accel_x, accel_y, accel_z] in m/s²

        Raises:
            MPU6050ConnectionError: If I2C read fails
        """
        # Read raw values
        accel_x_raw, accel_y_raw, accel_z_raw = self.read_accelerometer_raw()

        # Convert to g-force
        accel_x_g = self.convert_to_g(accel_x_raw)
        accel_y_g = self.convert_to_g(accel_y_raw)
        accel_z_g = self.convert_to_g(accel_z_raw)

        # Apply calibration offsets
        accel = np.array([accel_x_g, accel_y_g, accel_z_g]) - self.accel_offset

        # Convert to m/s²
        accel_mps2 = accel * self.gravity

        return accel_mps2

    def get_gyroscope(self) -> np.ndarray:
        """
        Get calibrated angular velocity in degrees/second.

        Returns:
            NumPy array [gyro_x, gyro_y, gyro_z] in °/s

        Raises:
            MPU6050ConnectionError: If I2C read fails
        """
        # Read raw values
        gyro_x_raw, gyro_y_raw, gyro_z_raw = self.read_gyroscope_raw()

        # Convert to degrees/second
        gyro_x = self.convert_to_deg_per_sec(gyro_x_raw)
        gyro_y = self.convert_to_deg_per_sec(gyro_y_raw)
        gyro_z = self.convert_to_deg_per_sec(gyro_z_raw)

        # Apply calibration offsets
        gyro = np.array([gyro_x, gyro_y, gyro_z]) - self.gyro_offset

        return gyro

    def close(self):
        """Close I2C bus connection."""
        if self.bus is not None:
            self.bus.close()
            self.bus = None


# Example usage and testing
if __name__ == "__main__":
    print("=== MPU6050 Sensor Test ===\n")

    try:
        # Initialize sensor
        sensor = MPU6050Sensor(bus_number=1, address=0x68)
        print("Initializing MPU6050...")

        if not sensor.initialize():
            print("Failed to initialize sensor")
            exit(1)

        print("Sensor initialized successfully!\n")

        # Calibrate sensor
        print("IMPORTANT: Keep sensor stationary and level!")
        input("Press Enter when ready to calibrate...")

        calibration = sensor.calibrate(samples=1000)
        print()

        # Read sensor data for 5 seconds
        print("Reading sensor data for 5 seconds...")
        print("Acceleration [X, Y, Z] in m/s²:\n")

        start_time = time.time()
        while time.time() - start_time < 5.0:
            accel = sensor.get_acceleration()
            print(f"Accel: [{accel[0]:7.3f}, {accel[1]:7.3f}, {accel[2]:7.3f}] m/s²", end='\r')
            time.sleep(0.1)

        print("\n\n=== Test Complete ===")

        # Cleanup
        sensor.close()

    except MPU6050Error as e:
        print(f"MPU6050 Error: {e}")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
