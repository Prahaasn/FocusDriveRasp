"""
Hardware Validation Test for MPU6050 Accelerometer

This script tests the MPU6050 sensor integration on Raspberry Pi hardware.
Requires physical MPU6050 connected via I2C.

Run this script on Raspberry Pi with MPU6050 connected:
    python tests/hardware_test_mpu6050.py

Author: FocusDrive Team
Date: December 2024
"""

import sys
import time
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.mpu6050_driver import MPU6050Sensor, MPU6050Error
from src.utils.accel_speed_estimator import AccelerometerSpeedEstimator
from src.utils.speed_monitor import SpeedMonitor


def test_i2c_connection():
    """Test 1: Verify I2C connection to MPU6050"""
    print("\n" + "="*60)
    print("TEST 1: I2C Connection")
    print("="*60)

    try:
        sensor = MPU6050Sensor(bus_number=1, address=0x68)
        if sensor.initialize():
            print("✓ MPU6050 connected successfully at address 0x68")
            print("✓ WHO_AM_I register verified")
            sensor.close()
            return True
        else:
            print("✗ Failed to initialize MPU6050")
            return False
    except MPU6050Error as e:
        print(f"✗ MPU6050 Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected Error: {e}")
        return False


def test_sensor_reading():
    """Test 2: Read raw sensor data"""
    print("\n" + "="*60)
    print("TEST 2: Sensor Reading")
    print("="*60)

    try:
        sensor = MPU6050Sensor(bus_number=1, address=0x68)
        sensor.initialize()

        print("Reading 10 samples from accelerometer...")
        for i in range(10):
            accel_raw = sensor.read_accelerometer_raw()
            accel_g = [sensor.convert_to_g(val) for val in accel_raw]

            print(f"  Sample {i+1}: X={accel_g[0]:6.3f}g, "
                  f"Y={accel_g[1]:6.3f}g, Z={accel_g[2]:6.3f}g")
            time.sleep(0.1)

        print("✓ Sensor reading successful")
        sensor.close()
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_calibration():
    """Test 3: Sensor calibration"""
    print("\n" + "="*60)
    print("TEST 3: Calibration")
    print("="*60)

    try:
        sensor = MPU6050Sensor(bus_number=1, address=0x68)
        sensor.initialize()

        print("IMPORTANT: Keep sensor stationary and level!")
        print("Calibration will start in 3 seconds...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)

        print("\nCalibrating...")
        calibration = sensor.calibrate(samples=500)

        print("\n✓ Calibration successful!")
        print(f"  Accel X offset: {calibration['accel_x_offset']:.3f} g")
        print(f"  Accel Y offset: {calibration['accel_y_offset']:.3f} g")
        print(f"  Accel Z offset: {calibration['accel_z_offset']:.3f} g (should be ~1.0g)")
        print(f"  Z-axis std dev: {calibration['accel_z_std']:.4f} g")

        # Validate calibration quality
        if abs(calibration['accel_z_offset'] - 1.0) > 0.2:
            print("⚠ Warning: Z-axis calibration may be inaccurate")
            print("  Ensure sensor is level on a flat surface")
        else:
            print("✓ Calibration quality: Good")

        sensor.close()
        return True

    except MPU6050Error as e:
        print(f"✗ Calibration failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_speed_estimation():
    """Test 4: Speed estimation with ZUPT"""
    print("\n" + "="*60)
    print("TEST 4: Speed Estimation")
    print("="*60)

    try:
        sensor = MPU6050Sensor(bus_number=1, address=0x68)
        sensor.initialize()

        print("Calibrating...")
        sensor.calibrate(samples=500)

        print("\n✓ Initializing speed estimator...")
        estimator = AccelerometerSpeedEstimator(
            sensor=sensor,
            stationary_threshold=0.05,
            stationary_duration=0.5,
            forward_axis=0  # X-axis
        )

        print("✓ Reading speed for 10 seconds...")
        print("  (Move sensor forward/backward to simulate acceleration)\n")
        print(f"{'Time':>6} | {'Speed (mph)':>12} | {'Accel (m/s²)':>13} | {'Stationary':>10} | {'ZUPT':>5}")
        print("-" * 60)

        start_time = time.time()
        last_update = start_time

        while time.time() - start_time < 10:
            current_time = time.time()
            dt = current_time - last_update
            last_update = current_time

            result = estimator.update(dt)

            elapsed = current_time - start_time
            print(f"{elapsed:6.1f} | {result['speed_mph']:12.2f} | "
                  f"{result['accel_forward']:13.3f} | "
                  f"{str(result['is_stationary']):>10} | "
                  f"{str(result['drift_corrected']):>5}")

            time.sleep(0.2)  # 5 Hz for readability

        print("\n✓ Speed estimation test complete")

        # Print statistics
        stats = estimator.get_statistics()
        print(f"\nStatistics:")
        print(f"  ZUPT corrections: {stats['zupt_count']}")
        print(f"  Current bias: {stats['current_bias']:.4f} m/s²")
        print(f"  Accel variance: {stats['accel_variance']:.6f}")

        sensor.close()
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_speed_monitor_integration():
    """Test 5: SpeedMonitor integration"""
    print("\n" + "="*60)
    print("TEST 5: SpeedMonitor Integration")
    print("="*60)

    try:
        print("Initializing SpeedMonitor with accelerometer mode...")
        print("(This will calibrate the sensor - keep it stationary!)\n")

        monitor = SpeedMonitor(
            method='accelerometer',
            speed_threshold=15.0,
            activation_duration=5.0
        )

        print("\n✓ SpeedMonitor initialized successfully!")
        print("✓ Reading speed for 10 seconds...\n")
        print(f"{'Time':>6} | {'Speed (mph)':>12} | {'Active':>7} | {'Time Above':>11}")
        print("-" * 45)

        start_time = time.time()

        while time.time() - start_time < 10:
            status = monitor.update()
            elapsed = time.time() - start_time

            print(f"{elapsed:6.1f} | {status['speed']:12.2f} | "
                  f"{str(status['is_active']):>7} | "
                  f"{status['time_above_threshold']:11.1f}")

            time.sleep(0.5)  # 2 Hz for readability

        print("\n✓ SpeedMonitor integration test complete")

        # Cleanup
        monitor.cleanup()
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all hardware tests"""
    print("\n" + "="*60)
    print("MPU6050 HARDWARE VALIDATION TEST SUITE")
    print("="*60)
    print("\nThis test requires:")
    print("  1. MPU6050 sensor connected via I2C")
    print("  2. I2C enabled on Raspberry Pi")
    print("  3. smbus2 library installed (pip install smbus2)")
    print("\nWiring:")
    print("  VCC → Pin 1 (3.3V)")
    print("  GND → Pin 6 (Ground)")
    print("  SDA → Pin 3 (GPIO 2)")
    print("  SCL → Pin 5 (GPIO 3)")

    input("\nPress Enter to start tests...")

    # Run tests
    results = []

    results.append(("I2C Connection", test_i2c_connection()))
    results.append(("Sensor Reading", test_sensor_reading()))
    results.append(("Calibration", test_calibration()))
    results.append(("Speed Estimation", test_speed_estimation()))
    results.append(("SpeedMonitor Integration", test_speed_monitor_integration()))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30} {status}")

    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! MPU6050 is working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Check hardware connections.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
