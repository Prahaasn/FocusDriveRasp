#!/usr/bin/env python3
"""
FocusDrive BLE Server - Standalone Test Runner

Starts the BLE GATT server and sends fake driver state data
every second for testing with mobile app.

This script must run with sudo privileges to access BlueZ Bluetooth stack.

Usage:
    sudo python3 ble_server_standalone.py

Or with virtual environment:
    sudo venv/bin/python3 ble_server_standalone.py

Author: FocusDrive Team
Date: December 2024
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from communication.ble_server import BLEServer
from communication.ble_config import (
    DEVICE_NAME,
    SERVICE_UUID,
    CHARACTERISTIC_UUID,
    UPDATE_INTERVAL
)


async def main():
    """Main entry point for BLE server"""
    print("="*60)
    print("FocusDrive BLE GATT Server")
    print("="*60)
    print(f"Device Name: {DEVICE_NAME}")
    print(f"Service UUID: {SERVICE_UUID}")
    print(f"Characteristic UUID: {CHARACTERISTIC_UUID}")
    print(f"Update Interval: {UPDATE_INTERVAL}s")
    print()
    print("Starting BLE server...")
    print("IMPORTANT: This script must run with sudo")
    print("="*60)

    server = BLEServer()

    try:
        await server.start()
        print("\n✓ BLE server started successfully!")
        print("✓ Device is now discoverable as 'FocusDrive'")
        print("✓ Sending test notifications every 1 second")
        print()
        print("To test with iPhone:")
        print("  1. Open iOS Settings → Bluetooth")
        print("  2. Look for 'FocusDrive' in Other Devices")
        print()
        print("Or use nRF Connect app (recommended):")
        print("  1. Install 'nRF Connect' from App Store")
        print("  2. Scan for devices")
        print("  3. Connect to 'FocusDrive'")
        print("  4. Enable notifications on characteristic")
        print("  5. Watch for: DISTRACTED|PHONE|0.91")
        print()
        print("Press Ctrl+C to stop")
        print()

        # Run test loop
        await server.run_test_loop()

    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("Shutting down BLE server...")
        server.stop()
        print("✓ BLE server stopped")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()
        print("Troubleshooting:")
        print("  - Is Bluetooth enabled? Run: bluetoothctl power on")
        print("  - Running with sudo? Must use: sudo python3 ble_server_standalone.py")
        print("  - Check Bluetooth service: systemctl status bluetooth")
        sys.exit(1)


if __name__ == "__main__":
    # Check for root privileges
    if os.geteuid() != 0:
        print("="*60)
        print("ERROR: This script must be run with sudo")
        print("="*60)
        print()
        print("BLE advertising and GATT server require root privileges.")
        print()
        print("Run with:")
        print("  sudo python3 ble_server_standalone.py")
        print()
        print("Or with virtual environment:")
        print("  sudo venv/bin/python3 ble_server_standalone.py")
        print()
        print("="*60)
        sys.exit(1)

    # Run server
    asyncio.run(main())
