#!/bin/bash

# FocusDrive - One-Command Startup Script
# Usage: ./start_focusdrive.sh

echo "=========================================="
echo "  🚗 FocusDrive System Starting"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  This script needs sudo permissions for Bluetooth"
    echo "Restarting with sudo..."
    sudo "$0" "$@"
    exit $?
fi

# Navigate to project directory
cd /home/prahaasn/focusdrive-ai-detection

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "✓ Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found at venv/"
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "=========================================="
echo "  Starting BLE GATT Server"
echo "=========================================="
echo "Device Name: FocusDrive"
echo "Service UUID: 9a1f0000-0000-1000-8000-00805f9b34fb"
echo ""
echo "To test on your iPhone:"
echo "  1. Open nRF Connect app"
echo "  2. Scan for devices"
echo "  3. Connect to 'FocusDrive'"
echo "  4. Enable notifications"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Run the BLE server
python3 ble_server_standalone.py
