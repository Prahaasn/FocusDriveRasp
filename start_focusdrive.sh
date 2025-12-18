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
echo "  Starting FocusDrive System"
echo "=========================================="
echo "  • AI Driver Distraction Detection"
echo "  • BLE Broadcasting to iPhone"
echo "  • Real-time Camera Feed"
echo "=========================================="
echo ""

# Run the integrated system
python3 run_focusdrive.py
