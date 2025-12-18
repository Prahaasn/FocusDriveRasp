#!/bin/bash

echo "=========================================="
echo "SIMPLE BLUETOOTH TEST FOR FOCUSDRIVE"
echo "=========================================="
echo ""

# Stop any existing processes
sudo pkill -9 -f ble_server 2>/dev/null
sleep 2

# Power on Bluetooth
echo "1. Powering on Bluetooth..."
bluetoothctl power on
sleep 1

# Set device name
echo "2. Setting device name to 'FocusDrive'..."
bluetoothctl system-alias FocusDrive
sleep 1

# Make discoverable
echo "3. Making device discoverable..."
bluetoothctl discoverable on
sleep 1

# Make pairable
echo "4. Making device pairable..."
bluetoothctl pairable on
sleep 1

echo ""
echo "=========================================="
echo "✓ BLUETOOTH IS NOW ACTIVE"
echo "=========================================="
echo ""
echo "Device Name: FocusDrive"
echo "Discoverable: YES"
echo "Pairable: YES"
echo ""
echo "NOW CHECK YOUR IPHONE:"
echo "  1. Open iPhone Settings → Bluetooth"
echo "  2. Look for 'FocusDrive'"
echo "  3. OR open nRF Connect app and scan"
echo ""
echo "The device will stay discoverable for 3 minutes"
echo "Press Ctrl+C to stop"
echo ""

# Show current status
bluetoothctl show | grep -E "Name|Alias|Powered|Discoverable|Pairable"

echo ""
echo "Keeping alive... (Press Ctrl+C to exit)"
sleep 180
