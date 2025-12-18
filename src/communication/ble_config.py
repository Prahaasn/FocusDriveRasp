"""
BLE Configuration for FocusDrive

Defines UUIDs, device name, and data format constants for
Bluetooth Low Energy communication.

Author: FocusDrive Team
Date: December 2024
"""

# Device Information
DEVICE_NAME = "FocusDrive"

# GATT Service and Characteristic UUIDs (LOCKED - DO NOT CHANGE)
SERVICE_UUID = "9a1f0000-0000-1000-8000-00805f9b34fb"
CHARACTERISTIC_UUID = "9a1f0001-0000-1000-8000-00805f9b34fb"

# Notification Configuration
UPDATE_INTERVAL = 1.0  # seconds (1 Hz)

# Driver State Values
STATE_ATTENTIVE = "ATTENTIVE"
STATE_DISTRACTED = "DISTRACTED"

# Distraction Reasons (example values for testing)
REASON_SAFE_DRIVING = "SAFE_DRIVING"
REASON_PHONE = "PHONE"
REASON_TEXTING = "TEXTING"
REASON_EATING = "EATING"
REASON_DRINKING = "DRINKING"
REASON_REACHING = "REACHING"
REASON_MAKEUP = "MAKEUP"
REASON_PASSENGER = "PASSENGER"

# Data Format: STATE|REASON|CONFIDENCE
# Example: DISTRACTED|PHONE|0.91


def format_driver_state(state: str, reason: str, confidence: float) -> str:
    """
    Format driver state data for BLE transmission.

    Args:
        state: "ATTENTIVE" or "DISTRACTED"
        reason: Distraction reason (e.g., "PHONE", "SAFE_DRIVING")
        confidence: Confidence score (0.0 to 1.0)

    Returns:
        Formatted string: STATE|REASON|CONFIDENCE

    Example:
        >>> format_driver_state("DISTRACTED", "PHONE", 0.91)
        'DISTRACTED|PHONE|0.91'
    """
    return f"{state}|{reason}|{confidence:.2f}"
