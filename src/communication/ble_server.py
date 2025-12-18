"""
BLE GATT Server for FocusDrive

Implements a Bluetooth Low Energy GATT Server using BlueZ D-Bus API.
Broadcasts driver distraction status via BLE notifications.

Author: FocusDrive Team
Date: December 2024
"""

import asyncio
import random
from datetime import datetime
from typing import Optional
from dbus_next.aio import MessageBus
from dbus_next.service import ServiceInterface, method, dbus_property
from dbus_next import Variant, BusType
from dbus_next.constants import PropertyAccess

from communication.ble_config import (
    DEVICE_NAME,
    SERVICE_UUID,
    CHARACTERISTIC_UUID,
    UPDATE_INTERVAL,
    STATE_ATTENTIVE,
    STATE_DISTRACTED,
    REASON_SAFE_DRIVING,
    REASON_PHONE,
    REASON_TEXTING,
    REASON_EATING,
    format_driver_state
)


# D-Bus constants
BLUEZ_SERVICE = 'org.bluez'
GATT_MANAGER_IFACE = 'org.bluez.GattManager1'
LE_ADVERTISING_MANAGER_IFACE = 'org.bluez.LEAdvertisingManager1'
GATT_SERVICE_IFACE = 'org.bluez.GattService1'
GATT_CHARACTERISTIC_IFACE = 'org.bluez.GattCharacteristic1'
LE_ADVERTISEMENT_IFACE = 'org.bluez.LEAdvertisement1'
DBUS_PROP_IFACE = 'org.freedesktop.DBus.Properties'
DBUS_OM_IFACE = 'org.freedesktop.DBus.ObjectManager'

ADAPTER_PATH = '/org/bluez/hci0'


class DriverStateCharacteristic(ServiceInterface):
    """
    BLE GATT Characteristic for driver state notifications.

    UUID: 9a1f0001-0000-1000-8000-00805f9b34fb
    Properties: NOTIFY
    Format: STATE|REASON|CONFIDENCE (UTF-8 string)
    """

    def __init__(self, service_path: str, char_index: int):
        """
        Initialize the driver state characteristic.

        Args:
            service_path: D-Bus path of parent service
            char_index: Index for unique D-Bus path
        """
        self.path = f"{service_path}/char{char_index:04d}"
        self.service = service_path
        self.uuid = CHARACTERISTIC_UUID
        self.flags = ['notify']
        self.notifying = False
        self.value = b''  # Empty bytes initially

        super().__init__(GATT_CHARACTERISTIC_IFACE)

    @method()
    def StartNotify(self):
        """Start sending notifications (called when client subscribes)"""
        print(f"✓ Client subscribed to notifications")
        self.notifying = True

    @method()
    def StopNotify(self):
        """Stop sending notifications (called when client unsubscribes)"""
        print(f"✗ Client unsubscribed from notifications")
        self.notifying = False

    @dbus_property(PropertyAccess.READ)
    def UUID(self) -> 's':
        """Return characteristic UUID"""
        return self.uuid

    @dbus_property(PropertyAccess.READ)
    def Service(self) -> 'o':
        """Return parent service D-Bus path"""
        return self.service

    @dbus_property(PropertyAccess.READ)
    def Flags(self) -> 'as':
        """Return characteristic flags (notify-only)"""
        return self.flags

    @dbus_property(PropertyAccess.READ)
    def Notifying(self) -> 'b':
        """Return whether characteristic is currently notifying"""
        return self.notifying

    @dbus_property(PropertyAccess.READ)
    def Value(self) -> 'ay':
        """Return current characteristic value as byte array"""
        return self.value

    def send_notification(self, state: str, reason: str, confidence: float):
        """
        Send driver state via BLE notification.

        Args:
            state: "ATTENTIVE" or "DISTRACTED"
            reason: Distraction reason (e.g., "PHONE", "SAFE_DRIVING")
            confidence: 0.0 to 1.0
        """
        if not self.notifying:
            return  # No clients subscribed

        # Format data
        payload = format_driver_state(state, reason, confidence)

        # Convert to bytes
        self.value = payload.encode('utf-8')

        # Emit PropertiesChanged signal (triggers BLE notification)
        self.emit_properties_changed(
            {'Value': Variant('ay', self.value)},
            []
        )

        # Log notification
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Sent: {payload}")


class DriverStateService(ServiceInterface):
    """
    BLE GATT Service for FocusDrive driver monitoring.

    UUID: 9a1f0000-0000-1000-8000-00805f9b34fb
    Contains: DriverStateCharacteristic
    """

    def __init__(self, bus: MessageBus, service_index: int):
        """
        Initialize the driver state service.

        Args:
            bus: D-Bus message bus
            service_index: Index for unique D-Bus path
        """
        self.path = f"/com/focusdrive/service{service_index:04d}"
        self.uuid = SERVICE_UUID
        self.primary = True
        self.bus = bus

        super().__init__(GATT_SERVICE_IFACE)

        # Create characteristic
        self.characteristic = DriverStateCharacteristic(self.path, 0)

    @dbus_property(PropertyAccess.READ)
    def UUID(self) -> 's':
        """Return service UUID"""
        return self.uuid

    @dbus_property(PropertyAccess.READ)
    def Primary(self) -> 'b':
        """Return whether this is a primary service"""
        return self.primary

    @dbus_property(PropertyAccess.READ)
    def Characteristics(self) -> 'ao':
        """Return list of characteristic D-Bus paths"""
        return [self.characteristic.path]


class FocusDriveAdvertisement(ServiceInterface):
    """
    BLE Advertisement for device discoverability.

    Type: peripheral
    LocalName: FocusDrive
    ServiceUUIDs: [SERVICE_UUID]
    """

    def __init__(self, bus: MessageBus, ad_index: int):
        """
        Initialize the BLE advertisement.

        Args:
            bus: D-Bus message bus
            ad_index: Index for unique D-Bus path
        """
        self.path = f"/com/focusdrive/advertisement{ad_index:04d}"
        self.bus = bus
        self.ad_type = 'peripheral'
        self.local_name = DEVICE_NAME
        self.service_uuids = [SERVICE_UUID]
        self.include_tx_power = True

        super().__init__(LE_ADVERTISEMENT_IFACE)

    @dbus_property(PropertyAccess.READ)
    def Type(self) -> 's':
        """Return advertisement type"""
        return self.ad_type

    @dbus_property(PropertyAccess.READ)
    def LocalName(self) -> 's':
        """Return advertised device name"""
        return self.local_name

    @dbus_property(PropertyAccess.READ)
    def ServiceUUIDs(self) -> 'as':
        """Return advertised service UUIDs"""
        return self.service_uuids

    @dbus_property(PropertyAccess.READ)
    def IncludeTxPower(self) -> 'b':
        """Return whether to include TX power in advertisement"""
        return self.include_tx_power

    @method()
    def Release(self):
        """Release advertisement (called by BlueZ when stopping)"""
        print("✗ Advertisement released")


class GATTApplication(ServiceInterface):
    """
    GATT Application that implements ObjectManager interface.

    Required by BlueZ for GATT service registration.
    """

    def __init__(self, bus: MessageBus):
        """
        Initialize GATT application.

        Args:
            bus: D-Bus message bus
        """
        self.path = '/com/focusdrive'
        self.bus = bus
        self.services = []

        super().__init__(DBUS_OM_IFACE)

    @method()
    def GetManagedObjects(self) -> 'a{oa{sa{sv}}}':
        """
        Return managed objects (services and characteristics).

        Required by ObjectManager interface.
        """
        response = {}

        for service in self.services:
            # Add service
            response[service.path] = {
                GATT_SERVICE_IFACE: {
                    'UUID': Variant('s', service.uuid),
                    'Primary': Variant('b', service.primary),
                    'Characteristics': Variant('ao', service.Characteristics)
                }
            }

            # Add characteristic
            char = service.characteristic
            response[char.path] = {
                GATT_CHARACTERISTIC_IFACE: {
                    'UUID': Variant('s', char.uuid),
                    'Service': Variant('o', char.service),
                    'Flags': Variant('as', char.flags),
                    'Notifying': Variant('b', char.notifying),
                    'Value': Variant('ay', char.value)
                }
            }

        return response

    def add_service(self, service):
        """Add a GATT service to this application"""
        self.services.append(service)


class BLEServer:
    """
    Main BLE GATT server for FocusDrive.

    Manages BlueZ D-Bus connection, GATT service registration,
    advertisement, and notification loop.
    """

    def __init__(self):
        """Initialize BLE server"""
        self.bus: Optional[MessageBus] = None
        self.application: Optional[GATTApplication] = None
        self.service: Optional[DriverStateService] = None
        self.advertisement: Optional[FocusDriveAdvertisement] = None
        self.running = False

    async def start(self):
        """
        Start BLE server and advertisement.

        Raises:
            RuntimeError: If Bluetooth adapter not found or registration fails
        """
        try:
            # Connect to system bus
            self.bus = await MessageBus(bus_type=BusType.SYSTEM).connect()

            # Get BlueZ adapter
            bluez_obj = self.bus.get_proxy_object(BLUEZ_SERVICE, ADAPTER_PATH, await self.bus.introspect(BLUEZ_SERVICE, ADAPTER_PATH))
            adapter = bluez_obj.get_interface('org.bluez.Adapter1')

            # Power on adapter if needed
            powered = await adapter.get_powered()
            if not powered:
                print("Powering on Bluetooth adapter...")
                await adapter.set_powered(True)

            # Create GATT application
            self.application = GATTApplication(self.bus)
            self.bus.export(self.application.path, self.application)

            # Create GATT service
            self.service = DriverStateService(self.bus, 0)

            # Add service to application
            self.application.add_service(self.service)

            # Export service and characteristic
            self.bus.export(self.service.path, self.service)
            self.bus.export(self.service.characteristic.path, self.service.characteristic)

            # Register GATT application
            gatt_manager = bluez_obj.get_interface(GATT_MANAGER_IFACE)
            await gatt_manager.call_register_application(
                self.application.path,
                {}
            )

            print("✓ GATT service registered")

            # Create and register advertisement
            self.advertisement = FocusDriveAdvertisement(self.bus, 0)
            self.bus.export(self.advertisement.path, self.advertisement)

            adv_manager = bluez_obj.get_interface(LE_ADVERTISING_MANAGER_IFACE)
            await adv_manager.call_register_advertisement(
                self.advertisement.path,
                {}
            )

            print("✓ Advertisement registered")
            self.running = True

        except Exception as e:
            raise RuntimeError(f"Failed to start BLE server: {e}")

    async def send_driver_state(self, state: str, reason: str, confidence: float):
        """
        Send driver state notification to connected clients.

        Args:
            state: "ATTENTIVE" or "DISTRACTED"
            reason: Distraction reason (e.g., "PHONE", "SAFE_DRIVING")
            confidence: 0.0 to 1.0
        """
        if self.service and self.service.characteristic:
            self.service.characteristic.send_notification(state, reason, confidence)

    async def run_test_loop(self):
        """
        Send fake test data every second for validation.

        Test sequence (cycles):
        1. ATTENTIVE|SAFE_DRIVING|0.95
        2. DISTRACTED|PHONE|0.91
        3. DISTRACTED|TEXTING|0.87
        4. DISTRACTED|EATING|0.82
        5. ATTENTIVE|SAFE_DRIVING|0.93
        """
        if not self.running:
            raise RuntimeError("BLE server not started")

        # Test data sequence
        test_data = [
            (STATE_ATTENTIVE, REASON_SAFE_DRIVING, 0.95),
            (STATE_DISTRACTED, REASON_PHONE, 0.91),
            (STATE_DISTRACTED, REASON_TEXTING, 0.87),
            (STATE_DISTRACTED, REASON_EATING, 0.82),
            (STATE_ATTENTIVE, REASON_SAFE_DRIVING, 0.93),
        ]

        index = 0
        while self.running:
            state, reason, confidence = test_data[index % len(test_data)]

            # Send notification
            await self.send_driver_state(state, reason, confidence)

            # Wait for next update
            await asyncio.sleep(UPDATE_INTERVAL)
            index += 1

    def stop(self):
        """Stop BLE server and unregister services"""
        self.running = False
        print("Stopping BLE server...")


# Example usage and testing
if __name__ == "__main__":
    import sys

    async def main():
        print("="*60)
        print("FocusDrive BLE GATT Server - Direct Test")
        print("="*60)
        print(f"Device Name: {DEVICE_NAME}")
        print(f"Service UUID: {SERVICE_UUID}")
        print(f"Characteristic UUID: {CHARACTERISTIC_UUID}")
        print("="*60)

        server = BLEServer()

        try:
            await server.start()
            print("\n✓ BLE server started successfully!")
            print("✓ Device is now discoverable as 'FocusDrive'")
            print("✓ Sending test notifications every 1 second")
            print("\nPress Ctrl+C to stop\n")

            await server.run_test_loop()

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            server.stop()
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    asyncio.run(main())
