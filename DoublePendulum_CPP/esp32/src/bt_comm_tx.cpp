//
// Created by Charles on 8/1/2024.
//

#include "bt_comm_tx.h"

const char* deviceServiceUuid = "19b10000-e8f2-537e-4f6c-d104768a1214";
const char* deviceServiceCharacteristicUuid = "19b10001-e8f2-537e-4f6c-d104768a1214";

BLEService angleVelocityService(deviceServiceUuid);
BLECharacteristic angleVelocityCharacteristic(deviceServiceCharacteristicUuid,
                                              BLERead | BLEWriteWithoutResponse | BLENotify,
                                              8);
union AngleData {
    float angleVelocity[2];
    uint8_t bytes[8];
};

void btInit()
{
#ifdef DEBUG

    Serial.begin(9600);
    while (!Serial);

    if (!BLE.begin()) {
        Serial.println("- Starting Bluetooth® Low Energy module failed!");
        while (1);
    }

    BLE.setLocalName("Arduino Nano ESP32 (Peripheral)");
    BLE.setAdvertisedService(angleVelocityService);
    angleVelocityService.addCharacteristic(angleVelocityCharacteristic);
    BLE.addService(angleVelocityService);

    AngleData angleData = {{-1.0f, -1.0f}};
    angleVelocityCharacteristic.writeValue(angleData.bytes, 8);
    BLE.advertise();

    Serial.println("Nano ESP32 (Peripheral Device)");
    Serial.println(" ");

#else

    if (!BLE.begin())
    {
        while (1);
    }

    BLE.setConnectionInterval(0x004, 0x008);

    BLE.setLocalName("Arduino Nano ESP32 (Peripheral)");
    BLE.setAdvertisedService(angleVelocityService);
    angleVelocityService.addCharacteristic(angleVelocityCharacteristic);
    BLE.addService(angleVelocityService);

    AngleData angleData = {{-1.0f, -1.0f}};
    angleVelocityCharacteristic.writeValue(angleData.bytes, 8);
    BLE.advertise();

#endif
}

void transmitCentral()
{
    BLEDevice central = BLE.central();
    redBlink(2, 250);

    if (central)
    {
        greenBlink(4, 250);
        AngleData receivedAngleData;
        AngleData sendAngleData = {{-1.0f, 1.0f}};
        while (central.connected())
        {
            angleVelocityCharacteristic.readValue(receivedAngleData.bytes, 8);
            if (receivedAngleData.angleVelocity[0] > 0.0f)
            {
                angleVelocityCharacteristic.writeValue(sendAngleData.bytes, 8);
            }
        }
    }
}
