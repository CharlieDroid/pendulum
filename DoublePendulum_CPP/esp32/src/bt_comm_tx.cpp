//
// Created by Charles on 8/1/2024.
//

#include "bt_comm_tx.h"

const char* deviceServiceUuid = "dec1be10-9063-4b54-ae16-24f2bf72a4c6";
const char* angleVelocityCharacteristicUuid = "dec1be11-9063-4b54-ae16-24f2bf72a4c6";
const char* resetEncoderCharacteristicUuid = "dec1be12-9063-4b54-ae16-24f2bf72a4c6";

BLEService angleVelocityService(deviceServiceUuid);
BLECharacteristic angleVelocityCharacteristic(angleVelocityCharacteristicUuid,
                                              BLERead | BLEWriteWithoutResponse | BLENotify,
                                              8);
BLECharacteristic resetEncoderCharacteristic(resetEncoderCharacteristicUuid,
                                             BLERead | BLEWrite,  // Adjust properties as needed
                                             1);

union AngleData {
    float angleVelocity[2];
    uint8_t bytes[8];
};

union ResetData {
    bool reset;
    uint8_t bytes[1];
};

static float currAngle{ 0.0f };
static float currAngleVelo{ 0.0f };

void btInit()
{
#ifdef DEBUG
    Serial.println("Bluetooth beginning");
#endif

    if (!BLE.begin()) { while (true) { redBlink(2, 250); } }

#ifdef DEBUG
    Serial.println("Bluetooth initialized");
#endif

    BLE.setConnectionInterval(0x004, 0x008);

    BLE.setLocalName("Nano ESP32 (Peripheral)");
    BLE.setAdvertisedService(angleVelocityService);
    angleVelocityService.addCharacteristic(angleVelocityCharacteristic);
    angleVelocityService.addCharacteristic(resetEncoderCharacteristic);
    BLE.addService(angleVelocityService);

#ifdef LATENCY_MEASUREMENT
    AngleData angleData = {{-1.0f, -1.0f}};
#else
    constexpr AngleData angleData = {{0.0f, 0.0f}};
    constexpr ResetData resetData = { false };
#endif
    angleVelocityCharacteristic.writeValue(angleData.bytes, 8);
    resetEncoderCharacteristic.writeValue(resetData.bytes, 1);
    BLE.advertise();
}

void pingPong(const BLEDevice& central)
{
    AngleData receivedAngleData{};
    constexpr AngleData sendAngleData = {{-1.0f, 1.0f}};
    while (central.connected())
    {
        angleVelocityCharacteristic.readValue(receivedAngleData.bytes, 8);
        if (receivedAngleData.angleVelocity[0] > 0.0f)
        {
            angleVelocityCharacteristic.writeValue(sendAngleData.bytes, 8);
        }
    }
}

void sendEncoderMeasurements()
{
    const AngleData sendAngleData = {{currAngle, currAngleVelo}};
    angleVelocityCharacteristic.writeValue(sendAngleData.bytes, 8);
}

void IRAM_ATTR updateEncoderMeasurements()
{
    currAngle = getAngle();
    currAngleVelo = getAngleVelo();
}

#ifdef DEBUG
void getAngleAndVelocity(float& angle, float& angleVelo)
{
    AngleData angleData{};
    angleVelocityCharacteristic.readValue(angleData.bytes, 8);
    angle = angleData.angleVelocity[0];
    angleVelo = angleData.angleVelocity[1];
}
#endif

bool getIsReset()
{
    ResetData resetData{};
    resetEncoderCharacteristic.readValue(resetData.bytes, 1);
    return resetData.reset;
}

void setReset(const bool& reset)
{
    const ResetData resetData{ reset };
    resetEncoderCharacteristic.writeValue(resetData.bytes, 1);
}

BLEDevice getCentral()
{
    while (true)
    {
        BLEDevice central = BLE.central();
        redBlink(2, 250);

        if (central)
        {
            greenBlink(4, 250);
            return central;
        }
    }
}


