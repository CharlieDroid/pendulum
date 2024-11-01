//
// Created by Charles on 8/1/2024.
//
#include "bt_comm_rx.h"

const char* deviceServiceUuid = "dec1be10-9063-4b54-ae16-24f2bf72a4c6";
const char* angleVelocityCharacteristicUuid = "dec1be11-9063-4b54-ae16-24f2bf72a4c6";
const char* resetEncoderCharacteristicUuid = "dec1be12-9063-4b54-ae16-24f2bf72a4c6";

BLEDevice peripheral;
BLECharacteristic angleVelocityCharacteristic;
BLECharacteristic resetEncoderCharacteristic;

#ifdef LATENCY_MEASUREMENT
#include <array>

constexpr int SAMPLES{ 10 };
std::array<unsigned long, SAMPLES> latencies = {};
#endif

union AngleData {
    float angleVelocity[2];
    uint8_t bytes[8];
};

union ResetData {
    bool reset;
    uint8_t bytes[1];
};


#ifdef LATENCY_MEASUREMENT
void measureLatency(BLEDevice& peripheral, BLECharacteristic& angleVelocityCharacteristic)
{
    AngleData sendAngleData = {{1.0f, -1.0f}};
    AngleData defaultAngleData = {{-1.0f, -1.0f}};
    AngleData receivedAngleData = defaultAngleData;
    while (peripheral.connected())
    {
        for (int i{ 0 }; i < SAMPLES; i++)
        {
            Serial.print("* Ping #");
            Serial.print(i);
            Serial.println(" ...");
            greenBlink(1, 100);
            unsigned long currentTime{ micros() };
            angleVelocityCharacteristic.writeValue(sendAngleData.bytes, 8);
            angleVelocityCharacteristic.readValue(receivedAngleData.bytes, 8);
            if (receivedAngleData.angleVelocity[1] > 0.0f)
            {
                unsigned long latency{ micros() - currentTime };
                latencies[i] = latency;

                angleVelocityCharacteristic.writeValue(defaultAngleData.bytes, 8);
            }
        }
        // disconnect after measurements
        peripheral.disconnect();
    }
}

void printStats()
{
    // Calculate minimum and maximum
    unsigned long minLatency = latencies[0];
    unsigned long maxLatency = latencies[0];
    for (int i = 1; i < SAMPLES; i++) {
        if (latencies[i] < minLatency) {
            minLatency = latencies[i];
        }
        if (latencies[i] > maxLatency) {
            maxLatency = latencies[i];
        }
    }

    // Calculate average
    unsigned long sum = 0;
    for (int i = 0; i < SAMPLES; i++) {
        sum += latencies[i];
    }
    float averageLatency = (float)sum / SAMPLES;

    // Calculate standard deviation
    float sumSquares = 0;
    for (int i = 0; i < SAMPLES; i++) {
        sumSquares += (latencies[i] - averageLatency) * (latencies[i] - averageLatency);
    }
    float stdDevLatency = sqrt((float)sumSquares / SAMPLES);

    // Print statistics
    Serial.print("Minimum latency: ");
    Serial.print(minLatency, 3);
    Serial.println(" us");
    Serial.print("Maximum latency: ");
    Serial.print(maxLatency / 1000.0, 3);
    Serial.println(" ms");
    Serial.print("Average latency: ");
    Serial.print(averageLatency, 3);
    Serial.println(" us");
    Serial.print("Standard deviation: ");
    Serial.print(stdDevLatency / 1000.0, 3);
    Serial.println(" ms");
}
#endif

int checkCharacteristics()
{
#ifdef DEBUG
    angleVelocityCharacteristic = peripheral.characteristic(angleVelocityCharacteristicUuid);
    resetEncoderCharacteristic = peripheral.characteristic(resetEncoderCharacteristicUuid);

    if (!angleVelocityCharacteristic)
    {
        Serial.println("* Peripheral device does not have characteristic!");
        redBlink(3, 100);
        peripheral.disconnect();
        return 0;
    }
    if (!angleVelocityCharacteristic.canRead())
    {
        Serial.println("* Peripheral does not have a readable characteristic!");
        redBlink(3, 100);
        peripheral.disconnect();
        return 0;
    }
    if (!angleVelocityCharacteristic.canSubscribe())
    {
        Serial.println("* Peripheral does not have a subscribable characteristic!");
        redBlink(3, 100);
        peripheral.disconnect();
        return 0;
    }

    if (!resetEncoderCharacteristic)
    {
        Serial.println("* Peripheral device does not have characteristic!");
        redBlink(3, 100);
        peripheral.disconnect();
        return 0;
    }
    if (!resetEncoderCharacteristic.canRead())
    {
        Serial.println("* Peripheral does not have a readable characteristic!");
        redBlink(3, 100);
        peripheral.disconnect();
        return 0;
    }

    angleVelocityCharacteristic.subscribe();
    greenBlink(4, 250);
    return 1;  // 1 = OK
#else

    BLECharacteristic valueCharacteristic = peripheral.characteristic(deviceServiceCharacteristicUuid);

    if (!valueCharacteristic)
    {
        redBlink(3, 100);
        peripheral.disconnect();
        return;
    }
    else if (!valueCharacteristic.canWrite())
    {
        redBlink(3, 100);
        peripheral.disconnect();
        return;
    }

    greenBlink(3, 250);
    while (peripheral.connected())
    {
        value = valueCharacteristic.read();
        if (valueCharacteristic.written())
        {
            value = valueCharacteristic.read();
        }
    }

#endif
}

void getAngleAndVelocity(float& angle, float& angleVelo)
{
    AngleData receivedAngleData{};
    angleVelocityCharacteristic.readValue(receivedAngleData.bytes, 8);
    angle = receivedAngleData.angleVelocity[0];
    angleVelo = receivedAngleData.angleVelocity[1];
}

bool getReset()
{
    ResetData receivedResetData{};
    resetEncoderCharacteristic.readValue(receivedResetData.bytes, 1);
    return receivedResetData.reset;
}

void setReset(const bool& reset)
{
    const ResetData resetData{ reset };
    resetEncoderCharacteristic.writeValue(resetData.bytes, 1);
}

int connectPeripheral()
{
#ifdef DEBUG

    Serial.println("- Discovering peripheral device...");

    do
    {
        redBlink(2, 250);
        BLE.scanForUuid(deviceServiceUuid);
        peripheral = BLE.available();
    } while (!peripheral);

    if (peripheral)
    {
        blueBlink(3, 100);
        Serial.println("* Peripheral device found!");
        Serial.print("* Device MAC address: ");
        Serial.println(peripheral.address());
        Serial.print("* Device name: ");
        Serial.println(peripheral.localName());
        Serial.print("* Advertised service UUID: ");
        Serial.println(peripheral.advertisedServiceUuid());
        Serial.println(" ");

        BLE.stopScan();

        Serial.println("- Connecting to peripheral device...");

        if (peripheral.connect())
        {
            Serial.println("* Connected to peripheral device!");
            Serial.println(" ");
        }
        else
        {
            Serial.println("* Connection to peripheral device failed!");
            Serial.println(" ");
            redBlink(5, 100);
            return 0;
        }

        Serial.println("- Discovering peripheral device attributes...");

        if (peripheral.discoverAttributes())
        {
            Serial.println("* Peripheral device attributes discovered!");
            Serial.println(" ");
        }
        else
        {
            Serial.println("* Peripheral device attributes discovery failed!");
            Serial.println(" ");
            redBlink(3, 100);
            peripheral.disconnect();
            return 0;
        }

        return 1;  // 1 = OK
    }

#else

    BLEDevice peripheral;

    do
    {
        redBlink(2, 250);
        BLE.scanForUuid(deviceServiceUuid);
        peripheral = BLE.available();
    } while (!peripheral);

    if (peripheral)
    {
        blueBlink(3, 100);

        BLE.stopScan();
        if (!peripheral.connect())
        {
            redBlink(5, 100);
            return;
        }

        if (!peripheral.discoverAttributes())
        {
            redBlink(3, 100);
            peripheral.disconnect();
            return;
        }

        readPeripheral(peripheral);
    }

#endif
}

void btInit()
{
#ifdef DEBUG

    if (!BLE.begin())
    {
        Serial.println("* Starting Bluetooth® Low Energy module failed!");
        while (true);
    }

    BLE.setConnectionInterval(0x004, 0x008);

    BLE.setLocalName("Giga R1 (Central)");
    BLE.advertise();

    Serial.println("Arduino Giga R1 (Central Device)");
    Serial.println(" ");

#else

    if (!BLE.begin()) while (true);

    BLE.setConnectionInterval(0x004, 0x008);

    BLE.setLocalName("Giga R1 (Central)");
    BLE.advertise();

#endif
}
