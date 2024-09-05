//
// Created by Charles on 8/1/2024.
//

#ifndef ESP32_BT_COMM_TX_H
#define ESP32_BT_COMM_TX_H

#include "globals.h"
#include "utils.h"

#include <ArduinoBLE.h>

void btInit();
BLEDevice getCentral();
void pingPong(const BLEDevice& central);
void sendEncoderMeasurements();
void IRAM_ATTR updateEncoderMeasurements();
bool getIsReset();
void setReset(const bool& reset);
#ifdef DEBUG
void getAngleAndVelocity(float& angle, float& angleVelo);
#endif

#endif //ESP32_BT_COMM_TX_H
