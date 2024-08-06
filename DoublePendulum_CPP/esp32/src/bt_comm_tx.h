//
// Created by Charles on 8/1/2024.
//

#ifndef ESP32_BT_COMM_TX_H
#define ESP32_BT_COMM_TX_H

#include "globals.h"
#include "utils.h"

#include <Arduino.h>
#include <ArduinoBLE.h>

void btInit();
void transmitCentral();

#endif //ESP32_BT_COMM_TX_H
