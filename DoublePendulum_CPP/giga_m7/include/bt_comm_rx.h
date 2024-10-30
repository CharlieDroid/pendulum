//
// Created by Charles on 8/1/2024.
//

#ifndef GIGA_M7_BT_COMM_RX_H
#define GIGA_M7_BT_COMM_RX_H

#include <ArduinoBLE.h>

#include "utils.h"

extern BLEDevice peripheral;
extern BLECharacteristic angleVelocityCharacteristic;
extern BLECharacteristic resetEncoderCharacteristic;

void btInit();
int connectPeripheral();
int checkCharacteristics();
void getAngleAndVelocity(float& angle, float& angleVelo);
bool getReset();
void setReset(const bool& reset);

#endif //GIGA_M7_BT_COMM_RX_H
