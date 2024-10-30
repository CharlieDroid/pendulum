//
// Created by Charles on 8/1/2024.
//

#ifndef GIGA_M7_UTILS_H
#define GIGA_M7_UTILS_H

#include <Arduino.h>

#include "globals.h"

void redBlink(const int& times, const int& delayTime);
void greenBlink(const int& times, const int& delayTime);
void blueBlink(const int& times, const int& delayTime);
void rgbLED(const int& red, const int& green, const int& blue);
void ledInit();

#endif //GIGA_M7_UTILS_H
