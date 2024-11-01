//
// Created by Charles on 8/1/2024.
//

#ifndef GIGA_M7_UTILS_H
#define GIGA_M7_UTILS_H

#include "globals.h"
#include <Arduino.h>

constexpr int encoderPinA{ 2 };
constexpr int encoderPinB{ 3 };

// for theta and theta_dot computation
constexpr float ANGLE_FACTOR{ PI / 300.0f };  // 2*pi / 599
constexpr float ANGLE_VELO_FACTOR{ PI / (300.0f * DT * 30.0f ) };

struct Encoder
{
    volatile long val{ 0 };
    volatile long val_{ 0 };
    volatile int levA{ 0 };
    volatile int levB{ 0 };
    volatile int lastPin{ -1 };
};

float getAngleVelo();
float getAngle();
[[noreturn]] void encInit(void* pvParameters);
void resetPendVals();

void redBlink(const int& times, const int& delayTime);
void greenBlink(const int& times, const int& delayTime);
void blueBlink(const int& times, const int& delayTime);
void yellowBlinkNonBlocking(const int& times, const int& delayTime);
void magentaBlinkNonBlocking(const int& times, const int& delayTime);
void rgbLED(const int& red, const int& green, const int& blue);
void ledInit();

#endif //GIGA_M7_UTILS_H
