//
// Created by Charles on 8/26/2024.
//

#ifndef GIGA_M7_ENVIRONMENT_H
#define GIGA_M7_ENVIRONMENT_H

#include "globals.h"
#include "utils.h"

#include <Arduino.h>
#define _PWM_LOGLEVEL_ 1
#include "Portenta_H7_PWM.h"

constexpr float USUAL_SPEED{ 12.0f };  // max = 100.0%
constexpr int PENDULUM_PIN_A{ 2 };
constexpr int PENDULUM_PIN_B{ 3 };
constexpr int CART_PIN_A{ 5 };
constexpr int CART_PIN_B{ 6 };
constexpr int BUTTON_PIN{ 7 };
constexpr int MOTOR_PIN_A{ 22 };  // pins in this area have not been tested yet
constexpr int MOTOR_PIN_B{ 23 };
constexpr float MOTOR_FREQUENCY{ 20000.0f };

// for x and x_dot computation
constexpr float MAX_POS_VAL{ 13491.0f };
constexpr float POS_FACTOR{ 2.0f / MAX_POS_VAL };
constexpr float POS_BIAS{ -1.0f };
constexpr float POS_VELO_FACTOR{ 2.0f / (MAX_POS_VAL * DT) };
// for theta and theta_dot computation
constexpr float ANGLE_FACTOR{ PI / 300.0f };  // 2*pi / 300
constexpr float ANGLE_VELO_FACTOR{ PI / (300.0f * DT) };

struct Encoder
{
    volatile int val{ 0 };
    volatile int val_{ 0 };
    volatile int levA{ 0 };
    volatile int levB{ 0 };
    volatile int lastPin{ -1 };
};

float getAngleVelo();
float getAngle();
void rotate(const float& dutyCycle);

#endif //GIGA_M7_ENVIRONMENT_H
