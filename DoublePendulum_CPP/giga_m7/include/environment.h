//
// Created by Charles on 8/26/2024.
//

#ifndef GIGA_M7_ENVIRONMENT_H
#define GIGA_M7_ENVIRONMENT_H

#include <Arduino.h>
#include "RPC.h"
#define _TIMERINTERRUPT_LOGLEVEL_ 0
#include "Portenta_H7_TimerInterrupt.h"

#include "globals.h"
#include "agent.h"
#include "utils.h"
#include "bt_comm_rx.h"


constexpr float USUAL_SPEED{ 13.0f };  // max = 100.0%
// PENDULUM AND CART PINS MOVED TO M4
constexpr int STOP_PIN{ 46 };
constexpr int BUTTON_PIN{ 4 };  // tested
constexpr int MOTOR_PIN_A{ 2 };  // tested and these are good PWM pins
constexpr int MOTOR_PIN_B{ 5 };
constexpr float MOTOR_FREQUENCY{ 20000.0f };

// for x and x_dot computation
constexpr float MAX_POS_VAL{ 13491.0f };
constexpr float POS_FACTOR{ 2.0f / MAX_POS_VAL };
constexpr float POS_BIAS{ -1.0f };
constexpr float POS_VELO_FACTOR{ 2.0f / (MAX_POS_VAL * DT) };
// for theta and theta_dot computation
constexpr float ANGLE_FACTOR{ PI / 300.0f };  // 2*pi / 300
#if defined(SINGLE_PENDULUM)
constexpr float ANGLE_VELO_FACTOR{ PI / (300.0f * DT) };
#elif defined(DOUBLE_PENDULUM)
constexpr float ANGLE_VELO_FACTOR{ PI / (300.0f * DT * 30.0f) };
#endif

void initEnv();
void killEnv();
void updateObservation(ObservationVector& observation);
void rotate(const float& dutyCycle);
void step(const float& mu);
void reset(ObservationVector& observation);

#if defined(PRECHECK) && defined(DEBUG)
void printEncoderValues();
bool getButtonState();
#endif

#endif //GIGA_M7_ENVIRONMENT_H
