//
// Created by Charles on 8/1/2024.
//

#ifndef GIGA_M7_GLOBALS_H
#define GIGA_M7_GLOBALS_H

#define ARDUINO_PORTENTA_H7_M7  // ako nalang mu define kay mag sige error

// #define DEBUG_TESTING  // for the testing stuff in main.cpp
#define DEBUG  // TODO: no debug for BT_Comm not yet implemented
/*
 * Step 1: Define both precheck and no motor, check vals
 * Step 2:
 * Things to check: (for single)
 * - clockwise rotation should be positive vice versa
 * - going to the right should be positive
 * - positive value for motor should move to the right
 * Step 3: Undefine no motor, check vals and move pendulum if motor moves correctly
 * Step 4: Undefine precheck and maybe DEBUG, then run
 */
#define PRECHECK
// #define NO_MOTOR

// #define SINGLE_PENDULUM
#define DOUBLE_PENDULUM  // !! ALSO CHANGE IN M4 !!

#if defined(SINGLE_PENDULUM) && defined(DOUBLE_PENDULUM)
#error "Please define either SINGLE_PENDULUM or DOUBLE_PENDULUM"
#endif

constexpr float DT{ 0.01f };
constexpr int DT_MS{ static_cast<int>(DT * 1000.0f) };
constexpr int TOTAL_TIMESTEPS{ static_cast<int>(40.0f / DT) };  // duration [s] / deltaT

#endif //GIGA_M7_GLOBALS_H
