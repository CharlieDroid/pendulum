//
// Created by Charles on 10/26/2024.
//

#ifndef GLOBALS_H
#define GLOBALS_H

#include <Arduino.h>

#define DEBUG

constexpr float DT{ 0.01f };
constexpr int preScaler{ 80 };
constexpr int timerTicks{ static_cast<int>((DT * APB_CLK_FREQ) / preScaler) };

#endif //GLOBALS_H
