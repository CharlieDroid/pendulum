//
// Created by Charles on 8/1/2024.
//

#ifndef ESP32_GLOBALS_H
#define ESP32_GLOBALS_H

// #define DEBUG
//#define LATENCY_MEASUREMENT

#include <Arduino.h>

// divider = Prescaler, alarm_value = TimerTicks, APB_CLK = 80MHz default (use 10^6 when computing)
// formula: DT = TimerTicks * (Prescaler / APB_CLK)
// prescaler is usually 80 or 800 to simplify calculations must be between 0 and 65536, [0, 2] will always be 2
// TimerTicks = DT * (APB_CLK / Prescaler)
// TimerTicks = 5ms * (80MHz / 80)
constexpr float DT{ 0.01f };
constexpr int preScaler{ 80 };
constexpr int timerTicks{ static_cast<int>((DT * APB_CLK_FREQ) / preScaler) };

#endif //ESP32_GLOBALS_H
