//
// Created by Charles on 8/1/2024.
//

#include "utils.h"

constexpr int LED_R{ 14 };
constexpr int LED_G{ 15 };
constexpr int LED_B{ 16 };

void ledInit()
{
    pinMode(LED_R, OUTPUT);
    pinMode(LED_G, OUTPUT);
    pinMode(LED_B, OUTPUT);
}

void redBlink(const int& times, const int& delayTime)
{
    for (int i{ 0 }; i < times; i++)
    {
        digitalWrite(LED_R, LOW);
        digitalWrite(LED_G, HIGH);
        digitalWrite(LED_B, HIGH);
        delay(static_cast<int>(delayTime / 2));
        digitalWrite(LED_R, HIGH);
        digitalWrite(LED_G, HIGH);
        digitalWrite(LED_B, HIGH);
        delay(static_cast<int>(delayTime / 2));
    }
}

void greenBlink(const int& times, const int& delayTime)
{
    for (int i{ 0 }; i < times; i++)
    {
        digitalWrite(LED_R, HIGH);
        digitalWrite(LED_G, LOW);
        digitalWrite(LED_B, HIGH);
        delay(static_cast<int>(delayTime / 2));
        digitalWrite(LED_R, HIGH);
        digitalWrite(LED_G, HIGH);
        digitalWrite(LED_B, HIGH);
        delay(static_cast<int>(delayTime / 2));
    }
}

void blueBlink(const int& times, const int& delayTime)
{
    for (int i{ 0 }; i < times; i++)
    {
        digitalWrite(LED_R, HIGH);
        digitalWrite(LED_G, HIGH);
        digitalWrite(LED_B, LOW);
        delay(static_cast<int>(delayTime / 2));
        digitalWrite(LED_R, HIGH);
        digitalWrite(LED_G, HIGH);
        digitalWrite(LED_B, HIGH);
        delay(static_cast<int>(delayTime / 2));
    }
}

void rgbLED(const int& red, const int& green, const int& blue)  // 0 = ON, 255 = OFF
{
    analogWrite(LED_R, red);
    analogWrite(LED_G, green);
    analogWrite(LED_B, blue);
}
