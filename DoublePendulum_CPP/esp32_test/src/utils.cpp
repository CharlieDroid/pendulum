//
// Created by Charles on 8/1/2024.
//

#include "utils.h"

// ============[ ENCODER FUNCTIONS ]================
static volatile Encoder pend2Enc{};
// NOTE TO SELF: never use port mux shit with bluetooth stuff

void IRAM_ATTR updateEncoderA()
{
    pend2Enc.levA = digitalRead(encoderPinA);

    if (encoderPinA != pend2Enc.lastPin)
    {
        pend2Enc.lastPin = encoderPinA;
        if (pend2Enc.levA)
        {
            if (pend2Enc.levB) pend2Enc.val++;
        }
    }
}

void IRAM_ATTR updateEncoderB()
{
    pend2Enc.levB = digitalRead(encoderPinB);

    if (encoderPinB != pend2Enc.lastPin)
    {
        pend2Enc.lastPin = encoderPinB;
        if (pend2Enc.levB)
        {
            if (pend2Enc.levA) pend2Enc.val--;
        }
    }
}

[[noreturn]] void encInit(void* pvParameters)
{
    pinMode(encoderPinA, INPUT_PULLUP);
    pinMode(encoderPinB, INPUT_PULLUP);

    attachInterrupt(encoderPinA, updateEncoderA, CHANGE);
    attachInterrupt(encoderPinB, updateEncoderB, CHANGE);

#ifdef DEBUG
    Serial.print("encInit() running on core ");
    Serial.println(xPortGetCoreID());
    Serial.println("Running infinite loop in here");
#endif
    while (true) delay(100);
}

float getAngleVelo()
{
    const float velo{ static_cast<float>(pend2Enc.val - pend2Enc.val_) * ANGLE_VELO_FACTOR };
    pend2Enc.val_ = pend2Enc.val;
    return velo;
}

// int mod(const int& value, const int& divisor) { return ((value % divisor) + divisor) % divisor; }

float getAngle()
{
    return static_cast<float>(pend2Enc.val) * ANGLE_FACTOR;
}

void resetPendVals()
{
    // reset to zero since this is second pendulum
    pend2Enc.val = 0;
    pend2Enc.val_ = 0;
    pend2Enc.levA = 0;
    pend2Enc.levB = 0;
    pend2Enc.lastPin = -1;
}

// ============[ LED FUNCTIONS ]================
constexpr int LED_R{ 14 };
constexpr int LED_G{ 15 };
constexpr int LED_B{ 16 };

void ledInit()
{
    pinMode(LED_R, OUTPUT);
    pinMode(LED_G, OUTPUT);
    pinMode(LED_B, OUTPUT);
}

void yellowBlinkNonBlocking(const int& times, const int& delayTime)
{
    unsigned long ledTimeNow{ millis() };
    bool isLedOn{ false };
    int counter{ 0 };
    while (true)
    {
        if (!isLedOn && (millis() - ledTimeNow > (delayTime / 2)))
        {
            digitalWrite(LED_R, LOW);
            digitalWrite(LED_G, LOW);
            digitalWrite(LED_B, HIGH);
            isLedOn = true;
            ledTimeNow = millis();
        }
        else if (isLedOn && (millis() - ledTimeNow > (delayTime / 2)))
        {
            digitalWrite(LED_R, HIGH);
            digitalWrite(LED_G, HIGH);
            digitalWrite(LED_B, HIGH);
            isLedOn = false;
            ledTimeNow = millis();
            if (++counter == times) break;
        }
    }
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
