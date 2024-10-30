#include "Arduino.h"
#include "RPC.h"

#include "utils.h"

constexpr int LED_R{ 86 };
constexpr int LED_G{ 87 };
constexpr int LED_B{ 88 };
constexpr int BUTTON_PIN{ 29 };
volatile bool buttonPressedFlag{ false };

bool getButtonState()
{
    if (buttonPressedFlag)
    {
        buttonPressedFlag = false;
        return true;
    }
    return false;
}

void updateButton()
{
    buttonPressedFlag = true;
}

void setup()
{
    RPC.begin();
    Serial.begin(115200);

    RPC.bind("getPend1EncValDiff", getPend1EncValDiff);
    RPC.bind("getSimpAngleValue", getSimpAngleValue);
    RPC.bind("getCartEncVal", getCartEncVal);
    RPC.bind("getCartEncValDiff", getCartEncValDiff);

    initEncoders();

    pinMode(LED_R, OUTPUT);
    pinMode(LED_G, OUTPUT);
    pinMode(LED_B, OUTPUT);
    digitalWrite(LED_R, LOW);
    digitalWrite(LED_G, HIGH);
    digitalWrite(LED_B, HIGH);
    delay(500);
    pinMode(BUTTON_PIN, INPUT);
    attachInterrupt(BUTTON_PIN, updateButton, RISING);

    delay(100);
    RPC.println("Core M4 Initialized");
}

void loop() {
    if (getButtonState())
    {
        digitalWrite(LED_R, HIGH);
        digitalWrite(LED_G, LOW);
        digitalWrite(LED_B, HIGH);
        delay(100);
    }
    else
    {
        digitalWrite(LED_R, HIGH);
        digitalWrite(LED_G, HIGH);
        digitalWrite(LED_B, LOW);
        delay(100);
    }
}
