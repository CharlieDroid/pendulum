//
// Created by Charles on 8/26/2024.
//

#include "environment.h"

mbed::PwmOut* motorA{ nullptr };
mbed::PwmOut* motorB{ nullptr };
static volatile Encoder pend1Enc{};
static volatile Encoder cartEnc{};
static volatile bool stopFlag{ false };

// ============[ INTERRUPT HANDLERS ]================
void updatePendulumEncoderA()
{
    pend1Enc.levA = digitalRead(PENDULUM_PIN_A);

    if (PENDULUM_PIN_A != pend1Enc.lastPin)
    {
        pend1Enc.lastPin = PENDULUM_PIN_A;
        if (pend1Enc.levA)
        {
            if (pend1Enc.levB) pend1Enc.val++;
        }
    }
}

void updatePendulumEncoderB()
{
    pend1Enc.levB = digitalRead(PENDULUM_PIN_B);

    if (PENDULUM_PIN_B != pend1Enc.lastPin)
    {
        pend1Enc.lastPin = PENDULUM_PIN_B;
        if (pend1Enc.levB)
        {
            if (pend1Enc.levA) pend1Enc.val--;
        }
    }
}

void updateCartEncoderA()
{
    cartEnc.levA = digitalRead(CART_PIN_A);

    if (CART_PIN_A != cartEnc.lastPin)
    {
        cartEnc.lastPin = CART_PIN_A;
        if (cartEnc.levA)
        {
            if (cartEnc.levB) cartEnc.val++;
        }
    }
}

void updateCartEncoderB()
{
    cartEnc.levB = digitalRead(CART_PIN_B);

    if (CART_PIN_B != cartEnc.lastPin)
    {
        cartEnc.lastPin = CART_PIN_B;
        if (cartEnc.levB)
        {
            if (cartEnc.levA) cartEnc.val--;
        }
    }
}

void buttonOff();  // forward declaration for function
void updateButton()
{
    cartEnc.val = 0;
    cartEnc.val_ = 0;
    buttonOff();
    cartEnc.levA = 0;
    cartEnc.levB = 0;
    cartEnc.lastPin = -1;
    rotate(-USUAL_SPEED);
    delay(500);
    rotate(0.0f);
    stopFlag = true;
}

// ============[ INITIALIZATION ]================
void initEncoders()
{
    pinMode(PENDULUM_PIN_A, INPUT_PULLUP);
    pinMode(PENDULUM_PIN_B, INPUT_PULLUP);
    pinMode(CART_PIN_A, INPUT_PULLUP);
    pinMode(CART_PIN_B, INPUT_PULLUP);

    // TODO: Double check if this should have digitalPinToInterrupt()
    attachInterrupt(PENDULUM_PIN_A, updatePendulumEncoderA, CHANGE);
    attachInterrupt(PENDULUM_PIN_B, updatePendulumEncoderB, CHANGE);
    attachInterrupt(CART_PIN_A, updateCartEncoderA, CHANGE);
    attachInterrupt(CART_PIN_B, updateCartEncoderB, CHANGE);
}

void initMotor()
{
    setPWM(motorA, MOTOR_PIN_A, MOTOR_FREQUENCY, 0.0f);
    setPWM(motorB, MOTOR_PIN_B, MOTOR_FREQUENCY, 0.0f);

#ifdef DEBUG
    Serial.print("Motor Frequency: ");
    Serial.println(MOTOR_FREQUENCY);
#endif
}

void initButton()
{
    pinMode(BUTTON_PIN, INPUT_PULLUP);
}

void buttonOn()
{
    // not sure if RISING or FALLING
    attachInterrupt(BUTTON_PIN, updateButton, RISING);
}

void initEnv()
{
//#warning "Encoder initialized inside main core, M7"
    initEncoders();
    initMotor();
    initButton();
}

// ============[ KILL ]================
void killMotor()
{
    setPWM(motorA, MOTOR_PIN_A, MOTOR_FREQUENCY, 0.0f);
    setPWM(motorB, MOTOR_PIN_B, MOTOR_FREQUENCY, 0.0f);
}

void killEncoders()
{
    detachInterrupt(PENDULUM_PIN_A);
    detachInterrupt(PENDULUM_PIN_B);
    detachInterrupt(CART_PIN_A);
    detachInterrupt(CART_PIN_B);
}

void buttonOff()
{
    detachInterrupt(BUTTON_PIN);
}

void killEnv()
{
    killMotor();
    killEncoders();
    buttonOff();
}

// ============[ ENVIRONMENT FUNCTIONS ]================
float getPos()
{ return static_cast<float>(cartEnc.val) * POS_FACTOR + POS_BIAS; }

float getAngleVelo()
{
    float velo{ static_cast<float>(pend1Enc.val - pend1Enc.val_) * ANGLE_VELO_FACTOR };
    pend1Enc.val_ = pend1Enc.val;
    return velo;
}

float getSimpAngleValue()
{
    // ((val + 300) % 600) - 300
    return static_cast<float>(((pend1Enc.val + 300) % 600) - 300);
}

void rotate(const float& dutyCycle)
{
    if (dutyCycle > 0.0f)
    {
        // forward
        setPWM(motorB, MOTOR_PIN_B, MOTOR_FREQUENCY, 0.0f);
        setPWM(motorA, MOTOR_PIN_A, MOTOR_FREQUENCY, dutyCycle);
    }
    else
    {
        // backward
        setPWM(motorA, MOTOR_PIN_A, MOTOR_FREQUENCY, 0.0f);
        setPWM(motorB, MOTOR_PIN_B, MOTOR_FREQUENCY, -dutyCycle);
    }
}

#if defined(PRECHECK) && defined(DEBUG)
void printEncoderValues()
{
    /* Things to check:
     * - clockwise rotation should be positive vice versa
     * - going to the right should be positive
     * - positive value for motor should move to the right
     */
    const float angle{ getSimpAngleValue() * ANGLE_FACTOR }
    const float posVelo{ static_cast<float>(cartEnc.val - cartEnc.val_) * POS_VELO_FACTOR };
    cartEnc.val_ = cartEnc.val;
    Serial.print(getPos());
    Serial.print(",");
    Serial.print(angle);
    Serial.print(",");
    Serial.print(posVelo);
    Serial.print(",");
    Serial.print(getAngleVelo());
    Serial.print("\t");
    Serial.print(pend1Enc.val);
    Serial.print(",");
    Serial.print(cartEnc.val);
    Serial.print("\t");
#if defined(PRECHECK) && !defined(NO_MOTOR)
    const float dutyCycle{ static_cast<int>(pend1Enc.val * (5.0f / 3.0f)) };
    rotate(dutyCycle);
    Serial.print("Duty Cycle: ");
    Serial.print(dutyCycle);
#endif
    Serial.println();
}
#endif

void resetCart()
{
    // go to leftmost side and reset
    buttonOn();
    rotate(-USUAL_SPEED);
    // checks if button has not been pressed for 40 seconds
    const unsigned long timeBefore{ millis() };
    while (!stopFlag)
    {
        if ((millis() - timeBefore) > 40000)
        {
#ifdef DEBUG
            Serial.println("Limit has not been found for more than 40 seconds\n.");
#endif
            rotate(0);
            while (true) redBlink(4, 250);
        }
    }
    stopFlag = false;

    // go to center-ish
    do { rotate(USUAL_SPEED); } while (getPos() < -0.05f);
    rotate(0);
}

void resetPendulum()
{
    // reset pendulum
    // create a timer
}
