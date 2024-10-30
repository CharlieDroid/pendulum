//
// Created by Charles on 10/19/2024.
//
#include "Arduino.h"

#include "utils.h"

static volatile Encoder pend1Enc{};
static volatile Encoder cartEnc{};


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

void initEncoders()
{
    pinMode(PENDULUM_PIN_A, INPUT_PULLUP);
    pinMode(PENDULUM_PIN_B, INPUT_PULLUP);
    pinMode(CART_PIN_A, INPUT_PULLUP);
    pinMode(CART_PIN_B, INPUT_PULLUP);

    attachInterrupt(CART_PIN_A, updateCartEncoderA, CHANGE);
    attachInterrupt(CART_PIN_B, updateCartEncoderB, CHANGE);
    attachInterrupt(PENDULUM_PIN_A, updatePendulumEncoderA, CHANGE);
    attachInterrupt(PENDULUM_PIN_B, updatePendulumEncoderB, CHANGE);
}


// ============[ GET VALUES AND UTILS ]================
// true modulo like python
int mod(const int& value, const int& divisor) { return ((value % divisor) + divisor) % divisor; }

// SAMPLE CODE IN M7 CORE
// RPC.call("getPend1EncValDiff").as<float>()
int getPend1EncValDiff()
{
    const float dummy{ static_cast<float>(pend1Enc.val - pend1Enc.val_) };
    pend1Enc.val_ = pend1Enc.val;
    return dummy;
}

int getCartEncValDiff()
{
    const float dummy{ static_cast<float>(cartEnc.val - cartEnc.val_) };
    cartEnc.val_ = cartEnc.val;
    return dummy;
}

int getCartEncVal()
{
    return cartEnc.val;
}

int getSimpAngleValue()
{
    // ((val + 300) % 600) - 300 in Python
    return mod(pend1Enc.val + 300, 600) - 300;
}
