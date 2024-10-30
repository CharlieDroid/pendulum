//
// Created by Charles on 10/19/2024.
//

#ifndef UTILS_H
#define UTILS_H

#define DOUBLE_PENDULUM

// pins are all tested below
constexpr int PENDULUM_PIN_A{ 8 };
constexpr int PENDULUM_PIN_B{ 9 };
constexpr int CART_PIN_A{ 6 };
constexpr int CART_PIN_B{ 7 };

#ifdef DOUBLE_PENDULUM
constexpr int PENDULUM_1_CTRL_PIN{ 52 };
constexpr int PENDULUM_2_CTRL_PIN{ 50 };

extern int pendulumCommand1;
extern int pendulumCommand2;

void initControlPins();
int getPendulumCommand();
#endif

struct Encoder
{
    volatile int val{ 0 };
    volatile int val_{ 0 };
    volatile int levA{ 0 };
    volatile int levB{ 0 };
    volatile int lastPin{ -1 };
};

// forward declaration of functions
void resetPend1EncData();
void resetCartEncVals();
void resetCartEncLvlsAndPin();
void resetCartEncVelocity();
void killEncoders();

void initEncoders();
int getPend1EncValDiff();
int getCartEncValDiff();
int getCartEncVal();
int getSimpAngleValue();

#endif //UTILS_H
