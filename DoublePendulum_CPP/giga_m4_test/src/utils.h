//
// Created by Charles on 10/19/2024.
//

#ifndef UTILS_H
#define UTILS_H

constexpr int PENDULUM_PIN_A{ 6 };
constexpr int PENDULUM_PIN_B{ 7 };
constexpr int CART_PIN_A{ 3 };
constexpr int CART_PIN_B{ 4 };

struct Encoder
{
    volatile int val{ 0 };
    volatile int val_{ 0 };
    volatile int levA{ 0 };
    volatile int levB{ 0 };
    volatile int lastPin{ -1 };
};

// forward declaration of functions
void initEncoders();
int getPend1EncValDiff();
int getCartEncValDiff();
int getCartEncVal();
int getSimpAngleValue();

#endif //UTILS_H
