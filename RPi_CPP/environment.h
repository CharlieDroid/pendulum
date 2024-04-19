//
// Created by Charles on 2/3/2024.
//

#ifndef RPI_CPP_ENVIRONMENT_H
#define RPI_CPP_ENVIRONMENT_H

#include <boost/math/constants/constants.hpp>
#include <Eigen/Dense>

// environment variables
constexpr int USUAL_SPEED{ 100 };  // minimum PWM duty cycle is about 90 / 1000 = 9%
constexpr float PHYSICAL_BOUND{ 0.95f };
//constexpr float REWARD_BOUND{ 0.9f };
constexpr float TWO_PI{ boost::math::constants::pi<float>() * 2.0f };  // compute 2*pi before runtime
// pins
constexpr int PENDULUM_PIN_A{ 5 };
constexpr int PENDULUM_PIN_B{ 6 };
constexpr int CART_PIN_A{ 19 };
constexpr int CART_PIN_B{ 26 };
constexpr int MOTOR_PIN_A{ 20 };
constexpr int MOTOR_PIN_B{ 21 };
constexpr int MOTOR_FREQUENCY{ 20000 };
constexpr int MOTOR_RANGE { 1000 };
constexpr int BUTTON_PIN{ 12 };
// for x and x_dot computation
constexpr float MAX_POS_VAL{ 13491.0f };
constexpr float POS_FACTOR{ 2.0f / MAX_POS_VAL };
constexpr float POS_BIAS{ -1.0f };
constexpr float POS_VELO_FACTOR{ 2.0f / (MAX_POS_VAL * DT) };
// for theta and theta_dot computation
constexpr float ANGLE_FACTOR{ boost::math::constants::pi<float>() / 299.5f };  // 2*pi / 599
constexpr float ANGLE_VELO_FACTOR{ boost::math::constants::pi<float>() / (299.5f * DT) };


struct Encoder
{
    volatile int val{ 0 };
    volatile int val_{ 0 };
    volatile int levA{ 0 };
    volatile int levB{ 0 };
    volatile int lastPin{ -1 };
};

void initEnv();
void killEnv();
void rotate(int dutyCycle);
Eigen::VectorXf reset();
void step(float mu);
Eigen::VectorXf getObservation();

#ifdef PRECHECK
void printEncoderValues();
#endif

#endif //RPI_CPP_ENVIRONMENT_H
