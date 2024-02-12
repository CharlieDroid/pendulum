//
// Created by Charles on 2/3/2024.
// I avoid using classes as it avoids overhead and improves speed
// I might have repeated code
//
#include "globals.h"
#include "environment.h"
#include "agent.h"
#include <Eigen/Dense>
#include <pigpio.h>
#include <boost/asio.hpp>
#include <iostream>
#include <chrono>

using namespace boost;
static volatile Encoder pendEnc{};
static volatile Encoder cartEnc{};
static bool stopFlag{ false };

// ============[ EXTERNAL INTERRUPT HANDLERS ]================
void pulsePendulumExternal(int gpio, int level, [[maybe_unused]] uint32_t tick, [[maybe_unused]] void *user)  // pulse external from C
{
    (gpio == PENDULUM_PIN_A) ? pendEnc.levA = level : pendEnc.levB = level;

    if (gpio != pendEnc.lastPin)
    {
        pendEnc.lastPin = gpio;

        if ((gpio == PENDULUM_PIN_A) && (level == 1))
        {
            if (pendEnc.levB) pendEnc.val++;
        }
        else if ((gpio == PENDULUM_PIN_B) && (level == 1))
        {
            if (pendEnc.levA) pendEnc.val--;
        }
    }
}

void pulseCartExternal(int gpio, int level, [[maybe_unused]] uint32_t tick, [[maybe_unused]] void *user)  // pulse external from C
{
    (gpio == CART_PIN_A) ? cartEnc.levA = level : cartEnc.levB = level;

    if (gpio != cartEnc.lastPin)  // debounce
    {
        cartEnc.lastPin = gpio;

        if ((gpio == CART_PIN_A) && (level == 1))
        {
            if (cartEnc.levB) cartEnc.val++;
        }
        else if ((gpio == CART_PIN_B) && (level == 1))
        {
            if (cartEnc.levA) cartEnc.val--;
        }
    }
}

void buttonOff();  // forward declaration for func
void pulseButtonExternal(int gpio, int level, [[maybe_unused]] uint32_t tick, [[maybe_unused]] void *user)  // pulse external from C
{
    if ((gpio == BUTTON_PIN && level == 1))
    {
        cartEnc.val = 0;
        cartEnc.val_ = 0;
        buttonOff();  // the order is like this to turn off button instantly and prevent debounce
        cartEnc.levA = 0;
        cartEnc.levB = 0;
        cartEnc.lastPin = -1;
        rotate(-USUAL_SPEED);
        gpioSleep(PI_TIME_RELATIVE, 0, 500000); // sleep for 0.5 seconds
        rotate(0);
        stopFlag = true;
    }
}

// ============[ INITIALIZATION ]================
void initEncoders()
{
    // Setup GPIO pins
    // pendulum
    gpioSetMode(PENDULUM_PIN_A, PI_INPUT);
    gpioSetMode(PENDULUM_PIN_B, PI_INPUT);
    gpioSetPullUpDown(PENDULUM_PIN_A, PI_PUD_UP);
    gpioSetPullUpDown(PENDULUM_PIN_B, PI_PUD_UP);
    // cart
    gpioSetMode(CART_PIN_A, PI_INPUT);
    gpioSetMode(CART_PIN_B, PI_INPUT);
    gpioSetPullUpDown(CART_PIN_A, PI_PUD_UP);
    gpioSetPullUpDown(CART_PIN_B, PI_PUD_UP);

    // Set interrupt handlers for each pin
    // pendulum
    gpioSetAlertFuncEx(PENDULUM_PIN_A, pulsePendulumExternal, nullptr);
    gpioSetAlertFuncEx(PENDULUM_PIN_B, pulsePendulumExternal, nullptr);
    // cart
    gpioSetAlertFuncEx(CART_PIN_A, pulseCartExternal, nullptr);
    gpioSetAlertFuncEx(CART_PIN_B, pulseCartExternal, nullptr);
}

void initMotor()
{
    gpioSetMode(MOTOR_PIN_A, PI_OUTPUT);
    gpioSetPWMfrequency(MOTOR_PIN_A, MOTOR_FREQUENCY);
    gpioSetPWMrange(MOTOR_PIN_A, MOTOR_RANGE);

    gpioSetMode(MOTOR_PIN_B, PI_OUTPUT);
    gpioSetPWMfrequency(MOTOR_PIN_B, MOTOR_FREQUENCY);
    gpioSetPWMrange(MOTOR_PIN_B, MOTOR_RANGE);

    std::cout << "Frequency: " << gpioGetPWMfrequency(MOTOR_PIN_A) << "\n";
}

void initButton()
{
    gpioSetMode(BUTTON_PIN, PI_INPUT);
    gpioSetPullUpDown(BUTTON_PIN, PI_PUD_UP);
}

void buttonOn()
{
    gpioSetAlertFuncEx(BUTTON_PIN, pulseButtonExternal, nullptr);
}

void initEnv()
{
    initEncoders();
    initMotor();
    initButton();
}

// ============[ KILL ]================
void killMotor()
{
    gpioPWM(MOTOR_PIN_A, 0);
    gpioPWM(MOTOR_PIN_B, 0);
}

void killEncoders()
{
    // Cancel the interrupts for the GPIO pins associated with the encoder
    gpioSetAlertFuncEx(PENDULUM_PIN_A, nullptr, nullptr);
    gpioSetAlertFuncEx(PENDULUM_PIN_B, nullptr, nullptr);
    gpioSetAlertFuncEx(CART_PIN_A, nullptr, nullptr);
    gpioSetAlertFuncEx(CART_PIN_B, nullptr, nullptr);
}


void buttonOff()
{
    gpioSetAlertFuncEx(BUTTON_PIN, nullptr, nullptr);
}

void killEnv()
{
    rotate(0);
    killMotor();
    killEncoders();
    buttonOff();
}

// ============[ ENVIRONMENT FUNCTIONS ]================
float getPos()
{ return static_cast<float>(cartEnc.val) * POS_FACTOR + POS_BIAS; }

float getAngleVelo()
{
    float velo{ static_cast<float>(pendEnc.val - pendEnc.val_) * ANGLE_VELO_FACTOR };
    pendEnc.val_ = pendEnc.val;
    return velo;
}

float simpAngle(float angle)
{
    if (angle > math::constants::pi<float>()) return simpAngle(angle - TWO_PI);
    else if (angle < -math::constants::pi<float>()) return simpAngle(angle + TWO_PI);
    return angle;
}

// upon testing the gpioGetPWMdutycycle is just the same with the function parameter
void rotate(int dutyCycle)
{
    if (dutyCycle > 0)
    {
        // forward
        gpioPWM(MOTOR_PIN_B, 0);
        gpioPWM(MOTOR_PIN_A, dutyCycle);
    }
    else
    {
        // backward
        gpioPWM(MOTOR_PIN_A, 0);
        gpioPWM(MOTOR_PIN_B, -dutyCycle);
    }
}

#ifdef PRECHECK
void printEncoderValues()
{
    /* Things to check:
     * - clockwise rotation should be positive vice versa
     * - going to the right should be positive
     * - positive value for motor should move to the right
     */
    float angle{ simpAngle(static_cast<float>(pendEnc.val) * ANGLE_FACTOR) };
    float posVelo{ static_cast<float>(cartEnc.val - cartEnc.val_) * POS_VELO_FACTOR };
    cartEnc.val_ = cartEnc.val;
    std::cout << getPos() << "," << angle << "," << posVelo << "," <<  getAngleVelo() << "\t";
    std::cout << pendEnc.val << "," << cartEnc.val << "\t";
#ifdef PRECHECK_NO_MOTOR
    const int dutyCycle{ static_cast<int>(pendEnc.val * (5.0f / 3.0f)) };
    rotate(dutyCycle);
    std::cout << "Duty Cycle: " << dutyCycle;
#endif
    std::cout << "\n";
}
#endif

void resetCart()
{
    // go to leftmost side and reset
    buttonOn();
    rotate(-USUAL_SPEED);
    // checks if button has not been pressed for 40 secs
    using ChronoTime = std::chrono::time_point<std::chrono::steady_clock>;
    ChronoTime timeBefore{ std::chrono::steady_clock::now() };
    std::chrono::seconds timeDiff{};
    while (!stopFlag)
    {
        timeDiff = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timeBefore);
        if (timeDiff.count() > 40)
        {
            std::cerr << "Limit has not been found for more than 40 seconds\n";
            rotate(0);
            std::exit(-1);
        }
    }
    stopFlag = false;

    // go to center-ish
    do { rotate(static_cast<int>(USUAL_SPEED * 1.2f)); } while (getPos() < -0.05f);
    rotate(0);
}

void doEvery(const system::error_code& ec)
{
    static int cntr{ 0 };
    static constexpr int cntrMax{ static_cast<int>(1.2f / DT) };
    // check if pendulum has zero velocity
    (getAngleVelo() < 0.001) ? cntr++ : cntr = 0;

    // if it has been zero velocity for 1.2 seconds then reset
    stopFlag = static_cast<bool>(cntr > cntrMax);

    if (!stopFlag)
    {
        // Create a new timer with the same interval (20ms)
        timer.expires_after(std::chrono::milliseconds(DT_MS));
        timer.async_wait(&doEvery);
    } else  // stop loop
    { cntr = 0; io.stop(); }
}

void resetPendulum()
{
    // reset pendulum
    // Create a timer with an initial interval of 20ms
    timer.expires_after(std::chrono::milliseconds(DT_MS));
    timer.async_wait(&doEvery);
    // Run episode
    io.run();

    work.reset();
    io.restart();

    // wait until it finishes resetting the pendulum
    stopFlag = false;
    pendEnc.val = 300;
    pendEnc.val_ = 300;
    pendEnc.levA = 0;
    pendEnc.levB = 0;
    pendEnc.lastPin = -1;
}

Eigen::VectorXf getObservation()
{
    Eigen::VectorXf observation(INPUT_SIZE);
    // x, theta, x_dot, theta_dot
    observation(0) = getPos();
    observation(1) = simpAngle(static_cast<float>(pendEnc.val) * ANGLE_FACTOR);
    observation(2) = static_cast<float>(cartEnc.val - cartEnc.val_) * POS_VELO_FACTOR;
    cartEnc.val_ = cartEnc.val;
    observation(3) = getAngleVelo();
    return observation;
}

Eigen::VectorXf reset()
{
    std::cout << "...resetting...\n";
    resetCart();
    resetPendulum();
    cartEnc.val_ = cartEnc.val;  // resets velocity to zero too
    return getObservation();
}

void step(float mu)  // receives mu which is -1 to 1
{
    int action{ static_cast<int>(mu * static_cast<float>(MOTOR_RANGE)) };
    rotate(action);
}


