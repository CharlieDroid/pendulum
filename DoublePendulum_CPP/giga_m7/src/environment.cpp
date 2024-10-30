//
// Created by Charles on 8/26/2024.
//
// Main change for this implementation compared to RPi is the dual-core usage.
// Interrupts for encoders are handled by M4 core for better performance.
//

#include "environment.h"

#define _PWM_LOGLEVEL_ 0
#include "Portenta_H7_PWM.h"

// Env timer is TIM4 and global timer is TIM8
Portenta_H7_Timer EnvTimer(TIM4);
mbed::PwmOut* motorA{ nullptr };
mbed::PwmOut* motorB{ nullptr };
static volatile bool stopFlag{ false };
float angle2{ 0.0f };
float angle2Velo{ 0.0f };

// ============[ INTERRUPT HANDLERS ]================
// Interrupt handlers for encoders are moved to M4 core

void buttonOff();  // forward declaration for button off function
void updateButton()
{
    stopFlag = true;
}

// ============[ INITIALIZATION ]================
// initEncoders moved to M4

void initMotor()
{
    setPWM(motorA, MOTOR_PIN_A, MOTOR_FREQUENCY, 0.0f);
    setPWM(motorB, MOTOR_PIN_B, MOTOR_FREQUENCY, 0.0f);

#ifdef DEBUG
    Serial.print("Set Motor Frequency: ");
    Serial.println(MOTOR_FREQUENCY);
#endif
}

void initButton()
{
    pinMode(BUTTON_PIN, INPUT_PULLDOWN);
}

void buttonOn()
{
    attachInterrupt(BUTTON_PIN, updateButton, RISING);
}

void initEnv()
{
    RPC.begin();  // init RPC for M4
    delay(500);  // wait for RPC to load, interferes with button interrupts if not so for some reason???
    initMotor();
    initButton();
    pinMode(STOP_PIN, INPUT_PULLUP);
}

// ============[ KILL ]================
void killMotor()
{
    setPWM_DCPercentage_manual(motorA, MOTOR_PIN_A, 0.0f);
    setPWM_DCPercentage_manual(motorB, MOTOR_PIN_B, 0.0f);
}

void buttonOff()
{
    detachInterrupt(BUTTON_PIN);
}

void killEnv()
{
    killMotor();
    buttonOff();
}

// ============[ ENVIRONMENT FUNCTIONS ]================
float getPos()
{ return RPC.call("getCartEncVal").as<float>() * POS_FACTOR + POS_BIAS; }

float getAngleVelo()
{ return RPC.call("getPend1EncValDiff").as<float>() * ANGLE_VELO_FACTOR; }

void rotate(const float& dutyCycle)  // from 0 to 100.0f
{
    if (dutyCycle > 0.0f)
    {
        // forward
        setPWM_DCPercentage_manual(motorB, MOTOR_PIN_B, 0.0f);
        setPWM_DCPercentage_manual(motorA, MOTOR_PIN_A, dutyCycle);
    }
    else
    {
        // backward
        setPWM_DCPercentage_manual(motorA, MOTOR_PIN_A, 0.0f);
        setPWM_DCPercentage_manual(motorB, MOTOR_PIN_B, -dutyCycle);
    }
}

#if defined(PRECHECK) && defined(DEBUG)
void printEncoderValues()
{
    /* Single Pendulum: x, theta, x_dot, theta_dot
     * Double Pendulum: pos, sin(a1), sin(a2), cos(a1), cos(a2), pos_vel, a1_vel, a2_vel
     * Things to check:
     * - clockwise rotation should be positive vice versa
     * - going to the right should be positive
     * - positive value for motor should move to the right
     */
    const float angle1{ RPC.call("getSimpAngleValue").as<float>() * ANGLE_FACTOR };
    const float posVelo{ RPC.call("getCartEncValDiff").as<float>() * POS_VELO_FACTOR };
    // actual observed information
#if defined(SINGLE_PENDULUM)
    Serial.print(getPos());
    Serial.print(",");
    Serial.print(angle1);
    Serial.print(",");
    Serial.print(posVelo);
    Serial.print(",");
    Serial.print(getAngleVelo());
#elif defined(DOUBLE_PENDULUM)
    getAngleAndVelocity(angle2, angle2Velo);  // get from bluetooth
    Serial.print(getPos());
    Serial.print(",");
    Serial.print(sin(angle1));
    Serial.print(",");
    Serial.print(sin(angle2));
    Serial.print(",");
    Serial.print(cos(angle1));
    Serial.print(",");
    Serial.print(cos(angle2));
    Serial.print(",");
    Serial.print(posVelo);
    Serial.print(",");
    Serial.print(getAngleVelo());
    Serial.print(",");
    Serial.print(angle2Velo);
#endif
    Serial.print("\t");
    Serial.print(RPC.call("getSimpAngleValue").as<int>());
    Serial.print(",");
    Serial.print(RPC.call("getCartEncVal").as<int>());
    Serial.print("\t");
#if defined(PRECHECK) && !defined(NO_MOTOR)
    // gets to 100% at 143 steps
    const float dutyCycle{ RPC.call("getSimpAngleValue").as<float>() * 0.7f };
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
    // When button has been pressed...
    RPC.call("resetCartEncVals");
    buttonOff();  // the order is like this to turn off button instantly and prevent debounce
    RPC.call("resetCartEncLvlsAndPin");
    rotate(-USUAL_SPEED);
    delay(500);  // wait for 0.5 seconds
    rotate(0.0f);
    stopFlag = false;

    // go to center-ish
    do { rotate(USUAL_SPEED * 1.2f); } while (getPos() < -0.05f);
    rotate(0);
}

volatile bool doAngleVelo{ true };

void resetPendulumHandler()
{
    doAngleVelo = true;
}

void resetPendulum1()
{
    // reset pendulum
    // create a timer with interval in us
    EnvTimer.attachInterruptInterval(DT_MS * 1000, resetPendulumHandler);

    // wait until it finishes resetting the pendulum
    while (!stopFlag)
    {
        static int cntr{ 0 };
        static constexpr int cntrMax{ static_cast<int>(1.2f / DT) };

        if (doAngleVelo)
        {
            // if no velocity then add 1 to counter
            (getAngleVelo() < 0.001) ? cntr++ : cntr = 0;

            // if it has been zero velocity for 1.2 seconds then reset
            stopFlag = (cntr > cntrMax);

            if (stopFlag)
            {
                cntr = 0;
                EnvTimer.stopTimer();
                EnvTimer.detachInterrupt();
            }

            doAngleVelo = false;
        }
    }

    stopFlag = false;
    RPC.call("resetPend1EncData");
}

#ifdef DOUBLE_PENDULUM
void resetPendulum2()
{
    setReset(true);
    while (getReset()) { delay(20); }
}
#endif

void step(const float& mu)  // mu is -1 to 1
{
    // scale it to 100%
    rotate(mu * 100.0f);
}

void updateObservation(ObservationVector& observation)
{
    // Single Pendulum: x, theta, x_dot, theta_dot
    // Double Pendulum: pos, sin(a1), sin(a2), cos(a1), cos(a2), pos_vel, a1_vel, a2_vel
#if defined(SINGLE_PENDULUM)
    observation(0) = getPos();
    observation(1) = RPC.call("getSimpAngleValue").as<float>() * ANGLE_FACTOR;
    observation(2) = RPC.call("getCartEncValDiff").as<float>() * POS_VELO_FACTOR;
    observation(3) = getAngleVelo();
#elif defined(DOUBLE_PENDULUM)
    const float angle1{ RPC.call("getSimpAngleValue").as<float>() * ANGLE_FACTOR };
    getAngleAndVelocity(angle2, angle2Velo);
    observation(0) = getPos();
    observation(1) = sin(angle1);
    observation(2) = sin(angle2);
    observation(3) = cos(angle1);
    observation(4) = cos(angle2);
    observation(5) = RPC.call("getCartEncValDiff").as<float>() * POS_VELO_FACTOR;
    observation(6) = getAngleVelo();
    observation(7) = angle2Velo;
#endif
}

void reset(ObservationVector& observation)
{
#ifdef DEBUG
    Serial.println("...resetting...");
#endif
    resetCart();
    resetPendulum1();
#ifdef DOUBLE_PENDULUM
    resetPendulum2();
#endif
    RPC.call("resetCartEncVelocity");
    updateObservation(observation);
}
