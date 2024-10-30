#include <Arduino.h>
#include <ArduinoEigenDense.h>

#include "globals.h"
#include "environment.h"
#include "utils.h"
#include "agents.h"

#ifdef DOUBLE_PENDULUM
#include "bt_comm_rx.h"

bool pendulum2Connected{ false };
#endif

Portenta_H7_Timer GlobalTimer(TIM8);
static volatile int timesteps{ 0 };
static volatile bool runningFlag{ false };
static volatile bool doStep{ false };
ObservationVector observation{};

void episodeEnd()
{
    rotate(0.0f);
    timesteps = 0;
    GlobalTimer.stopTimer();
    GlobalTimer.detachInterrupt();
    killEnv();
    runningFlag = false;
}

#if defined(DEBUG) && defined(PRECHECK)
static volatile bool printCurrState{ false };

void episodeHandler()
{
    printCurrState = true;

    timesteps++;
    if (timesteps > (2 * TOTAL_TIMESTEPS)) episodeEnd();
}
#elif !defined(PRECHECK)
void episodeHandler()
{
    doStep = true;

    timesteps++;
    if (timesteps > TOTAL_TIMESTEPS) episodeEnd();
}
#endif

void setup()
{
    ledInit();
    blueBlink(5, 100);
#ifdef DEBUG
    Serial.begin(9600);
    while (!Serial) {}
    Serial.println("Starting");
#endif
    initEnv();
    rotate(0.0f);  // make sure motor is not moving
    greenBlink(5, 100);

#ifdef DEBUG_TESTING
    // Sample Input for Single Pendulum, Last Val is Action
    // 0.021824982916313303,0.9158823284749271,1.2344219540678782,2.7968668538338433     -0.8724172115325928
    // Sample Input for Double Pendulum
    // -0.29252750822761436, 0.9375753648773371, -0.25744433535020583, 0.3477821662695318, 0.9662931305748224, -2.34774561341996, 1.029626542789722, -2.2365320510569275,   0.8682127594947815
    // 0.29232077165530646,0.9230528327915285,0.309718989993921,-0.384673196201054,-0.9508281375922494,-1.9212146699016848,0.07388261938254895,-0.49556366232830096,       -0.4947131276130676

    observation(0) = 0.29232077165530646;
    observation(1) = 0.9230528327915285;
    observation(2) = 0.309718989993921;
    observation(3) = -0.384673196201054;
    observation(4) = -0.9508281375922494;
    observation(5) = -1.9212146699016848;
    observation(6) = 0.07388261938254895;
    observation(7) = -0.49556366232830096;

    const float mu0{ feedForward(agents[0], observation) };
    Serial.print("Mu0: ");
    Serial.print(mu0, 6);
    Serial.print("\t");

    const float mu1{ feedForward(agents[1], observation) };
    Serial.print("Mu1: ");
    Serial.print(mu1, 6);
    Serial.print("\t");

    const float mu2{ feedForward(agents[2], observation) };
    Serial.print("Mu2: ");
    Serial.print(mu2, 6);
    Serial.print("\t");

    const float mu3{ feedForward(agents[3], observation) };
    Serial.print("Mu3: ");
    Serial.print(mu3, 6);
    Serial.println();

    while (true) { blueBlink(1, 500); }
#endif

#ifdef DOUBLE_PENDULUM
    btInit();
    if (connectPeripheral() && checkCharacteristics())
        { pendulum2Connected = true; }
    else
        { while (true) redBlink(5, 1000); }
#endif

#ifndef PRECHECK
    // initialize environment and get init obs
    reset(observation);
#if defined(SINGLE_PENDULUM)
    const float mu{ feedForward(agents[0], observation) };
#elif defined(DOUBLE_PENDULUM)
    const float mu{ feedForward(agents[RPC.call("getPendulumCommand").as<int>()], observation) };
#endif
    step(mu);
#endif

    // start timer and step through the environment
    runningFlag = true;
    GlobalTimer.attachInterruptInterval(DT_MS * 1000, episodeHandler);
}

void loop()
{
#ifndef PRECHECK
    if (doStep)
    {
        doStep = false;
        updateObservation(observation);
#if defined(SINGLE_PENDULUM)
        const float mu{ feedForward(agents[0], observation) };
#elif defined(DOUBLE_PENDULUM)
        const float mu{ feedForward(agents[RPC.call("getPendulumCommand").as<int>()], observation) };
#endif
        step(mu);
    }
#endif

    // kill everything if disconnected or stopped
    // 0 = Stop, 1 = Not clicked
#if defined(SINGLE_PENDULUM)
    if (!digitalRead(STOP_PIN))
    {
#elif defined(DOUBLE_PENDULUM)
    if (!digitalRead(STOP_PIN) || (!peripheral.connected() && pendulum2Connected))
    {
        angleVelocityCharacteristic.unsubscribe();
        pendulum2Connected = false;
#endif
        timesteps = 2 * TOTAL_TIMESTEPS + 1;
        killEnv();
    }

#ifdef DEBUG
    if (!runningFlag)
    {
        Serial.println("Shit ended");
        delay(1000);
    }
#endif

#if defined(DEBUG) && defined(PRECHECK)
    if (printCurrState)
    {
        printCurrState = false;
        printEncoderValues();
    }
#endif
}
