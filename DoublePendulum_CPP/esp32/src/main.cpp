#include "utils.h"
#include "bt_comm_tx.h"

#include <Arduino.h>

TaskHandle_t encoderTaskHandle;
hw_timer_t *timer0{ nullptr };
static volatile bool stopResetFlag{ false };
static bool isTimerMeasureAttached{ false };
unsigned long currTime{ micros() };
constexpr int DT_HALF_MICROS{ static_cast<int>(0.5f * DT * 1e6) };
#ifdef DEBUG
unsigned long timeNow{ millis() };
#endif

void stop_timer0()
{
    timerAlarmDisable(timer0);
    timerDetachInterrupt(timer0);
    timer0 = nullptr;
}

void IRAM_ATTR timer0_reset()
{
    static int cntr{ 0 };
    static constexpr int cntrMax{ static_cast<int>(1.2f / DT) };
    // check if pendulum has zero velocity
    (getAngleVelo() < 0.001) ? cntr++ : cntr = 0;

    // if it has been zero velocity for 1.2 seconds then stop timer interrupt and reset
    stopResetFlag = (cntr > cntrMax);
}

void IRAM_ATTR timer0_measure()
{
    updateEncoderMeasurements();
}

void setup()
{
#ifdef DEBUG
    Serial.begin(9600);
    while (!Serial) {}
    delay(100);
    Serial.print("Timer ticks: ");
    Serial.println(timerTicks);
    Serial.print("setup() running on core ");
    Serial.println(xPortGetCoreID());
#endif
    ledInit();
    blueBlink(5, 100);  // initializing

    // put encoder task in core 0
    xTaskCreatePinnedToCore(
            encInit,
            "encoderTask",
            10000,
            nullptr,
            1,
            &encoderTaskHandle,
            0);

    btInit();
    greenBlink(4, 250);
}
void loop()
{
#ifdef DEBUG
    if (micros() - currTime > DT_HALF_MICROS)
    {
        sendEncoderMeasurements();
        currTime = micros();
    }
    if (millis() - timeNow > 100)
    {
        // Serial.print(getAngle());
        // Serial.print(",");
        // Serial.println(getAngleVelo());
        float angle{};
        float angleVelo{};
        getAngleAndVelocity(angle, angleVelo);
        Serial.print(angle);
        Serial.print(",");
        Serial.println(angleVelo);
        timeNow = millis();
    }
    if (!getIsReset())
    {
        if (!isTimerMeasureAttached)
        {
            // attach timer interrupt for normal measurement
            timer0 = timerBegin(0, preScaler, true);
            timerAttachInterrupt(timer0, &timer0_measure, true);
            timerAlarmWrite(timer0, timerTicks, true);
            timerAlarmEnable(timer0);
            isTimerMeasureAttached = true;
        }
    }
    else
    {
        blueBlink(5, 100);

        // detach timer for measure
        if (isTimerMeasureAttached)
        {
            stop_timer0();
            isTimerMeasureAttached = false;
        }

        // attach timer interrupt for reset
        stopResetFlag = false;
        timer0 = timerBegin(0, preScaler, true);
        timerAttachInterrupt(timer0, &timer0_reset, true);
        timerAlarmWrite(timer0, timerTicks, true);
        timerAlarmEnable(timer0);
        while (!stopResetFlag) {}
        // detach timer interrupt for reset
        stop_timer0();
        // then reset pendulum enc values
        resetPendVals();

        // set reset back to false
        setReset(false);
        greenBlink(5, 100);
    }
#else
    BLEDevice central{ getCentral() };
#ifdef LATENCY_MEASUREMENT
    pingPong(central);
#else
    while (central.connected())
    {
        // update bluetooth encoder measurements every half of DT
        if (micros() - currTime > DT_HALF_MICROS)
        {
            sendEncoderMeasurements();
            currTime = micros();
        }
        // if not reset attach measure
        if (!getIsReset())
        {
            if (!isTimerMeasureAttached)
            {
                // attach timer interrupt for normal measurement
                timer0 = timerBegin(0, preScaler, true);
                timerAttachInterrupt(timer0, &timer0_measure, true);
                timerAlarmWrite(timer0, timerTicks, true);
                timerAlarmEnable(timer0);
                isTimerMeasureAttached = true;
            }
        }
        else
        {
            blueBlink(5, 100);

            // detach timer for measure
            if (isTimerMeasureAttached)
            {
                stop_timer0();
                isTimerMeasureAttached = false;
            }

            // attach timer interrupt for reset
            stopResetFlag = false;
            timer0 = timerBegin(0, preScaler, true);
            timerAttachInterrupt(timer0, &timer0_reset, true);
            timerAlarmWrite(timer0, timerTicks, true);
            timerAlarmEnable(timer0);
            while (!stopResetFlag) {}
            // detach timer interrupt for reset
            stop_timer0();
            // then reset pendulum enc values
            resetPendVals();

            // set reset back to false
            setReset(false);
            greenBlink(5, 100);
        }
    }
#endif
#endif
}
