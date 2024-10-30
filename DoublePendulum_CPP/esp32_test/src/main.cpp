#include <Arduino.h>

#include "globals.h"
#include "utils.h"

TaskHandle_t encoderTaskHandle;
hw_timer_t *timer0{ nullptr };
volatile int cntr0{ 0 };
volatile int cntr1{ 0 };

void IRAM_ATTR timer0_ISR()
{
    cntr0++;
}

void IRAM_ATTR timer1_ISR()
{
    cntr1++;
}

void printCntrs()
{
    Serial.print("cntr0: ");
    Serial.println(cntr0);
    Serial.print("cntr1: ");
    Serial.println(cntr1);
}

void setup() {
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

    greenBlink(4, 250);

    yellowBlinkNonblocking(6, 250);

    timer0 = timerBegin(0, preScaler, true);
    timerAttachInterrupt(timer0, &timer0_ISR, true);
    timerAlarmWrite(timer0, timerTicks, true);
    timerAlarmEnable(timer0);

    Serial.println("Starting");
    printCntrs();
    delay(500);

    // print again after and then disable
    Serial.println("Disabling");
    printCntrs();
    timerAlarmDisable(timer0);
    timerDetachInterrupt(timer0);
    timer0 = nullptr;

    Serial.println("Disabled there should be no changes");
    printCntrs();

    timer0 = timerBegin(1, preScaler, true);
    timerAttachInterrupt(timer0, &timer1_ISR, true);
    timerAlarmWrite(timer0, timerTicks, true);
    timerAlarmEnable(timer0);
    delay(500);

    Serial.println("cntr1 should increase now");
    printCntrs();

    timerAlarmDisable(timer0);
    timerDetachInterrupt(timer0);
    timer0 = nullptr;

    timer0 = timerBegin(0, preScaler, true);
    timerAttachInterrupt(timer0, &timer0_ISR, true);
    timerAlarmWrite(timer0, timerTicks, true);
    timerAlarmEnable(timer0);
    delay(500);

    Serial.println("cntr0 should increase now");
    printCntrs();
}

void loop() {
}