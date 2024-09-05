#include "utils.h"
#include "bt_comm_rx.h"
#include "agent.h"

#include <Arduino.h>
#include <array>
#include <ArduinoEigenDense.h>

//#define _TIMERINTERRUPT_LOGLEVEL_ 4

// Can be included as many times as necessary, without `Multiple Definitions` Linker Error
//#include "Portenta_H7_TimerInterrupt.h"

void timerHandler()
{
    static bool toggle{ false };
    (toggle) ? rgbLED(255, 255, 0) : rgbLED(255, 255, 255);
    toggle = !toggle;
}
//Portenta_H7_Timer ITimer0(TIM15);

Agent actor3s{ initActor() };
Agent actor3b{ initActor() };
Agent actor2{ initActor() };
Agent actor1{ initActor() };
Agent actor0{ initActor() };
constexpr int ITERATIONS{ 1000 };
std::array<unsigned long, ITERATIONS> executionTimes = {};
using namespace Eigen;

void setup()
{
    ledInit();
    Serial.begin(9600);
    while (!Serial) {}
    blueBlink(5, 100);
    Serial.println("Starting");

    btInit();
    // last val is action
    // 0.034260649 -0.361462331 0.927125206	0.932386714	0.37475172	-0.765315233 -1.371879705 19.35203646 1 1 -0.602274238 -0.927081764
//    static Eigen::Matrix4d foo = (Eigen::Matrix4d() << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16).finished();
//    static Vector2f input = (Vector2f() << 0.034260649, -0.361462331, 0.927125206, 0.932386714, 0.37475172, -0.765315233, -1.371879705, 19.35203646, 1, 1).finished();
    InputVector input{ 0.034260649, -0.361462331, 0.927125206, 0.932386714, 0.37475172, -0.765315233, -1.371879705, 19.35203646, 1, 1 };

//    if (ITimer0.attachInterruptInterval(5000 * 1000, timerHandler))
//    {
//        Serial.print(F("Starting ITimer0 OK, millis() = "));
//        delay(50);
//        Serial.println(millis());
//    }
//    else
//        Serial.println(F("Can't set ITimer0. Select another freq. or timer"));

//    timerAttachInterrupt(TIMER1, timerHandler, 1000);

//     unsigned long currentTime;
//     for (int i{ 0 }; i < ITERATIONS; i++)
//     {
//         if (i % 100 == 0) Serial.println(i);
// //        actor3 = initActor();
//         currentTime = micros();
//         float a{ feedForward(actor2, input) };
//         executionTimes[i] = micros() - currentTime;
//     }
//
//     // Calculate minimum and maximum
//     unsigned long minLatency = executionTimes[0];
//     unsigned long maxLatency = executionTimes[0];
//     for (int i = 1; i < ITERATIONS; i++) {
//         if (executionTimes[i] < minLatency) {
//             minLatency = executionTimes[i];
//         }
//         if (executionTimes[i] > maxLatency) {
//             maxLatency = executionTimes[i];
//         }
//     }
//
//     // Calculate average
//     unsigned long sum = 0;
//     for (int i = 0; i < ITERATIONS; i++) {
//         sum += executionTimes[i];
//     }
//     float averageLatency = (float)sum / ITERATIONS;
//
//     // Calculate standard deviation
//     float sumSquares = 0;
//     for (int i = 0; i < ITERATIONS; i++) {
//         sumSquares += (executionTimes[i] - averageLatency) * (executionTimes[i] - averageLatency);
//     }
//     float stdDevLatency = sqrt((float)sumSquares / ITERATIONS);
//
//     // Print statistics
//     Serial.print("Minimum Execution Time: ");
//     Serial.print(minLatency / 1000.0, 3);
//     Serial.println(" ms");
//     Serial.print("Maximum Execution Time: ");
//     Serial.print(maxLatency / 1000.0, 3);
//     Serial.println(" ms");
//     Serial.print("Average Execution Time: ");
//     Serial.print(averageLatency / 1000.0, 3);
//     Serial.println(" ms");
//     Serial.print("Standard deviation: ");
//     Serial.print(stdDevLatency, 3);
//     Serial.println(" us");
}

void loop()
{
//    for (int i{ 0 }; i < ((int)(100.0f / 5.0f) + 1); i++)
//    {
//        dutycyclePercent = (float)i * 5.0f;
//        if (dutycyclePercent > 100.0f) dutycyclePercent = 100.0f;
//        setPWM_DCPercentage_manual(pwm, myPin, dutycyclePercent);
//        Serial.println(dutycyclePercent);
//        (dutycyclePercent > 85.0f) ? delay(1000) : delay(100);
//    }

    connectPeripheral();
}
