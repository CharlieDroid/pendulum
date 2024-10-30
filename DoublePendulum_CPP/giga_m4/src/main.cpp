#include "Arduino.h"
#include "RPC.h"

#include "utils.h"

#ifdef DOUBLE_PENDULUM
// 0 = down, 1 = up
int pendulumCommand1{ 1 };
int pendulumCommand2{ 1 };
#endif

void setup()
{
    RPC.begin();
    // Serial.begin(115200);  // not sure if this is useful ?? but it is in the tutorial

    RPC.bind("resetPend1EncData", resetPend1EncData);
    RPC.bind("resetCartEncVals", resetCartEncVals);
    RPC.bind("resetCartEncLvlsAndPin", resetCartEncLvlsAndPin);
    RPC.bind("resetCartEncVelocity", resetCartEncVelocity);
    RPC.bind("killEncoders", killEncoders);

    RPC.bind("getPend1EncValDiff", getPend1EncValDiff);
    RPC.bind("getSimpAngleValue", getSimpAngleValue);
    RPC.bind("getCartEncVal", getCartEncVal);
    RPC.bind("getCartEncValDiff", getCartEncValDiff);

#ifdef DOUBLE_PENDULUM
    RPC.bind("getPendulumCommand", getPendulumCommand);

    initControlPins();
#endif

    initEncoders();

    delay(500);
    RPC.println("Core M4 Initialized");
    delay(100);
}

void loop()
{
    pendulumCommand1 = digitalRead(PENDULUM_1_CTRL_PIN);
    pendulumCommand2 = digitalRead(PENDULUM_2_CTRL_PIN);
    delay(1);
}
