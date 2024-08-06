#include "utils.h"
#include "bt_comm_tx.h"

#include <Arduino.h>

void setup()
{
    btInit();
    ledInit();
}

void loop()
{
    transmitCentral();
}
