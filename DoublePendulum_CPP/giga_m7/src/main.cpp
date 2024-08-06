#include "utils.h"
#include "bt_comm_rx.h"

#include <Arduino.h>

void setup()
{
    btInit();
    ledInit();
}

void loop()
{
    connectPeripheral();
}
