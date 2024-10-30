#include "Arduino.h"
#include "RPC.h"

constexpr int START_PIN{ 48 };
constexpr int STOP_PIN{ 46 };

void setup() {
  RPC.begin();
  Serial.begin(9600);
  while (!Serial) {}
  Serial.println("Beginning");

  pinMode(START_PIN, INPUT_PULLUP);
  pinMode(STOP_PIN, INPUT_PULLUP);
}

void loop()
{
  Serial.print("Pendulum Command: ");
  Serial.print(RPC.call("getPendulumCommand").as<int>());
  Serial.print("\tStop Pin: ");
  Serial.print(digitalRead(STOP_PIN));
  Serial.print("\tStart Pin: ");
  Serial.println(digitalRead(START_PIN));
  delay(100);
}
