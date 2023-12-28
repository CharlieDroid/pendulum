import pigpio
import time
import numpy as np


class Motor:
    def __init__(self, pi: pigpio.pi, pin1, pin2, freq=50):
        self.pin1 = pin1
        self.pin2 = pin2
        self.pi = pi
        range_ = 1000
        self.pi.set_mode(pin1, pigpio.OUTPUT)
        self.pi.set_PWM_frequency(pin1, freq)
        self.pi.set_PWM_range(pin1, range_)

        self.pi.set_mode(pin2, pigpio.OUTPUT)
        self.pi.set_PWM_frequency(pin2, freq)
        self.pi.set_PWM_range(pin2, range_)

        self.rotate(0.0)
        self.duty_cycle = 0.0

    def forward(self, duty_cycle):
        self.pi.set_PWM_dutycycle(self.pin2, 0.0)
        self.pi.set_PWM_dutycycle(self.pin1, duty_cycle)

    def backward(self, duty_cycle):
        self.pi.set_PWM_dutycycle(self.pin1, 0.0)
        self.pi.set_PWM_dutycycle(self.pin2, duty_cycle)

    def rotate(self, duty_cycle):
        # duty_cycle is in float 0. to 1000.
        if duty_cycle > 0.0:
            self.forward(duty_cycle)
        else:
            self.backward(-duty_cycle)
        self.duty_cycle = duty_cycle


class Encoder:
    def __init__(self, pi: pigpio.pi, a, b, dt):
        self.pi = pi
        self.a = a
        self.b = b
        self.dt = dt

        self.levA = 0
        self.levB = 0
        self.lastGpio = None

        self.val = 0  # val now
        self.val_ = 0  # last value

        self.pi.set_mode(a, pigpio.INPUT)
        self.pi.set_mode(b, pigpio.INPUT)

        self.pi.set_pull_up_down(a, pigpio.PUD_UP)
        self.pi.set_pull_up_down(b, pigpio.PUD_UP)

        self.cbA = self.pi.callback(a, pigpio.EITHER_EDGE, self._pulse)
        self.cbB = self.pi.callback(b, pigpio.EITHER_EDGE, self._pulse)

    def _pulse(self, gpio, level, _):
        """
        Decode the rotary encoder pulse.

                     +---------+         +---------+      0
                     |         |         |         |
           A         |         |         |         |
                     |         |         |         |
           +---------+         +---------+         +----- 1

               +---------+         +---------+            0
               |         |         |         |
           B   |         |         |         |
               |         |         |         |
           ----+         +---------+         +---------+  1
        """

        if gpio == self.a:
            self.levA = level
        else:
            self.levB = level

        if gpio != self.lastGpio:  # debounce
            self.lastGpio = gpio

            if gpio == self.a and level:
                if self.levB:
                    self.val += 1
            elif gpio == self.b and level:
                if self.levA:
                    self.val -= 1

    def reset_values(self):
        self.levA = 0
        self.levB = 0
        self.lastGpio = None

        self.val = 0  # val now
        self.val_ = 0  # last value

    def set_value(self, val):
        self.val = val
        self.val_ = val

    def off(self):
        self.cbA.cancel()
        self.cbB.cancel()


class PendulumEncoder(Encoder):
    def __init__(self, pi: pigpio.pi, a, b, dt):
        super().__init__(pi, a, b, dt)
        self.angle_factor = np.pi / 300
        self.velo_factor = np.pi / (300 * self.dt)

    def angle(self):
        return self.val * self.angle_factor

    def velo(self):
        velo = (self.val - self.val_) * self.velo_factor
        self.val_ = self.val
        return velo


class CartEncoder(Encoder):
    def __init__(self, pi: pigpio.pi, a, b, dt):
        super().__init__(pi, a, b, dt)
        max_pos_val = 13410
        self.pos_factor = 2 / max_pos_val
        self.pos_bias = -1
        self.velo_factor = 2 / (max_pos_val * self.dt)
        self.velo_bias = -1

    def pos(self):
        return self.val * self.pos_factor + self.pos_bias

    def velo(self):
        velo = (self.val - self.val_) * self.velo_factor
        self.val_ = self.val
        return velo


if __name__ == "__main__":
    raspberry_pi = pigpio.pi()
    print("running...")
    pendulum = PendulumEncoder(raspberry_pi, 21, 20, 0.02)
    # cart = CartEncoder(raspberry_pi, 13, 6, 0.02)
    motor = Motor(raspberry_pi, 13, 6)
    try:
        while True:
            # 1000 / 600
            pwm = pendulum.val * (5 / 3)
            motor.rotate(pwm)
            print(f"val={pendulum.val}\tdc={motor.duty_cycle:.2f}")
            time.sleep(0.02)
    except KeyboardInterrupt:
        pendulum.off()
        raspberry_pi.stop()
