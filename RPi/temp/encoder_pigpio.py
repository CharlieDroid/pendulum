import numpy as np
import pigpio
import time

belt_pulley_radius = 6e-3


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


def printer():
    global cart, pendulum
    print(
        f"pos={cart.pos():.2f}\tvelo={cart.velo():.2f}\t\tangle={pendulum.angle():.2f}\tvelo={pendulum.velo():.2f}"
    )


def do_every(period, f, max_time, *args):
    def g_tick():
        t = time.time()
        while True:
            t += period
            yield max(t - time.time(), 0)

    start = time.time()
    g = g_tick()
    while True:
        if time.time() - start > max_time:
            break
        time.sleep(next(g))
        f(*args)


if __name__ == "__main__":
    pi = pigpio.pi()
    print("running")
    pendulum = PendulumEncoder(pi, 6, 5, 0.02)
    cart = CartEncoder(pi, 26, 19, 0.02)
    try:
        do_every(0.02, printer, 300)
        pendulum.off()
        cart.off()
        pi.stop()
    except KeyboardInterrupt:
        pendulum.off()
        cart.off()
        pi.stop()
