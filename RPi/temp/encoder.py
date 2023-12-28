import RPi.GPIO as GPIO
import time
from numpy import pi


class Encoder:
    def __init__(self, pin_a, pin_b, dt):
        self.a = pin_a
        self.b = pin_b
        self.dt = dt

        self.encoded_ = 0  # last encoded
        self.val = 0  # val now
        self.val_ = 0  # last value

        GPIO.setup(self.a, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.b, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        GPIO.add_event_detect(self.a, GPIO.BOTH, callback=self.update_encoder)
        GPIO.add_event_detect(self.b, GPIO.BOTH, callback=self.update_encoder)

    def update_encoder(self, _):
        msb = GPIO.input(self.a)
        lsb = GPIO.input(self.b)
        encoded = (msb << 1) | lsb
        sum_ = (self.encoded_ << 2) | encoded

        if (sum_ == 0b1110) or (sum_ == 0b0111) or (sum_ == 0b0001) or (sum_ == 0b1000):
            self.val -= 1
        if (sum_ == 0b1101) or (sum_ == 0b0100) or (sum_ == 0b0010) or (sum_ == 0b1011):
            self.val += 1

        self.encoded_ = encoded

    def angle(self):
        return self.val * (pi / 1200)

    def velo(self):
        # print(self.val, self.val_)
        velo = ((self.val - self.val_) / self.dt) * (
            pi / 1200
        )  # simplify this part in the future
        self.val_ = self.val
        return velo


def do_every(period, f, *args):
    def g_tick():
        t = time.time()
        while True:
            t += period
            yield max(t - time.time(), 0)

    g = g_tick()
    while True:
        time.sleep(next(g))
        f(*args)


def print_angle(e: Encoder):
    # e.velo()
    print(f"angle={e.angle():.2f}\tvelo={e.velo():.2f}")


if __name__ == "__main__":
    GPIO.setmode(GPIO.BCM)
    dt = 0.02
    encoder1 = Encoder(21, 20, dt)
    encoder2 = Encoder(26, 19, dt)
    do_every(
        dt,
        print_angle,
        encoder1,
    )
    GPIO.cleanup()
