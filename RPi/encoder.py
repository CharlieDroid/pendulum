import pigpio
import time
from environment import PendulumEncoder, CartEncoder
from utils import get_pins


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
    # prints values of pendulum encoder and cart encoder for 300 seconds
    pi = pigpio.pi()
    cart_pins, pendulum_pins, _, _ = get_pins(grouped=True)
    print("...running...")
    pendulum = PendulumEncoder(pi, *pendulum_pins, dt=0.02)
    cart = CartEncoder(pi, *cart_pins, dt=0.02)
    try:
        do_every(0.02, printer, 300)
        pendulum.off()
        cart.off()
        pi.stop()
    except KeyboardInterrupt:
        pendulum.off()
        cart.off()
        pi.stop()
