import pigpio
import time
import argparse
from utils import get_pins
from environment import PendulumEncoder, CartEncoder, Motor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--rotate",
        help="rotate for 300 ms (pulse width range is -1000. to 1000.)",
        default=0.,
        type=float,
    )
    parser.add_argument(
        "--dt",
        help="1 sample per 1 dt",
        default=0.02,
        type=float,
    )
    parser.add_argument(
        "--motor_control",
        help="control motor through the pendulum encoder",
        default=False,
        type=bool,
    )
    args = parser.parse_args()
    raspberry_pi = pigpio.pi()
    cart_pins, pendulum_pins, motor_pins, _ = get_pins(grouped=True)
    print("...running...")
    pendulum = PendulumEncoder(raspberry_pi, *pendulum_pins, dt=0.02)
    # cart = CartEncoder(raspberry_pi, *cart_pins, dt=0.02)
    motor = Motor(raspberry_pi, *motor_pins, freq=50)
    if args.motor_control:
        try:
            while True:
                # rotate pendulum encoder to control motor speed/pwm
                # 1000 / 600
                pwm = pendulum.val * (5 / 3)
                motor.rotate(pwm)
                print(f"val={pendulum.val}\tdc={motor.duty_cycle:.2f}")
                time.sleep(0.02)
        except KeyboardInterrupt:
            pendulum.off()
            raspberry_pi.stop()
    else:
        print(f"rotating at pwm={args.rotate} for 0.3 seconds")
        motor.rotate(args.rotate)
        time.sleep(0.3)
        motor.rotate(0.0)
