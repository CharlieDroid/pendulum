import time
import argparse
import numpy as np

from environment import Pendulum
from utils import get_pins



class PendulumTest(Pendulum):
    def __init__(self, ceg, cew, peg, pew, ml, mr, me, bt, dt):
        super().__init__(ceg, cew, peg, pew, ml, mr, me, bt, dt)

    def direction_test(self):
        self.print_values()
        self.motor.rotate(-self.usual_speed * 0.8)
        print("Moving left")
        time.sleep(2)

        self.print_values()
        self.motor.rotate(self.usual_speed * 0.8)
        print("Moving right")
        time.sleep(4)

        self.print_values()
        self.motor.rotate(-self.usual_speed * 0.8)
        print("Moving left")
        time.sleep(2)

        self.print_values()
        self.motor.rotate(0.0)

    def goto(self, pos, speed_=None):
        if not speed_:
            speed_ = self.usual_speed * 1.2
        current_pos = self.cart_obs.pos()
        direction = -1
        if pos - current_pos > 0:
            direction = 1
        self.motor.rotate(direction * speed_)
        if direction > 0:
            while self.cart_obs.pos() < pos:
                pass
        else:
            while self.cart_obs.pos() > pos:
                pass
        self.motor.rotate(0.0)

    def print_values(self):
        pos = self.cart_obs.val
        angle_ = self.pendulum_obs.val
        print(f"{pos=}\t{angle_=}")

    def print_actual(self):
        pos = self.cart_obs.pos()
        pos_dot = self.cart_obs.velo()
        theta = self.pendulum_obs.angle()
        theta_dot = self.pendulum_obs.velo()
        print(f"{pos=:.3f} {pos_dot=:.3f}\t{theta=:.4f} {theta_dot=:.3f}")


class EpisodeTest:
    def __init__(self, steps_, episode_time_steps_, env: PendulumTest):
        self.cntr = 0
        self.steps = steps_
        self.episode_time_steps = episode_time_steps_
        self.env = env
        self.observation = None
        self.total_reward = 0
        self.var = float("inf")

    def do_every(self, period, f, *args_):
        def g_tick():
            t = time.time()
            while True:
                t += period
                yield max(t - time.time(), 0)

        g = g_tick()
        while self.steps < (self.episode_time_steps + 1):
            time.sleep(next(g))
            f(*args_)

    def run_env(self):
        if self.steps > 0:
            obs = self.observation
            if self.steps < 400:
                action = 1000.
            elif self.steps > 400:
                action = -1000.
            else:
                action = 0.
            self.env.step(action)
            bound = self.env.bound[1]
            obs_ = self.env.get_obs()
            reward = np.cos(obs_[1]) - (
                    10 * int(abs(obs_[0]) > bound) + 10 * int(abs(obs_[3]) > 20)
            )

            print(
                f"pos:{obs[0]:.2f} angle:{obs[1]:.4f} lin_vel:{obs[2]:.2f} ang_vel:{obs[3]:.2f} reward:{reward:.2f}"
            )
            if abs(obs[1]) < self.var:
                self.var = abs(obs[1])
            self.total_reward += reward
            self.observation = obs_
        else:
            self.observation = self.env.get_obs()
        self.steps += 1

    def env_start(self):
        self.env.bound = (0.8, 0.9)
        self.do_every(self.env.dt, self.run_env)
        print(self.total_reward)
        print(self.var)
        self.env.motor.rotate(0.)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direction_test",
        help="figure out if the cart pin and motor pin is correctly configured",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--find_end_val",
        help="find end value of the track length",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--pendulum_orientation_test",
        help="find out orientation of the pendulum, left side negative, right side positive",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--goto_test",
        help="goes to different points along the track specifically -.9 -.5 0 .5 .9",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--drift_test",
        help="oscillates between the two boundaries of the track to see how much the drift affects its accuracy",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--random_test",
        help="for random stuffs",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--run_env_test",
        help="running environment random actions",
        default=False,
        type=bool,
    )
    args = parser.parse_args()

    pins = get_pins()
    pendulum = PendulumTest(*pins, dt=0.02)
    try:
        if args.direction_test:
            input("Continue?")
            print("Starting direction test")
            pendulum.direction_test()

        if args.find_end_val:
            input("Continue?")
            print("Finding end value")
            pendulum.reset_cart()
            pendulum.motor.rotate(pendulum.usual_speed * 0.8)
            time.sleep(12)
            pendulum.motor.rotate(0.0)
            input("Press enter if at the end of the pendulum.")
            print(pendulum.cart_obs.val)

        if args.pendulum_orientation_test:
            input("Continue?")
            print("Starting pendulum orientation test")
            pendulum.reset()
            steps = int(60.0 / pendulum.dt)
            for _ in range(steps):
                val = pendulum.pendulum_obs.val
                angle = pendulum.pendulum_obs.angle()
                velo = pendulum.pendulum_obs.velo()
                print(f"{val=}\t{angle=:.4f}\t{velo=:.2f}")
                time.sleep(pendulum.dt)

        if args.goto_test:
            input("Continue?")
            print("Zeroing out")
            pendulum.reset_cart()
            print("Going to positions -.9 -.5 0 .5 .9")
            points = (-0.9, -0.5, 0, 0.5, 0.9)
            speed = 600.0
            for point in points:
                pendulum.goto(point, speed_=speed)
                print(f"Arrived at {point}")
                time.sleep(1.0)

            print("Going to positions .9 .5 0 -.5 -.9")
            points = points[::-1]
            for point in points:
                pendulum.goto(point, speed_=speed)
                print(f"Arrived at {point}")
                time.sleep(1.0)

        if args.drift_test:
            speed = 1000.0
            input(f"Continue at {speed=}?")
            print("Zeroing out")
            pendulum.reset_cart()
            bounds = 0.6
            print(f"Oscillating between -{bounds} and {bounds} at {speed=}")
            for _ in range(5):
                pendulum.goto(-bounds, speed_=speed)
                time.sleep(2.0)
                pendulum.goto(bounds, speed_=speed)
                time.sleep(2.0)

        if args.run_env_test:
            pendulum.reset()
            episode = EpisodeTest(-5, 1_000, pendulum)
            episode.env_start()

        if args.random_test:
            print("Starting random test")
            pendulum.reset()
    except KeyboardInterrupt:
        pass
    pendulum.kill()
