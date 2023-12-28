from environment import Pendulum
import time


class PendulumTest(Pendulum):
    def __init__(self, ceg, cew, peg, pew, ml, mr, bt, dt):
        super().__init__(ceg, cew, peg, pew, ml, mr, bt, dt)

    def direction_test(self):
        self.print_values()
        self.motor.rotate(-300.0)
        print("Moving left")
        time.sleep(2)

        self.print_values()
        self.motor.rotate(300.0)
        print("Moving right")
        time.sleep(4)

        self.print_values()
        self.motor.rotate(-300)
        print("Moving left")
        time.sleep(2)

        self.print_values()
        self.motor.rotate(0.0)

    def goto(self, pos, speed_=300.0):
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
        angle = self.pendulum_obs.val
        print(f"{pos=}\t{angle=}")

    def print_actual(self):
        pos = self.cart_obs.pos()
        pos_dot = self.cart_obs.velo()
        theta = self.pendulum_obs.angle()
        theta_dot = self.pendulum_obs.velo()
        print(f"{pos=:.3f} {pos_dot=:.3f}\t{theta=:.4f} {theta_dot=:.3f}")


if __name__ == "__main__":
    cart_pins = [21, 20]
    button_pin = 26
    motor_pins = [13, 6]
    pendulum_pins = [12, 16]
    pendulum = PendulumTest(*cart_pins, *pendulum_pins, *motor_pins, button_pin, dt=0.1)
    direction_test = True
    find_end_val = False
    pendulum_orientation_test = False
    goto_test = False
    drift_test = False
    try:
        print(
            f"{direction_test=} {find_end_val=} {pendulum_orientation_test=} {goto_test=} {drift_test=}"
        )
        if direction_test:
            input("Continue?")
            print("Starting direction test")
            pendulum.direction_test()

        if find_end_val:
            input("Continue?")
            print("Finding end value")
            pendulum.reset_zero()
            pendulum.motor.rotate(500.0)
            time.sleep(12)
            pendulum.motor.rotate(0.0)
            input("Press enter if at the end of the pendulum.")
            print(pendulum.cart_obs.val)

        if pendulum_orientation_test:
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

        if goto_test:
            input("Continue?")
            print("Zeroing out")
            pendulum.reset_zero()
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

        if drift_test:
            speed = 800.0
            input(f"Continue at {speed=}?")
            print("Zeroing out")
            pendulum.reset_zero()
            print(f"Oscillating between -.9 and .9 at {speed=}")
            for _ in range(10):
                pendulum.goto(-0.9, speed_=speed)
                time.sleep(2.0)
                pendulum.goto(0.9, speed_=speed)
                time.sleep(2.0)
    except KeyboardInterrupt:
        pass
    pendulum.kill()
