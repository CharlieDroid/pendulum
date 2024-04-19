import pigpio
import numpy as np
import time


class Motor:
    def __init__(self, pi: pigpio.pi, in1, in2, freq):
        self.in1 = in1
        self.in2 = in2
        self.pi = pi
        range_ = 1000
        self.pi.set_mode(in1, pigpio.OUTPUT)
        self.pi.set_PWM_frequency(in1, freq)
        self.pi.set_PWM_range(in1, range_)

        self.pi.set_mode(in2, pigpio.OUTPUT)
        self.pi.set_PWM_frequency(in2, freq)
        self.pi.set_PWM_range(in2, range_)

        self.duty_cycle = None
        self.rotate(0.0)

    def forward(self, duty_cycle):
        self.pi.set_PWM_dutycycle(self.in2, 0.0)
        self.pi.set_PWM_dutycycle(self.in1, duty_cycle)

    def backward(self, duty_cycle):
        self.pi.set_PWM_dutycycle(self.in1, 0.0)
        self.pi.set_PWM_dutycycle(self.in2, duty_cycle)

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

        self.pi.set_pull_up_down(self.a, pigpio.PUD_UP)
        self.pi.set_pull_up_down(self.b, pigpio.PUD_UP)

        self.cbA = self.pi.callback(self.a, pigpio.EITHER_EDGE, self._pulse)
        self.cbB = self.pi.callback(self.b, pigpio.EITHER_EDGE, self._pulse)

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
        self.val = 0  # val now
        self.val_ = 0  # last value

        self.levA = 0
        self.levB = 0
        self.lastGpio = None

    def set_value(self, val):
        self.val = val
        self.val_ = val

    def off(self):
        self.pi.set_pull_up_down(self.a, pigpio.PUD_OFF)
        self.pi.set_pull_up_down(self.b, pigpio.PUD_OFF)

        self.cbA.cancel()
        self.cbB.cancel()


class PendulumEncoder(Encoder):
    def __init__(self, pi: pigpio.pi, a, b, dt):
        super().__init__(pi, a, b, dt)
        # 2*pi / 599
        self.angle_factor = np.pi / 299.5
        self.velo_factor = np.pi / (299.5 * self.dt)

    def angle(self):
        return self.val * self.angle_factor

    def velo(self):
        velo = (self.val - self.val_) * self.velo_factor
        self.val_ = self.val
        return velo


class CartEncoder(Encoder):
    def __init__(self, pi: pigpio.pi, a, b, dt):
        super().__init__(pi, a, b, dt)
        max_pos_val = 13491
        self.pos_factor = 2 / max_pos_val
        self.pos_bias = -1
        self.velo_factor = 2 / (max_pos_val * self.dt)

    def pos(self):
        return self.val * self.pos_factor + self.pos_bias

    def velo(self):
        velo = (self.val - self.val_) * self.velo_factor
        self.val_ = self.val
        return velo


class Button:
    def __init__(self, pi: pigpio.pi, pin):
        self.pi = pi
        self.pin = pin
        self.callback = None

        self.pi.set_mode(pin, pigpio.INPUT)

        self.pi.set_pull_up_down(pin, pigpio.PUD_UP)

    def on(self, callback):
        self.callback = self.pi.callback(self.pin, pigpio.RISING_EDGE, callback)

    def off(self):
        self.callback.cancel()
        self.callback = None


class ObservationSpace:
    def __init__(self):
        # pos, angle, pos_velo, angle_velo
        # velo is max mechanical rpm of encoder which is 5000 rpm * 10%
        self.shape = (4,)
        # max_horizontal_speed = according to testing is 18.
        # max_rad_sec = (500 * np.pi) / 3 = 104.7197551 ~ 100.
        self.high = (
            1.1,
            np.pi,
            20,
            100.0,
        )
        self.low = (
            -1.1,
            -np.pi,
            -20,
            -100.0,
        )


class ActionSpace:
    def __init__(self):
        self.shape = (1,)
        self.high = [1000.0]
        self.low = [-1000.0]

    def sample(self):
        return np.random.uniform(low=self.low[0], high=self.high[0])


class DummyPendulum:
    def __init__(self):
        # self.bound = (physical bound, reward bound)
        self.bound = (0.93, 0.9)
        self.time_step = 0
        self.reward_range = (float("-inf"), float("inf"))

        self.observation_space = ObservationSpace()
        self.action_space = ActionSpace()


class Pendulum(DummyPendulum):
    def __init__(self, ceg, cew, peg, pew, ml, mr, bt, dt=0.02):
        super().__init__()
        # cart encoder green, pendulum encoder white, motor left, motor right, etc.
        self.pi = pigpio.pi()
        self.reset_flag = False
        self.dt = dt
        # just save it in memory to not calculate it every step
        self._2pi = 2 * np.pi
        # frequency is limited to https://abyz.me.uk/rpi/pigpio/python.html#set_PWM_frequency
        freq = 20_000

        self.cart_obs = CartEncoder(self.pi, ceg, cew, self.dt)
        self.pendulum_obs = PendulumEncoder(self.pi, peg, pew, self.dt)
        self.motor = Motor(self.pi, ml, mr, freq)
        print(f"PWM Freq:{self.pi.get_PWM_frequency(mr)}")
        self.limit_switch = Button(self.pi, bt)
        self.cntr = 0
        self.usual_speed = 160.0

    def do_every(self, period, f, *args):
        def g_tick():
            t = time.time()
            while True:
                t += period
                yield max(t - time.time(), 0)

        g = g_tick()
        while not self.reset_flag:
            time.sleep(next(g))
            f(*args)

    def end_limit_pressed(self, *_):
        # there might be error here, if so, just add _ in parameters/args
        self.cart_obs.reset_values()
        self.limit_switch.off()
        self.motor.rotate(-self.usual_speed)
        time.sleep(0.5)
        self.motor.rotate(0.0)
        self.reset_flag = True

    def check_pendulum(self):
        # check if pendulum has zero velocity
        if self.pendulum_obs.velo() < 0.001:
            self.cntr += 1
        else:
            self.cntr = 0
        # if it has been zero velocity for 1 second then reset
        if self.cntr > int(1.0 / self.dt):
            self.reset_flag = True

    def reset_cart(self):
        # go back to -1. or find 0 val or leftmost side
        self.limit_switch.on(callback=self.end_limit_pressed)
        self.motor.rotate(-self.usual_speed)
        start = time.time()
        while not self.reset_flag:
            assert (
                time.time() - start
            ) < 40.0, "Limit has not been found for more than 40 seconds"
        self.reset_flag = False

        # go to center-ish
        self.motor.rotate(self.usual_speed * 1.2)
        while self.cart_obs.pos() < -0.05:
            pass
        self.motor.rotate(0.0)

    def reset_pendulum(self):
        self.do_every(self.dt, self.check_pendulum)
        self.reset_flag = False
        self.cntr = 0
        self.pendulum_obs.reset_values()
        self.pendulum_obs.set_value(300)

    def reset(self):
        print("...resetting...")
        self.reset_cart()
        self.reset_pendulum()

    def step(self, action):
        if self.cart_obs.pos() < -self.bound[0]:
            action = self.usual_speed * 1.2
        elif self.cart_obs.pos() > self.bound[0]:
            action = -self.usual_speed * 1.2
        self.motor.rotate(action)

    def simp_angle(self, a):
        if a > np.pi:
            return self.simp_angle(a - self._2pi)
        elif a < -np.pi:
            return self.simp_angle(a + self._2pi)
        return a

    def get_obs(self):
        angle = self.simp_angle(self.pendulum_obs.angle())
        ob = (
            self.cart_obs.pos(),
            angle,
            self.cart_obs.velo(),
            self.pendulum_obs.velo(),
        )
        return ob

    def kill(self):
        self.motor.rotate(0.0)
        self.pendulum_obs.off()
        self.cart_obs.off()
        self.pi.stop()
