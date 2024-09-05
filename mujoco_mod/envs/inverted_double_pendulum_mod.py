import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from lxml import etree


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 3,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}


def simp_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class InvertedDoublePendulumEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description

    This environment originates from control theory and builds on the cartpole
    environment based on the work done by Barto, Sutton, and Anderson in
    ["Neuronlike adaptive elements that can solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
    powered by the Mujoco physics simulator - allowing for more complex experiments
    (such as varying the effects of gravity or constraints). This environment involves a cart that can
    moved linearly, with a pole fixed on it and a second pole fixed on the other end of the first one
    (leaving the second pole as the only one with one free end). The cart can be pushed left or right,
    and the goal is to balance the second pole on top of the first pole, which is in turn on top of the
    cart, by applying continuous forces on the cart.

    ## Action Space
    The agent take a 1-element vector for actions.
    The action space is a continuous `(action)` in `[-1, 1]`, where `action` represents the
    numerical force applied to the cart (with magnitude representing the amount of force and
    sign representing the direction)

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit      |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -1          | 1           | slider                           | slide | Force (N) |

    ## Observation Space

    The state space consists of positional values of different body parts of the pendulum system,
    followed by the velocities of those individual parts (their derivatives) with all the
    positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(11,)` where the elements correspond to the following:

    | Num | Observation                                                       | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ----------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | position of the cart along the linear surface                     | -Inf | Inf | slider                           | slide | position (m)             |
    | 1   | sine of the angle between the cart and the first pole             | -Inf | Inf | sin(hinge)                       | hinge | unitless                 |
    | 2   | sine of the angle between the two poles                           | -Inf | Inf | sin(hinge2)                      | hinge | unitless                 |
    | 3   | cosine of the angle between the cart and the first pole           | -Inf | Inf | cos(hinge)                       | hinge | unitless                 |
    | 4   | cosine of the angle between the two poles                         | -Inf | Inf | cos(hinge2)                      | hinge | unitless                 |
    | 5   | velocity of the cart                                              | -Inf | Inf | slider                           | slide | velocity (m/s)           |
    | 6   | angular velocity of the angle between the cart and the first pole | -Inf | Inf | hinge                            | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of the angle between the two poles               | -Inf | Inf | hinge2                           | hinge | angular velocity (rad/s) |
    | 8   | constraint force - 1                                              | -Inf | Inf |                                  |       | Force (N)                |
    | 9   | constraint force - 2                                              | -Inf | Inf |                                  |       | Force (N)                |
    | 10  | constraint force - 3                                              | -Inf | Inf |                                  |       | Force (N)                |


    There is physical contact between the robots and their environment - and Mujoco
    attempts at getting realistic physics simulations for the possible physical contact
    dynamics by aiming for physical accuracy and computational efficiency.

    There is one constraint force for contacts for each degree of freedom (3).
    The approach and handling of constraints by Mujoco is unique to the simulator
    and is based on their research. Once can find more information in their
    [*documentation*](https://mujoco.readthedocs.io/en/latest/computation.html)
    or in their paper
    ["Analytically-invertible dynamics with contacts and constraints: Theory and implementation in MuJoCo"](https://homes.cs.washington.edu/~todorov/papers/TodorovICRA14.pdf).


    ## Rewards

    The reward consists of two parts:
    - *alive_bonus*: The goal is to make the second inverted pendulum stand upright
    (within a certain angle limit) as long as possible - as such a reward of +10 is awarded
     for each timestep that the second pole is upright.
    - *distance_penalty*: This reward is a measure of how far the *tip* of the second pendulum
    (the only free end) moves, and it is calculated as
    *0.01 * x<sup>2</sup> + (y - 2)<sup>2</sup>*, where *x* is the x-coordinate of the tip
    and *y* is the y-coordinate of the tip of the second pole.
    - *velocity_penalty*: A negative reward for penalising the agent if it moves too
    fast *0.001 *  v<sub>1</sub><sup>2</sup> + 0.005 * v<sub>2</sub> <sup>2</sup>*

    The total reward returned is ***reward*** *=* *alive_bonus - distance_penalty - velocity_penalty*

    ## Starting State
    All observations start in state
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) with a uniform noise in the range
    of [-0.1, 0.1] added to the positional values (cart position and pole angles) and standard
    normal force with a standard deviation of 0.1 added to the velocity values for stochasticity.

    ## Episode End
    The episode ends when any of the following happens:

    1.Truncation:  The episode duration reaches 1000 timesteps.
    2.Termination: Any of the state space values is no longer finite.
    3.Termination: The y_coordinate of the tip of the second pole *is less than or equal* to 1. The maximum standing height of the system is 1.196 m when all the parts are perpendicularly vertical on top of each other).

    ## Arguments

    No additional arguments are currently supported.

    ```python
    import gymnasium as gym
    env = gym.make('InvertedDoublePendulum-v4')
    ```
    There is no v3 for InvertedPendulum, unlike the robot environments where a v3 and
    beyond take `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc.

    ```python
    import gymnasium as gym
    env = gym.make('InvertedDoublePendulum-v2')
    ```

    ## Version History

    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3
    * v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco-py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks (including inverted pendulum)
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 200,
    }

    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float64)
        # self.xml_file_pth = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/doublependulum/code/Users/csosmea/mujoco_mod/assets/inverted_double_pendulum.xml"
        self.xml_file_pth = "./mujoco_mod/assets/inverted_double_pendulum.xml"
        MujocoEnv.__init__(
            self,
            self.xml_file_pth,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        self.swingup = True
        self.all_in_one = False
        self.pendulum_command = 3
        # actual_p2_angle = desired_p2_angle - p1_angle
        if self.swingup or self.all_in_one:
            self.init_qpos = np.array([0.0, np.pi, 0.0])
        else:
            self.init_qpos = np.array([0.0, 0.0, 0.0])
        self.init_qposes = (
            np.array([0.0, np.pi, 0.0]),  # 00
            np.array([0.0, np.pi, -np.pi]),  # 01
            np.array([0.0, 0.0, np.pi]),  # 10
            np.array([0.0, 0.0, 0.0]),  # 11
        )
        # (first pendulum, second pendulum)
        self.pendulum_command_list = ((0, 0), (0, 1), (1, 0), (1, 1))
        self.steps_change_command = 500
        self.steps = 0
        self.curr_pendulum_length = 0.6

        self.a1_rot_max = 2.5
        self.a2_rot_max = 5
        utils.EzPickle.__init__(self, **kwargs)

    def step(self, a):
        self.steps += 1
        max_action = 3.0
        a = np.clip(a * max_action, -max_action, max_action)
        for _ in range(10):
            self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        v1, v2, v3 = self.data.qvel
        x, a1, a2 = self.data.qpos[:]
        tip_x, _, tip_y = self.data.site_xpos[0]

        if self.all_in_one:
            # ALL IN ONE
            vel_rotation_penalty = 5

            if self.pendulum_command == 0:
                vel_penalty = min(
                    10 * abs(v2) + 10 * abs(v3) + 0.1 * a[0] ** 2,
                    vel_rotation_penalty,
                )
            else:
                vel_penalty = min(
                    0.01 * (v2**2) + 0.01 * (v3**2) + 0.01 * a[0] ** 2,
                    vel_rotation_penalty,
                )

            if self.pendulum_command == 0:
                pos_penalty = min(10 * x**2, 5)
            else:
                # it was observed pendulum_commands 0,1, and 2 would not settle in the center
                pos_penalty = 5 * x**2

            if self.pendulum_command == 0:
                angle_reward = self.compute_angle_reward(a1, a2)
            else:
                angle_reward = 3.0 * self.compute_angle_reward(a1, a2)

            a1_rot = (a1 - np.pi) / (2 * np.pi)
            a2_rot = (a2 + a1 - np.pi) / (2 * np.pi)
            rotation_penalty = min(
                (1 / (self.a1_rot_max**4)) * (a1_rot**4)
                + (1 / (self.a2_rot_max**4)) * (a2_rot**4),
                vel_rotation_penalty,
            )
            r = angle_reward - pos_penalty - vel_penalty - rotation_penalty

            terminated = (
                False
                or bool(abs(a1_rot) > (self.a1_rot_max * 2))
                or bool(abs(a2_rot) > (self.a2_rot_max * 2))
            )
        elif self.swingup:
            vel_rotation_penalty = 5

            if self.pendulum_command == 0:
                vel_penalty = min(
                    3 * (abs(v2) + abs(v3) + abs(a[0])),
                    # 10 * abs(v2) + 10 * abs(v3) + 0.1 * a[0] ** 2,
                    vel_rotation_penalty,
                )
            else:
                vel_penalty = min(
                    0.015 * (v2**2) + 0.015 * (v3**2) + 0.015 * a[0] ** 2,
                    vel_rotation_penalty,
                )

            if self.pendulum_command == 0:
                pos_penalty = min(4 * x**2, 4)
                # pos_penalty = min(10 * x**2, 5)
            else:
                # it was observed pendulum_commands 0,1, and 2 would not settle in the center
                pos_penalty = 6 * x**2

            if self.pendulum_command == 0:
                angle_reward = self.compute_angle_reward(a1, a2)
            else:
                angle_reward = 3.0 * self.compute_angle_reward(a1, a2)

            a1_rot = (a1 - np.pi) / (2 * np.pi)
            a2_rot = (a2 + a1 - np.pi) / (2 * np.pi)
            rotation_penalty = min(
                (1 / (self.a1_rot_max**4)) * (a1_rot**4)
                + (1 / (self.a2_rot_max**4)) * (a2_rot**4),
                vel_rotation_penalty,
            )
            r = angle_reward - pos_penalty - vel_penalty - rotation_penalty

            terminated = (
                False
                or bool(abs(a1_rot) > self.a1_rot_max)
                or bool(abs(a2_rot) > self.a2_rot_max)
            )
        else:
            dist_penalty = 0.01 * (tip_x**2) + 14 * (
                (tip_y - (self.curr_pendulum_length * 2)) ** 2
            )
            vel_penalty = 1e-3 * (v1**2) + 5e-3 * (v2**2) + 0.01 * (a[0] ** 2)
            pos_penalty = 2 * x**2
            alive_bonus = 1
            r = alive_bonus - dist_penalty - vel_penalty - pos_penalty
            terminated = bool(tip_y <= (self.curr_pendulum_length * 2 - 0.4))

        # print(self.steps, self.steps_change_command)
        if self.all_in_one and (self.steps == self.steps_change_command):
            self.pendulum_command = np.random.choice([0, 1, 2, 3, 3])

        if self.all_in_one:
            self.steps += 1
        truncated = False
        if self.render_mode == "human":
            self.render()
        return ob, r, terminated, truncated, {}

    def _get_obs(self):
        # observed info (12): pos, sin(a1), sin(a2), cos(a1), cos(a2), pos_vel, a1_vel, a2_vel, 00, 01, 10, 11
        # privileged info (14): tip_x, tip_y, a1, a2, friction_cart, friction_pendulums, damping, armature, gear, length, density_p1, density_p2, force, noise_std
        x, _, y = self.data.site_xpos[0]
        a1, a2 = self.data.qpos[1:]
        a1 = (a1 - np.pi) / (2 * np.pi * self.a1_rot_max)
        a2 = (a2 + a1 - np.pi) / (2 * np.pi * self.a2_rot_max)
        comm = np.zeros(
            4,
        )
        comm[self.pendulum_command] = 1.0
        system_params = np.zeros(
            10,
        )
        obs = np.concatenate(
            [
                self.data.qpos[:1],  # cart x pos
                np.sin(self.data.qpos[1:]),  # link angles
                np.cos(self.data.qpos[1:]),
                np.clip(self.data.qvel, -150, 150),
                comm,
                np.array([x]),
                np.array([y]),
                np.array([a1]),
                np.array([a2]),
                system_params,
            ]
        ).ravel()
        return obs

    def reset_model(self):
        if self.all_in_one:
            # x_noise = self.np_random.uniform(low=-0.2, high=0.2)
            # theta1_noise = self.np_random.uniform(low=-np.pi, high=np.pi)
            # theta2_noise = self.np_random.uniform(low=-np.pi, high=np.pi)
            # x_dot_noise = self.np_random.uniform(low=-1, high=1)
            # theta1_dot_noise = self.np_random.uniform(low=-10, high=10)
            # theta2_dot_noise = self.np_random.uniform(low=-20, high=20)
            # x_noise = self.np_random.uniform(low=-0.1, high=0.1)
            # theta1_noise = self.np_random.uniform(low=-0.1, high=0.1)
            # theta2_noise = self.np_random.uniform(low=-0.1, high=0.1)
            # x_dot_noise = self.np_random.uniform(low=-0.2, high=0.2)
            # theta1_dot_noise = self.np_random.uniform(low=-0.2, high=0.2)
            # theta2_dot_noise = self.np_random.uniform(low=-0.2, high=0.2)
            if self.pendulum_command == 0:
                x_noise = self.np_random.uniform(low=-0.2, high=0.2)
                theta1_noise = self.np_random.uniform(low=-np.pi, high=np.pi)
                theta2_noise = self.np_random.uniform(low=-np.pi, high=np.pi)
                x_dot_noise = self.np_random.uniform(low=-1, high=1)
                theta1_dot_noise = self.np_random.uniform(low=-10, high=10)
                theta2_dot_noise = self.np_random.uniform(low=-20, high=20)

                self.init_qpos = self.init_qposes[0]
            else:
                x_noise = self.np_random.uniform(low=-0.4, high=0.4)
                theta1_noise = self.np_random.uniform(low=-0.2, high=0.2)
                theta2_noise = self.np_random.uniform(low=-0.2, high=0.2)
                x_dot_noise = self.np_random.uniform(low=-0.2, high=0.2)
                theta1_dot_noise = self.np_random.uniform(low=-1.0, high=1.0)
                theta2_dot_noise = self.np_random.uniform(low=-2.0, high=2.0)

                self.init_qpos = self.init_qposes[np.random.randint(low=0, high=4)]
        elif self.swingup:
            if self.pendulum_command in [0, 1, 2]:
                x_noise = self.np_random.uniform(low=-0.2, high=0.2)
                theta1_noise = self.np_random.uniform(low=-np.pi, high=np.pi)
                theta2_noise = self.np_random.uniform(low=-np.pi, high=np.pi)
                x_dot_noise = self.np_random.uniform(low=-1, high=1)
                theta1_dot_noise = self.np_random.uniform(low=-10, high=10)
                theta2_dot_noise = self.np_random.uniform(low=-20, high=20)

                self.init_qpos = self.init_qposes[0]
            else:
                x_noise = self.np_random.uniform(low=-0.4, high=0.4)
                theta1_noise = self.np_random.uniform(low=-0.2, high=0.2)
                theta2_noise = self.np_random.uniform(low=-0.2, high=0.2)
                x_dot_noise = self.np_random.uniform(low=-0.2, high=0.2)
                theta1_dot_noise = self.np_random.uniform(low=-1.0, high=1.0)
                theta2_dot_noise = self.np_random.uniform(low=-2.0, high=2.0)

                choice = np.random.randint(low=0, high=4)
                while choice == self.pendulum_command:
                    choice = np.random.randint(low=0, high=4)

                self.init_qpos = self.init_qposes[np.random.randint(low=0, high=4)]
                # self.init_qpos = self.init_qposes[0]
        else:
            # extra 0.01 margin
            x_noise = self.np_random.uniform(low=-0.4, high=0.4)
            theta1_noise = self.np_random.uniform(low=-0.15, high=0.15)
            theta2_noise = self.np_random.uniform(low=-0.15, high=0.15)
            x_dot_noise = self.np_random.uniform(low=-0.4, high=0.4)
            theta1_dot_noise = self.np_random.uniform(low=-2, high=2)
            theta2_dot_noise = self.np_random.uniform(low=-2, high=2)

            tree = etree.parse(self.xml_file_pth)
            root = tree.getroot()
            self.curr_pendulum_length = float(
                root.xpath('.//geom[@name="cpole2"]')[0].attrib["size"].split()[1]
            )

        qpos = self.init_qpos + np.array([x_noise, theta1_noise, theta2_noise])
        qvel = self.init_qvel + np.array(
            [x_dot_noise, theta1_dot_noise, theta2_dot_noise]
        )

        self.set_state(qpos, qvel)

        if self.all_in_one:
            self.steps = 0
            # has to be odd for some reason and the max steps is 1999
            self.steps_change_command = np.random.randint(low=0, high=999) * 2 + 1
            self.pendulum_command = np.random.choice([0, 1, 2, 3, 3])

        return self._get_obs()

    def compute_angle_reward(self, a1, a2):
        if self.pendulum_command == 3:
            angle_reward = np.cos(a1) + np.cos(a2 + a1)
        elif self.pendulum_command == 2:
            angle_reward = 1.2 * np.cos(a1) - 0.8 * np.cos(a2 + a1)
        elif self.pendulum_command == 1:
            angle_reward = -0.8 * np.cos(a1) + 1.2 * np.cos(a2 + a1)
        elif self.pendulum_command == 0:
            angle_reward = -np.cos(a1) - np.cos(a2 + a1)
        return angle_reward
