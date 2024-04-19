import paramiko
import os
import configparser
import pickle
import time
import socket
import zipfile
import csv

import numpy as np

from environment import Pendulum, DummyPendulum
from td3_fork import Agent, AgentActor, ReplayBuffer


class RPIConnect:
    def __init__(
        self, username="charles", hostname="raspberrypi", password="charlesraspberrypi"
    ):
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        tries = 0
        while True:
            connected = True
            try:
                self.ssh.connect(
                    hostname=hostname, username=username, password=password
                )
            except socket.gaierror as error:
                print(f"ERROR: {error}")
                connected = False
                tries += 1
            if connected:
                print("Connected to Raspberry PI!")
                break
            elif tries > 100:
                print("ERROR: Couldn't connect to Raspberry PI after 100 tries!")
                break

    def ssh_command(self, command_):
        _, stdout, _ = self.ssh.exec_command(command_)
        return stdout.read().decode()

    def sys_command(self, command_):
        return os.system(command_)


def get_pins(grouped=False):
    config = configparser.ConfigParser()
    config.read("./config.ini")

    button_pin = config.getint("PINS", "button_pin")
    pendulum_pins = config.get("PINS", "pendulum_pins")
    cart_pins = config.get("PINS", "cart_pins")
    motor_pins = config.get("PINS", "motor_pins")

    pendulum_pins = [int(pin) for pin in pendulum_pins.split(",")]
    cart_pins = [int(pin) for pin in cart_pins.split(",")]
    motor_pins = [int(pin) for pin in motor_pins.split(",")]
    if grouped:
        return cart_pins, pendulum_pins, motor_pins, button_pin
    return *cart_pins, *pendulum_pins, *motor_pins, button_pin


def get_paths():
    config = configparser.ConfigParser()
    config.read("./config.ini")

    pc_pth = os.path.abspath(os.getcwd()).replace("\\", "/")
    rpi_pth = config.get("PATHS", "absolute_path_rpi").replace("\\", "/")
    return pc_pth, rpi_pth


class Episode:
    def __init__(
        self,
        steps,
        episode_time_steps,
        warmup,
        agent: (Agent, AgentActor),
        env: (Pendulum, DummyPendulum),
    ):
        self.cntr = 0
        self.steps = steps
        self.warmup = warmup
        # number of time steps in an episode
        # 1 episode = 1k time steps
        self.episode_time_steps = episode_time_steps
        self.env = env
        self.agent = agent

        self.observation = None
        self.observations = np.zeros(
            (self.episode_time_steps, env.observation_space.shape[0])
        )
        self.observations_ = np.zeros(
            (self.episode_time_steps, env.observation_space.shape[0])
        )
        self.actions = np.zeros((self.episode_time_steps, env.action_space.shape[0]))

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

    def run(self):
        if self.steps > 0:
            # should be added if starting from scratch
            # if self.env.time_step < self.warmup:
            #     # 0 to 1000
            #     action = self.env.action_space.sample()
            # else:

            # 0 to 1000
            action = self.agent.choose_action(self.observation)
            self.env.step(action[0])
            observation_ = self.env.get_obs()
            action = self.env.motor.duty_cycle
            self.observation = observation_

            # save them for post-processing later
            self.observations[self.cntr] = self.observation
            self.observations_[self.cntr] = observation_
            self.actions[self.cntr] = action
            self.cntr += 1
        else:
            self.observation = self.env.get_obs()
        self.steps += 1

    def run_eval(self):
        if self.steps > 0:
            action = self.agent.choose_action(self.observation)
            self.env.step(action[0])
            bound = self.env.bound[1]
            obs_ = self.env.get_obs()
            # reward = np.cos(obs_[1]) - (
            #     10 * int(abs(obs_[0]) > bound) + 10 * int(abs(obs_[3]) > 20)
            # )
            self.observations[self.cntr] = self.observation
            self.observations_[self.cntr] = obs_
            self.actions[self.cntr] = action
            self.observation = obs_
            self.cntr += 1
        else:
            self.observation = self.env.get_obs()
        self.steps += 1

    def post_process(self):
        memory = (self.observations, self.actions, self.observations_)
        with open(self.agent.memory_file_pth, "wb") as outfile:
            pickle.dump(memory, outfile, pickle.HIGHEST_PROTOCOL)

    def pre_process(self):
        # bounds for reward, change in environment.py
        bound = self.env.bound[1]

        # unzip memory and delete the original zip file
        with zipfile.ZipFile(self.agent.memory_zipfile_pth, "r") as infile:
            infile.extractall(self.agent.memory_dir)
        os.remove(self.agent.memory_zipfile_pth)

        with open(os.path.join(self.agent.memory_dir, "observations.csv"), "r") as f:
            observations = np.array(list(csv.reader(f)), dtype=np.float32)
        os.remove(os.path.join(self.agent.memory_dir, "observations.csv"))

        with open(os.path.join(self.agent.memory_dir, "actions.csv"), "r") as f:
            actions = np.array(list(csv.reader(f)), dtype=np.float32)
        os.remove(os.path.join(self.agent.memory_dir, "actions.csv"))

        observations_ = observations[1:]
        observations = observations[:-1]
        # create fake dones cuz it'll always be like this anyway
        dones = np.zeros((self.episode_time_steps, 1), dtype=np.bool_)
        dones[-1] = [True]
        score_ = 0
        for obs, action, obs_, done in zip(observations, actions, observations_, dones):
            # from 0. to 1000.
            # to 0. to 1.
            # compute for reward
            reward = np.cos(obs_[1]) - (
                2 * obs_[0] ** 2
                + 0.001 * obs_[2] ** 2
                + 0.001 * obs_[3] ** 2
                + 0.01 * action[0] ** 2
            )
            # reward = np.cos(obs_[1]) - (
            #     10 * int(abs(obs_[0]) > bound) + 10 * int(abs(obs_[3]) > 17)
            # )
            # action = np.clip(action * 2.0, -2.0, 2.0)
            self.agent.remember(obs, action, reward, obs_, done)
            self.env.time_step += 1
            score_ += reward
        return score_.item()

    def train_start(self):
        self.do_every(self.env.dt, self.run)
        self.env.motor.rotate(0.0)
        self.post_process()

    def eval_start(self):
        self.do_every(self.env.dt, self.run_eval)
        self.env.motor.rotate(0.0)
        for obs, action, obs_ in zip(
            self.observations, self.actions, self.observations_
        ):
            print(obs, action, obs_)
        import matplotlib.pyplot as plt

        # print(self.actions)
        plt.hist(self.actions)
        plt.title("Actions")
        plt.savefig("actions.png")
        plt.clf()


def reprocess_warmup(
    episode_time_steps,
    reward_bound,
    episodes=20,
    max_size=1_000_000,
    input_shape=(4,),
    n_actions=1,
):
    memory = ReplayBuffer(max_size, input_shape, n_actions)
    for i in range(episodes):
        bound = reward_bound

        with open(f"./memory/episode_{i}_data.pkl", "rb") as infile:
            result = pickle.load(infile)
        (observations, actions, observations_) = result
        # create fake dones cuz it'll always be like this anyway
        dones = np.zeros((episode_time_steps, 1), dtype=np.bool_)
        dones[-1] = [True]
        for obs, action, obs_, done in zip(observations, actions, observations_, dones):
            # from 0. to 1000.
            # to 0. to 1.
            action *= 0.001
            # compute for reward
            reward = np.cos(obs_[1]) - (
                10 * int(abs(obs_[0]) > bound) + 10 * int(abs(obs_[3]) > 17)
            )
            memory.store_transition(obs, action, reward, obs_, done)
    return memory


def process_data_console():
    import pickle
    import numpy as np
    import os
    from td3_fork import ReplayBuffer
    import matplotlib.pyplot as plt

    os.chdir("RPi")
    memory = reprocess_warmup(1_000, 0.9, episodes=6)
    # plt.hist(memory.action_memory[:10, :])
    memory.action_memory[:10, :]
    memory.reward_memory[:10]
    memory.state_memory[:10]
    memory.state_memory[:10, 0]


def open_agent_console():
    from RPi.td3_fork import Agent
    from RPi.environment import DummyPendulum
    import os
    import numpy as np

    os.chdir("RPi")

    env = DummyPendulum()
    agent = Agent(env)


def check_environment():
    import gymnasium as gym
    from gymnasium.envs.registration import register
    import numpy as np
    import torch as T
    import os

    register(
        id="InvertedPendulumModded",
        entry_point="mujoco_mod.envs.inverted_pendulum_mod:InvertedPendulumEnv",
        max_episode_steps=1000,
        reward_threshold=0.0,
    )

    game_id = "InvertedPendulumModded"
    env = gym.make(game_id, render_mode="human")
    env.unwrapped.model.geom_size
    env.unwrapped.model.body_mass
