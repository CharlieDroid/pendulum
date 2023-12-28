import os
import pickle
import platform
import time
import traceback
import shutil

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from environment import Pendulum, DummyPendulum
from td3_fork import Agent, AgentActor
from utils import RPIConnect, get_pins, get_paths


class Episode:
    def __init__(
        self,
        steps,
        episode_time_steps,
        _agent: (Agent, AgentActor),
        _env: (Pendulum, DummyPendulum),
    ):
        self.cntr = 0
        self.steps = steps
        # number of time steps in an episode
        # 1 episode = 1k time steps
        self.episode_time_steps = episode_time_steps
        self.env = _env
        self.agent = _agent

        self.observation = None
        self.observations = np.zeros(
            (self.episode_time_steps, env.observation_space.shape[0])
        )
        self.observations_ = np.zeros(
            (self.episode_time_steps, env.observation_space.shape[0])
        )
        self.actions = np.zeros((self.episode_time_steps, env.action_space.shape[0]))

    def do_every(self, period, f, *args):
        def g_tick():
            t = time.time()
            while True:
                t += period
                yield max(t - time.time(), 0)

        g = g_tick()
        while self.steps < self.episode_time_steps:
            time.sleep(next(g))
            f(*args)

    def run(self):
        if self.steps > 0:
            if self.env.time_step < self.agent.warmup:
                # 0 to 1000
                action = self.env.action_space.sample()
            else:
                # 0 to 1000
                action = self.agent.choose_action(self.observation)
            self.env.step(action)
            observation_ = self.env.get_obs()
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
            self.env.step(action)
        else:
            self.observation = self.env.get_obs()
        self.steps += 1

    def post_process(self):
        # create fake dones cuz it'll always be like this anyway
        dones = np.zeros((self.episode_time_steps, 1), dtype=np.bool_)
        dones[-1] = [True]
        # bounds for reward
        bound = 0.7
        actions = np.zeros((self.episode_time_steps, 1))
        rewards = np.zeros((self.episode_time_steps, 1))
        for j, data in enumerate(
            zip(self.observations, self.actions, self.observations_)
        ):
            obs, action, obs_ = data
            # from 0. to 1000.
            # to 0. to 1.
            actions[j] = action[0] * 0.001
            # compute for reward
            rewards[j] = np.cos(obs[1]) - (
                10 * int(abs(obs[0]) > bound) + 10 * int(abs(obs[3]) > 13)
            )
        memory = (self.observations, actions, rewards, self.observations_, dones)
        with open(self.agent.memory_file_pth, "wb") as outfile:
            pickle.dump(memory, outfile, pickle.HIGHEST_PROTOCOL)

    def pre_process(self):
        with open(self.agent.memory_file_pth, "rb") as infile:
            result = pickle.load(infile)
        (observations, actions, rewards, observations_, dones) = result
        score_ = 0
        for obs, action, reward, obs_, done in zip(
            observations, actions, rewards, observations_, dones
        ):
            self.agent.remember(obs, action, reward, obs_, done)
            self.env.time_step += 1
            score_ += reward
        return score_

    def train_start(self):
        self.do_every(self.env.dt, self.run)
        self.env.motor.rotate(0.0)
        self.post_process()

    def eval_start(self):
        self.do_every(self.env.dt, self.run_eval)
        self.env.motor.rotate(0.0)


if __name__ == "__main__":
    # setup environment and agent
    # sudo nice -n -20 ./pendulum/venv/bin/python ./pendulum/main.py
    # make sure there is no ./models/td3_fork_actor.chkpt in raspberrypi and PC that is important
    # make sure there is no ./memory/td3_fork_memory.pkl in raspberrypi and PC that is important
    """
    --- ensure connection between PC and raspberry first ---
    --- main() should be run first in PC ---
    """
    filename = "testing run"
    log_dir = f"runs/{filename}"
    pc_pth, pi_pth = get_paths()
    evaluate = False

    if platform.node() == "raspberrypi":
        pins = get_pins()
        env = Pendulum(*pins)
        agent = AgentActor(env=env)
        agent.load_model()
        n_episodes = 1000
        try:
            input("Start?")
            if evaluate:
                for i in range(5):
                    env.reset()
                    Episode(-5, 1_000, agent, env).eval_start()
                    print(f"episode {i + 1}/5")
            else:
                os.remove(agent.chkpt_file_pth)
                for i in range(n_episodes):
                    # reset episode
                    env.reset()
                    # -5 to prime the velo() function
                    episode = Episode(-5, 1_000, agent, env)
                    # saving episode data
                    episode.train_start()

                    while True:
                        if agent.actor_file_name in os.listdir(agent.chkpt_dir):
                            agent.load_model()
                            os.remove(agent.actor_file_pth)
                            break
                        time.sleep(0.5)
        except Exception:
            traceback.print_exc()
        env.kill()
    else:  # if PC or laptop
        env = DummyPendulum()
        agent = Agent(env=env)
        rpi = RPIConnect()

        # send initialized actor parameters to rpi
        agent.save_model()
        print("...sending actor params...")
        send_actor_command = f"scp {agent.actor_file_pth} charles@raspberrypi:{os.path.join(pi_pth, agent.chkpt_dir)}"
        get_memory_command = f"scp charles@raspberrypi:{os.path.join(pi_pth, agent.memory_file_pth)} {agent.memory_dir}"
        rpi.sys_command(send_actor_command)

        # check if there is memory file in pi then delete
        files = rpi.ssh_command("cd pendulum/memory ; ls -a").split("\n")
        if agent.memory_file_name in files:
            print("...deleting data file in pi...")
            rpi.ssh_command(f"cd pendulum ; sudo rm -f {agent.memory_file_pth}")

        # setup data logging
        writer = SummaryWriter(log_dir=log_dir)
        n_episodes = 1000
        best_score = env.reward_range[0]
        best_avg_score = best_score
        score_history = []

        print("Start!")
        for i in range(n_episodes):
            # reset values before episode
            critic_loss_count = 0
            actor_loss_count = 0
            critic_loss = 0
            actor_loss = 0

            episode = Episode(-5, 1_000, agent, env)
            # wait for pre-processing
            while True:
                files = rpi.ssh_command("cd pendulum/memory ; ls -a").split("\n")
                if agent.memory_file_name in files:
                    print(get_memory_command)
                    rpi.sys_command(get_memory_command)
                    score = episode.pre_process()
                    shutil.copy(
                        agent.memory_file_pth,
                        os.path.join(agent.memory_dir, f"episode_{i}_data.pkl"),
                    )
                    os.remove(agent.memory_file_pth)
                    break
                time.sleep(0.01)

            # learn the episode (monte carlo)
            if env.time_step > agent.warmup:
                for step in range(episode.episode_time_steps):
                    c_loss, a_loss = agent.learn()

                    if c_loss is not None:
                        critic_loss_count += 1
                        critic_loss += c_loss
                    if a_loss is not None:
                        actor_loss_count += 1
                        actor_loss += a_loss

            # send new agent
            agent.save_model()
            rpi.sys_command(send_actor_command)

            # log training data
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            if critic_loss_count > 0:
                critic_loss /= critic_loss_count
            if actor_loss_count > 0:
                actor_loss /= actor_loss_count
            if avg_score > best_score:
                best_score = avg_score
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save_models()

            writer.add_scalar("train/reward", score, i)
            writer.add_scalar("train/critic_loss", critic_loss, i)
            writer.add_scalar("train/actor_loss", actor_loss, i)
            print(
                "episode",
                i,
                "score %.1f" % score,
                "avg score %.1f" % avg_score,
                "critic loss %.5f" % critic_loss,
                "actor loss %.5f" % actor_loss,
            )
            writer.flush()
