import os
import pickle
import platform
import time
import traceback
import shutil
import argparse

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from environment import Pendulum, DummyPendulum
from td3_fork import Agent, AgentActor
from utils import RPIConnect, get_pins, get_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_file_name",
        help="file name of the log file",
        default="fixed memory again, 0.9 bounds",
        type=str,
    )
    parser.add_argument(
        "--evaluate",
        help="if true it will evaluate current actor model",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--preload_buffer",
        help="if true it will preload 10,000 time steps of warmup data",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--save_episode_data",
        help="if true it will save each episode data",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--continue_training",
        help="if true it will save each episode data",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--preload_trained_sim_agent",
        help="if true it will load simulation trained agent (will use td3_fork.chkpt)",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--cooldown",
        help="if true it will take extra COOLDOWN seconds to start running again",
        default=120.0,
        type=float,
    )
    return parser.parse_args()


class Episode:
    def __init__(
        self,
        steps,
        episode_time_steps_,
        _agent: (Agent, AgentActor),
        _env: (Pendulum, DummyPendulum),
    ):
        self.cntr = 0
        self.steps = steps
        # number of time steps in an episode
        # 1 episode = 1k time steps
        self.episode_time_steps = episode_time_steps_
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
            if self.env.time_step < self.agent.warmup:
                # 0 to 1000
                action = self.env.action_space.sample()
            else:
                # 0 to 1000
                action = self.agent.choose_action(self.observation)
            self.env.step(action)
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
            self.env.step(action)
            bound = self.env.bound - 0.15
            obs_ = self.env.get_obs()
            reward = np.cos(obs_[1]) - (
                10 * int(abs(obs_[0]) > bound) + 10 * int(abs(obs_[3]) > 13)
            )
            print(obs_, )
            self.observation = obs_
        else:
            self.observation = self.env.get_obs()
        self.steps += 1

    def post_process(self):
        memory = (self.observations, self.actions, self.observations_)
        with open(self.agent.memory_file_pth, "wb") as outfile:
            pickle.dump(memory, outfile, pickle.HIGHEST_PROTOCOL)

    def pre_process(self):
        # bounds for reward, change in environment.py
        bound = self.env.bound - 0.15

        with open(self.agent.memory_file_pth, "rb") as infile:
            result = pickle.load(infile)
        (observations, actions, observations_) = result
        # create fake dones cuz it'll always be like this anyway
        dones = np.zeros((self.episode_time_steps, 1), dtype=np.bool_)
        dones[-1] = [True]
        score_ = 0
        for obs, action, obs_, done in zip(observations, actions, observations_, dones):
            # from 0. to 1000.
            # to 0. to 1.
            action *= 0.001
            # compute for reward
            reward = np.cos(obs_[1]) - (
                10 * int(abs(obs_[0]) > bound) + 10 * int(abs(obs_[3]) > 13)
            )
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


if __name__ == "__main__":
    # TODO: fix this nesting problem
    # edit config.ini, ssh-keygen and copy keys to rpi
    # sudo nice -n -20 ./pendulum/venv/bin/python ./pendulum/main.py
    # make sure there is no ./models/td3_fork_actor.chkpt in raspberrypi and PC that is important
    # make sure there is no ./memory/td3_fork_memory.pkl in raspberrypi and PC that is important
    """
    --- ensure connection between PC and raspberry first ---
    --- main() should be run first in PC ---
    --- set --log_file_name before running ---
    """
    args = parse_args()
    log_dir = f"runs/{args.log_file_name}"
    pc_pth, pi_pth = get_paths()

    if platform.node() == "raspberrypi":
        pins = get_pins()
        env = Pendulum(*pins)
        agent = AgentActor(env=env)
        agent.load_model()
        n_episodes = 1000
        try:
            if args.evaluate:
                for i in range(5):
                    env.reset()
                    Episode(-5, 1_000, agent, env).eval_start()
                    print(f"episode {i + 1}/5")
            else:
                os.remove(agent.actor_file_pth)
                for i in range(n_episodes):
                    # reset episode
                    env.reset()
                    # -5 to prime the velo() function
                    episode = Episode(-5, 1_000, agent, env)
                    # saving episode data
                    episode.train_start()

                    while True:
                        if agent.actor_file_name in os.listdir(agent.chkpt_dir):
                            # try and try again until you succeed :))
                            # mag bonakid preschool three plus
                            try:
                                # cool the motors and drivers
                                time.sleep(args.cooldown)
                                agent.load_model()
                            except (RuntimeError, OSError, EOFError):
                                continue
                            os.remove(agent.actor_file_pth)
                            break
                        time.sleep(1.0)
                    print("episode", i)
        except Exception:
            traceback.print_exc()
        env.kill()
    else:  # if PC or laptop
        env = DummyPendulum()
        # loads buffer and it gets override by args.continue_training parameter
        agent = Agent(
            env=env, save_buffer=True, load_buffer=(True or args.continue_training)
        )
        rpi = RPIConnect()
        episode_time_steps = 1000
        if args.evaluate:
            agent.load_models()
            agent.save_model()
            rpi.sys_command(send_actor_command)
        else:
            # preload warmup data to replay buffer
            if args.preload_buffer:
                agent.preload_buffer()
                env.time_step = agent.memory.mem_cntr

            episode_jump_start = 0
            if args.continue_training:
                print("...continuing...")
                agent.load_models()
                env.time_step = agent.memory.mem_cntr
                episode_jump_start = int(agent.memory.mem_cntr / episode_time_steps)

            # send initialized actor parameters to rpi
            # TODO: try and see if the saved actor parameters is the same to the sent actor parameters for at least 5 episodes with learning
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
                i += episode_jump_start
                # reset values before episode
                critic_loss_count = 0
                actor_loss_count = 0
                critic_loss = 0
                actor_loss = 0

                episode = Episode(-5, episode_time_steps, agent, env)
                # wait for pre-processing
                while True:
                    files = rpi.ssh_command("cd pendulum/memory ; ls -a").split("\n")
                    if agent.memory_file_name in files:
                        rpi.sys_command(get_memory_command)
                        rpi.ssh_command(f"cd pendulum ; sudo rm -f {agent.memory_file_pth}")
                        score = episode.pre_process()
                        if args.save_episode_data:
                            shutil.copy(
                                agent.memory_file_pth,
                                os.path.join(agent.memory_dir, f"episode_{i}_data.pkl"),
                            )
                        os.remove(agent.memory_file_pth)
                        break
                    time.sleep(0.1)

                # learn the episode (monte carlo)
                if env.time_step > agent.warmup:
                    for step in range(episode_time_steps):
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
                if i % 10 == 0:
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
