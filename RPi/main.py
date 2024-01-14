import os
import platform
import time
import traceback
import shutil
import argparse

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from environment import Pendulum, DummyPendulum
from td3_fork import Agent, AgentActor
from utils import RPIConnect, Episode, get_pins, get_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_file_name",
        help="file name of the log file",
        default="21st run, resetting optimizer weights,no freezing, 1e-6 lr,update actor interval=5, 30Hz",
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
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--start_retraining",
        help="if true it will start retraining without loading optimizer weights",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--cooldown",
        help="it will take extra COOLDOWN seconds to start an episode",
        default=150.,
        type=float,
    )
    parser.add_argument(
        "--warmup",
        help="warmup before learning starts",
        default=10_000,
        type=int,
    )
    return parser.parse_args()


def main_pi(args, n_episodes, episode_time_steps):
    pins = get_pins()
    env = Pendulum(*pins)
    agent = AgentActor(env=env)
    agent.load_model()
    try:
        if args.evaluate:
            for i in range(1):
                env.reset()
                Episode(-5, episode_time_steps, args.warmup, agent, env).eval_start()
                print(f"episode {i + 1}/1")
        else:
            os.remove(agent.actor_file_pth)
            for i in range(n_episodes):
                # reset episode
                env.reset()
                # -5 to prime the velo() function
                episode = Episode(-5, episode_time_steps, args.warmup, agent, env)
                # saving episode data
                episode.train_start()

                while True:
                    if agent.actor_file_name in os.listdir(agent.chkpt_dir):
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


def main_pc(args, n_episodes, episode_time_steps, pi_pth, log_dir):
    env = DummyPendulum()
    # so that I can save when I stop main
    global agent
    # choose whether to save or load buffer
    agent = Agent(lr=1e-6, env=env, update_actor_interval=5, save_buffer=True, load_buffer=True)
    rpi = RPIConnect()
    send_actor_command = f"scp {agent.actor_file_pth} charles@raspberrypi:{os.path.join(pi_pth, agent.chkpt_dir)}"
    get_memory_command = f"scp charles@raspberrypi:{os.path.join(pi_pth, agent.memory_file_pth)} {agent.memory_dir}"
    if args.evaluate:
        agent.load_models()
        agent.save_model()
        rpi.sys_command(send_actor_command)
        print("...finished sending trained actor...")
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
            episode_jump_start = int(env.time_step / episode_time_steps)
            print(f"{episode_jump_start=}")

        if args.start_retraining:
            print("...starting transfer learning...")
            agent.load_models(reset=True, freeze=False)
            env.time_step = agent.memory.mem_cntr
            # print("...training warmup steps might take awhile...")
            # for i in range(episode_time_steps * 20):
            #     if i % 1000:
            #         print("step", i)
            #     agent.learn()

        # send initialized actor parameters to rpi
        agent.save_model()
        print("...sending actor params...")
        rpi.sys_command(send_actor_command)

        # check if there is memory file in pi then delete
        files = rpi.ssh_command("cd pendulum/memory ; ls -a").split("\n")
        if agent.memory_file_name in files:
            print("...deleting data file in pi...")
            rpi.ssh_command(f"cd pendulum ; sudo rm -f {agent.memory_file_pth}")

        # setup data logging
        writer = SummaryWriter(log_dir=log_dir)
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

            episode = Episode(-5, episode_time_steps, args.warmup, agent, env)
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
            if env.time_step > args.warmup:
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
            if score > 150.:
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


if __name__ == "__main__":
    # sampling rate default for pigpio daemon is 5 us: https://abyz.me.uk/rpi/pigpio/pigpiod.html
    """
    --- edit config.ini, ssh-keygen and copy keys to rpi ---
    --- ensure connection between PC and raspberry first ---
    --- main() should be run first in PC ---
    --- set --log_file_name before running ---
    raspberry pi commands:
    >>> cd pendulum ; source ./venv/bin/activate ; sudo pigpiod -s 5
    >>> sudo nice -n -20 ./venv/bin/python main.py
    """
    args_ = parse_args()
    log_dir_ = f"runs/{args_.log_file_name}"
    _, pi_pth_ = get_paths()
    episode_time_steps_ = 1000
    n_episodes_ = 1000

    if platform.node() == "raspberrypi":
        main_pi(args_, n_episodes_, episode_time_steps_)
    else:
        main_pc(args_, n_episodes_, episode_time_steps_, pi_pth_, log_dir_)
