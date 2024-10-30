import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
import csv
from mujoco_mod.envs.domain_randomization import DomainRandomization, simp_angle
from mujoco_mod.envs.domain_randomization_double import (
    DomainRandomization as DomainRandomizationDouble,
)
import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt


def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text(
        (im.size[0] / 20, im.size[1] / 18), f"Episode: {episode_num+1}", fill=text_color
    )

    return im


def save_random_agent_gif(env):
    frames = []
    for i in range(5):
        state = env.reset()
        for t in range(500):
            action = env.action_space.sample()

            frame = env.render(mode="rgb_array")
            frames.append(_label_with_episode_number(frame, episode_num=i))

            state, _, done, _ = env.step(action)
            if done:
                break

    env.close()

    imageio.mimwrite(os.path.join("./videos/", "random_agent.gif"), frames, fps=60)


def run_env(game_id="InvertedPendulumModded", eps=1):
    from gymnasium.envs.registration import register

    register(
        id="InvertedPendulumModded",
        entry_point="mujoco_mod.envs.inverted_pendulum_mod:InvertedPendulumEnv",
        max_episode_steps=500,
        reward_threshold=1000.0,
    )

    env = gym.make(game_id, render_mode="human")

    for _ in range(eps):
        steps = 0
        obs, info = env.reset()
        done = False
        max_action = [1.6]
        max_speed = float("inf")
        while not done:
            # action = env.action_space.sample()
            if steps < 200:
                action = max_action
                if obs[0] < -0.8:
                    action = [max_action[0] / 2 * 0.6]
                elif obs[0] > 0.8:
                    action = [-max_action[0] / 2 * 0.6]
            elif steps > 200:
                action = [-max_action[0]]
                if obs[0] < -0.8:
                    action = [max_action[0] / 2 * 0.6]
                elif obs[0] > 0.8:
                    action = [-max_action[0] / 2 * 0.6]

            observation_, reward, terminated, truncated, info = env.step(action)
            obs = observation_
            print(
                f"pos:{obs[0]:.2f} angle:{obs[1]:.4f} lin_vel:{obs[2]:.2f} ang_vel:{obs[3]:.2f} reward:{reward:.2f}"
            )
            if abs(obs[1]) < max_speed:
                max_speed = abs(obs[1])
            done = terminated or truncated
            steps += 1
    print(max_speed)
    env.close()


def agent_play(game_id, agent, eps=3, save=True, find_best=False, save_data=False):
    if save:
        render_mode = "rgb_array"
    else:
        render_mode = "human"
    if game_id == "Pendulum-v1":
        env = gym.make(game_id, render_mode=render_mode, g=9.80665)
    else:
        env = gym.make(game_id, render_mode=render_mode)
    if save:
        env = RecordVideo(
            env,
            "recordings",
            name_prefix=game_id,
            episode_trigger=lambda _: True,
        )
    if save_data:
        eps = 1
        data = []
    if not find_best:
        rewards = 0
        domain_randomizer = DomainRandomization()
        domain_randomizer.reset_environment()
        for ep in range(eps):
            frames = []
            observation, info = env.reset()
            obs = observation
            obs[1] = simp_angle(obs[1])
            print(
                f"pos:{obs[0]:.2f} angle:{obs[1]:.2f} lin_vel:{obs[2]:.2f} ang_vel:{obs[3]:.2f}"
            )
            if save:
                env.start_video_recorder()
            done = False
            max_speed = float("-inf")
            time_steps = 0
            while not done:
                action = agent.choose_action(observation, evaluate=True)
                observation_, reward, terminated, truncated, info = env.step(action)
                observation_ = domain_randomizer.observation(observation_, env)
                observation_[1] = simp_angle(observation_[1])
                if not save:
                    obs = observation_
                    print(
                        f"pos:{obs[0]:.2f} angle:{obs[1]:.2f} lin_vel:{obs[2]:.2f} ang_vel:{obs[3]:.2f} reward:{reward:.2f} action:{action.item():.2f}"
                    )
                if save_data:
                    data.append(
                        [
                            observation[0],
                            observation[1],
                            observation[2],
                            observation[3],
                            reward,
                            action.item(),
                        ]
                    )
                rewards += reward
                observation = observation_
                done = terminated or truncated
                time_steps += 1
            print(time_steps)
        env.close()

        if save_data:
            with open("recordings/sample_data_16x16.csv", "w") as file:
                writer = csv.writer(file)
                for d in data:
                    writer.writerow(d)

        return rewards / eps


def agent_play_double(
    game_id,
    agent_balance,
    agent_swingup=None,
    agent_0=None,
    agent_1=None,
    agent_2=None,
    eps=3,
    save=True,
    find_best=False,
    save_data=False,
    combined=False,
    teachers=False,
):
    if save:
        render_mode = "rgb_array"
    else:
        render_mode = "human"
    if game_id == "Pendulum-v1":
        env = gym.make(game_id, render_mode=render_mode, g=9.80665)
    else:
        env = gym.make(game_id, render_mode=render_mode)
    if save:
        env = RecordVideo(
            env,
            "recordings",
            name_prefix=game_id,
            episode_trigger=lambda _: True,
            # disable_logger=True,
        )
    if save_data:
        eps = 1
        data = []
    if not find_best:
        rewards = 0
        domain_randomizer = DomainRandomizationDouble(4000)
        domain_randomizer.reset_environment()
        for ep in range(eps):
            domain_randomizer.environment()
            env = gym.make(game_id, render_mode=render_mode)
            observation, info = env.reset()
            obs = observation
            a1, a2 = env.data.qpos[1:]
            a2 += a1
            a1 = simp_angle(a1)
            a2 = simp_angle(a2)
            print(
                f"pos:{obs[0]:.2f} a1:{a1:.2f} a2:{a2:.2f} lin_vel:{obs[5]:.2f} a1_vel:{obs[6]:.2f} a2_vel:{obs[7]:.2f} tip_x:{obs[8]:.2f} tip_y:{obs[9]:.2f} a1:{obs[10]:.2f} a2:{obs[11]:.2f}"
            )
            if save:
                env.start_video_recorder()
            done = False
            max_speed = float("-inf")
            time_steps = 0
            ready_balance = False
            # import time
            #
            # time.sleep(2)
            observation = domain_randomizer.observation(observation, env)
            while not done:
                action = agent_balance.choose_action(observation, evaluate=True)
                if not save_data:
                    action = domain_randomizer.action(action)
                # action[0] = 0.0
                observation_, reward, terminated, truncated, info = env.step(action)
                observation_ = domain_randomizer.observation(observation_, env)
                observation_ = np.clip(
                    observation_, agent_balance.obs_lower_bound_ideal, agent_balance.obs_upper_bound_ideal)

                if not save:
                    obs = observation_
                    a1, a2 = env.data.qpos[1:]
                    a2 += a1
                    a1 = simp_angle(a1)
                    a2 = simp_angle(a2)
                    if teachers:
                        print(
                            f"pos:{obs[0]:.2f} a1:{a1:.2f} a2:{a2:.2f} lin_vel:{obs[5]:.2f} a1_vel:{obs[6]:.2f} a2_vel:{obs[7]:.2f} p1:{obs[8]:.0f} p2:{obs[9]:.0f} tip_x:{obs[10]:.2f} tip_y:{obs[11]:.2f} a1:{obs[12]:.2f} a2:{obs[13]:.2f} reward:{reward:.2f} action:{action[0]:.2f}"
                        )
                    else:
                        print(
                            f"pos:{obs[0]:.2f} a1:{a1:.2f} a2:{a2:.2f} lin_vel:{obs[5]:.2f} a1_vel:{obs[6]:.2f} a2_vel:{obs[7]:.2f} tip_x:{obs[8]:.2f} tip_y:{obs[9]:.2f} a1:{obs[10]:.2f} a2:{obs[11]:.2f} reward:{reward:.2f} action:{action[0]:.2f}"
                        )
                if save_data:
                    if teachers:
                        data.append(
                            [
                                observation[0],
                                observation[1],
                                observation[2],
                                observation[3],
                                observation[4],
                                observation[5],
                                observation[6],
                                observation[7],
                                observation[8],
                                observation[9],
                                reward,
                                action.item(),
                            ]
                        )
                    else:
                        data.append(
                            [
                                observation[0],
                                observation[1],
                                observation[2],
                                observation[3],
                                observation[4],
                                observation[5],
                                observation[6],
                                observation[7],
                                reward,
                                action.item(),
                            ]
                        )
                rewards += reward
                observation = observation_
                done = terminated or truncated
                time_steps += 1
            print(time_steps)
            # if save:
            #     imageio.mimwrite(
            #         os.path.join("./recordings/", f"{game_id} {ep}.gif"),
            #         frames,
            #         fps=60,
            #     )
        env.close()
        domain_randomizer.reset_environment()

        if save_data:
            with open("recordings/sample_data_16x16.csv", "w") as file:
                writer = csv.writer(file)
                for d in data:
                    writer.writerow(d)

        return rewards / eps


def testing(env):
    for _ in range(2):
        env.reset()
        done = False
        env.unwrapped.state = np.array([2 * np.pi, 1.0])
        while not done:
            _, _, terminated, truncated, _ = env.step([env.max_torque])
            done = terminated or truncated
    env.close()
