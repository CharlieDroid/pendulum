import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


def run_env(game_id="InvertedPendulumModded", eps=1):
    from gymnasium.envs.registration import register

    register(
        id="InvertedPendulumModded",
        entry_point="mujoco_mod.envs.inverted_pendulum_mod:InvertedPendulumEnv",
        max_episode_steps=500,
        reward_threshold=1000.0,
    )

    env = gym.make(game_id, render_mode="human")
    max_speed = float('-inf')
    for _ in range(eps):
        steps = 0
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            if steps > 300:
                action = [3.]
            else:
                action = [-3.]
            observation_, reward, terminated, truncated, info = env.step(action)
            if abs(observation_[2]) > max_speed:
                max_speed = abs(observation_[2])
            print(f"{observation_} | {reward}")
            done = terminated or truncated
            steps += 1
    env.close()
    print(max_speed)


def agent_play(game_id, agent, eps=3, save=True):
    if save:
        render_mode = "rgb_array"
    else:
        render_mode = "human"
    if game_id == "Pendulum-v1":
        env = gym.make(game_id, render_mode=render_mode, g=9.80665)
    else:
        env = gym.make(game_id, render_mode=render_mode)
    if save:
        env = RecordVideo(env, "recordings", name_prefix="InvertedPendulum",
                          episode_trigger=lambda _: True)
    rewards = 0
    for ep in range(eps):
        observation, info = env.reset()
        if save:
            env.start_video_recorder()
        done = False
        while not done:
            action = agent.choose_action(observation, evaluate=True)
            observation_, reward, terminated, truncated, info = env.step(action)
            rewards += reward
            observation = observation_
            done = terminated or truncated
    env.close()
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
