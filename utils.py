import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 scores")
    # plt.savefig(figure_file)
    plt.show()
    plt.clf()


def agent_play(game_id, agent, eps=3):
    env = gym.make(game_id, render_mode="human", g=9.80665)
    rewards = 0
    for _ in range(eps):
        observation, info = env.reset()
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
