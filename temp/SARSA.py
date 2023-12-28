from gridworld import actionSpace, create_grid
from windy_gridworld import windy_grid_penalized
from matplotlib import pyplot as plt
import numpy as np


def print_policy(policy, g):
    for r in range(g.rows):
        print("------" * g.cols)
        for c in range(g.cols):
            print(f"  {policy.get((c, r), ' ')}  |", end="")
        print()


def print_values(value, g):
    for r in range(g.rows):
        print("------" * g.cols)
        for c in range(g.cols):
            val = value.get((c, r), 0.0)
            if val >= 0:
                print(f" {val:0.2f}|", end="")
            else:
                print(f"{val:0.2f}|", end="")
        print()


def max_dict(d):
    maxVal = max(d.values())
    maxKeys = [key for key, val in d.items() if val == maxVal]
    return np.random.choice(maxKeys), maxVal


def epsilon_greedy(Q, state, eps=0.1):
    if np.random.random() < eps:
        return np.random.choice(actionSpace)
    else:
        return max_dict(Q[state])[0]


if __name__ == "__main__":
    grid = windy_grid_penalized(-0.1)
    # grid = create_grid()

    print("rewards:")
    print_values(grid.rewards, grid)

    Q = {}
    for state in grid.all_states():
        Q[state] = {}
        for action in actionSpace:
            Q[state][action] = 0.0
    alpha = 0.1
    gamma = 0.9
    threshold = 1e-3
    update_counts = {}
    reward_per_episode = []

    for i in range(10_000):
        if i % 2000 == 0 and i > 0:
            print(f"{i} episodes")

        s = grid.reset()
        a = epsilon_greedy(Q, s)

        episode_reward = 0
        while not grid.game_over():
            r = grid.move(a)
            s_ = grid.current_state()

            episode_reward += r

            a_ = epsilon_greedy(Q, s_)
            Q[s][a] = Q[s][a] + alpha * (r + (gamma * Q[s_][a_]) - Q[s][a])
            update_counts[s] = update_counts.get(s, 0) + 1
            s = s_
            a = a_
        reward_per_episode.append(episode_reward)

    plt.plot(reward_per_episode)
    plt.title("reward_per_episode")
    plt.show()

    policy = {}
    V = {}
    for state in grid.all_states():
        a, maxQ = max_dict(Q[state])
        V[state] = maxQ
        policy[state] = a

    print("update counts:")
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
