""" Upon this rock, I will build my church """
import gym
import numpy as np
from math import pi
from ppo_mod import Agent
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    plt.clf()


def agent_play(env, agent, eps=5):
    for _ in range(eps):
        observation = env.reset()
        done = False
        while not done:
            env.render()
            action, prob, val = agent.choose_action(observation)
            observation_, _, done, _ = env.step(action)
            observation = observation_
    env.close()


if __name__ == "__main__":
    # tensorboard --logdir=runs/mod
    env = gym.make('Pendulum-v1', g=9.81)
    horizon = 1024
    batch_size = 128
    n_epochs = 10
    alpha = 0.000001
    n_games = 1000
    gamma = 0.9
    policy_clip = 0.1
    entropy = 0.001
    agent = Agent(n_actions=1, batch_size=batch_size, n_games=n_games, gamma=gamma, policy_clip=policy_clip,
                  alpha=alpha, entropy=entropy, n_epochs=n_epochs, input_dims=(3,))
    writer = SummaryWriter(log_dir=f"runs/mod/{batch_size=},{n_epochs=},{horizon=},{alpha=},{n_games=},{gamma=},"
                                   f"{entropy=},{policy_clip=},neurons=64,relu,tanh activation,1")
    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []
    losses = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    perfect_score_count = 0

    for i in range(n_games):
        episode_data = []
        observation = env.reset()
        done = False
        score = 0
        loss = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            action_ = (action * 4) - 2  # action * 4 - 2
            observation_, reward, done, info = env.step([action_])

            score += reward
            episode_data.append((observation, action, prob, val, reward, done))
            observation = observation_

        loss_count = 0
        loss = 0
        for data in episode_data:
            agent.remember(*data)
            n_steps += 1
            if n_steps % horizon == 0:
                loss += agent.learn()
                n_steps = 0
                loss_count += 1
                learn_iters += 1
        if loss_count > 0:
            loss /= loss_count
        agent.decay_lr()
        agent.decay_clip()
        writer.add_scalar("train/reward", score, i)
        writer.add_scalar("train/loss", loss, i)
        writer.add_scalar("train/learning rate", agent.alpha, i)
        writer.add_scalar("train/policy clip", agent.policy_clip, i)
        writer.add_scalar("train/epochs", agent.n_epochs, i)
        score_history.append(score)
        losses.append(loss)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        if score >= -200.:
            perfect_score_count += 1
            if perfect_score_count >= 20:
                print("Early Stopping")
                break
        else:
            perfect_score_count = 0

        print("episode", i, "learning_steps", learn_iters, "score %.1f" % score, "avg score %.1f" % avg_score,
              "learning_rate %.6f" % agent.alpha, "loss %.6f" % loss)
        writer.flush()
    agent.save_models()
    x1 = [i + 1 for i in range(len(losses))]
    x2 = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x1, losses, 'plots/norm_reward_losses.png')
    plot_learning_curve(x2, score_history, 'plots/norm_reward_scores.png')
    writer.close()
