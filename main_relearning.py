""" Upon this rock, I will build my church """
import numpy as np
from ppo_relearning import Agent
from environment import Env
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    # make sure to run in ipython console
    env = Env("./data")
    horizon = 64
    batch_size = 16
    n_epochs = 30
    alpha = 0.00025
    n_games = 300
    gamma = 0.99
    policy_clip = 0.2
    agent = Agent(n_actions=9, batch_size=batch_size, n_games=n_games, gamma=gamma, policy_clip=policy_clip,
                  alpha=alpha, n_epochs=n_epochs, input_dims=(2,))
    agent.actor.checkpoint_file = "./models/actor_torch_ppo_learn_3"
    agent.critic.checkpoint_file = "./models/critic_torch_ppo_learn_3"
    agent.load_models()
    agent.actor.checkpoint_file = "./tmp/ppo_learn/actor_torch_ppo_learn"
    agent.critic.checkpoint_file = "./tmp/ppo_learn/critic_torch_ppo_learn"
    # writer = SummaryWriter(log_dir=f"runs/robot/train 17,relearning,256 neurons")
    writer = SummaryWriter(log_dir=f"runs/robot/relearning 5")
    best_score = 0
    score_history = []
    losses = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    loss_ = 0

    for i in range(n_games):
        episode_data = []
        observation = env.reset()
        done = False
        score = 0
        while not done:
            observation_, reward, done, action = env.step()
            action, prob, val = agent.choose_action(observation, action)

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
        else:
            loss = loss_
        agent.decay_lr()
        agent.decay_clip()
        writer.add_scalar("train/reward", score, i)
        writer.add_scalar("train/loss", loss, i)
        writer.add_scalar("train/learning rate", agent.alpha, i)
        writer.add_scalar("train/policy clip", agent.policy_clip, i)
        writer.add_scalar("train/epochs", agent.n_epochs, i)
        score_history.append(score)
        losses.append(loss)
        loss_ = loss
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        print("episode", i, "learning_steps", learn_iters, "score %.1f" % score, "avg score %.1f" % avg_score,
              "learning_rate %.6f" % agent.alpha, "loss %.6f" % loss)
        writer.flush()
    agent.save_models()
    writer.close()
