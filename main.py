"""
Notes:
    tensorboard command is: tensorboard --logdir=runs/pendulum_sim
    edit "C:\ProgramData\anaconda3\Lib\site-packages\tensorboard\_vendor\html5lib\_trie\_base.py" imports to
    from collections.abc import Mapping
    edit "C:\ProgramData\anaconda3\Lib\site-packages\gymnasium\core.py"
    indent line 310 and 311
"""
import gymnasium as gym
import numpy as np
from td3 import Agent
from torch.utils.tensorboard import SummaryWriter
import torch as T
from utils import plot_learning_curve, agent_play


if __name__ == "__main__":
    # tensorboard --logdir=runs/pendulum_sim
    # r_coeff = 0.004 / (max_torque ** 2)
    game_id = "Pendulum-v1"
    torque = 0.071
    p_len = 0.25
    mass = 0.2
    env = gym.make(
        game_id, g=9.80665, t=torque, l=p_len, m=mass, max_s=10, r_coeff=0.793
    )
    seed = None
    if seed is not None:
        np.random.seed(seed)
        T.manual_seed(seed)
    buffer_size = 200_000
    lr = 0.001
    fc1 = 400
    fc2 = 300
    agent = Agent(
        alpha=lr,
        beta=lr,
        input_dims=env.observation_space.shape,
        tau=0.005,
        env=env,
        batch_size=100,
        layer1_size=fc1,
        layer2_size=fc2,
        noise=0.1,
        max_size=buffer_size,
        gamma=0.98,
        update_actor_interval=1,
        warmup=10_000,
        n_actions=env.action_space.shape[0],
    )
    # filename = (
    #     f"{seed=},{lr=},tau={agent.tau},batch_size={agent.batch_size},{fc1=},{fc2=},noise={agent.noise},"
    #     f"buffer_size={agent.memory.mem_size},gamma={agent.gamma},train_freq={agent.update_actor_iter},"
    #     f"warmup={agent.warmup}"
    # )
    filename = f"length={p_len},{torque=},{mass=}"
    writer = SummaryWriter(log_dir=f"runs/pendulum_sim/{filename}")
    n_games = 200
    filename = "plots/" + "PendulumContinuous_" + str(n_games) + "_games.png"

    best_score = env.reward_range[0]
    score_history = []

    # agent.load_models()
    # agent_play(game_id, agent)

    for i in range(n_games):
        critic_loss_count = 0
        actor_loss_count = 0
        critic_loss = 0
        actor_loss = 0

        observation, info = env.reset(seed=seed)
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            c_loss, a_loss = agent.learn()
            score += reward
            observation = observation_
            done = terminated or truncated

            if c_loss is not None:
                critic_loss_count += 1
                critic_loss += c_loss
            if a_loss is not None:
                actor_loss_count += 1
                actor_loss += a_loss
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if critic_loss_count > 0:
            critic_loss /= critic_loss_count
        if actor_loss_count > 0:
            actor_loss /= actor_loss_count

        if avg_score > best_score:
            best_score = avg_score
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

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, filename)
    writer.close()
