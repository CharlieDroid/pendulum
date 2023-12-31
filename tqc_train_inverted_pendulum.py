import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from tqc import Agent, RescaleAction
from torch.utils.tensorboard import SummaryWriter
import torch as T
from utils import agent_play

# Mujoco Modded by Charles
# ----------------------------------------
# modded inverted_pendulum.xml
# changed inverted_pendulum.xml file path in inverted_pendulum_mod.py
# changed to fromto="0 0 0 0.001 0 -0.6" in "cpole" (pole resets downwards)
# changed to range="-180 180" in "pole"
# rewards to cos(theta)
# does not terminate until max timesteps reached
# changed reset_model to have a min and high of -0.1 and 0.1 respectively
# changed reward function to -(theta^2 + 10*cart^2 + 0.1*theta_dt^2 + 0.1*cart_dt^2 + 0.001*torque^2)

register(
    id="InvertedPendulumModded",
    entry_point="mujoco_mod.envs.inverted_pendulum_mod:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=0.0,
)

if __name__ == "__main__":
    game_id = "InvertedPendulumModded"
    filename = "tqc agent"
    env = gym.make(game_id).unwrapped
    env = RescaleAction(env, -1.0, 1.0)

    seed = None
    if seed is not None:
        np.random.seed(seed)
        T.manual_seed(seed)

    agent = Agent(
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        game_id=game_id,
    )
    writer = SummaryWriter(log_dir=f"runs/inverted_pendulum_sim/{filename}")
    n_games = 1000

    best_score = env.reward_range[0]
    best_avg_score = best_score
    score_history = []

    # agent.load_models()
    # agent.time_step = agent.warmup + 1

    for i in range(n_games):
        critic_loss_count = 0
        alpha_loss_count = 0
        actor_loss_count = 0
        alpha_loss = 0
        critic_loss = 0
        actor_loss = 0

        observation, info = env.reset(seed=seed)
        steps = 0
        done = False
        score = 0
        while (steps < 100) or (not done):
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            # add actor and critic loss
            c_loss, a_loss, al_loss = agent.learn()
            score += reward
            observation = observation_
            done = terminated or truncated
            steps += 1

            if c_loss is not None:
                critic_loss_count += 1
                critic_loss += c_loss
            if al_loss is not None:
                alpha_loss_count += 1
                alpha_loss += al_loss
            if a_loss is not None:
                actor_loss_count += 1
                actor_loss += a_loss
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if critic_loss_count > 0:
            critic_loss /= critic_loss_count
        if alpha_loss_count > 0:
            alpha_loss /= alpha_loss_count
        if actor_loss_count > 0:
            actor_loss /= actor_loss_count

        if avg_score > best_score:
            best_score = avg_score

        if i % 10 == 0:
            agent.save_models()
        elif score > 880:
            agent.save_models()

        writer.add_scalar("train/reward", score, i)
        writer.add_scalar("train/critic_loss", critic_loss, i)
        writer.add_scalar("train/alpha_loss", alpha_loss, i)
        writer.add_scalar("train/actor_loss", actor_loss, i)
        print(
            "episode",
            i,
            "score %.1f" % score,
            "avg score %.1f" % avg_score,
            "critic loss %.5f" % critic_loss,
            "actor loss %.5f" % actor_loss,
        )

        if score >= 880:
            perfect_score_count += 1
            if perfect_score_count >= 10:
                print("...environment solved...")
                agent.save_models()
                break
        else:
            perfect_score_count = 0

        writer.flush()

    writer.close()
