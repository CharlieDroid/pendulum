import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from td3_fork import Agent
from torch.utils.tensorboard import SummaryWriter
import torch as T
import os

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
    filename = "testing"
    chkpt_dir = "./tmp/td3 fork"
    log_dir = f"runs/inverted_pendulum_sim/{filename}"
    # chkpt_dir = "./drive/MyDrive/pendulum/tmp/td3 fork"
    # log_dir = f"./drive/MyDrive/pendulum/runs/inverted_pendulum_sim/{filename}"
    env = gym.make(game_id)

    seed = None
    if seed is not None:
        np.random.seed(seed)
        T.manual_seed(seed)

    # a warmup of 20,000 steps irl
    # don't forget to change max action back to 3.
    lr = 0.001
    agent = Agent(
        alpha=lr,
        beta=lr,
        input_dims=env.observation_space.shape,
        tau=0.005,
        env=env,
        gamma=0.98,
        noise=0.1,
        policy_noise=0.2,
        layer1_size=256,
        layer2_size=256,
        sys_weight=0.6,
        update_actor_interval=1,
        max_size=1_000_000,
        n_actions=env.action_space.shape[0],
        game_id=game_id,
        chkpt_dir=chkpt_dir,
    )
    # agent.partial_load_models()
    writer = SummaryWriter(log_dir=log_dir)
    n_timesteps = 1_000_000
    episode = 1_000  # 1 episode = 1k timesteps

    best_score = env.reward_range[0]
    best_avg_score = best_score
    score_history = []
    critic_loss_count = 0
    actor_loss_count = 0
    critic_loss = 0
    actor_loss = 0
    score = 0
    steps = 0
    done = True

    for step in range(n_timesteps):
        if done:
            for _ in range(steps):
                c_loss, a_loss = agent.learn()

                if c_loss is not None:
                    critic_loss_count += 1
                    critic_loss += c_loss
                if a_loss is not None:
                    actor_loss_count += 1
                    actor_loss += a_loss
            steps = 0
            observation, info = env.reset(seed=seed)

        action = agent.choose_action(observation)
        observation_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.remember(observation, action, reward, observation_, done)
        steps += 1
        # c_loss, a_loss = agent.learn()
        observation = observation_
        writer.add_scalar("train/return", reward, step)
        score += reward

        # if c_loss is not None:
        #     critic_loss_count += 1
        #     critic_loss += c_loss
        # if a_loss is not None:
        #     actor_loss_count += 1
        #     actor_loss += a_loss

        if (step + 1) % episode == 0:
            i = int(step / episode)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            if critic_loss_count > 0:
                critic_loss /= critic_loss_count
            if actor_loss_count > 0:
                actor_loss /= actor_loss_count
            if avg_score > best_score:
                best_score = avg_score

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

            if score >= 990:
                perfect_score_count += 1
                if perfect_score_count >= 10:
                    print("...environment solved...")
                    agent.save_models()
                    break
            else:
                perfect_score_count = 0

            if avg_score >= best_avg_score:
                best_avg_score = avg_score
                agent.save_models()
            critic_loss_count = 0
            actor_loss_count = 0
            critic_loss = 0
            actor_loss = 0
            score = 0
            writer.flush()

    writer.close()
