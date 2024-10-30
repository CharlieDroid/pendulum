import td3_fork_double as td3_fork

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from mujoco_mod.envs.domain_randomization import simp_angle
from mujoco_mod.envs.domain_randomization_double import DomainRandomization
import torch as T
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

register(
    id="InvertedDoublePendulumModded",
    entry_point="mujoco_mod.envs.inverted_double_pendulum_mod:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=0.0,
)


def make_env(rank, game_id, seed=0):
    def _init():
        register(
            id="InvertedDoublePendulumModded",
            entry_point="mujoco_mod.envs.inverted_double_pendulum_mod:InvertedDoublePendulumEnv",
            max_episode_steps=1000,
            reward_threshold=0.0,
        )

        env_ = gym.make(game_id)
        env_.reset(seed=seed + rank)
        return env_

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    isDomainRandomization = True
    game_id = "InvertedDoublePendulumModded"
    filename = "testing"
    chkpt_dir = "./tmp/td3 fork"
    log_dir = f"runs/double_inverted_pendulum_sim/{filename}"
    # chkpt_dir = "./drive/MyDrive/pendulum/tmp/td3 fork"
    # log_dir = f"./drive/MyDrive/pendulum/runs/double_inverted_pendulum_sim/{filename}"
    num_envs = 10
    env = gym.make(game_id)
    # comm 0 about 1400 gradient_clip=1.0 beta_ep_max=100
    # comm 1 about 5400 gradient_clip=1.0 beta_ep_max=150
    # comm 2 about 5400 gradient_clip=1.0 beta_ep_max=200
    # comm 3 about 1700 gradient_clip=1.0 beta_ep_max=100 sys_threshold=0.018
    # beta_ep_max = 200
    highest_score = 4000
    domain_randomizer = DomainRandomization(highest_score)

    seed = None
    if seed is not None:
        np.random.seed(seed)
        T.manual_seed(seed)
    # balance: gamma=0.98, sys_threshold=0.001
    # swingup: gamma=0.99, sys_threshold=0.03
    # swingup comm 2: gamma=0.98, sys_threshold=0.03
    lr = 0.001
    agent = td3_fork.Agent(
        alpha=lr,
        beta=lr,
        ln=True,
        warmup=10000,  # can be changed now
        input_dims=env.observation_space.shape,
        tau=0.005,
        env=env,
        n_extra_obs=14,
        gamma=0.99,
        noise=0.2,  # choose action noise
        noise_clip=0.5,
        gradient_clip=1.0,
        policy_noise=0.2,  # learning action noise
        layer1_size=80,
        layer2_size=60,
        critic1_size=400,
        critic2_size=300,
        sys1_size=400,
        sys2_size=300,
        r1_size=256,
        r2_size=256,
        sys_weight=0.6,
        update_actor_interval=1,
        max_size=1_000_000,
        sys_threshold=0.01,
        l2_regularization=1e-5,
        distill=False,
        n_actions=env.action_space.shape[0],
        game_id=game_id,
        chkpt_dir=chkpt_dir,
    )
    # agent.load_models(load_all_weights=True, load_optimizers=True)
    # agent.memory.load(agent.buffer_file_pth)
    # agent.chkpt_file_pth = os.path.join(chkpt_dir, f"{game_id} td3 fork.chkpt")

    # agent.partial_load_models()
    writer = SummaryWriter(log_dir=log_dir)
    n_timesteps = 1_000_000
    episode = 1_000  # 1 episode = 1k timesteps

    best_score = env.reward_range[0]
    best_avg_score = best_score
    step_skip = (
        agent.memory.mem_cntr
    )  # extra step when loading, episode also gets affected
    score_history = []
    critic_loss_count = 0
    actor_loss_count = 0
    critic_loss = 0
    actor_loss = 0
    system_loss_count = 0
    reward_loss_count = 0
    system_loss = 0
    reward_loss = 0
    score = 0
    steps = 0
    done = True
    domain_randomizer.reset_environment()

    for step in range(n_timesteps):
        step += step_skip
        if done:
            steps = 0
            if isDomainRandomization:
                ep = int(step / episode)
                if domain_randomizer.check_level_up(np.mean(score_history[-10:]), ep):
                    best_avg_score -= highest_score * 0.5
                    best_score -= highest_score * 0.5
                if domain_randomizer.difficulty_level >= 1:
                    agent.memory.tree.beta = 0.6
                    agent.gradient_clip = 0.5
                if domain_randomizer.difficulty_level >= 3:
                    agent.sys_threshold = 0.035
                domain_randomizer.environment()
            env = gym.make(game_id)
            observation, info = env.reset(seed=seed)
            if isDomainRandomization:
                observation = domain_randomizer.observation(observation, env)
                observation = np.clip(
                    observation,
                    agent.obs_lower_bound_ideal,
                    agent.obs_upper_bound_ideal,
                )

        action = agent.choose_action(observation)

        if isDomainRandomization:
            action = domain_randomizer.action(action)

        observation_, reward, terminated, truncated, info = env.step(action)

        if isDomainRandomization:
            observation_ = domain_randomizer.observation(observation_, env)
            observation_ = np.clip(
                observation_, agent.obs_lower_bound_ideal, agent.obs_upper_bound_ideal
            )

        done = terminated or truncated
        agent.remember(observation, action, reward, observation_, done)
        observation = observation_
        writer.add_scalar("train/return", reward, step)
        score += reward

        steps += 1
        c_loss, a_loss, s_loss, r_loss = agent.learn()

        if c_loss is not None:
            critic_loss_count += 1
            critic_loss += c_loss
        if a_loss is not None:
            actor_loss_count += 1
            actor_loss += a_loss
        if s_loss is not None:
            system_loss_count += 1
            system_loss += s_loss
        if r_loss is not None:
            reward_loss_count += 1
            reward_loss += r_loss

        if (step + 1) % episode == 0:
            i = int(step / episode)
            # agent.memory.tree.anneal_beta(i, beta_ep_max)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            if critic_loss_count > 0:
                critic_loss /= critic_loss_count
            if actor_loss_count > 0:
                actor_loss /= actor_loss_count
            if system_loss_count > 0:
                system_loss /= system_loss_count
            if reward_loss_count > 0:
                reward_loss /= reward_loss_count
            if avg_score > best_score:
                best_score = avg_score

            writer.add_scalar("train/reward", score, i)
            writer.add_scalar("train/critic_loss", critic_loss, i)
            writer.add_scalar("train/actor_loss", actor_loss, i)
            writer.add_scalar("train/system_loss", system_loss, i)
            writer.add_scalar("train/reward_loss", reward_loss, i)
            print(
                "episode",
                i,
                "score %.1f" % score,
                "avg score %.1f" % avg_score,
                "critic loss %.3f" % critic_loss,
                "actor loss %.2f" % actor_loss,
                "system loss %.6f" % system_loss,
                "reward loss %.6f" % reward_loss,
            )

            if avg_score >= best_avg_score:
                best_avg_score = avg_score
                agent.save_models()
            elif i % 10 == 0:
                agent.save_models()
            critic_loss_count = 0
            actor_loss_count = 0
            critic_loss = 0
            actor_loss = 0
            system_loss_count = 0
            reward_loss_count = 0
            system_loss = 0
            reward_loss = 0
            score = 0
            writer.flush()

    writer.close()
