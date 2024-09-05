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
import os

register(
    id="InvertedDoublePendulumModded",
    entry_point="mujoco_mod.envs.inverted_double_pendulum_mod:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=0.0,
)


def init_teachers(is_combining_balancing_swingup, is_all_in_one):
    if is_all_in_one:
        obs_shape = env.observation_space.shape
    else:
        obs_shape = (env.observation_space.shape[0] + 2,)

    teacher_agent_0 = td3_fork.Agent(
        alpha=lr,
        beta=lr,
        ln=True,
        input_dims=obs_shape,
        tau=0.005,
        n_extra_obs=4,
        env=env,
        teacher=True,
        layer1_size=100,  # constant don't change orig now it is 16x16
        layer2_size=75,
        n_actions=env.action_space.shape[0],
    )
    teacher_agent_0.chkpt_file_pth = os.path.join(
        chkpt_dir, f"{game_id} td3 fork comm 0.chkpt"
    )

    teacher_agent_1 = td3_fork.Agent(
        alpha=lr,
        beta=lr,
        ln=True,
        input_dims=obs_shape,
        tau=0.005,
        n_extra_obs=4,
        env=env,
        teacher=True,
        layer1_size=100,  # constant don't change orig now it is 16x16
        layer2_size=75,
        n_actions=env.action_space.shape[0],
    )
    teacher_agent_1.chkpt_file_pth = os.path.join(
        chkpt_dir, f"{game_id} td3 fork comm 1.chkpt"
    )

    teacher_agent_2 = td3_fork.Agent(
        alpha=lr,
        beta=lr,
        ln=True,
        input_dims=obs_shape,
        tau=0.005,
        n_extra_obs=4,
        env=env,
        teacher=True,
        layer1_size=100,  # constant don't change orig now it is 16x16
        layer2_size=75,
        n_actions=env.action_space.shape[0],
    )
    teacher_agent_2.chkpt_file_pth = os.path.join(
        chkpt_dir, f"{game_id} td3 fork comm 2.chkpt"
    )
    if is_combining_balancing_swingup:
        teacher_agent_3_swingup = td3_fork.Agent(
            alpha=lr,
            beta=lr,
            ln=True,
            input_dims=obs_shape,
            tau=0.005,
            n_extra_obs=4,
            env=env,
            teacher=True,
            layer1_size=100,  # constant don't change orig now it is 16x16
            layer2_size=75,
            n_actions=env.action_space.shape[0],
        )
        teacher_agent_3_swingup.chkpt_file_pth = os.path.join(
            chkpt_dir, f"{game_id} td3 fork comm 3 swingup.chkpt"
        )

        teacher_agent_3_balance = td3_fork.Agent(
            alpha=lr,
            beta=lr,
            ln=True,
            input_dims=obs_shape,
            tau=0.005,
            n_extra_obs=4,
            env=env,
            teacher=True,
            layer1_size=100,  # constant don't change orig now it is 16x16
            layer2_size=75,
            n_actions=env.action_space.shape[0],
        )
        teacher_agent_3_balance.chkpt_file_pth = os.path.join(
            chkpt_dir, f"{game_id} td3 fork comm 3 balance.chkpt"
        )
        teacher_agent_3_swingup.load_actor_model()
        teacher_agent_3_balance.load_actor_model()
    else:
        teacher_agent_3 = td3_fork.Agent(
            alpha=lr,
            beta=lr,
            ln=True,
            input_dims=obs_shape,
            tau=0.005,
            n_extra_obs=4,
            env=env,
            teacher=True,
            layer1_size=100,  # constant don't change orig now it is 16x16
            layer2_size=75,
            n_actions=env.action_space.shape[0],
        )
        teacher_agent_3.chkpt_file_pth = os.path.join(
            chkpt_dir, f"{game_id} td3 fork comm 3.chkpt"
        )
        teacher_agent_3.load_actor_model()

    teacher_agent_0.load_actor_model()
    teacher_agent_1.load_actor_model()
    teacher_agent_2.load_actor_model()

    if is_combining_balancing_swingup:
        return (
            teacher_agent_0,
            teacher_agent_1,
            teacher_agent_2,
            teacher_agent_3_swingup,
            teacher_agent_3_balance,
        )
    else:
        return (
            teacher_agent_0,
            teacher_agent_1,
            teacher_agent_2,
            teacher_agent_3,
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
    isDomainRandomization = False
    isParallelActors = False
    isCombiningBalanceSwingup = False
    isAllInOne = False
    # TODO: when combining balance swingup, teacher_action should only use balance actions not balance swingup action
    game_id = "InvertedDoublePendulumModded"
    filename = "testing"
    chkpt_dir = "./tmp/td3 fork"
    log_dir = f"runs/double_inverted_pendulum_sim/{filename}"
    # chkpt_dir = "./drive/MyDrive/pendulum/tmp/td3 fork"
    # log_dir = f"./drive/MyDrive/pendulum/runs/double_inverted_pendulum_sim/{filename}"
    num_envs = 2
    env = gym.make(game_id)
    # comm 0 about 100 teacher_weight=0.2
    # comm 1 about 2500 teacher_weight=0.2
    # comm 2 about 2500 teacher_weight=0.2
    # comm 3 about 5200 teacher_weight=0.2
    # Average is about 3000
    highest_score = 3000
    teacher_weight = 0.6
    domain_randomizer = DomainRandomization(highest_score)

    seed = None
    if seed is not None:
        np.random.seed(seed)
        T.manual_seed(seed)
    # balance: gamma=0.98, sys_threshold=0.001
    # swingup: gamma=0.99, sys_threshold=0.03
    # swingup comm 2: gamma=0.98, sys_threshold=0.03
    # Combining Balance and Swingup: teacher_weight=0.2
    lr = 0.001
    agent = td3_fork.Agent(
        alpha=lr,
        beta=lr,
        ln=True,
        warmup=10000,
        input_dims=env.observation_space.shape,
        tau=0.005,
        env=env,
        n_extra_obs=4,
        teacher_weight=teacher_weight,
        teacher_temperature=0.05,
        episodes_teacher_anneal=200,
        teacher_weight_end=0.2,
        student=True,
        gamma=0.98,
        noise=0.2,  # choose action noise
        noise_clip=0.5,
        gradient_clip=1.0,
        policy_noise=0.2,  # learning action noise
        layer1_size=100,  # constant don't change orig now it is 16x16
        layer2_size=75,
        sys1_size=400,
        sys2_size=300,
        r1_size=256,
        r2_size=256,
        sys_weight=0.6,
        update_actor_interval=1,
        max_size=1_000_000,
        sys_threshold=0.04,
        n_actions=env.action_space.shape[0],
        game_id=game_id,
        chkpt_dir=chkpt_dir,
    )
    # agent.load_models(load_all_weights=True, load_optimizers=True)
    # agent.memory.load(agent.buffer_file_pth)
    # agent.chkpt_file_pth = os.path.join(chkpt_dir, f"{game_id} td3 fork.chkpt")
    # teacher_file_pth = os.path.join(
    #     chkpt_dir, f"{game_id} td3 fork comm {env.unwrapped.pendulum_command}.chkpt"
    # )
    # agent.load_supplemental_networks(teacher_file_pth)

    if isCombiningBalanceSwingup:
        (
            teacher_0,
            teacher_1,
            teacher_2,
            teacher_3_swingup,
            teacher_3_balance,
        ) = init_teachers(isCombiningBalanceSwingup, isAllInOne)
    else:
        (
            teacher_0,
            teacher_1,
            teacher_2,
            teacher_3,
        ) = init_teachers(isCombiningBalanceSwingup, isAllInOne)
    writer = SummaryWriter(log_dir=log_dir)
    n_timesteps = 1_000_000
    episode = 1_000  # 1 episode = 1k timesteps

    best_score = env.reward_range[0]
    best_avg_score = best_score
    # extra step when loading, episode also gets affected
    step_skip = agent.memory.mem_cntr
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
            ready_balance = False
            steps = 0
            if isDomainRandomization:
                ep = int(step / episode)
                if domain_randomizer.check_level_up(np.mean(score_history[-10:]), ep):
                    best_avg_score -= highest_score * 0.8
                    best_score -= highest_score * 0.8
                if (domain_randomizer.difficulty_level >= 1) and (
                    agent.teacher_weight_end > 0.0
                ):
                    agent.memory.tree.beta = 1.0
                    agent.teacher_weight = 0.0
                    agent.teacher_weight_end = 0.0
                    print(
                        f"...teacher weight {agent.teacher_weight} and teacher weight end {agent.teacher_weight_end}..."
                    )
                domain_randomizer.environment()

            if (
                isCombiningBalanceSwingup
                and (0.15 < agent.teacher_weight < 0.21)
                and (np.mean(score_history[-5:]) > 2_000)
            ):
                best_avg_score -= 2_000
                best_score -= 2_000
                agent.teacher_weight = 0.1
                print("...teacher weight 0.1...")

            if isParallelActors:
                env = SubprocVecEnv([make_env(i, game_id) for i in range(num_envs)])
                observations = env.reset()
            else:
                env = gym.make(game_id)
                observation, info = env.reset(seed=seed)
            if isDomainRandomization:
                observation = domain_randomizer.observation(observation, env)
                observation = np.clip(
                    observation,
                    agent.obs_lower_bound_ideal,
                    agent.obs_upper_bound_ideal,
                )

        if isParallelActors:
            actions = np.zeros(shape=(num_envs, agent.n_actions))
            for i in range(num_envs):
                actions[i][0] = agent.choose_action(observations[i])
        else:
            # TEACHER STUDENT TRAINING
            action = agent.choose_action(observation)
            obs = observation
            if not isAllInOne:
                obs = np.insert(
                    observation,
                    8,
                    [
                        env.unwrapped.pendulum_command_list[
                            env.unwrapped.pendulum_command
                        ][0],
                        env.unwrapped.pendulum_command_list[
                            env.unwrapped.pendulum_command
                        ][1],
                    ],
                )
            p1 = obs[8]
            p2 = obs[9]
            if (p1 == 0) and (p2 == 0):
                teacher_action = teacher_0.choose_action(obs, evaluate=True)
                ready_balance = False
            elif (p1 == 0) and (p2 == 1):
                teacher_action = teacher_1.choose_action(obs, evaluate=True)
                ready_balance = False
            elif (p1 == 1) and (p2 == 0):
                teacher_action = teacher_2.choose_action(obs, evaluate=True)
                ready_balance = False
            elif (p1 == 1) and (p2 == 1):
                if not isCombiningBalanceSwingup:
                    teacher_action = teacher_3.choose_action(obs, evaluate=True)
                    ready_balance = False
                else:
                    a1, a2 = env.data.qpos[1:]
                    tip_x, _, tip_y = env.data.site_xpos[0]
                    a2 += a1
                    x_goal = bool(abs(obs[0]) < 0.3)
                    theta1_goal = bool(abs(simp_angle(a1)) < 0.1)
                    theta2_goal = bool(abs(simp_angle(a2)) < 0.1)
                    x_dot_goal = bool(abs(obs[5]) < 0.3)
                    theta1_dot_goal = bool(abs(obs[6]) < 1.0)
                    theta2_dot_goal = bool(abs(obs[7]) < 1.0)
                    if (
                        not ready_balance
                        and x_goal
                        and theta1_goal
                        and theta2_goal
                        and x_dot_goal
                        and theta1_dot_goal
                        and theta2_dot_goal
                    ):
                        ready_balance = True
                    if ready_balance and tip_y < 0.8:
                        ready_balance = False

                    if ready_balance:
                        teacher_action = teacher_3_balance.choose_action(
                            obs, evaluate=True
                        )
                    else:
                        teacher_action = teacher_3_swingup.choose_action(
                            obs, evaluate=True
                        )

        if isDomainRandomization:
            action_alt = domain_randomizer.action(action)
        else:
            if isParallelActors:
                action_alt = actions
            else:
                action_alt = action

        if isParallelActors:
            observations_, rewards, terminateds, truncateds, info = env.step(action_alt)
        else:
            observation_, reward, terminated, truncated, info = env.step(action_alt)

        if isDomainRandomization:
            observation_ = domain_randomizer.observation(observation_, env)
            observation_ = np.clip(
                observation_, agent.obs_lower_bound_ideal, agent.obs_upper_bound_ideal
            )

        if isParallelActors:
            for observation, action, reward, observation_, terminated, truncated in zip(
                observations, actions, rewards, observations_, terminateds, truncateds
            ):
                done = terminated or truncated
                agent.remember(observation, action, reward, observation_, done)
            observations = observation_
            avg_reward = np.mean(rewards)
            writer.add_scalar("train/return", avg_reward, step)
            score += avg_reward
        else:
            done = terminated or truncated
            agent.remember(
                observation, action, reward, observation_, done, teacher_action
            )
            observation = observation_
            writer.add_scalar("train/return", reward, step)
            score += reward

        steps += 1
        if isParallelActors:
            for _ in range(num_envs):
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
        else:
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
                "critic loss %.5f" % critic_loss,
                "actor loss %.5f" % actor_loss,
                "system loss %.5f" % system_loss,
                "reward loss %.5f" % reward_loss,
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
