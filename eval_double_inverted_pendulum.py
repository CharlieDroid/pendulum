import gymnasium as gym
from gymnasium.envs.registration import register
import td3_fork_double as td3_fork
from utils import agent_play_double, agent_play
import os

register(
    id="InvertedDoublePendulumModded",
    entry_point="mujoco_mod.envs.inverted_double_pendulum_mod:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=0.0,
)

if __name__ == "__main__":
    find_best = False  # find best video (not needed now)
    save = False  # save video
    # save episode data: obs, reward, action
    # !! if false then no DR action noise !!
    save_data = False
    combined = False
    teachers = False
    game_id = "InvertedDoublePendulumModded"
    eps = 1
    env = gym.make(game_id, render_mode="human")
    lr = 0.001
    if combined:
        agent_balance = td3_fork.Agent(
            alpha=lr,
            beta=lr,
            ln=True,
            input_dims=env.observation_space.shape,
            tau=0.005,
            env=env,
            gamma=0.98,
            noise=0.1,
            n_extra_obs=14,
            layer1_size=100,
            layer2_size=75,
            update_actor_interval=1,
            max_size=10_000,
            n_actions=env.action_space.shape[0],
            game_id=game_id,
            chkpt_dir=".\\tmp\\td3_fork_learned",
        )
        agent_balance.chkpt_file_pth = os.path.join(
            ".\\tmp\\td3_fork_learned", f"{game_id} td3 fork comm 3 balance.chkpt"
        )
        agent_balance.load_models()
        agent_balance.time_step = agent_balance.warmup + 1

        agent_swingup = td3_fork.Agent(
            alpha=lr,
            beta=lr,
            ln=True,
            input_dims=env.observation_space.shape,
            tau=0.005,
            env=env,
            gamma=0.98,
            noise=0.1,
            n_extra_obs=14,
            layer1_size=100,
            layer2_size=75,
            update_actor_interval=1,
            max_size=10_000,
            n_actions=env.action_space.shape[0],
            game_id=game_id,
            chkpt_dir=".\\tmp\\td3_fork_learned",
        )
        agent_swingup.chkpt_file_pth = os.path.join(
            ".\\tmp\\td3_fork_learned", f"{game_id} td3 fork comm 3 swingup.chkpt"
        )
        agent_swingup.load_models()
        agent_swingup.time_step = agent_swingup.warmup + 1

        agent_0 = td3_fork.Agent(
            alpha=lr,
            beta=lr,
            ln=True,
            input_dims=env.observation_space.shape,
            tau=0.005,
            env=env,
            gamma=0.98,
            noise=0.1,
            n_extra_obs=14,
            layer1_size=100,
            layer2_size=75,
            update_actor_interval=1,
            max_size=10_000,
            n_actions=env.action_space.shape[0],
            game_id=game_id,
            chkpt_dir=".\\tmp\\td3_fork_learned",
        )
        agent_0.chkpt_file_pth = os.path.join(
            ".\\tmp\\td3_fork_learned", f"{game_id} td3 fork comm 0.chkpt"
        )
        agent_0.load_models()
        agent_0.time_step = agent_0.warmup + 1

        agent_1 = td3_fork.Agent(
            alpha=lr,
            beta=lr,
            ln=True,
            input_dims=env.observation_space.shape,
            tau=0.005,
            env=env,
            gamma=0.98,
            noise=0.1,
            n_extra_obs=14,
            layer1_size=100,
            layer2_size=75,
            update_actor_interval=1,
            max_size=10_000,
            n_actions=env.action_space.shape[0],
            game_id=game_id,
            chkpt_dir=".\\tmp\\td3_fork_learned",
        )
        agent_1.chkpt_file_pth = os.path.join(
            ".\\tmp\\td3_fork_learned", f"{game_id} td3 fork comm 1.chkpt"
        )
        agent_1.load_models()
        agent_1.time_step = agent_1.warmup + 1

        agent_2 = td3_fork.Agent(
            alpha=lr,
            beta=lr,
            ln=True,
            input_dims=env.observation_space.shape,
            tau=0.005,
            env=env,
            gamma=0.98,
            noise=0.1,
            n_extra_obs=14,
            layer1_size=32,
            layer2_size=32,
            update_actor_interval=1,
            max_size=10_000,
            n_actions=env.action_space.shape[0],
            game_id=game_id,
            chkpt_dir=".\\tmp\\td3_fork_learned",
        )
        agent_2.chkpt_file_pth = os.path.join(
            ".\\tmp\\td3_fork_learned", f"{game_id} td3 fork comm 2.chkpt"
        )
        agent_2.load_models()
        agent_2.time_step = agent_2.warmup + 1

        print(
            agent_play_double(
                game_id,
                agent_balance,
                agent_swingup=agent_swingup,
                agent_0=agent_0,
                agent_1=agent_1,
                agent_2=agent_2,
                save=save,
                find_best=find_best,
                eps=eps,
                save_data=save_data,
                combined=combined,
                teachers=teachers,
            )
        )
    else:
        if teachers:
            obs_shape = (env.observation_space.shape[0] + 2,)
        else:
            obs_shape = env.observation_space.shape
        agent = td3_fork.Agent(
            alpha=lr,
            beta=lr,
            ln=True,
            input_dims=obs_shape,
            tau=0.005,
            env=env,
            gamma=0.98,
            noise=0.1,
            n_extra_obs=14,
            layer1_size=80,
            layer2_size=60,
            critic1_size=256,
            critic2_size=256,
            update_actor_interval=1,
            max_size=10_000,
            n_actions=env.action_space.shape[0],
            game_id=game_id,
            chkpt_dir=".\\tmp\\td3_fork_learned",
        )
        # agent.chkpt_file_pth = os.path.join(
        #     ".\\tmp\\td3_fork_learned", f"{game_id} td3 fork comm 0.chkpt"
        # )
        agent.load_models()
        agent.time_step = agent.warmup + 1
        agent.save_model_txt()

        print(
            agent_play_double(
                game_id,
                agent,
                save=save,
                find_best=find_best,
                eps=eps,
                save_data=save_data,
                combined=combined,
                teachers=teachers,
            )
        )
