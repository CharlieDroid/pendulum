import gymnasium as gym
from gymnasium.envs.registration import register
import td3
import sac_new as sac
import td3_fork
import tqc
from utils import agent_play

register(
    id="InvertedPendulumModded",
    entry_point="mujoco_mod.envs.inverted_pendulum_mod:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=0.0,
)

if __name__ == "__main__":
    find_best = False  # find best video (not needed now)
    save = False  # save video
    # obs, reward, action
    save_data = True  # save episode data
    algo = "td3-fork"
    game_id = "InvertedPendulumModded"
    eps = 1
    env = gym.make(game_id, render_mode="human")
    if algo == "td3":
        buffer_size = 200_000
        lr = 0.001
        fc1 = 400
        fc2 = 300
        agent = td3.Agent(
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
            game_id=game_id,
            chkpt_dir=".\\tmp\\td3_learned",
        )
    elif algo == "sac":
        agent = sac.Agent(
            q_lr=0.00073,
            a_lr=0.00073,
            layer1_size=16,
            layer2_size=16,
            input_dims=env.observation_space.shape,
            env=env,
            n_actions=env.action_space.shape[0],
            game_id=game_id,
            chkpt_dir=".\\tmp\\sac_learned",
        )
    elif algo == "td3-fork":
        lr = 0.001
        agent = td3_fork.Agent(
            alpha=lr,
            beta=lr,
            ln=True,
            input_dims=env.observation_space.shape,
            tau=0.005,
            env=env,
            gamma=0.98,
            noise=0.1,
            layer1_size=16,
            layer2_size=16,
            update_actor_interval=1,
            max_size=10_000,
            n_actions=env.action_space.shape[0],
            game_id=game_id,
            chkpt_dir=".\\tmp\\td3_fork_learned",
        )
    elif algo == "tqc":
        agent = tqc.Agent(
            input_dims=env.observation_space.shape,
            env=env,
            n_actions=env.action_space.shape[0],
            game_id=game_id,
            chkpt_dir=".\\tmp\\tqc_learned",
        )
    agent.load_models()
    agent.time_step = agent.warmup + 1
    print(
        agent_play(
            game_id, agent, save=save, find_best=find_best, eps=eps, save_data=save_data
        )
    )
