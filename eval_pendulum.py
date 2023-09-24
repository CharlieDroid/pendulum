import gymnasium as gym
from td3 import Agent
from utils import agent_play


if __name__ == "__main__":
    game_id = "Pendulum-v1"
    torque = 2.0
    p_len = 1.0
    mass = 1.0
    env = gym.make(
        game_id, g=9.80665, t=torque, l=p_len, m=mass, max_s=8, r_coeff=0.001
    )
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
        game_id=game_id,
        chkpt_dir=".\\tmp\\td3_learned",
    )
    agent.load_models()
    agent_play(game_id, agent)
