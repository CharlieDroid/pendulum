import numpy as np
from RPi.environment import DummyPendulum
from RPi.td3_fork import Agent
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    chkpt_dir = "./drive/MyDrive/pendulum/tmp/td3 fork"
    log_dir = f"./drive/MyDrive/pendulum/runs/inverted_pendulum_sim/testing"

    env = DummyPendulum()
    agent = Agent(lr=0.005, env=env, chkpt_dir=chkpt_dir)
    dummy = agent.chkpt_file_pth
    agent.chkpt_file_pth = "./drive/MyDrive/pendulum/RPi/models/td3_fork.chkpt"
    agent.load_models(reset=True, freeze=False)
    agent.chkpt_file_pth = dummy
    agent.memory.load("./drive/MyDrive/pendulum/RPi/memory/td3_fork_buffer_irl.pkl")
    episode_timesteps = 1_000

    # setup data logging
    writer = SummaryWriter(log_dir=log_dir)

    for i in range(int((agent.memory.mem_cntr / episode_timesteps) * 0.4)):
        critic_loss_count = 0
        actor_loss_count = 0
        system_loss_count = 0
        reward_loss_count = 0
        critic_loss = 0
        actor_loss = 0
        system_loss = 0
        reward_loss = 0
        for t in range(episode_timesteps):
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

        # log training data
        if critic_loss_count > 0:
            critic_loss /= critic_loss_count
        if actor_loss_count > 0:
            actor_loss /= actor_loss_count
        if system_loss_count > 0:
            system_loss /= system_loss_count
        if reward_loss_count > 0:
            reward_loss /= reward_loss_count
        agent.save_models()

        writer.add_scalar("train/critic_loss", critic_loss, i)
        writer.add_scalar("train/actor_loss", actor_loss, i)
        writer.add_scalar("train/system_loss", system_loss, i)
        writer.add_scalar("train/reward_loss", reward_loss, i)
        print(
            "episode",
            i,
            "score 0.0",
            "avg score 0.0",
            "critic loss %.5f" % critic_loss,
            "actor loss %.5f" % actor_loss,
            "system loss %.5f" % system_loss,
            "reward loss %.5f" % reward_loss,
        )
        writer.flush()
