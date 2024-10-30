from td3_fork import Agent, AgentActor
from environment import Pendulum, DummyPendulum

if __name__ == "__main__":
    env = DummyPendulum()
    agent = Agent(
        lr=0.001,
        env=env,
        save_buffer=True,
        load_buffer=False,
        ln=True,
        update_actor_interval=1,
    )
    agent.load_models()
    agent.save_model_txt()
