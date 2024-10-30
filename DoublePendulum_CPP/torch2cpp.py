from td3_fork import Agent
from td3_fork_double import Agent as AgentDouble
import gymnasium as gym
from gymnasium.envs.registration import register
import os

register(
    id="InvertedPendulumModded",
    entry_point="mujoco_mod.envs.inverted_pendulum_mod:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=0.0,
)

register(
    id="InvertedDoublePendulumModded",
    entry_point="mujoco_mod.envs.inverted_double_pendulum_mod:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=0.0,
)

tab = "        "


def python_vector_to_cpp_vector(vector):
    """
    From python to CPP
    :param vector:
    :return:
    """
    text = "{"
    for i, data in enumerate(vector):
        # for beautification
        if i < len(vector) - 1:
            text += f" {{{data}}},"
        else:
            text += f" {{{data}}} "
    text += "},"
    return text


def python_matrix_to_cpp_matrix(matrix):
    """
    From python to CPP
    :param matrix:
    :return:
    """
    text = "{"
    for row in range(len(matrix)):
        if row == 0:
            text += " {"
        else:
            text += f"{2 * tab}    {{"

        for col in range(len(matrix[row])):
            data = matrix[row][col]
            # for beautification
            if col < len(matrix[row]) - 1:
                text += f" {data},"
            else:
                text += f" {data} "

        # if last row then don't add ,\n (for beautification)
        if row < len(matrix) - 1:
            text += "},\n"
        else:
            text += "}"
    text += " },"
    return text

def get_agent_header_file_string(agent, last=False):
    text = "    Agent{\n"
    dummy_text = python_matrix_to_cpp_matrix(agent.actor.fc1.weight.detach().tolist())
    text += f"{tab}FC1_Weight{dummy_text}\n"

    dummy_text = python_vector_to_cpp_vector(agent.actor.fc1.bias.detach().tolist())
    text += f"{tab}FC1_Bias{dummy_text}\n"

    dummy_text = python_matrix_to_cpp_matrix(agent.actor.fc2.weight.detach().tolist())
    text += f"{tab}FC2_Weight{dummy_text}\n"

    dummy_text = python_vector_to_cpp_vector(agent.actor.fc2.bias.detach().tolist())
    text += f"{tab}FC2_Bias{dummy_text}\n"

    dummy_text = python_matrix_to_cpp_matrix(agent.actor.mu.weight.detach().tolist())
    text += f"{tab}MU_Weight{dummy_text}\n"

    dummy_text = python_vector_to_cpp_vector(agent.actor.mu.bias.detach().tolist())
    text += f"{tab}MU_Bias{dummy_text}\n"

    text += f"\n{tab}// LN weights and biases\n"

    dummy_text = python_vector_to_cpp_vector(agent.actor.ln1.weight.detach().tolist())
    text += f"{tab}FC1_Bias{dummy_text}\n"

    dummy_text = python_vector_to_cpp_vector(agent.actor.ln1.bias.detach().tolist())
    text += f"{tab}FC1_Bias{dummy_text}\n"

    dummy_text = python_vector_to_cpp_vector(agent.actor.ln2.weight.detach().tolist())
    text += f"{tab}FC2_Bias{dummy_text}\n"

    dummy_text = python_vector_to_cpp_vector(agent.actor.ln2.bias.detach().tolist())
    text += f"{tab}FC2_Bias{dummy_text}\n"

    if not last:
        text += "        },\n"
    else:
        text += "        }\n"
    return text


def create_agent_s_header_file(header_file_pth_, agent_s: list):
    """
    1. Delete the file
    2. Create the file using first_lines
    3. Create the agent/s arrays depending on single or double
    4. End the file with end_lines
    :param agent_s: must be list and can be multiple agents
    :return:
    """
    first_lines = f"//\n// Created by Charles on 10/19/2024.\n//\n\n#ifndef AGENTS_H\n#define AGENTS_H\n\n#include \"agent.h\"\n\nconst Agent agents[{len(agent_s)}] =\n{{\n"
    end_lines = f"}};\n\n// compile time checking if there is correct number of agents\n#if defined(SINGLE_PENDULUM)\n#define NUM_AGENTS 1\n#elif defined(DOUBLE_PENDULUM)\n#define NUM_AGENTS 4\n#endif\n\n#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))\n\n#define VALIDATE_AGENTS_ARRAY(arr) \\\n{tab}static_assert(ARRAY_SIZE(arr) == NUM_AGENTS, \\\n{tab}\"Array size mismatch: \" \\\n{tab}\"SINGLE_PENDULUM requires 1 agent, \" \\\n{tab}\"DOUBLE_PENDULUM requires 4 agents\")\n\nVALIDATE_AGENTS_ARRAY(agents);\n\n#endif //AGENTS_H\n"

    # delete file
    os.remove(header_file_pth_)

    # create file
    with open(header_file_pth, "w") as agents_file:
        agents_file.write(first_lines)

        # create agent/s arrays
        for ind, agent in enumerate(agent_s):
            agents_file.write(get_agent_header_file_string(agent, last=ind == len(agent_s) - 1))

        agents_file.write(end_lines)


if __name__ == "__main__":
    isSingle = False
    header_file_pth = r"C:\Users\Charles\Documents\Python Scripts\Personal\Artificial Intelligence\pendulum\DoublePendulum_CPP\giga_m7\include\agents.h"
    os.chdir("..")
    if isSingle:
        game_id = "InvertedPendulumModded"

        # load model
        chkpt_dir = "./DoublePendulum_CPP/models"
        env = gym.make(game_id)
        agent_ = Agent(
            alpha=0.001,
            beta=0.001,
            input_dims=env.observation_space.shape,
            tau=0.005,
            layer1_size=16,
            layer2_size=16,
            env=env,
            ln=True,
            update_actor_interval=1,
            n_actions=env.action_space.shape[0],
            chkpt_dir=chkpt_dir,
            game_id=game_id,
        )
        agent_.load_models(load_all_weights=True, load_optimizers=False)  # optimizers aren't needed anyways

        create_agent_s_header_file(header_file_pth, [agent_])
    else:
        game_id = "InvertedDoublePendulumModded"
        chkpt_dir = "./DoublePendulum_CPP/models"

        agents = []
        # load models four commands
        for i in range(4):
            env = gym.make(game_id)
            agent_ = AgentDouble(
                alpha=0.001,
                beta=0.001,
                input_dims=env.observation_space.shape,
                n_extra_obs=14,
                tau=0.005,
                layer1_size=80,
                layer2_size=60,
                env=env,
                ln=True,
                update_actor_interval=1,
                n_actions=env.action_space.shape[0],
                chkpt_dir=chkpt_dir,
                game_id=game_id,
            )
            agent_.chkpt_file_pth = os.path.join(chkpt_dir, f"{game_id} td3 fork comm {i}.chkpt")
            agent_.load_models(load_all_weights=True, load_optimizers=False)  # optimizers aren't needed anyways
            agents.append(agent_)

        create_agent_s_header_file(header_file_pth, agents)

    print("done")
