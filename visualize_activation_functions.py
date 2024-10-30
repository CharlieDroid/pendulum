import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
from mujoco_mod.envs.domain_randomization import DomainRandomization, simp_angle


activations = {"relu1": [], "relu2": []}

register(
    id="InvertedPendulumModded",
    entry_point="mujoco_mod.envs.inverted_pendulum_mod:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=0.0,
)


class ActorNetwork(nn.Module):
    def __init__(
        self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, ln=True
    ):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.ln1 = None
        self.ln2 = None
        if ln:
            self.ln1 = nn.LayerNorm(self.fc1_dims)
            self.ln2 = nn.LayerNorm(self.fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        a = F.relu(self.fc1(state))
        activations["relu1"].append(a.detach())
        if self.ln1:
            a = self.ln1(a)

        a = F.relu(self.fc2(a))
        activations["relu2"].append(a.detach())
        if self.ln2:
            a = self.ln2(a)

        # activation is tanh because it bounds it between +- 1
        # just multiply this according to the maximum action of the environment
        mu = T.tanh(self.mu(a))
        return mu

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        mu = self.forward(state).to(self.device)
        return mu.cpu().detach().numpy()


# Function to plot activation histograms
def plot_activation_histogram(activation_list, layer_name):
    # Concatenate all activations in the list into a single tensor
    all_activations = T.cat(activation_list, dim=0)

    # Convert to NumPy array and flatten
    activation_np = all_activations.cpu().numpy().flatten()

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(activation_np, bins=50, color='blue', alpha=0.7, edgecolor='black', density=True)

    # Titles and labels
    plt.title(f'Activation Distribution for {layer_name}')
    plt.xlabel('Activation Value')
    plt.ylabel('Density')

    plt.grid(True)
    plt.show()


def calculate_sparsity(activation_list):
    # Concatenate all activations in the list into a single tensor
    all_activations = T.cat(activation_list, dim=0)

    # Count number of zeros
    num_zeros = T.sum(all_activations == 0).item()

    # Total number of elements
    total_elements = all_activations.numel()

    # Calculate sparsity as a percentage
    sparsity = (num_zeros / total_elements) * 100
    return sparsity

def agent_play_episode(_agent, _env, isDomainRandomization):
    total_reward = 0
    observation, info = _env.reset()
    if isDomainRandomization:
        domain_randomizer.observation(observation, _env)
    observation[1] = simp_angle(observation[1])

    done = False
    while not done:
        action = _agent.choose_action(observation)
        if isDomainRandomization:
            domain_randomizer.action(action)

        observation_, reward, terminated, truncated, info = _env.step(action)
        total_reward += reward

        if isDomainRandomization:
            domain_randomizer.observation(observation_, _env)
        observation_[1] = simp_angle(observation_[1])

        observation = observation_
        done = terminated or truncated
    return total_reward

def agents_get_rewards(agents, num_episodes, domain_randomizer, isDomainRandomization, game_id):
    domain_randomizer.reset_environment()
    env = gym.make(game_id)
    total_scores = [0. for _ in range(len(agents))]
    for ep in range(num_episodes):
        if ep % 10 == 0:
            print(f"Episode: {ep}")
        if isDomainRandomization:
            domain_randomizer.environment()
            env = gym.make(game_id)

        for i, agent in enumerate(agents):
            total_scores[i] += agent_play_episode(agent, env, isDomainRandomization)

    env.close()
    return [total_score / num_episodes for total_score in total_scores]

def init_agent(file_name, game_id):
    chkpt_file_pth = f"./tmp/td3_fork_learned/{game_id} td3 fork{file_name}.chkpt"
    agent = ActorNetwork(0.001, (4,), 16, 16, 1, "actor")
    checkpoint = T.load(chkpt_file_pth, map_location=agent.device)
    agent.load_state_dict(checkpoint["actor"])
    return agent


if __name__ == "__main__":
    # 123 balanced bias, relu1 = 71.80%, relu2 = 48.46%, r = 583.965
    # 123 external bias, relu1 = 56.70%, relu2 = 59.02%, r = 539.405
    # 3 torque new, relu1 = 48.37%, relu2 = 71.36%, r = 613.742

    # 12 r = 415.32

    # filenames must have whitespace in beginning since that is how I named them
    # filenames = ("", " 3.0 torque new", " 123 balanced bias")
    filenames = ("", " 3.0 torque new")
    num_episodes = 50
    game_id = "InvertedPendulumModded"
    isDomainRandomization = True

    agents = tuple([init_agent(filename, game_id) for filename in filenames])
    # env = gym.make(game_id, render_mode="human")
    domain_randomizer = DomainRandomization()
    if isDomainRandomization:
        domain_randomizer.difficulty_level = 2

    total_scores = agents_get_rewards(agents, num_episodes, domain_randomizer, isDomainRandomization, game_id)

    # reset everything back to normal after all episodes
    domain_randomizer.reset_environment()

    for i, filename in enumerate(filenames):
        print(f"Agent {i} \"{filename}\":\t\t\t{total_scores[i]:.2f}")



    # # Plot for ReLU1
    # plot_activation_histogram(activations['relu1'], 'ReLU1')
    #
    # # Plot for ReLU2
    # plot_activation_histogram(activations['relu2'], 'ReLU2')
    #
    # sparsity_relu1 = calculate_sparsity(activations['relu1'])
    # sparsity_relu2 = calculate_sparsity(activations['relu2'])
    #
    # print(f"Sparsity in ReLU1: {sparsity_relu1:.2f}%")
    # print(f"Sparsity in ReLU2: {sparsity_relu2:.2f}%")
