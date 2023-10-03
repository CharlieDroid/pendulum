import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import os


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state, action):
        q = F.relu(self.fc1(T.cat([state, action], dim=1)))
        q = F.relu(self.fc2(q))

        q1 = self.q1(q)

        return q1


class ValueNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, name):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))

        # activation is tanh because it bounds it between +- 1
        # just multiply this according to the maximum action of the environment
        v = self.v(prob)
        return v


class ActorNetwork(nn.Module):
    def __init__(
        self,
        alpha,
        input_dims,
        max_action,
        fc1_dims,
        fc2_dims,
        n_actions,
        name,
    ):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))

        mu = self.mu(prob)  # mean
        sigma = self.sigma(prob)  # standard deviation

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        # take care of this in real world
        action = T.tanh(actions)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(-1, keepdim=True)

        return action, log_probs


class Agent:
    def __init__(
        self,
        alpha=0.0003,
        beta=0.0003,
        input_dims=[8],
        gradient_steps=8,
        env=None,
        gamma=0.99,
        n_actions=2,
        max_size=1000000,
        tau=0.005,
        layer1_size=256,
        layer2_size=256,
        batch_size=256,
        reward_scale=2,
        chkpt_dir=".\\tmp\\sac",
        game_id="Pendulum-v2",
    ):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.gradient_steps = gradient_steps
        self.time_step = 0
        self.chkpt_file_pth = os.path.join(chkpt_dir, f"{game_id} sac.chkpt")

        self.actor = ActorNetwork(
            alpha,
            input_dims,
            env.action_space.high,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="actor",
        )
        self.critic_1 = CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="critic_1",
        )
        self.critic_2 = CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="critic_2",
        )
        self.value = ValueNetwork(
            beta, input_dims, layer1_size, layer2_size, name="value"
        )
        self.target_value = ValueNetwork(
            beta, input_dims, layer1_size, layer2_size, name="target_value"
        )

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        action, _ = self.actor.sample(state, reparameterize=False)
        return action.cpu().detach().numpy()[0] * self.actor.max_action

    def remember(self, state, action, reward, new_state, done):
        self.time_step += 1
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = (
                tau * value_state_dict[name].clone()
                + (1 - tau) * target_value_state_dict[name].clone()
            )

        self.target_value.load_state_dict(value_state_dict)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return None, None, None

        total_critic_loss = 0
        total_actor_loss = 0
        total_value_loss = 0
        for _ in range(self.gradient_steps):
            state, action, reward, new_state, done = self.memory.sample_buffer(
                self.batch_size
            )

            reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
            done = T.tensor(done).to(self.actor.device)
            state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
            state = T.tensor(state, dtype=T.float).to(self.actor.device)
            action = T.tensor(action, dtype=T.float).to(self.actor.device)

            value = self.value(state).view(-1)
            value_ = self.target_value(state_).view(-1)
            value_[done] = 0.0

            actions, log_probs = self.actor.sample(state, reparameterize=False)
            log_probs = log_probs.view(-1)
            q1_new_policy = self.critic_1.forward(state, actions)
            q2_new_policy = self.critic_2.forward(state, actions)
            critic_value = T.min(q1_new_policy, q2_new_policy)
            critic_value = critic_value.view(-1)

            self.value.optimizer.zero_grad()
            value_target = critic_value - log_probs
            value_loss = 0.5 * F.mse_loss(value, value_target)
            value_loss.backward(retain_graph=True)
            self.value.optimizer.step()

            actions, log_probs = self.actor.sample(state, reparameterize=True)
            log_probs = log_probs.view(-1)
            q1_new_policy = self.critic_1.forward(state, actions)
            q2_new_policy = self.critic_2.forward(state, actions)
            critic_value = T.min(q1_new_policy, q2_new_policy)
            critic_value = critic_value.view(-1)

            actor_loss = log_probs - critic_value
            actor_loss = T.mean(actor_loss)
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()
            q_hat = self.scale * reward + self.gamma * value_
            q1_old_policy = self.critic_1.forward(state, action).view(-1)
            q2_old_policy = self.critic_2.forward(state, action).view(-1)
            critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

            critic_loss = critic_1_loss + critic_2_loss
            critic_loss.backward()
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()

            self.update_network_parameters()
            total_critic_loss += critic_loss
            total_actor_loss += actor_loss
            total_value_loss += value_loss
        # returns critic, value, actor loss respectively
        return [
            loss / self.gradient_steps
            for loss in [total_critic_loss, total_value_loss, total_actor_loss]
        ]

    def save_models(self):
        print("...saving checkpoint...")
        T.save(
            {
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "value": self.value.state_dict(),
                "target_value": self.target_value.state_dict(),
                "actor_optimizer": self.actor.optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1.optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2.optimizer.state_dict(),
                "value_optimizer": self.value.optimizer.state_dict(),
                "target_value_optimizer": self.target_value.optimizer.state_dict(),
            },
            self.chkpt_file_pth,
        )

    def load_models(self):
        print("...loading checkpoint...")
        checkpoint = T.load(self.chkpt_file_pth)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.value.load_state_dict(checkpoint["value"])
        self.target_value.load_state_dict(checkpoint["target_value"])
        self.actor.optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_1.optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2.optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        self.value.optimizer.load_state_dict(checkpoint["value_optimizer"])
        self.target_value.optimizer.load_state_dict(
            checkpoint["target_value_optimizer"]
        )
