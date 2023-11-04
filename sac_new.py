import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
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
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, optimizer=True):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        if optimizer:
            self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state, action):
        q = F.relu(self.fc1(T.cat([state, action], dim=1)))
        q = F.relu(self.fc2(q))

        q1 = self.q1(q)

        return q1


class ActorNetwork(nn.Module):
    def __init__(
        self,
        alpha,
        input_dims,
        max_action,
        fc1_dims,
        fc2_dims,
        n_actions,
    ):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = T.tensor(max_action, dtype=T.float32)
        self.log_std_max = 2
        self.log_std_min = -5
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
        log_std = self.sigma(prob)  # standard deviation

        log_std = T.tanh(log_std)
        sigma = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )
        return mu, sigma

    def get_action(self, state):
        mu, log_std = self.forward(state)
        sigma = log_std.exp()
        probabilities = T.distributions.Normal(mu, sigma)
        x_t = probabilities.rsample()
        y_t = T.tanh(x_t)
        action = y_t * self.max_action

        log_probs = probabilities.log_prob(x_t)
        log_probs -= T.log(self.max_action * (1 - y_t.pow(2)) + self.reparam_noise)
        log_probs = log_probs.sum(-1, keepdim=True)
        mu = T.tanh(mu) * self.max_action
        return action, log_probs, mu


class Agent:
    def __init__(
        self,
        q_lr,
        a_lr,
        env,
        tau=0.005,
        warmup=25_000,
        batch_size=256,
        max_size=1_000_000,
        input_dims=[8],
        gamma=0.99,
        layer1_size=256,
        layer2_size=256,
        n_actions=2,
        autotune=True,
        alpha=0.2,
        policy_freq=1,
        target_network_frequency=1,
        chkpt_dir="./tmp/sac new",
        game_id="Pendulum-v2",
    ):
        self.gamma = gamma
        self.tau = tau
        self.warmup = warmup
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.time_step = 0
        self.policy_freq = policy_freq
        self.autotune = autotune
        self.target_network_frequency = target_network_frequency
        self.chkpt_file_pth = os.path.join(chkpt_dir, f"{game_id} sac new.chkpt")

        self.actor = ActorNetwork(
            a_lr,
            input_dims,
            env.action_space.high,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
        )
        self.critic_1 = CriticNetwork(
            q_lr, input_dims, layer1_size, layer2_size, n_actions
        )
        self.critic_2 = CriticNetwork(
            q_lr, input_dims, layer1_size, layer2_size, n_actions
        )
        self.critic_target_1 = CriticNetwork(
            q_lr, input_dims, layer1_size, layer2_size, n_actions, optimizer=False
        )
        self.critic_target_2 = CriticNetwork(
            q_lr, input_dims, layer1_size, layer2_size, n_actions, optimizer=False
        )
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        if autotune:
            self.target_entropy = -T.prod(
                T.tensor(env.action_space.shape).to(self.actor.device)
            ).item()
            self.log_alpha = T.zeros(1, requires_grad=True, device=self.actor.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)
        else:
            self.alpha = alpha

    def choose_action(self, observation, evaluate=True):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        action, _, _ = self.actor.get_action(state)
        return action.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.time_step += 1
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if (self.memory.mem_cntr < self.batch_size) or (self.time_step < self.warmup):
            return None, None

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done, dtype=T.float).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        with T.no_grad():
            states_actions_, state_log_pi_, _ = self.actor.get_action(state_)
            q1_target_ = self.critic_1.forward(state_, states_actions_)
            q2_target_ = self.critic_2.forward(state_, states_actions_)
            min_q_target_ = T.min(q1_target_, q2_target_)
            q_value_ = reward + (1 - done) * self.gamma * min_q_target_.view(-1)

        q1_a_val = self.critic_1.forward(state, action).view(-1)
        q2_a_val = self.critic_2.forward(state, action).view(-1)
        q1_loss = F.mse_loss(q1_a_val, q_value_)
        q2_loss = F.mse_loss(q2_a_val, q_value_)
        q_loss = q1_loss + q2_loss

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        if (self.time_step % self.policy_freq) == 0:
            for _ in range(self.policy_freq):
                pi, log_pi, _ = self.actor.get_action(state)
                qf1_pi = self.critic_1.forward(state, pi)
                qf2_pi = self.critic_2.forward(state, pi)
                min_qf_pi = T.min(qf1_pi, qf2_pi)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                if self.autotune:
                    with T.no_grad():
                        _, log_pi, _ = self.actor.get_action(state)
                    alpha_loss = (
                        -self.log_alpha.exp() * (log_pi + self.target_entropy)
                    ).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()
            actor_loss = actor_loss.detach().item() / self.policy_freq
        else:
            actor_loss = None

        if (self.time_step % self.target_network_frequency) == 0:
            for param, target_param in zip(
                self.critic_1.parameters(), self.critic_target_1.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(
                self.critic_2.parameters(), self.critic_target_2.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        return q_loss.detach().item(), actor_loss

    def save_models(self):
        print("...saving checkpoint...")
        dictionary = {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_target_1": self.critic_target_1.state_dict(),
            "critic_target_2": self.critic_target_2.state_dict(),
            "actor_optimizer": self.actor.optimizer.state_dict(),
            "critic_1_optimizer": self.critic_1.optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2.optimizer.state_dict(),
            "time_step": self.time_step,
            "alpha": self.alpha,
        }
        if self.autotune:
            dictionary["target_entropy"] = self.target_entropy
            dictionary["log_alpha"] = self.log_alpha
            dictionary["a_optimizer"] = self.a_optimizer.state_dict()
        T.save(dictionary, self.chkpt_file_pth)

    def load_models(self):
        print("...loading checkpoint...")
        checkpoint = T.load(self.chkpt_file_pth, map_location=self.actor.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.critic_target_1.load_state_dict(checkpoint["critic_target_1"])
        self.critic_target_2.load_state_dict(checkpoint["critic_target_2"])
        self.actor.optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_1.optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2.optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        self.time_step = checkpoint["time_step"]
        self.alpha = checkpoint["alpha"]
        if self.autotune:
            self.target_entropy = checkpoint["target_entropy"]
            self.log_alpha = checkpoint["log_alpha"]
            self.a_optimizer.load_state_dict(checkpoint["a_optimizer"])
