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
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))

        # activation is tanh because it bounds it between +- 1
        # just multiply this according to the maximum action of the environment
        mu = T.tanh(self.mu(prob))
        return mu


class Agent:
    def __init__(
        self,
        alpha,
        beta,
        input_dims,
        tau,
        env,
        gamma=0.99,
        update_actor_interval=2,
        warmup=10_000,
        n_actions=2,
        max_size=1_000_000,
        layer1_size=400,
        layer2_size=300,
        batch_size=100,
        noise=0.1,
        chkpt_dir=".\\tmp\\td3",
        game_id="Pendulum-v2",
    ):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.chkpt_file_pth = os.path.join(chkpt_dir, f"{game_id} td3.chkpt")

        self.actor = ActorNetwork(
            alpha,
            input_dims,
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
        self.target_actor = ActorNetwork(
            alpha,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="target_actor",
        )
        self.target_critic_1 = CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="target_critic_1",
        )
        self.target_critic_2 = CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="target_critic_2",
        )

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, evaluate=False):
        if (self.time_step < self.warmup) and not evaluate:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)))
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        # your choice optional shit, remove when moving to real world cuz we already have noise lmao
        # if not evaluate:
        #     mu_prime = mu + T.tensor(
        #         np.random.normal(scale=self.noise), dtype=T.float
        #     ).to(self.actor.device)
        #     mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        # else:
        #     mu_prime = mu
        mu_prime = mu
        mu_prime = T.clamp(
            mu_prime * self.max_action[0], self.min_action[0], self.max_action[0]
        )
        self.time_step += 1
        # should be list if moving to real world now or like basta optional ra ang numpy
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return None, None
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + T.clamp(
            T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5
        )
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)
        target = reward + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return critic_loss, None

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
        return critic_loss, actor_loss

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = (
                tau * critic_1[name].clone() + (1 - tau) * target_critic_1[name].clone()
            )

        for name in critic_2:
            critic_2[name] = (
                tau * critic_2[name].clone() + (1 - tau) * target_critic_2[name].clone()
            )

        for name in actor:
            actor[name] = (
                tau * actor[name].clone() + (1 - tau) * target_actor[name].clone()
            )

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def save_models(self):
        print("...saving checkpoint...")
        T.save(
            {
                "actor": self.actor.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "target_critic_1": self.target_critic_1.state_dict(),
                "target_critic_2": self.target_critic_2.state_dict(),
                "actor_optimizer": self.actor.optimizer.state_dict(),
                "target_actor_optimizer": self.target_actor.optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1.optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2.optimizer.state_dict(),
                "target_critic_1_optimizer": self.target_critic_1.optimizer.state_dict(),
                "target_critic_2_optimizer": self.target_critic_2.optimizer.state_dict(),
                "timestep": self.time_step,
            },
            self.chkpt_file_pth,
        )

    def load_models(self):
        print("...loading checkpoint...")
        checkpoint = T.load(self.chkpt_file_pth)
        self.actor.load_state_dict(checkpoint["actor"])
        self.target_actor.load_state_dict(checkpoint["target_actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.target_critic_1.load_state_dict(checkpoint["target_critic_1"])
        self.target_critic_2.load_state_dict(checkpoint["target_critic_2"])
        self.actor.optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.target_actor.optimizer.load_state_dict(
            checkpoint["target_actor_optimizer"]
        )
        self.critic_1.optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2.optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        self.target_critic_1.optimizer.load_state_dict(
            checkpoint["target_critic_1_optimizer"]
        )
        self.target_critic_2.optimizer.load_state_dict(
            checkpoint["target_critic_2_optimizer"]
        )
        self.time_step = checkpoint["timestep"]
