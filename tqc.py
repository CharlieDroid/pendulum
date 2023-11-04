import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Distribution, Normal
from gymnasium import spaces
import gymnasium as gym
import copy
import os


T.autograd.set_detect_anomaly(True)


class RescaleAction(gym.ActionWrapper):
    def __init__(self, env, a, b):
        assert isinstance(
            env.action_space, spaces.Box
        ), "expected Box action space, got {}".format(type(env.action_space))
        assert np.less_equal(a, b).all(), (a, b)
        super(RescaleAction, self).__init__(env)
        self.a = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + a
        self.b = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + b
        self.action_space = spaces.Box(
            low=a, high=b, shape=env.action_space.shape, dtype=env.action_space.dtype
        )

    def action(self, action):
        assert np.all(np.greater_equal(action, self.a)), (action, self.a)
        assert np.all(np.less_equal(action, self.b)), (action, self.b)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * ((action - self.a) / (self.b - self.a))
        action = np.clip(action, low, high)
        return action


class TanhNormal(Distribution):
    arg_constraints = {}

    def __init__(self, normal_mean, normal_std):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.standard_normal = Normal(
            T.zeros_like(self.normal_mean, device=self.device),
            T.ones_like(self.normal_std, device=self.device),
        )
        self.normal = Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh):
        log_det = (
            2 * np.log(2) + F.logsigmoid(2 * pre_tanh) + F.logsigmoid(-2 * pre_tanh)
        )
        result = self.normal.log_prob(pre_tanh) - log_det
        return result

    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return T.tanh(pretanh), pretanh


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.transition_names = ("state", "action", "state_", "reward", "not_done")
        sizes = (state_dim[0], action_dim, state_dim[0], 1, 1)
        for name, size in zip(self.transition_names, sizes):
            setattr(self, name, np.empty((max_size, size)))

    def store_transition(self, state, action, reward, state_, done):
        values = (state, action, state_, reward, 1.0 - done)
        for name, value in zip(self.transition_names, values):
            getattr(self, name)[self.ptr] = value

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_buffer(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        names = self.transition_names
        return (
            T.FloatTensor(getattr(self, name)[ind]).to(self.device) for name in names
        )


# class ReplayBuffer:
#     def __init__(self, max_size, input_shape, n_actions):
#         self.mem_size = max_size
#         self.mem_cntr = 0
#         self.state_memory = np.zeros((self.mem_size, *input_shape))
#         self.new_state_memory = np.zeros((self.mem_size, *input_shape))
#         self.action_memory = np.zeros((self.mem_size, n_actions))
#         self.reward_memory = np.zeros(self.mem_size)
#         self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
#
#     def store_transition(self, state, action, reward, state_, done):
#         index = self.mem_cntr % self.mem_size
#         self.state_memory[index] = state
#         self.action_memory[index] = action
#         self.reward_memory[index] = reward
#         self.new_state_memory[index] = state_
#         self.terminal_memory[index] = 1.0 - done
#
#         self.mem_cntr += 1
#
#     def sample_buffer(self, batch_size):
#         max_mem = min(self.mem_cntr, self.mem_size)
#
#         batch = np.random.choice(max_mem, batch_size)
#
#         states = self.state_memory[batch]
#         states_ = self.new_state_memory[batch]
#         actions = self.action_memory[batch]
#         rewards = self.reward_memory[batch]
#         dones = self.terminal_memory[batch]
#
#         return states, actions, rewards, states_, dones


class Network(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(Network, self).__init__()
        self.fcs = []
        in_sz = input_dims
        for i, hidden_ in enumerate(hidden_dims):
            fc = nn.Linear(in_sz, hidden_)
            self.add_module(f"fc{i}", fc)
            self.fcs.append(fc)
            in_sz = hidden_
        self.last_fc = nn.Linear(in_sz, output_dims)

    def forward(self, state_action):
        h = state_action
        for fc in self.fcs:
            h = F.relu(fc(h))
        output = self.last_fc(h)
        return output


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, n_quantiles, n_nets):
        super(CriticNetwork, self).__init__()
        self.nets = []
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets

        for i in range(n_nets):
            net = Network(input_dims[0] + n_actions, [512, 512, 512], n_quantiles)
            self.add_module(f"qf{i}", net)
            self.nets.append(net)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state, action):
        sa = T.cat([state, action], dim=1)
        quantiles = T.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.net = Network(input_dims[0], [256, 256], 2 * n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    def forward(self, state, training=True):
        mean, log_std = self.net(state).split([self.n_actions, self.n_actions], dim=1)
        log_std = T.clamp(log_std, -20, 2)

        if training:
            std = T.exp(log_std)
            tanh_normal = TanhNormal(mean, std)
            action, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            action = T.tanh(mean)
            log_prob = None
        return action, log_prob


class Agent:
    def __init__(
        self,
        input_dims,
        n_actions,
        n_quantiles=25,
        n_nets=5,
        env=None,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        top_quantiles_to_drop_per_net=2,
        batch_size=256,
        max_size=int(1e6),
        chkpt_dir="./tmp/tqc",
        game_id="Pendulum-v2",
    ):
        self.warmup = 0  # for consistency
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -np.prod(env.action_space.shape).item()
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.top_quantiles_to_drop = top_quantiles_to_drop_per_net * n_nets
        self.batch_size = batch_size
        self.time_step = 0
        self.chkpt_file_pth = os.path.join(chkpt_dir, f"{game_id} tqc.chkpt")

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.actor = ActorNetwork(lr, input_dims, n_actions)
        self.critic = CriticNetwork(lr, input_dims, n_actions, n_quantiles, n_nets)
        self.critic_target = copy.deepcopy(self.critic)

        self.quantiles_total = self.critic.n_quantiles * self.critic.n_nets

        self.log_alpha = T.zeros((1,), requires_grad=True, device=self.device)
        self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=lr)

    def choose_action(self, observation, evaluate=False):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)[None, :]
        action, _ = self.actor.forward(state, training=not evaluate)
        return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.time_step += 1
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.ptr < self.batch_size:
            return None, None, None

        state, action, state_, reward, not_done = self.memory.sample_buffer(
            self.batch_size
        )

        alpha = T.exp(self.log_alpha)
        with T.no_grad():
            new_next_action, next_log_pi = self.actor.forward(state_)

            next_z = self.critic_target(state, new_next_action)
            sorted_z, _ = T.sort(next_z.reshape(self.batch_size, -1))
            sorted_z_part = sorted_z[
                :, : self.quantiles_total - self.top_quantiles_to_drop
            ]

            target = reward + not_done * self.gamma * (
                sorted_z_part - alpha * next_log_pi
            )
        cur_z = self.critic.forward(state, action)
        critic_loss = self.quantile_huber_loss_f(cur_z, target)

        new_action, log_pi = self.actor.forward(state, True)
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
        actor_loss = (
            alpha * log_pi
            - self.critic(state, new_action).mean(2).mean(1, keepdim=True)
        ).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return critic_loss, actor_loss, alpha_loss

    def quantile_huber_loss_f(self, quantiles, samples):
        pairwise_delta = (
            samples[:, None, None, :] - quantiles[:, :, :, None]
        )  # batch x nets x quantiles x samples
        abs_pairwise_delta = T.abs(pairwise_delta)
        huber_loss = T.where(
            abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5
        )

        n_quantiles = quantiles.shape[2]
        tau = (
            T.arange(n_quantiles, device=self.device).float() / n_quantiles
            + 1 / 2 / n_quantiles
        )
        loss = (
            T.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss
        ).mean()
        return loss

    def save_models(self):
        print("...saving checkpoint...")
        T.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "log_alpha": self.log_alpha,
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "actor_optimizer": self.actor.optimizer.state_dict(),
                "critic_optimizer": self.critic.optimizer.state_dict(),
                "time_step": self.time_step,
            },
            self.chkpt_file_pth,
        )

    def load_models(self):
        print("...loading checkpoint...")
        checkpoint = T.load(self.chkpt_file_pth)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor.optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic.optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.log_alpha = checkpoint["log_alpha"]
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.time_step = checkpoint["time_step"]
