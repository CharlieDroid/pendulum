import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from time import time


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))

        # activation is tanh because it bounds it between +- 1
        # just multiply this according to the maximum action of the environment
        mu = T.tanh(self.mu(a))
        return mu


if __name__ == "__main__":
    actor = ActorNetwork(
        3e-3,
        (4,),
        256,
        256,
        n_actions=1,
    )
    samples = 1000
    durations = np.zeros(samples)
    for i in range(samples):
        start = time()
        actor.forward(T.rand(4))
        duration = time() - start
        durations[i] = duration
    print(f"max={max(durations)}")
    print(f"average={np.average(durations)}")
    print(f"minimum_time=0.02")
