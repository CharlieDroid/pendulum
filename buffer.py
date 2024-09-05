import numpy as np
import pickle
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Node:
    value: float = 0.1
    total: float = 0.1

    def update_priority(self, priority: float):
        delta = priority - self.value
        self.value = priority
        self.total += delta
        return delta

    def update_total(self, delta: float):
        self.total += delta

    def set_value(self, value: float):
        self.value = value
        self.total = value


class SumTree:
    def __init__(
        self,
        max_size,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        self.counter = 0
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.alpha_start = alpha
        self.beta_start = beta
        self.sum_tree = []

    def _insert(self, value=None):
        if self.counter < self.max_size:
            self.sum_tree.append(Node())
        if value:
            # self.sum_tree[self.counter].set_value(value)
            self.propagate_changes([self.counter % self.max_size], [value])
        self.counter += 1

    def store_transition(self, vals):
        self._insert(vals)

    def calculate_parents(self, index: int):
        parents = []
        while index > 0:
            parents.append(int((index - 1) // 2))
            index = int((index - 1) // 2)
        return parents

    def update_priorities(self, indices: List, priorities: List):
        self.propagate_changes(indices, priorities)

    def propagate_changes(self, indices: List, priorities: List):
        for idx, p in zip(indices, priorities):
            delta = self.sum_tree[idx].update_priority(p**self.alpha)
            parents = self.calculate_parents(idx)
            for parent in parents:
                self.sum_tree[parent].update_total(delta)

    def _sample(self, batch_size):
        total_weight = self.sum_tree[0].total
        # if we're not using our actors to come up with initial estimates for
        # priorities, then we will default to uniform sampling.
        if total_weight == 0.1:
            samples = np.random.choice(self.counter, batch_size, replace=False)
            probs = [1 / batch_size for _ in range(batch_size)]
            return samples, probs

        samples, probs, n_samples = [], [], 1
        index = self.counter % self.max_size - 1
        samples.append(index)
        probs.append(self.sum_tree[index].value / self.sum_tree[0].total)
        while n_samples < batch_size:
            index = 0
            target = total_weight * np.random.random()
            while True:
                left = 2 * index + 1
                right = 2 * index + 2
                if left > len(self.sum_tree) - 1 or right > len(self.sum_tree) - 1:
                    break
                left_sum = self.sum_tree[left].total
                if target < left_sum:
                    index = left
                    continue
                target -= left_sum
                right_sum = self.sum_tree[right].total
                if target < right_sum:
                    index = right
                    continue
                target -= right_sum
                break
            # no guarantee there won't be repeat indices in samples!
            samples.append(index)
            n_samples += 1
            probs.append(max(self.sum_tree[index].value, 1e-6) / self.sum_tree[0].total)
        return samples, probs

    def sample(self, batch_size):
        samples, probs = self._sample(batch_size)
        weights = self._calculate_weights(probs)
        return samples, weights

    def _calculate_weights(self, probs: List):
        weights = np.array(
            [(1 / self.counter * 1 / prob) ** self.beta for prob in probs]
        )
        weights *= 1 / max(weights)
        return weights

    def anneal_beta(self, ep: int, ep_max: int):
        self.beta = self.beta_start + ep / ep_max * (1 - self.beta_start)
        if self.beta > 1.0:
            self.beta = 1.0


class PrioritizedReplayBuffer:
    def __init__(
        self,
        max_size,
        input_shape,
        n_actions,
        distill,
        n_extra_obs=0,
        alpha=0.6,
        beta=0.4,
    ):
        self.tree = SumTree(max_size, alpha, beta)
        self.mem_size = max_size
        self.mem_cntr = 0
        self.distill = distill

        # Asymmetric params
        self.n_extra_obs = n_extra_obs

        # transition: state, action, reward, next_state, done
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        if distill:
            self.teacher_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(
        self, state, action, reward, state_, done, teacher_action=None, vals=None
    ):
        index = self.mem_cntr % self.mem_size

        self.tree.store_transition(vals)

        self.state_memory[index] = state
        self.action_memory[index] = action
        if self.distill:
            self.teacher_memory[index] = teacher_action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        indices, weights = self.tree.sample(batch_size)

        states = self.state_memory[indices]
        states_ = self.new_state_memory[indices]
        actions = self.action_memory[indices]
        if self.distill:
            teacher_actions = self.teacher_memory[indices]
        rewards = self.reward_memory[indices]
        dones = self.terminal_memory[indices]

        actor_states = states[:, : -self.n_extra_obs]
        actor_states_ = states_[:, : -self.n_extra_obs]

        if self.distill:
            batch = (
                states,
                actions,
                teacher_actions,
                rewards,
                states_,
                dones,
                actor_states,
                actor_states_,
            )
        else:
            batch = (
                states,
                actions,
                rewards,
                states_,
                dones,
                actor_states,
                actor_states_,
            )
        return batch, weights, indices

    def update_priorities(self, indices, values):
        self.tree.update_priorities(indices, values)

    def save(self, file_pth):
        print("...saving memory...")
        if self.distill:
            memory = (
                self.state_memory,
                self.new_state_memory,
                self.action_memory,
                self.terminal_memory,
                self.reward_memory,
                self.mem_cntr,
                self.teacher_memory,
                self.tree,
            )
        else:
            memory = (
                self.state_memory,
                self.new_state_memory,
                self.action_memory,
                self.terminal_memory,
                self.reward_memory,
                self.mem_cntr,
                self.tree,
            )
        with open(file_pth, "wb") as outfile:
            pickle.dump(memory, outfile, pickle.HIGHEST_PROTOCOL)

    def load(self, file_pth):
        print("...loading memory...")
        with open(file_pth, "rb") as infile:
            result = pickle.load(infile)
        if self.distill:
            (
                self.state_memory,
                self.new_state_memory,
                self.action_memory,
                self.terminal_memory,
                self.reward_memory,
                self.mem_cntr,
                self.teacher_memory,
                self.tree,
            ) = result
        else:
            (
                self.state_memory,
                self.new_state_memory,
                self.action_memory,
                self.terminal_memory,
                self.reward_memory,
                self.mem_cntr,
                self.tree,
            ) = result


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, n_extra_obs=0):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_extra_obs = n_extra_obs
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.teacher_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done, teacher_action):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.teacher_memory[index] = teacher_action
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
        teacher_actions = self.teacher_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        actor_states = states[:, : -self.n_extra_obs]
        actor_states_ = states_[:, : -self.n_extra_obs]

        return (
            states,
            actions,
            rewards,
            states_,
            dones,
            actor_states,
            actor_states_,
            teacher_actions,
        )

    def save(self, file_pth):
        print("...saving memory...")
        memory = (
            self.state_memory,
            self.new_state_memory,
            self.action_memory,
            self.terminal_memory,
            self.reward_memory,
            self.mem_cntr,
        )
        with open(file_pth, "wb") as outfile:
            pickle.dump(memory, outfile, pickle.HIGHEST_PROTOCOL)

    def load(self, file_pth):
        print("...loading memory...")
        with open(file_pth, "rb") as infile:
            result = pickle.load(infile)
        (
            self.state_memory,
            self.new_state_memory,
            self.action_memory,
            self.terminal_memory,
            self.reward_memory,
            self.mem_cntr,
        ) = result
