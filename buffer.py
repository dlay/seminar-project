import numpy as np
import random
from tree import SumTree

# based on https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py
class PrioritizedReplayBuffer(object):

    def __init__(self, max_size, action_dim, state_dim, eps=1e-2, alpha=0.1, beta=0.1):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.real_size = 0

        self.tree = SumTree(size=max_size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        self.user_memory = np.zeros((self.mem_size), dtype=int)
        self.items_memory = np.zeros((self.mem_size, *[state_dim]), dtype=int)
        self.new_items_memory = np.zeros((self.mem_size, *[state_dim]), dtype=int)
        self.action_memory = np.zeros((self.mem_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def add(self, user, items, action, reward, items_, done):
        self.tree.add(self.max_priority, self.mem_cntr)

        self.user_memory[self.mem_cntr] = user
        self.items_memory[self.mem_cntr] = items
        self.new_items_memory[self.mem_cntr] = items_
        self.action_memory[self.mem_cntr] = action
        self.reward_memory[self.mem_cntr] = reward
        self.terminal_memory[self.mem_cntr] = done

        self.mem_cntr = (self.mem_cntr + 1) % self.mem_size
        self.real_size = min(self.mem_size, self.real_size + 1)

    def sample(self, batch_size):
        idxs = []
        weights = np.empty(batch_size)
        tree_idxs = []

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            min_seg, max_seg = segment * i, segment * (i + 1)
            rnd = random.uniform(min_seg, max_seg)
            tree_index, priority, sample_index = self.tree.get(rnd)

            tree_idxs.append(tree_index)
            idxs.append(sample_index)
            prob = priority / self.tree.total
            weights[i] = (self.real_size * prob) ** -self.beta

        weights = weights / weights.max()

        user = self.user_memory[idxs]
        items = self.items_memory[idxs]
        actions = self.action_memory[idxs]
        rewards = self.reward_memory[idxs]
        items_ = self.new_items_memory[idxs]
        terminal = self.terminal_memory[idxs]

        return user, items, actions, rewards, items_, terminal, weights, tree_idxs
    
    def update_priority(self, priority, index):
        priority = (priority + self.eps) ** self.alpha
        self.tree.update(index, priority)
        self.max_priority = max(self.max_priority, priority)

class ReplayBuffer(object):
    def __init__(self, max_size, action_dim, state_dim):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.real_size = 0

        self.user_memory = np.zeros((self.mem_size), dtype=int)
        self.items_memory = np.zeros((self.mem_size, *[state_dim]), dtype=int)
        self.new_items_memory = np.zeros((self.mem_size, *[state_dim]), dtype=int)
        self.action_memory = np.zeros((self.mem_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def add(self, user, items, action, reward, items_, done):
        self.user_memory[self.mem_cntr] = user
        self.items_memory[self.mem_cntr] = items
        self.new_items_memory[self.mem_cntr] = items_
        self.action_memory[self.mem_cntr] = action
        self.reward_memory[self.mem_cntr] = reward
        self.terminal_memory[self.mem_cntr] = done

        self.mem_cntr = (self.mem_cntr + 1) % self.mem_size
        self.real_size = min(self.mem_size, self.real_size + 1)

    def sample(self, batch_size):
        idxs = np.random.choice(self.real_size, batch_size, replace=False)

        user = self.user_memory[idxs]
        items = self.items_memory[idxs]
        actions = self.action_memory[idxs]
        rewards = self.reward_memory[idxs]
        items_ = self.new_items_memory[idxs]
        terminal = self.terminal_memory[idxs]

        return user, items, actions, rewards, items_, terminal