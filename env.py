import numpy as np

class Env(object):
    def __init__(self, user_items_dict: dict[int, tuple[int, int]], item_count, state_size) -> None:
        self.user_items_dict = user_items_dict
        self.item_count = item_count
        self.state_size = state_size
        self.state_storage = {}

    def reset(self, user):
        self.user = user
        if self.user in self.state_storage.keys():
            self.memory_items = self.state_storage[self.user]
            return self.memory_items
        self.rel_items = np.array(self.user_items_dict[user][self.state_size:])
        self.memory_items = np.array(self.user_items_dict[user][:self.state_size])
        self.state_storage[user] = self.memory_items
        self.scope = len(self.rel_items)
        irrel_items = np.random.choice(list(set(range(self.item_count)) - set(self.rel_items)),
                                       self.scope, replace=False)
        self.avail_items = np.concatenate([self.rel_items, irrel_items])
        assert(len(np.unique(self.avail_items)) == len(self.avail_items))
        self.done = False
        self.recommended_items = set(self.memory_items)
        return self.memory_items
    
    def step(self, action, eval=False):
        if eval:
            k_rewards = []
            for a in action:
                if a in self.rel_items and a not in self.recommended_items:
                    k_rewards.append(1)
                    self.memory_items = np.roll(self.memory_items, -1)
                    self.memory_items[-1] = a
                    self.state_storage[self.user] = self.memory_items
                else:
                    k_rewards.append(0)
            return self.memory_items, k_rewards
        reward = 0.0
        if action in self.rel_items and action not in self.recommended_items:
            reward = 1.0
            self.memory_items = np.roll(self.memory_items, -1)
            self.memory_items[-1] = action
            self.state_storage[self.user] = self.memory_items
        self.recommended_items.add(action)
        self.avail_items = self.avail_items[self.avail_items != action]

        if len(self.rel_items) == len(self.recommended_items):
            self.done = True

        return self.memory_items, reward, reward # the last return would normally be the done flag, we are imitating the authors implementation here