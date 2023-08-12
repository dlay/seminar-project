import numpy as np
import torch as T
import torch.utils.data as td

class EvalSet(td.Dataset):
    def __init__(self, users, user_dict, items_count, state_size):
        super(EvalSet, self).__init__()
        self.data = []
        for user in users:
            pos_items = user_dict[user]
            neg_items = list(set(i for i in range(items_count)) - set(pos_items))
            cntr = 0
            for pos_item in pos_items:
                if cntr < state_size:
                    cntr += 1
                    continue
                self.data.append([user, pos_item, 1])
                rnd_neg_items = np.random.choice(neg_items, 99)
                for neg_item in rnd_neg_items:
                    self.data.append([user, neg_item, 0])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        user, item, label = self.data[index]
        return {
            "user": user,
            "item": item,
            "label": label,
        }

def run_eval(agent, env, loader):
        hits = []
        ndcgs= []
        prev_user = -1
        for batch in loader:
            with T.no_grad():
                user = batch['user'][0].item()
                if user != prev_user:
                    items = env.reset(user)
                k_actions = agent.get_action(user, items, batch['item'], eval=True)
                actual = batch['item'][0].item()
                prediction = k_actions.detach().cpu().numpy().tolist()
                items, k_rewards = env.step(k_actions.detach().numpy(), eval=True)
                prev_user = user
                hits.append(calc_hit_rate(prediction, actual))
                ndcgs.append(calc_ndcg(prediction, actual))
        
        return np.round(np.mean(hits), 3), np.round(np.mean(ndcgs), 3)

def calc_hit_rate(rec_items, actual_item):
    return int(actual_item in rec_items)

def calc_ndcg(rel, irel):
    if irel in rel:
        index = rel.index(irel)
        return np.reciprocal(np.log2(index + 2))
    return 0