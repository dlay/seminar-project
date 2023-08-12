import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from buffer import PrioritizedReplayBuffer, ReplayBuffer
from networks import ActorNetwork, CriticNetwork
from noise import OUActionNoise
from state_rep_model import DRRAveStateRepresentation

class DDPGAgent(object):
    def __init__(self, embed_dim, action_dim, hidden_dim, users_count, items_count, state_window_size, buffer_size,
                 batch_size, gamma, tau, lr_actor, lr_critic, lr_decay_actor, lr_decay_critic,
                 noise=True, sigma=.2, theta=.15, use_pretrained_embeds = False, use_per_buffer = False):
        state_dim = embed_dim * 3
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.items_count = items_count
        self.use_per_buffer = use_per_buffer
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        print(self.device)

        self.actor_network = ActorNetwork(embed_dim, action_dim, hidden_dim, users_count, items_count).to(self.device)
        self.target_actor_network = ActorNetwork(embed_dim, action_dim, hidden_dim, users_count, items_count).to(self.device)
        self.critic_network = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic_network = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        if use_pretrained_embeds:
            if embed_dim == 8:
                pretrained_weights = T.load('ml-1m/pre_embeds_8.pt')
            if embed_dim == 100:
                pretrained_weights = T.load('ml-1m/pre_embeds_100.pt')
            self.actor_network.srm.load_state_dict(pretrained_weights)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=lr_actor)#, weight_decay=lr_decay_actor)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=lr_critic)#, weight_decay=lr_decay_critic)

        if use_per_buffer:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size, action_dim, state_window_size)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size, action_dim, state_window_size)

        self.noise = None
        if (noise):
            self.noise = OUActionNoise(np.zeros(action_dim), sigma, theta)

        # initial hard copy weights
        self.update_target_networks(1)

    def noise_reset(self):
        self.noise.reset()

    def get_action(self, user, items, avail_items, eval=False):
        with T.no_grad():
            user_embed = self.actor_network.srm.user_embeds(T.tensor(user).to(self.device))
            items_embed = self.actor_network.srm.item_embeds(T.tensor(items).to(self.device))
            action = self.actor_network(user_embed, items_embed)
            if self.noise is not None and not eval:
                action = action + T.tensor(self.noise(), dtype=T.float).to(self.device)
            if not eval:
                avail_item_embs = self.actor_network.srm.item_embeds(T.tensor(avail_items).to(self.device))
            else:
                avail_item_embs = self.actor_network.srm.item_embeds(avail_items.to(self.device))
            scores = T.tensordot(avail_item_embs, action, dims=([1], [0]))
            if eval:
                _, indices = scores.topk(10)
                return avail_items[indices.cpu().detach().numpy()]
            idx = T.argmax(scores)
            return avail_items[idx], action.cpu().detach().numpy()

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        for tp, sp in zip(self.actor_network.parameters(), self.target_actor_network.parameters()):
            tp.data.copy_((1.0 - tau) * tp.data + tau * sp.data)
        for tp, sp in zip(self.critic_network.parameters(), self.target_critic_network.parameters()):
            tp.data.copy_((1.0 - tau) * tp.data + tau * sp.data)

        return

    def train(self, writer, step):
        # Randomly sample a batch of size N from Buffer
        if self.use_per_buffer:
            users, items, actions, rewards, next_items, dones, weights, tree_idxs = self.replay_buffer.sample(self.batch_size)
            weights = T.tensor(weights, dtype=T.float).to(self.device)
        else:
            users, items, actions, rewards, next_items, dones = self.replay_buffer.sample(self.batch_size)
        users = T.tensor(users, dtype=T.int).to(self.device)
        items = T.tensor(items, dtype=T.int).to(self.device)
        actions = T.tensor(actions, dtype=T.float).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.device)
        next_items = T.tensor(next_items, dtype=T.int).to(self.device)
        dones = T.tensor(dones, dtype=T.float).to(self.device)
        rewards = T.unsqueeze(rewards, -1)
        dones = T.unsqueeze(dones, -1)

        users_embeds = self.actor_network.srm.user_embeds(users)
        items_embeds = self.actor_network.srm.item_embeds(items)
        states = self.actor_network.srm(users_embeds, items_embeds)

        next_items_embeds = self.actor_network.srm.item_embeds(next_items)
        next_states = self.actor_network.srm(users_embeds, next_items_embeds)

        # 8. Update yi = ri +γQ′(si+1, π ′(si+1)) and then update critic, minimizing L = 1/N*MSELoss(yi − QθQ (si , ai ))
        next_actions = self.target_actor_network(users_embeds, next_items_embeds)
        q_next = self.target_critic_network(next_states, next_actions)
        q_targets = rewards + self.gamma * q_next * (1.0 - dones)
        q = self.critic_network(states, actions)

        if self.use_per_buffer:
            td_error = T.abs(q_targets - q).detach()
            critic_loss = T.mean((q_targets.detach() - q)**2 * weights)

            for err, idx in zip(td_error, tree_idxs):
                self.replay_buffer.update_priority(err.detach().cpu().numpy(), idx)
        else:
            critic_loss = T.mean((q_targets.detach() - q)**2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        

        # 9. Update actor, using sampled policy gradient
        new_actions = self.actor_network(users_embeds, items_embeds)
        actor_loss = - T.mean(self.critic_network(states, new_actions))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        '''
        writer.add_histogram('q', q, step)
        writer.add_histogram('q target', q_targets, step)
        writer.add_histogram('q next', q_next, step)

        writer.add_histogram('actor fc1 weight', self.actor_network.fc1.weight, step)
        writer.add_histogram('actor fc2 weight', self.actor_network.fc2.weight, step)
        writer.add_histogram('srm weight', self.actor_network.srm.drr_ave.weight, step)
        writer.add_histogram('actor out weight', self.actor_network.out.weight, step)

        writer.add_histogram('critic fc1 weight', self.critic_network.fc1.weight, step)
        writer.add_histogram('critic fc2 weight', self.critic_network.fc2.weight, step)
        writer.add_histogram('critic out weight', self.critic_network.out.weight, step)
        '''

        # 10. Update weights
        self.update_target_networks()

        return actor_loss, critic_loss

    def remember(self, user, items, action, reward, items_, done):
        self.replay_buffer.add(user, items, action, reward, items_, done)

    def save_model(self, path):
        print('saving model...')
        T.save({
            'actor_model': self.actor_network.state_dict(),
            'critic_model': self.critic_network.state_dict(),
        }, path)
        print('model saved')

    def load_model(self, path):
        data = T.load(path)
        self.actor_network.load_state_dict(data['actor_model'])
        self.critic_network.load_state_dict(data['critic_model'])
        self.update_target_networks()