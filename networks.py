import torch as T
import torch.nn as nn
import torch.nn.functional as F

from state_rep_model import DRRAveStateRepresentation

# Actor network
class ActorNetwork(nn.Module):
    def __init__(self, embed_dim, action_dim, hidden_dim, user_count, item_count):
        super(ActorNetwork, self).__init__()
        
        
        self.fc1 = nn.Linear(embed_dim * 3, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

        self.srm = DRRAveStateRepresentation(embed_dim, user_count, item_count)

        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.out.weight)

    def forward(self, user, items):
        state = self.srm(user, items)
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.out(x)
        return T.tanh(x)


# Critic network
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.out.weight)

    def forward(self, state, action):
        x = T.concat([state, action], dim=-1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.out(x)
        return x