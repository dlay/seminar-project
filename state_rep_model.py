import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DRRAveStateRepresentation(nn.Module):
    def __init__(self, embedding_dim, user_count, item_count):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.user_embeds = nn.Embedding(user_count, embedding_dim)
        self.item_embeds = nn.Embedding(item_count, embedding_dim)
        self.drr_ave = nn.Conv1d(5, 1, 1)

        nn.init.normal_(self.user_embeds.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embeds.weight, mean=0.0, std=0.01)
        nn.init.uniform_(self.drr_ave.weight)
        self.drr_ave.bias.data.zero_()
        
    def forward(self, user_embed, item_embed):
        # user_embed = self.user_embeds(user.long())
        # item_embed = self.item_embeds(items.long())
        drr_ave = T.squeeze(self.drr_ave(item_embed))
        mult = T.multiply(user_embed, drr_ave)
        concat = T.concat([user_embed, mult, drr_ave], dim=-1)
        return concat
    