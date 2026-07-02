
import torch
import torch.nn as nn

class BPRModel(nn.Module):

    def __init__(self, n_users, n_items, latent_dim=128):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, latent_dim)
        self.item_emb = nn.Embedding(n_items, latent_dim)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, users, items):

        pu = self.user_emb(users)
        qi = self.item_emb(items)

        bu = self.user_bias(users).squeeze(-1)
        bi = self.item_bias(items).squeeze(-1)

        return (pu * qi).sum(dim=1) + bu + bi

    def forward(self, users, pos_items, neg_items):

        pos_score = self.score(users, pos_items)
        neg_score = self.score(users, neg_items)

        return pos_score, neg_score

