import torch
import torch.nn as nn
import torch.nn.functional as F
from data import Dataset
import numpy as np


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        r"""Predict the scores between users and items. the size of users and items are the same.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size,]
        """
        raise NotImplementedError

    def rating(self, *args, **kwargs):
        """
        Given users, calculate the scores between users and all candidate items.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users, m_all_candidate_items]
        """
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class PureMF(BasicModel):
    def __init__(self,
                 embed_dim: int,
                 dataset: Dataset,
                 device):
        super(PureMF, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.embed_dim = embed_dim
        self.device = device
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self.reset_parameters()

    def reset_parameters(self, pretrain=0, init_method="uniform", dir=None):
        if pretrain:
            pretrain_user_embedding = np.load(dir + 'user_embeddings.npy')
            pretrain_item_embedding = np.load(dir + 'item_embeddings.npy')
            pretrain_user_tensor = torch.FloatTensor(pretrain_user_embedding)
            pretrain_item_tensor = torch.FloatTensor(pretrain_item_embedding)
            self.user_embeddings = nn.Embedding.from_pretrained(pretrain_user_tensor)
            self.item_embeddings = nn.Embedding.from_pretrained(pretrain_item_tensor)
        else:
            nn.init.uniform_(self.user_embeddings.weight)
            nn.init.uniform_(self.item_embeddings.weight)

    def predict(self, users, items):
        users = torch.LongTensor(users).to(self.device)
        items = torch.LongTensor(items).to(self.device)

        u_embeddings = self.user_embeddings[users]
        i_embeddings = self.item_embeddings[items]

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def rating(self, users):
        users = torch.LongTensor(users).to(self.device)
        users_emb = self.user_embeddings(users)
        items_emb = self.item_embeddings.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return scores

    def forward(self, users, pos_items, neg_items):
        users_emb = self.user_embeddings(users)
        pos_emb = self.item_embeddings(pos_items)
        neg_emb = self.item_embeddings(neg_items)
        return users_emb, pos_emb, neg_emb
