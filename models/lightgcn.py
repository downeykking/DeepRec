import torch
import torch.nn as nn
import torch.nn.functional as F
from data import Interaction
import numpy as np
from .base_model import BasicModel
from layers import LightGCNConv


class LightGCN(BasicModel):
    def __init__(self,
                 embed_dim: int,
                 num_layers: int,
                 dataset: Interaction,
                 device):
        super(LightGCN, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.device = device
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self.lightgcn_conv = LightGCNConv(dim=self.embed_dim)

        self.edge_index, self.edge_weight = dataset.get_norm_adj_mat()
        self.edge_index, self.edge_weight = self.edge_index.to(device), self.edge_weight.to(device)

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.reset_parameters()

    def reset_parameters(self, pretrain=0, init_method="xavier_normal_", dir=None):
        if pretrain:
            pretrain_user_embedding = np.load(dir + 'user_embeddings.npy')
            pretrain_item_embedding = np.load(dir + 'item_embeddings.npy')
            pretrain_user_tensor = torch.FloatTensor(pretrain_user_embedding)
            pretrain_item_tensor = torch.FloatTensor(pretrain_item_embedding)
            self.user_embeddings = nn.Embedding.from_pretrained(pretrain_user_tensor)
            self.item_embeddings = nn.Embedding.from_pretrained(pretrain_item_tensor)
        else:
            nn.init.normal_(self.user_embeddings.weight, std=0.1)
            nn.init.normal_(self.item_embeddings.weight, std=0.1)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embeddings.weight
        item_embeddings = self.item_embeddings.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def computer(self):
        # E^0  size (num_users+num_items, d)
        emb_0 = self.get_ego_embeddings()
        embeddings_list = [emb_0]
        emb_k = emb_0

        for _ in range(self.num_layers):
            emb_k = self.lightgcn_conv(emb_k, self.edge_index, self.edge_weight)
            embeddings_list.append(emb_k)
        all_embeddings = torch.stack(embeddings_list, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)

        user_final_emb, item_final_emb = torch.split(final_embeddings, [self.num_users, self.num_items])
        return user_final_emb, item_final_emb

    def predict(self, users, items):
        users = torch.LongTensor(users).to(self.device)
        items = torch.LongTensor(items).to(self.device)
        user_final_emb, item_final_emb = self.computer()

        u_embeddings = user_final_emb[users]
        i_embeddings = item_final_emb[items]

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def rating(self, users):
        users = torch.LongTensor(users).to(self.device)

        # used for getting batch user embedding, for many batches we just computer embedding once
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.computer()

        # get user embedding from storage variable
        users_emb = self.restore_user_e[users]

        # dot with all item embedding to accelerate
        scores = torch.matmul(users_emb, self.restore_item_e.t())
        return scores

    def forward(self, users, pos_items, neg_items):

        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_final_emb, item_final_emb = self.computer()

        users_emb_lightgcn = user_final_emb[users]
        pos_emb_lightgcn = item_final_emb[pos_items]
        neg_emb_lightgcn = item_final_emb[neg_items]

        users_emb = self.user_embeddings(users)
        pos_emb = self.item_embeddings(pos_items)
        neg_emb = self.item_embeddings(neg_items)

        return users_emb_lightgcn, pos_emb_lightgcn, neg_emb_lightgcn, users_emb, pos_emb, neg_emb
