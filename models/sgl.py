import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BasicModel
from layers import LightGCNConv
from utils import xavier_uniform_initialization
from losses import BPRLoss, EmbLoss
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class SGL(BasicModel):
    def __init__(self, config, dataset):
        super(SGL, self).__init__(config, dataset)
        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.aug_type = config["aug_type"]
        self.drop_ratio = config["drop_ratio"]
        self.ssl_tau = config["ssl_tau"]
        self.reg_weight = config["reg_weight"]
        self.ssl_weight = config["ssl_weight"]

        self.user_embeddings = nn.Embedding(self.num_users, self.embedding_size)
        self.item_embeddings = nn.Embedding(self.num_items, self.embedding_size)

        self.lightgcn_conv = LightGCNConv(dim=self.embedding_size)

        self.users, self.items = torch.LongTensor(dataset.to_user_item_pairs().T)
        self.edge_index, self.edge_weight = dataset.get_norm_adj_mat()
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)

        self.bpr_loss = BPRLoss(reduction="sum")
        self.reg_loss = EmbLoss(reg_weight=self.reg_weight, reduction="sum")

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(xavier_uniform_initialization)

    def train(self, mode: bool = True):
        r"""Override train method of base class. The subgraph is reconstructed each time it is called."""
        T = super().train(mode=mode)
        if mode:
            self.graph_construction()
        return T

    def graph_construction(self):
        r"""Devise three operators to generate the views — node dropout, edge dropout, and random walk of a node."""
        if self.aug_type == "ND" or self.aug_type == "ED":
            self.sub_graph1 = [self.random_graph_augment()] * self.n_layers
            self.sub_graph2 = [self.random_graph_augment()] * self.n_layers
        elif self.aug_type == "RW":
            self.sub_graph1 = [self.random_graph_augment() for _ in range(self.n_layers)]
            self.sub_graph2 = [self.random_graph_augment() for _ in range(self.n_layers)]

    def random_graph_augment(self):
        def rand_sample(high, size=None, replace=True):
            return np.random.choice(np.arange(high), size=size, replace=replace)

        if self.aug_type == "ND":
            drop_user = rand_sample(self.num_users, size=int(self.num_users * self.drop_ratio), replace=False)
            drop_item = rand_sample(self.num_items, size=int(self.num_items * self.drop_ratio), replace=False)

            mask = np.isin(self.users.numpy(), drop_user)
            mask |= np.isin(self.items.numpy(), drop_item)
            # keep = np.where(~mask)
            keep = ~mask

            row = self.users[keep]
            col = self.items[keep] + self.num_users

        elif self.aug_type == "ED" or self.aug_type == "RW":
            keep = rand_sample(len(self.users), size=int(len(self.users) * (1 - self.drop_ratio)), replace=False)
            row = self.users[keep]
            col = self.items[keep] + self.num_users

        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        edge_index, edge_weight = gcn_norm(edge_index, num_nodes=self.num_users + self.num_items, add_self_loops=False)

        return edge_index.to(self.device), edge_weight.to(self.device)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embeddings.weight
        item_embeddings = self.item_embeddings.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, sub_graph=None):
        # E^0  size (num_users+num_items, d)
        emb_0 = self.get_ego_embeddings()
        embeddings_list = [emb_0]
        emb_k = emb_0

        if sub_graph is None:
            for _ in range(self.n_layers):
                emb_k = self.lightgcn_conv(emb_k, self.edge_index, self.edge_weight)
                embeddings_list.append(emb_k)
        else:
            for edge_index, edge_weight in sub_graph:
                emb_k = self.lightgcn_conv(emb_k, edge_index, edge_weight)
                embeddings_list.append(emb_k)
        all_embeddings = torch.stack(embeddings_list, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)

        user_final_emb, item_final_emb = torch.split(final_embeddings, [self.num_users, self.num_items])
        return user_final_emb, item_final_emb

    def predict(self, users, items):
        users = torch.LongTensor(users).to(self.device)
        items = torch.LongTensor(items).to(self.device)
        user_final_emb, item_final_emb = self.forward()

        u_embeddings = user_final_emb[users]
        i_embeddings = item_final_emb[items]

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def rating(self, users):
        users = torch.LongTensor(users).to(self.device)

        # used for getting batch user embedding, for many batches we just computer embedding once
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        # get user embedding from storage variable
        users_emb = self.restore_user_e[users]

        # dot with all item embedding to accelerate
        scores = torch.matmul(users_emb, self.restore_item_e.t())
        return scores

    def calc_bpr_loss(self, user_emd, item_emd, users, pos_items, neg_items):
        r"""Calculate the the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.

        Args:
            user_emd (torch.Tensor): Ego embedding of all users after forwarding.
            item_emd (torch.Tensor): Ego embedding of all items after forwarding.
            users (torch.Tensor): List of the user.
            pos_items (torch.Tensor): List of positive examples.
            neg_items (torch.Tensor): List of negative examples.

        Returns:
            torch.Tensor: Loss of BPR tasks and parameter regularization.
        """
        u_e = user_emd[users]
        pi_e = item_emd[pos_items]
        ni_e = item_emd[neg_items]
        _bpr_loss = self.bpr_loss(u_e, pi_e, ni_e)

        u_e_p = self.user_embeddings(users)
        pi_e_p = self.item_embeddings(pos_items)
        ni_e_p = self.item_embeddings(neg_items)
        _reg_loss = self.reg_loss(u_e_p, pi_e_p, ni_e_p)

        return _bpr_loss + _reg_loss

    def calc_ssl_loss(self, users, pos_items, user_sub1, user_sub2, item_sub1, item_sub2):
        r"""Calculate the loss of self-supervised tasks.

        Args:
            users (torch.Tensor): List of the user.
            pos_items (torch.Tensor): List of positive items.
            user_sub1 (torch.Tensor): Ego embedding of all users in the first subgraph after forwarding.
            user_sub2 (torch.Tensor): Ego embedding of all users in the second subgraph after forwarding.
            item_sub1 (torch.Tensor): Ego embedding of all items in the first subgraph after forwarding.
            item_sub2 (torch.Tensor): Ego embedding of all items in the second subgraph after forwarding.

        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """
        user_sub1 = F.normalize(user_sub1, dim=1)
        user_sub2 = F.normalize(user_sub2, dim=1)
        item_sub1 = F.normalize(item_sub1, dim=1)
        item_sub2 = F.normalize(item_sub2, dim=1)

        u_emd1 = user_sub1[users]
        u_emd2 = user_sub2[users]
        all_user2 = user_sub2
        v1 = torch.sum(u_emd1 * u_emd2, dim=1)
        v2 = u_emd1.matmul(all_user2.T)
        v1 = torch.exp(v1 / self.ssl_tau)
        v2 = torch.sum(torch.exp(v2 / self.ssl_tau), dim=1)
        ssl_user = -torch.sum(torch.log(v1 / v2))

        i_emd1 = item_sub1[pos_items]
        i_emd2 = item_sub2[pos_items]
        all_item2 = item_sub2
        v3 = torch.sum(i_emd1 * i_emd2, dim=1)
        v4 = i_emd1.matmul(all_item2.T)
        v3 = torch.exp(v3 / self.ssl_tau)
        v4 = torch.sum(torch.exp(v4 / self.ssl_tau), dim=1)
        ssl_item = -torch.sum(torch.log(v3 / v4))

        return (ssl_item + ssl_user) * self.ssl_weight

    def calculate_loss(self, users, pos_items, neg_items):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_emd, item_emd = self.forward()
        user_sub1, item_sub1 = self.forward(self.sub_graph1)
        user_sub2, item_sub2 = self.forward(self.sub_graph2)

        total_loss = self.calc_bpr_loss(user_emd, item_emd, users, pos_items, neg_items) + \
            self.calc_ssl_loss(users, pos_items, user_sub1, user_sub2, item_sub1, item_sub2)
        return total_loss
