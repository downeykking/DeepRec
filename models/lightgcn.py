import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BasicModel
from layers import LightGCNConv
from losses import BPRLoss, EmbLoss
from utils import xavier_uniform_initialization


class LightGCN(BasicModel):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.require_pow = config['require_pow']

        self.user_embeddings = nn.Embedding(self.num_users, self.embedding_size)
        self.item_embeddings = nn.Embedding(self.num_items, self.embedding_size)

        self.lightgcn_conv = LightGCNConv(dim=self.embedding_size)

        self.edge_index, self.edge_weight = dataset.get_norm_adj_mat()
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)

        # loss
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss(reg_weight=self.reg_weight, require_pow=self.require_pow)

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(xavier_uniform_initialization)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embeddings.weight
        item_embeddings = self.item_embeddings.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        # E^0  size (num_users+num_items, d)
        emb_0 = self.get_ego_embeddings()
        embeddings_list = [emb_0]
        emb_k = emb_0

        for _ in range(self.n_layers):
            emb_k = self.lightgcn_conv(emb_k, self.edge_index, self.edge_weight)
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

    def calculate_loss(self, users, pos_items, neg_items):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_final_emb, item_final_emb = self.forward()

        users_emb_lightgcn = user_final_emb[users]
        pos_emb_lightgcn = item_final_emb[pos_items]
        neg_emb_lightgcn = item_final_emb[neg_items]

        users_emb = self.user_embeddings(users)
        pos_emb = self.item_embeddings(pos_items)
        neg_emb = self.item_embeddings(neg_items)

        _bpr_loss = self.bpr_loss(users_emb_lightgcn, pos_emb_lightgcn, neg_emb_lightgcn)
        _reg_loss = self.reg_loss(users_emb, pos_emb, neg_emb)

        loss = _bpr_loss + _reg_loss

        return loss
