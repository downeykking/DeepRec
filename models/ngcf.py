import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BasicModel
from layers import BiGNNConv
from losses import BPRLoss, EmbLoss
from torch_geometric.utils import dropout_node
from utils import xavier_normal_initialization


class NGCF(BasicModel):
    def __init__(self, config, dataset):
        super(NGCF, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size_list = config['hidden_size_list']
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.node_dropout = config['node_dropout']
        self.message_dropout = config['message_dropout']
        self.reg_weight = config['reg_weight']

        self.user_embeddings = nn.Embedding(self.num_users, self.embedding_size)
        self.item_embeddings = nn.Embedding(self.num_items, self.embedding_size)

        self.ngcf_convs = nn.ModuleList()
        for input_size, output_size in zip(self.hidden_size_list[:-1], self.hidden_size_list[1:]):
            self.ngcf_convs.append(BiGNNConv(input_size, output_size))

        self.edge_index, self.edge_weight = dataset.get_norm_adj_mat()
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)

        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss(reg_weight=self.reg_weight)

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(xavier_normal_initialization)

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
        if self.node_dropout == 0:
            edge_index, edge_weight = self.edge_index, self.edge_weight
        else:
            edge_index, edge_weight = self.edge_index, self.edge_weight
            edge_index, edge_mask, _ = dropout_node(edge_index=self.edge_index, p=self.node_dropout)
            edge_weight = edge_weight[edge_mask]

        # E^0  size (num_users+num_items, d)
        emb_0 = self.get_ego_embeddings()
        embeddings_list = [emb_0]
        emb_k = emb_0

        for conv in self.ngcf_convs:
            emb_k = conv(emb_k, edge_index, edge_weight)
            emb_k = F.leaky_relu(emb_k, negative_slope=0.2)
            emb_k = F.dropout(emb_k, p=self.message_dropout, training=self.training)
            emb_k = F.normalize(emb_k, p=2, dim=1)
            embeddings_list += [emb_k]

        all_embeddings = torch.cat(embeddings_list, dim=1)
        user_final_emb, item_final_emb = torch.split(all_embeddings, [self.num_users, self.num_items])

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

        # get user embedding from storage variable
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        users_emb = self.restore_user_e[users]

        # dot with all item embedding to accelerate
        scores = torch.matmul(users_emb, self.restore_item_e.t())
        return scores

    def calculate_loss(self, users, pos_items, neg_items):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_final_emb, item_final_emb = self.forward()

        users_emb_ngcf = user_final_emb[users]
        pos_emb_ngcf = item_final_emb[pos_items]
        neg_emb_ngcf = item_final_emb[neg_items]

        _bpr_loss = self.bpr_loss(users_emb_ngcf, pos_emb_ngcf, neg_emb_ngcf)  # calculate BPR Loss
        _reg_loss = self.reg_loss(users_emb_ngcf, pos_emb_ngcf, neg_emb_ngcf)  # L2 regularization of embeddings

        loss = _bpr_loss + _reg_loss

        return loss
