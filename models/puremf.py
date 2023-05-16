import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BasicModel
from losses import BPRLoss
from utils import xavier_normal_initialization


class PureMF(BasicModel):
    def __init__(self, config, dataset):
        super(PureMF, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']

        self.user_embeddings = nn.Embedding(self.num_users, self.embedding_size)
        self.item_embeddings = nn.Embedding(self.num_items, self.embedding_size)

        self.loss = BPRLoss()

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(xavier_normal_initialization)

    def forward(self):
        pass

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

    def calculate_loss(self, users, pos_items, neg_items):
        users_emb = self.user_embeddings(users)
        pos_emb = self.item_embeddings(pos_items)
        neg_emb = self.item_embeddings(neg_items)

        loss = self.loss(users_emb, pos_emb, neg_emb)
        return loss
