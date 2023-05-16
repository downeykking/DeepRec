import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Interaction


class BasicModel(nn.Module):
    def __init__(self, config, dataset: Interaction):
        super(BasicModel, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.device = config['device']

    def reset_parameters(self):
        r"""Initialnize embedding parameters."""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        r"""Get the embedding of model."""
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        r"""Predict the scores between users and items. the size of users and items are the same.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size,]
        """
        raise NotImplementedError

    def rating(self, users):
        """
        Given users, calculate the scores between users and all candidate items.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users, m_all_candidate_items]
        """
        raise NotImplementedError

    def calculate_loss(self, user, pos_items, neg_items):
        r"""Calculate the training loss for a batch data.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError
