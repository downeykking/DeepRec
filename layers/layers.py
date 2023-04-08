import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class LightGCNConv(MessagePassing):
    def __init__(self, dim):
        super(LightGCNConv, self).__init__(aggr='add')
        self.dim = dim

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)


class BiGNNConv(MessagePassing):
    r"""Propagate a layer of Bi-interaction GNN, used for NGCF.
    .. math::
        output = (L+I)E dot W_1 + (LE \otimes E) dot W_2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lin1 = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.lin2 = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, edge_index, edge_weight):
        x_prop = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        x_trans = self.lin1(x_prop + x)
        x_inter = self.lin2(torch.mul(x_prop, x))
        return x_trans + x_inter

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
