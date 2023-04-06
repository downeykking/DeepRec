from .dataset import Interaction
import torch
from torch_geometric.utils import get_laplacian, to_undirected


class GraphInteraction(Interaction):
    def __init__(self):
        super().__init__()

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """
        users_items = torch.from_numpy(self._data.to_user_item_pairs().T)
        row, col = users_items
        col = col + self.num_users
        num_nodes = self.num_users + self.num_items
        edge_index = torch.stack([row, col], dim=0)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)

        return edge_index, edge_weight
