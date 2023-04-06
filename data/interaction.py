import scipy.sparse as sp
import os
import warnings
import pandas as pd
import numpy as np
from utils import typeassert, pad_sequences
from collections import OrderedDict
from copy import deepcopy
import torch
from torch_geometric.utils import get_laplacian, to_undirected


# meta info
_USER = "user_id"
_ITEM = "item_id"
_RATING = "rating"
_TIME = "timestamp"
_column_dict = {"UI": [_USER, _ITEM],
                "UIR": [_USER, _ITEM, _RATING],
                "UIT": [_USER, _ITEM, _TIME],
                "UIRT": [_USER, _ITEM, _RATING, _TIME]
                }


class Interaction(object):
    @typeassert(data=(pd.DataFrame, None), num_users=(int, None), num_items=(int, None))
    def __init__(self, data=None, num_users=None, num_items=None):
        if data is None or data.empty:
            self._data = pd.DataFrame()
            self.num_users = 0
            self.num_items = 0
            self.num_ratings = 0
        else:
            self._data = data
            self.num_users = num_users if num_users is not None else max(data[_USER]) + 1
            self.num_items = num_items if num_items is not None else max(data[_ITEM]) + 1
            self.num_ratings = len(data)

        self._buffer = dict()

    def to_user_item_pairs(self):
        if self._data.empty:
            warnings.warn("self._data is empty.")
            return None
        # users_np = self._data[_USER].to_numpy(copy=True, dtype=np.int32)
        # items_np = self._data[_ITEM].to_numpy(copy=True, dtype=np.int32)
        ui_pairs = self._data[[_USER, _ITEM]].to_numpy(copy=True, dtype=np.int32)
        return ui_pairs

    def to_coo_matrix(self):
        if self._data.empty:
            warnings.warn("self._data is empty.")
            return None
        users, items = self._data[_USER].to_numpy(), self._data[_ITEM].to_numpy()
        ratings = np.ones(len(users), dtype=np.float32)
        coo_mat = sp.coo_matrix((ratings, (users, items)), shape=(self.num_users, self.num_items))
        return coo_mat

    def to_csr_matrix(self):
        if self._data.empty:
            warnings.warn("self._data is empty.")
            return None
        return self.to_coo_matrix().tocsr()

    def to_dok_matrix(self):
        if self._data.empty:
            warnings.warn("self._data is empty.")
            return None
        return self.to_coo_matrix().todok()

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """
        users_items = torch.LongTensor(self.to_user_item_pairs().T)
        row, col = users_items
        col = col + self.num_users
        num_nodes = self.num_users + self.num_items
        edge_index = torch.stack([row, col], dim=0)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)

        return edge_index, edge_weight

    def to_user_dict(self, by_time=False):
        """
            dict: {user: pos_items}
        """
        if self._data.empty:
            warnings.warn("self._data is empty.")
            return None

        if by_time and _TIME not in self._data:
            raise ValueError("This dataset do not have timestamp.")

        # read from buffer
        if by_time is True and "user_dict_byt" in self._buffer:
            return deepcopy(self._buffer["user_dict_byt"])
        if by_time is False and "user_dict" in self._buffer:
            return deepcopy(self._buffer["user_dict"])

        user_dict = OrderedDict()
        user_grouped = self._data.groupby(_USER)
        for user, user_data in user_grouped:
            if by_time:
                user_data = user_data.sort_values(by=[_TIME])
            user_dict[user] = user_data[_ITEM].to_numpy(dtype=np.int32)

        # write to buffer
        if by_time is True:
            self._buffer["user_dict_byt"] = deepcopy(user_dict)
        else:
            self._buffer["user_dict"] = deepcopy(user_dict)
        return user_dict

    def to_item_dict(self):
        """
            dict: {item: corresponding users}
        """
        if self._data.empty:
            warnings.warn("self._data is empty.")
            return None

        # read from buffer
        if "item_dict" in self._buffer:
            return deepcopy(self._buffer["item_dict"])

        item_dict = OrderedDict()
        item_grouped = self._data.groupby(_ITEM)
        for item, item_data in item_grouped:
            item_dict[item] = item_data[_USER].to_numpy(dtype=np.int32)

        # write to buffer
        self._buffer["item_dict"] = deepcopy(item_dict)
        return item_dict

    def to_truncated_seq_dict(self, max_len, pad_value=0, padding='post', truncating='post'):
        """Get the truncated item sequences of each user.

        Args:
            max_len (int or None): Maximum length of all sequences.
            pad_value: Padding value. Defaults to `0.`.
            padding (str): `"pre"` or `"post"`: pad either before or after each
                sequence. Defaults to `post`.
            truncating (str): `"pre"` or `"post"`: remove values from sequences
                larger than `max_len`, either at the beginning or at the end of
                the sequences. Defaults to `post`.

        Returns:
            OrderedDict: key is user and value is truncated item sequences.

        """
        user_seq_dict = self.to_user_dict(by_time=True)
        if max_len is None:
            max_len = max([len(seqs) for seqs in user_seq_dict.values()])
        item_seq_list = [item_seq[-max_len:] for item_seq in user_seq_dict.values()]
        item_seq_arr = pad_sequences(item_seq_list, value=pad_value, max_len=max_len,
                                     padding=padding, truncating=truncating, dtype=np.int32)

        seq_dict = OrderedDict([(user, item_seq) for user, item_seq in
                                zip(user_seq_dict.keys(), item_seq_arr)])
        return seq_dict

    def _clean_buffer(self):
        self._buffer.clear()

    def update(self, other):
        """Update this object with the union of itself and other.
        Args:
            other (Interaction): An object of Interaction

        """
        if not isinstance(other, Interaction):
            raise TypeError("'other' must be a object of 'Interaction'")
        other_data = other._data
        if other_data.empty:
            warnings.warn("'other' is empty and update nothing.")
        elif self._data.empty:
            self._data = other_data.copy()
            self.num_users = other.num_users
            self.num_items = other.num_items
            self.num_ratings = other.num_items
            self._clean_buffer()
        elif self._data is other_data:
            warnings.warn("'other' is equal with self and update nothing.")
        else:
            self._data = pd.concat([self._data, other_data])
            self._data.drop_duplicates(inplace=True)
            self.num_users = max(self._data[_USER]) + 1
            self.num_items = max(self._data[_ITEM]) + 1
            self.num_ratings = len(self._data)
            self._clean_buffer()

    def union(self, other):
        """Return the union of self and other as a new Interaction.

        Args:
            other (Interaction): An object of Interaction.

        Returns:
            Interaction: The union of self and other.

        """
        if not isinstance(other, Interaction):
            raise TypeError("'other' must be a object of 'Interaction'")
        result = Interaction()
        result.update(self)
        result.update(other)
        return result

    def __add__(self, other):
        return self.union(other)

    def __bool__(self):
        return self.__len__() > 0

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        statistic = ["data statistics:",
                     "The number of users: %d" % self.num_users,
                     "The number of items: %d" % self.num_items,
                     "The number of ratings: %d" % self.num_ratings,
                     ]
        statistic = "\n".join(statistic)
        return statistic
