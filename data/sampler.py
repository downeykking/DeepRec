from .dataloader import DataIterator
from utils import randint_choice
from utils import typeassert
from utils import pad_sequences
from collections import Iterable
from collections import OrderedDict, defaultdict
from .interaction import Interaction
import numpy as np
import itertools


class Sampler(object):
    """Base class for all sampler to sample negative items.
    """

    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


@typeassert(user_pos_dict=dict)
def _generate_positive_items(user_pos_dict):
    if not user_pos_dict:
        raise ValueError("'user_pos_dict' cannot be empty.")

    users_list, pos_items_list = [], []
    user_n_pos = OrderedDict()

    for user, pos_items in user_pos_dict.items():
        users_list.append(np.full_like(pos_items, user))
        pos_items_list.append(pos_items)
        user_n_pos[user] = len(pos_items)
    users_arr = np.concatenate(users_list)
    items_arr = np.concatenate(pos_items_list)
    return user_n_pos, users_arr, items_arr


@typeassert(user_n_pos=OrderedDict, num_neg=int, num_items=int, user_pos_dict=dict)
def _sampling_negative_items(user_n_pos, num_neg, num_items, user_pos_dict):
    if num_neg <= 0:
        raise ValueError("'neg_num' must be a positive integer.")

    neg_items_list = []
    for user, n_pos in user_n_pos.items():
        neg_items = randint_choice(num_items, size=n_pos * num_neg, exclusion=user_pos_dict[user])
        if num_neg == 1:
            neg_items = neg_items if isinstance(neg_items, Iterable) else [neg_items]
            neg_items_list.append(neg_items)
        else:
            neg_items = np.reshape(neg_items, newshape=[n_pos, num_neg])
            neg_items_list.append(neg_items)

    return np.concatenate(neg_items_list)


@typeassert(user_pos_dict=dict, len_seqs=int, len_next=int, pad=(int, None), aug=bool, return_seq_len=bool)
def _generative_time_order_positive_items(user_pos_dict, len_seqs=1, len_next=1, pad=None, aug=False, return_seq_len=False):
    if not user_pos_dict:
        raise ValueError("'user_pos_dict' cannot be empty.")

    users_list, item_seqs_list, item_seqs_len_list, next_items_list = [], [], [], []
    user_n_pos = OrderedDict()

    tot_len = len_seqs + len_next
    for user, seq_items in user_pos_dict.items():
        if not isinstance(seq_items, np.ndarray):
            seq_items = np.array(seq_items, dtype=np.int32)
        user_n_pos[user] = 0
        # add augment like https://recbole.io/docs/get_started/started/sequential.html
        if aug and pad is not None:
            for next_item_idx in range(1, min(len(seq_items), len_seqs)):
                tmp_seqs = seq_items[:next_item_idx]
                item_seqs_len_list.append(len(tmp_seqs))
                tmp_seqs = pad_sequences([tmp_seqs], value=pad, max_len=len_seqs,
                                         padding='post', truncating='post', dtype=np.int32)
                item_seqs_list.append(tmp_seqs.squeeze().reshape([1, len_seqs]))
                next_items_list.append(seq_items[next_item_idx].reshape([1, len_next]))
                users_list.append(user)
                user_n_pos[user] += 1
        if len(seq_items) >= tot_len:
            for idx in range(len(seq_items) - tot_len + 1):
                tmp_seqs = seq_items[idx:idx + tot_len]
                item_seqs_list.append(tmp_seqs[:len_seqs].reshape([1, len_seqs]))
                item_seqs_len_list.append(len_seqs)
                next_items_list.append(tmp_seqs[len_seqs:].reshape([1, len_next]))
                users_list.append(user)
                user_n_pos[user] += 1
        elif len(seq_items) > len_next and not aug and pad is not None:  # padding
            next_items_list.append(seq_items[-len_next:].reshape([1, len_next]))
            tmp_seqs = pad_sequences([seq_items[:-len_next]], value=pad, max_len=len_seqs,
                                     padding='post', truncating='post', dtype=np.int32)
            item_seqs_list.append(tmp_seqs.squeeze().reshape([1, len_seqs]))
            item_seqs_len_list.append(len(seq_items) - len_next)
            users_list.append(user)
            user_n_pos[user] = 1
        else:  # discard
            continue
    users_arr = np.int32(users_list)
    item_seqs_arr = np.concatenate(item_seqs_list).squeeze()
    item_seqs_len_arr = np.int32(item_seqs_len_list)
    next_items_arr = np.concatenate(next_items_list).squeeze()
    if return_seq_len:
        return user_n_pos, users_arr, item_seqs_arr, item_seqs_len_arr, next_items_arr
    else:
        return user_n_pos, users_arr, item_seqs_arr, next_items_arr


@typeassert(user_pos_dict=OrderedDict, num_samples=int, num_neg=int, num_item=int)
def _pairwise_sampling_v2(user_pos_dict, num_samples, num_neg, num_item):
    if not isinstance(user_pos_dict, dict):
        raise TypeError("'user_pos_dict' must be a dict.")

    if not user_pos_dict:
        raise ValueError("'user_pos_dict' cannot be empty.")

    user_arr = np.array(list(user_pos_dict.keys()), dtype=np.int32)
    user_idx = randint_choice(len(user_arr), size=num_samples, replace=True)
    users_list = user_arr[user_idx]

    pos_items_list, neg_items_list = [0] * len(users_list), [0] * len(users_list)

    for idx, user in enumerate(users_list):
        pos_all_items = user_pos_dict[user]
        pos_idx = randint_choice(len(pos_all_items), size=1)
        pos_items = pos_all_items[pos_idx]
        pos_items_list[idx] = pos_items

        neg_items = randint_choice(num_item, size=num_neg, replace=True, exclusion=pos_all_items)
        neg_items_list[idx] = neg_items

    return users_list, pos_items_list, neg_items_list


class PointwiseSampler(Sampler):
    """Sampling negative items and construct pointwise training instances.

    The training instances consist of `batch_user`, `batch_item` and
    `batch_label`, which are lists of users, items and labels. All lengths of
    them are `batch_size`.
    Positive and negative items are labeled as `1` and  `0`, respectively.
    """
    @typeassert(dataset=Interaction, num_neg=int, batch_size=int, shuffle=bool, drop_last=bool)
    def __init__(self, dataset, num_neg=1, batch_size=1024, shuffle=True, drop_last=False):
        """Initializes a new `PointwiseSampler` instance.

        Args:
            dataset (data.Interaction): An instance of `Interaction`.
            num_neg (int): How many negative items for each positive item.
                Defaults to `1`.
            batch_size (int): How many samples per batch to load.
                Defaults to `1`.
            shuffle (bool): Whether reshuffling the samples at every epoch.
                Defaults to `False`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        super(Sampler, self).__init__()
        if num_neg <= 0:
            raise ValueError("'num_neg' must be a positive integer.")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_neg = num_neg
        self.num_items = dataset.num_items
        self.user_pos_dict = dataset.to_user_dict()
        self.user_n_pos, users_arr, self.pos_items = \
            _generate_positive_items(self.user_pos_dict)

        self.all_users = np.tile(users_arr, self.num_neg + 1)
        len_pos = len(self.pos_items)
        pos_labels = np.full(len_pos, 1.0, dtype=np.float32)
        neg_labels = np.full(len_pos * self.num_neg, 0.0, dtype=np.float32)
        self.all_labels = np.concatenate([pos_labels, neg_labels])

    def __iter__(self):
        neg_items = _sampling_negative_items(self.user_n_pos, self.num_neg,
                                             self.num_items, self.user_pos_dict)

        neg_items = neg_items.transpose().reshape([-1])
        all_items = np.concatenate([self.pos_items, neg_items])

        data_iter = DataIterator(self.all_users, all_items, self.all_labels,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle, drop_last=self.drop_last)

        for bat_users, bat_items, bat_labels in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_items), np.asarray(bat_labels)

    def __len__(self):
        n_sample = len(self.all_users)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size


class PointwiseSamplerV2(Sampler):
    """construct pointwise training instances without negative samples. Uniformly sample from the observed instances.

    The training instances consist of `batch_user` and `batch_item`, which are lists of users, items in the training set. All lengths of them are `batch_size`.
    """
    @typeassert(dataset=Interaction, batch_size=int, shuffle=bool, drop_last=bool)
    def __init__(self, dataset, batch_size=1024, shuffle=True, drop_last=False):
        """Initializes a new `PointwiseSampler` instance.

        Args:
            dataset (data.Dataset): An instance of `Dataset`.
            batch_size (int): How many samples per batch to load.
                Defaults to `1024`.
            shuffle (bool): Whether reshuffling the samples at every epoch.
                Defaults to `True`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        super(Sampler, self).__init__()

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_items = dataset.num_items
        self.user_pos_dict = dataset.to_user_dict()
        self.num_trainings = sum([len(item) for u, item in self.user_pos_dict.items()])
        self.user_pos_len, self.users_list, self.pos_items_list = \
            _generate_positive_items(self.user_pos_dict)

    def __iter__(self):
        data_iter = DataIterator(self.users_list, self.pos_items_list,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle, drop_last=self.drop_last)

        for bat_users, bat_items in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_items)

    def __len__(self):
        n_sample = len(self.users_list)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size


class PairwiseSampler(Sampler):
    """Sampling negative items and construct pairwise training instances.

    The training instances consist of `batch_user`, `batch_pos_item` and
    `batch_neg_items`, where `batch_user` and `batch_pos_item` are lists
    of users and positive items with length `batch_size`, and `neg_items`
    does not interact with `user`.

    If `neg_num == 1`, `batch_neg_items` is also a list of negative items
    with length `batch_size`;  If `neg_num > 1`, `batch_neg_items` is an
    array like list with shape `(batch_size, neg_num)`.
    """

    @typeassert(dataset=Interaction, num_neg=int, batch_size=int, shuffle=bool, drop_last=bool)
    def __init__(self, dataset, num_neg=1, batch_size=1024, shuffle=True, drop_last=False):
        """Initializes a new `PairwiseSampler` instance.

        Args:
            dataset (data.Interaction): An instance of `data.Interaction`.
            num_neg (int): How many negative items for each positive item.
                Defaults to `1`.
            batch_size (int): How many samples per batch to load.
                Defaults to `1`.
            shuffle (bool): Whether reshuffling the samples at every epoch.
                Defaults to `False`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        super(PairwiseSampler, self).__init__()
        if num_neg <= 0:
            raise ValueError("'num_neg' must be a positive integer.")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_neg = num_neg
        self.num_items = dataset.num_items
        self.user_pos_dict = dataset.to_user_dict()

        self.user_n_pos, self.all_users, self.pos_items = \
            _generate_positive_items(self.user_pos_dict)

    def __iter__(self):
        neg_items = _sampling_negative_items(self.user_n_pos, self.num_neg,
                                             self.num_items, self.user_pos_dict)

        data_iter = DataIterator(self.all_users, self.pos_items, neg_items,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle, drop_last=self.drop_last)
        for bat_users, bat_pos_items, bat_neg_items in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_pos_items), np.asarray(bat_neg_items)

    def __len__(self):
        n_sample = len(self.all_users)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size


class PairwiseSamplerV2(Sampler):
    """Sampling negative items and construct pairwise training instances.

    The training instances consist of `batch_user`, `batch_pos_item` and
    `batch_neg_items`, where `batch_user` and `batch_pos_item` are lists
    of users and positive items with length `batch_size`, and `neg_items`
    does not interact with `user`.

    If `neg_num == 1`, `batch_neg_items` is also a list of negative items
    with length `batch_size`;  If `neg_num > 1`, `batch_neg_items` is an
    array like list with shape `(batch_size, neg_num)`.
    """
    @typeassert(dataset=Interaction, num_neg=int, batch_size=int, shuffle=bool, drop_last=bool)
    def __init__(self, dataset, num_neg=1, batch_size=1024, shuffle=True, drop_last=False):
        """Initializes a new `PairwiseSampler` instance.

        Args:
            dataset (data.Dataset): An instance of `Dataset`.
            num_neg (int): How many negative items for each positive item.
                Defaults to `1`.
            batch_size (int): How many samples per batch to load.
                Defaults to `1024`.
            shuffle (bool): Whether reshuffling the samples at every epoch.
                Defaults to `True`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        super(PairwiseSamplerV2, self).__init__()
        if num_neg <= 0:
            raise ValueError("'num_neg' must be a positive integer.")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_neg = num_neg
        self.num_items = dataset.num_items
        self.user_pos_dict = dataset.to_user_dict()
        # see https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/utils.py#L73 and https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/dataloader.py#L252
        # self.num_trainings = sum([len(item) for _, item in self.user_pos_dict.items()])
        self.num_trainings = dataset.num_ratings
        # self.user_pos_dict = {u: np.array(item) for u, item in user_pos_dict.items()}

    def __iter__(self):

        users_list, pos_items_list, neg_items_list = \
            _pairwise_sampling_v2(self.user_pos_dict, self.num_trainings, self.num_neg, self.num_items)

        data_iter = DataIterator(users_list, pos_items_list, neg_items_list,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle, drop_last=self.drop_last)
        for bat_users, bat_pos_items, bat_neg_items in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_pos_items), np.asarray(bat_neg_items)
            # yield torch.LongTensor(bat_users), torch.LongTensor(bat_pos_items), torch.LongTensor(bat_neg_items)

    def __len__(self):
        n_sample = self.num_trainings
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size


# TODO add augment
class TimeOrderPointwiseSampler(Sampler):
    """Sampling negative items and construct time ordered pointwise instances.

    The training instances consist of `batch_user`, `batch_recent_items`,
    `batch_item` and `batch_label`. For each instance, positive `label`
    indicates that `user` interacts with `item` immediately following
    `recent_items`; and negative `label` indicates that `item` does not
    interact with `user`.

    If `len_seqs == 1`, `batch_recent_items` is a list of items with length
    `batch_size`; If `high_order > 1`, `batch_recent_items` is an array like
    list with shape `(batch_size, high_order)`.
    Positive and negative items are labeled as `1` and  `0`, respectively.
    """

    @typeassert(dataset=Interaction, len_seqs=int, len_next=int, pad=(int, None),
                aug=False, num_neg=int, batch_size=int, shuffle=bool, drop_last=bool)
    def __init__(self, dataset, len_seqs=1, len_next=1, pad=None, aug=False, num_neg=1,
                 batch_size=1024, shuffle=True, drop_last=False):
        """
        Args:
            dataset (data.Interaction): An instance of `data.Interaction`.
            len_seqs (int): The length of item sequence. Default to 1.
            len_next (int): The length/number of next items. Default to 1.
            pad (int, None): The pad value of item sequence. None means
                discarding the item sequences whose length less than
                'len_seqs'. Otherwise, the length of item sequence will
                be padded to 'len_seqs' with the specified pad value.
                Default to None.
            aug (bool): Whether to augment data like
                <https://recbole.io/docs/get_started/started/sequential.html>
            num_neg (int): How many negative items for each item sequence.
                Default to `1`.
            batch_size (int): How many samples per batch to load.
                Defaults to `1`.
            shuffle (bool): Whether reshuffling the samples at every epoch.
                Defaults to `False`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        super(TimeOrderPointwiseSampler, self).__init__()
        if len_seqs <= 0:
            raise ValueError("'len_seqs' must be a positive integer.")
        if len_next <= 0:
            raise ValueError("'len_next' must be a positive integer.")
        if num_neg <= 0:
            raise ValueError("'num_neg' must be a positive integer.")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_neg = num_neg
        self.num_items = dataset.num_items
        self.len_next = len_next
        self.user_pos_dict = dataset.to_user_dict(by_time=True)

        self.user_n_pos, users_arr, item_seqs_arr, self.pos_next_items = \
            _generative_time_order_positive_items(self.user_pos_dict, len_seqs=len_seqs,
                                                  len_next=len_next, pad=pad, aug=aug)

        self.all_users = np.tile(users_arr, self.num_neg + 1)
        self.all_item_seqs = np.tile(item_seqs_arr, [self.num_neg + 1, 1])

        len_pos = len(self.pos_next_items)
        pos_labels = np.full([len_pos, len_next], 1.0, dtype=np.float32)
        neg_labels = np.full([len_pos * self.num_neg, len_next], 0.0, dtype=np.float32)
        self.all_labels = np.concatenate([pos_labels, neg_labels]).squeeze()

    def __iter__(self):
        neg_next_items = _sampling_negative_items(self.user_n_pos, self.num_neg * self.len_next,
                                                  self.num_items, self.user_pos_dict)
        neg_item_split = np.hsplit(neg_next_items, self.num_neg)
        neg_next_items = np.vstack(neg_item_split).squeeze()
        all_next_items = np.concatenate([self.pos_next_items, neg_next_items])

        data_iter = DataIterator(self.all_users, self.all_item_seqs, all_next_items, self.all_labels,
                                 batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)

        for bat_users, bat_item_seqs, bat_next_items, bat_labels in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_item_seqs), np.asarray(bat_next_items), np.asarray(bat_labels)

    def __len__(self):
        n_sample = len(self.all_users)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size


class TimeOrderPairwiseSampler(Sampler):
    """Sampling negative items and construct time ordered pairwise instances.

    The training instances consist of `batch_user`, `batch_recent_items`,
    `batch_next_item` and `batch_neg_items`. For each instance, `user`
    interacts with `next_item` immediately following `recent_items`, and
    `neg_items` does not interact with `user`.

    If `high_order == 1`, `batch_recent_items` is a list of items with length
    `batch_size`; If `high_order > 1`, `batch_recent_items` is an array like
    list with shape `(batch_size, high_order)`.

    If `neg_num == 1`, `batch_neg_items` is a list of negative items with length
    `batch_size`; If `neg_num > 1`, `batch_neg_items` is an array like list with
    shape `(batch_size, neg_num)`.
    """

    @typeassert(dataset=Interaction, len_seqs=int, len_next=int, pad=(int, None),
                aug=bool, num_neg=int, batch_size=int, shuffle=bool, drop_last=bool)
    def __init__(self, dataset, len_seqs=1, len_next=1, pad=None, aug=False, return_seq_len=False,
                 num_neg=1, batch_size=1024, shuffle=True, drop_last=False):
        """Initializes a new `TimeOrderPairwiseSampler` instance.

        Args:
            dataset (data.Interaction): An instance of `data.Interaction`.
            len_seqs (int): The length of item sequence. Default to 1.
            len_next (int): The length/number of next items. Default to 1.
            pad (int, None): The pad value of item sequence. None means
                discarding the item sequences whose length less than
                'len_seqs'. Otherwise, the length of item sequence will
                be padded to 'len_seqs' with the specified pad value.
                Default to None.
            aug (bool): Whether to augment data like:
                <https://recbole.io/docs/get_started/started/sequential.html>
            return_seq_len (bool): used for garther index. see:
                <https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/gru4rec.py#L78>
            num_neg (int): How many negative items for each item sequence.
                Default to `1`.
            batch_size (int): How many samples per batch to load.
                Defaults to `1`.
            shuffle (bool): Whether reshuffling the samples at every epoch.
                Defaults to `False`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        super(TimeOrderPairwiseSampler, self).__init__()
        if len_seqs <= 0:
            raise ValueError("'len_seqs' must be a positive integer.")
        if len_next <= 0:
            raise ValueError("'len_next' must be a positive integer.")
        if num_neg <= 0:
            raise ValueError("'num_neg' must be a positive integer.")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_neg = num_neg
        self.num_items = dataset.num_items
        self.len_next = len_next
        self.user_pos_dict = dataset.to_user_dict(by_time=True)
        self.return_seq_len = return_seq_len

        self.user_n_pos, self.all_users, self.all_item_seqs, \
            self.all_item_seqs_len, self.pos_next_items = \
            _generative_time_order_positive_items(self.user_pos_dict, len_seqs=len_seqs,
                                                  len_next=len_next, pad=pad, aug=aug,
                                                  return_seq_len=True)

    def __iter__(self):
        neg_next_items = _sampling_negative_items(self.user_n_pos, self.num_neg,
                                                  self.num_items, self.user_pos_dict)

        data_iter = DataIterator(self.all_users, self.all_item_seqs,
                                 self.all_item_seqs_len, self.pos_next_items, neg_next_items,
                                 batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)

        for bat_users, bat_item_seqs, bat_item_seqs_len, bat_pos_items, bat_neg_items in data_iter:
            if self.return_seq_len:
                yield np.asarray(bat_users), np.asarray(bat_item_seqs), \
                    np.asarray(bat_item_seqs_len), np.asarray(bat_pos_items), np.asarray(bat_neg_items)
            else:
                yield np.asarray(bat_users), np.asarray(bat_item_seqs), np.asarray(bat_pos_items), np.asarray(bat_neg_items)

    def __len__(self):
        n_sample = len(self.all_users)
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size


class FISMPointwiseSampler(Sampler):
    @typeassert(dataset=Interaction, pad=int, batch_size=int, shuffle=bool, drop_last=bool)
    def __init__(self, dataset, pad, batch_size=1024, shuffle=True, drop_last=False):
        super(FISMPointwiseSampler, self).__init__()
        self.pad_value = pad
        self.user_pos_dict = dataset.to_user_dict()
        self.point_iter = PointwiseSampler(dataset, batch_size=batch_size,
                                           shuffle=shuffle, drop_last=drop_last)

    def __iter__(self):
        for bat_users, bat_items, bat_labels in self.point_iter:
            bat_his_items = []
            bat_his_len = []
            for user, pos_item in zip(bat_users, bat_items):
                his_items = self.user_pos_dict[user]
                his_len = len(his_items) - 1 if len(his_items) - 1 > 0 else 1
                bat_his_len.append(his_len)
                bat_his_items.append(np.where(his_items == pos_item, self.pad_value, his_items))
            bat_his_items = pad_sequences(bat_his_items, value=self.pad_value, max_len=None,
                                          padding='post', truncating='post', dtype=np.int32)
            yield np.asarray(bat_users), np.asarray(bat_his_items), np.asarray(bat_his_len), np.asarray(bat_items), np.asarray(bat_labels)

    def __len__(self):
        return len(self.point_iter)


class FISMPairwiseSampler(Sampler):
    @typeassert(dataset=Interaction, pad=int, batch_size=int, shuffle=bool, drop_last=bool)
    def __init__(self, dataset, pad, batch_size=1024, shuffle=True, drop_last=False):
        super(FISMPairwiseSampler, self).__init__()
        self.pad_value = pad
        self.user_pos_dict = dataset.to_user_dict()
        self.pair_iter = PairwiseSampler(dataset, batch_size=batch_size,
                                         shuffle=shuffle, drop_last=drop_last)

    def __iter__(self):
        for bat_users, bat_pos_items, bat_neg_items in self.pair_iter:
            bat_his_items = []
            bat_his_len = []
            for user, pos_item in zip(bat_users, bat_pos_items):
                his_items = self.user_pos_dict[user]
                his_len = len(his_items) - 1 if len(his_items) - 1 > 0 else 1
                bat_his_len.append(his_len)
                flag = his_items == pos_item
                bat_his_items.append(np.where(flag, self.pad_value, his_items))
            bat_his_items = pad_sequences(bat_his_items, value=self.pad_value, max_len=None,
                                          padding='post', truncating='post', dtype=np.int32)
            yield np.asarray(bat_users), np.asarray(bat_his_items), np.asarray(bat_his_len), np.asarray(bat_pos_items), np.asarray(bat_neg_items)

    def __len__(self):
        return len(self.pair_iter)
