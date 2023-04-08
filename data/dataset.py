from .interaction import Interaction
import pandas as pd
import numpy as np
from collections import OrderedDict
import os
import warnings

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


class Dataset(object):
    def __init__(self, root_dir, dataset_name, base_dir='processed', sep='\t', columns='UI'):
        """Dataset

        Notes:
            The prefix name of data files is same as the root_dir, and the
            suffix/extension names are 'train', 'test', 'user2id', 'item2id'.
            Directory structure:
            root_dir
                ├──
                    base_dir
                        ├──
                            dataset_name
                                ├── dataset_name.train      // training data
                                ├── dataset_name.valid      // validation data, optional
                                ├── dataset_name.test       // test data
                                ├── dataset_name.user2id    // user to id, optional
                                ├── dataset_name.item2id    // item to id, optional

        Args:
            root_dir: The root directory of dataset.
            dataset_name: which dataset to use.
            base_dir: the type of dataset (raw, atom, processed), in here is processed.
            sep: The separator/delimiter of file columns.
            columns: The format of columns, must be one of 'UI', 'UIR', 'UIT' and 'UIRT'
        """

        self._data_dir = os.path.join(root_dir, base_dir, dataset_name)
        self._dataset_name = dataset_name
        assert columns in ['UI', 'UIR', 'UIT', 'UIRT']

        # metadata
        self._train_data = Interaction()
        self._valid_data = Interaction()
        self._test_data = Interaction()
        self.user2id = None
        self.item2id = None
        self.id2user = None
        self.id2item = None

        # statistic
        self.num_users = 0
        self.num_items = 0
        self.num_ratings = 0
        self._load_data(self._data_dir, sep, columns)

        self.train_csr_mat = self._train_data.to_csr_matrix()
        self.user_group, self.item_group = {}, {}
        self.item_frequency = self._count_item_frequency()
        self.user_frequency = self._count_user_frequency()

        # self._group_item_by_frequency()
        # self._group_user_by_frequency()
        print('Data loading finished')

    def _load_data(self, data_dir, sep, columns):
        if columns not in _column_dict:
            key_str = ", ".join(_column_dict.keys())
            raise ValueError("'columns' must be one of '%s'." % key_str)

        columns = _column_dict[columns]

        if not os.listdir(data_dir):
            raise ValueError("the processed dataset has not been generated, \
                             please turn to `scripts/` to process raw data and get processed data")

        files = os.listdir(data_dir)

        # get the detailed prefix, eg. get `ml-100k_ratio_u5_i5` from `ml-100k_ratio_u5_i5.train`
        file_prefix = os.path.basename(files[0]).split('.')[0]

        file_prefix = os.path.join(data_dir, file_prefix)

        # load data
        train_file = file_prefix + ".train"
        if os.path.isfile(train_file):
            _train_data = pd.read_csv(train_file, sep=sep, usecols=columns)
            # num_train_users = max(_train_data[_USER]) + 1
            # num_train_items = max(_train_data[_ITEM]) + 1
        else:
            raise FileNotFoundError("%s does not exist." % train_file)

        valid_file = file_prefix + ".valid"
        if os.path.isfile(valid_file):
            _valid_data = pd.read_csv(valid_file, sep=sep, usecols=columns)
            # num_valid_users = max(_valid_data[_USER]) + 1
            # num_valid_items = max(_valid_data[_ITEM]) + 1
        else:
            _valid_data = pd.DataFrame()
            # num_valid_users, num_valid_items = None, None
            warnings.warn("%s does not exist." % valid_file)

        test_file = file_prefix + ".test"
        if os.path.isfile(test_file):
            _test_data = pd.read_csv(test_file, sep=sep, usecols=columns)
            # num_test_users = max(_test_data[_USER]) + 1
            # num_test_items = max(_test_data[_ITEM]) + 1
        else:
            raise FileNotFoundError("%s does not exist." % test_file)

        user2id_file = file_prefix + ".user2id"
        if os.path.isfile(user2id_file):
            _user2id = pd.read_csv(user2id_file, sep=sep).to_numpy()
            self.user2id = OrderedDict(_user2id)
            self.id2user = OrderedDict([(idx, user) for user, idx in self.user2id.items()])
        else:
            warnings.warn("%s does not exist." % user2id_file)

        item2id_file = file_prefix + ".item2id"
        if os.path.isfile(item2id_file):
            _item2id = pd.read_csv(item2id_file, sep=sep).to_numpy()
            self.item2id = OrderedDict(_item2id)
            self.id2item = OrderedDict([(idx, item) for item, idx in self.item2id.items()])
        else:
            warnings.warn("%s does not exist." % item2id_file)

        # statistical information
        data_list = [data for data in [_train_data, _valid_data, _test_data] if not data.empty]
        all_data = pd.concat(data_list)
        self.num_users = max(all_data[_USER]) + 1
        self.num_items = max(all_data[_ITEM]) + 1
        self.num_ratings = len(all_data)
        self.num_train_ratings = len(_train_data)

        # convert to to the object of Interaction
        # self._train_data = Interaction(_train_data, num_users=num_train_users, num_items=num_train_items)
        # self._valid_data = Interaction(_valid_data, num_users=num_valid_users, num_items=num_valid_items)
        # self._test_data = Interaction(_test_data, num_users=num_test_users, num_items=num_test_items)
        self._train_data = Interaction(_train_data, num_users=self.num_users, num_items=self.num_items)
        self._valid_data = Interaction(_valid_data, num_users=self.num_users, num_items=self.num_items)
        self._test_data = Interaction(_test_data, num_users=self.num_users, num_items=self.num_items)

    def _group_item_by_frequency(self):
        i_degree = np.array(self.train_csr_mat.sum(0))[0].astype(np.int32)
        i_degree_sort = np.argsort(i_degree)    # in ascend order
        i_degree_cumsum = i_degree.copy()
        cum_sum = 0
        for x in i_degree_sort:
            cum_sum += i_degree_cumsum[x]
            i_degree_cumsum[x] = cum_sum

        split_idx = np.linspace(0, self.train_csr_mat.sum(), 11)
        self.item_group_idx = np.searchsorted(split_idx[1:-1], i_degree_cumsum)

        print('Item degree grouping...')
        for i in range(10):
            self.item_group[i] = i_degree[self.item_group_idx == i]
            # print('Size of group %d:' % i, self.item_group[i].size)
            # print('Sum degree of group %d:' % i, self.item_group[i].sum())
            # print('Min degree of group %d:' % i, self.item_group[i].min())
            # print('Max degree of group %d:' % i, self.item_group[i].max())

    def _group_user_by_frequency(self):
        u_degree = np.array(self.train_csr_mat.sum(1))[:, 0].astype(np.int32)
        u_degree_sort = np.argsort(u_degree)    # in ascend order
        u_degree_cumsum = u_degree.copy()
        cum_sum = 0
        for x in u_degree_sort:
            cum_sum += u_degree_cumsum[x]
            u_degree_cumsum[x] = cum_sum

        split_idx = np.linspace(0, self.train_csr_mat.sum(), 11)
        self.user_group_idx = np.searchsorted(split_idx[1:-1], u_degree_cumsum)

        print('User degree grouping...')
        for i in range(10):
            self.user_group[i] = u_degree[self.user_group_idx == i]

    def _count_item_frequency(self):
        colsum = np.array(self.train_csr_mat.sum(0))
        return np.squeeze(colsum)

    def _count_user_frequency(self):
        rowsum = np.array(self.train_csr_mat.sum(1))
        return np.squeeze(rowsum)

    @property
    def train_data(self):
        """
        Returns:
            Interaction: DataFrame of train data
        """
        return self._train_data

    @property
    def valid_data(self):
        """
        Returns:
            Interaction: DataFrame of valid data
        """
        return self._valid_data

    @property
    def test_data(self):
        """
        Returns:
            Interaction: DataFrame of test data
        """
        return self._test_data

    def __str__(self):
        """The statistic of dataset.

        Returns:
            str: The summary of statistic
        """
        if 0 in {self.num_users, self.num_items, self.num_ratings}:
            return "statistical information is unavailable now"
        else:
            num_users, num_items = self.num_users, self.num_items
            num_ratings = self.num_ratings
            density = 1.0 * num_ratings / (num_users * num_items)
            sparsity = 1 - density

            statistic = ["Dataset statistics:",
                         "Name: %s" % self._dataset_name,
                         "The number of users: %d" % num_users,
                         "The number of items: %d" % num_items,
                         "The number of ratings: %d" % num_ratings,
                         "Average actions of users: %.2f" % (1.0 * num_ratings / num_users),
                         "Average actions of items: %.2f" % (1.0 * num_ratings / num_items),
                         "The density of the dataset: %.5f" % density,
                         "The sparsity of the dataset: %.5f" % sparsity,
                         "The number of training: %d" % len(self._train_data),
                         "The number of validation: %d" % len(self._valid_data),
                         "The number of testing: %d" % len(self._test_data)
                         ]
            statistic = "\n".join(statistic)
            return statistic

    def __repr__(self):
        return self.__str__()
