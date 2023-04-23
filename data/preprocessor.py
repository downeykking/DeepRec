import os
import os.path as osp
import math
import pandas as pd
import numpy as np
from utils import typeassert, setup_logger
from collections import OrderedDict
import warnings


class Preprocessor(object):
    _USER = "user_id"
    _ITEM = "item_id"
    _RATING = "rating"
    _TIME = "timestamp"

    def __init__(self):
        """
        A class for data preprocessing
        """

        self._column_dict = {"UI": [self._USER, self._ITEM],
                             "UIR": [self._USER, self._ITEM, self._RATING],
                             "UIT": [self._USER, self._ITEM, self._TIME],
                             "UIRT": [self._USER, self._ITEM, self._RATING, self._TIME]}

        self._column_name = None
        self._config = OrderedDict()
        self.all_data = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.user2id = None
        self.item2id = None
        self._dir_path = None
        self._data_name = ""
        self._split_manner = ""
        self._user_min = 0
        self._item_min = 0

    @typeassert(dataset_name=str, sep=str)
    def load_data(self, root_dir, dataset_name, file_name=None, base_dir='atom', sep='\t', columns=None):
        """Load data

        Args:
            atom_root_dir (str): The root dir of atom.
            dataset_name (str): the dataset name.
            file_name (str): usually 'dataset_name.inter'
            sep (str): The separator/delimiter of columns.
            columns (str): One of 'UI', 'UIR', 'UIT' and 'UIRT'.

        """
        _data_dir = os.path.join(root_dir, base_dir, dataset_name)
        _file_name = dataset_name + ".inter" if file_name is None else file_name
        _file_path = osp.join(_data_dir, _file_name)

        if not osp.isfile(_file_path):
            raise FileNotFoundError("There is no file named '%s'." % _file_name)
        if columns not in self._column_dict:
            key_str = ", ".join(self._column_dict.keys())
            raise ValueError("'columns' must be one of '%s'." % key_str)
        self._config["columns"] = columns

        self._column_name = self._column_dict[columns]

        print("loading data...")
        self._config["filename"] = _file_name
        self._config["sep"] = repr(sep)

        self.all_data = pd.read_csv(_file_path, sep=sep, usecols=self._column_name, engine='python')
        if self.all_data.empty:
            raise ValueError('dataframe is empty, load data failed')
        self.all_data.dropna(inplace=True)
        self._dataset_name = dataset_name

    def drop_duplicates(self, keep="first"):
        """Drop duplicate user-item interactions.

        Args:
            keep (str): 'first' or 'last', default 'first'.
                Drop duplicates except for the first or last occurrence.

        Returns:
            An object of pd.DataFrame without duplicates.

        Raises:
            ValueError: If 'keep' is not 'first' or 'last'.
        """

        if keep not in {'first', 'last'}:
            raise ValueError("'keep' must be 'first' or 'last', but '%s'" % keep)
        print("dropping duplicate interactions...")

        if self._TIME in self._column_name:
            sort_key = [self._USER, self._TIME]
        else:
            sort_key = [self._USER, self._ITEM]

        self.all_data.sort_values(by=sort_key, inplace=True)

        self.all_data.drop_duplicates(subset=[self._USER, self._ITEM], keep=keep, inplace=True)

    @typeassert(user_min=int, item_min=int)
    def data_filtering(self, user_min=0, item_min=0, user_value=None, item_value=None, rating_value=None, time_value=None):
        """
            - Filter missing user_id or item_id
            - Remove duplicated user-item interaction
            - Value-based data filtering
            - Remove interaction by user or item
            - K-core data filtering

        Args:
            user_min (int): The users with less interactions than 'user_min' will be filtered.
            item_min (int): The items with less interactions than 'item_min' will be filtered.
        """
        self.filter_nan_user_or_item()
        self.drop_duplicates()
        self.filter_by_field_value(user_value, item_value, rating_value, time_value)
        self.filter_by_inter_num(user_min, item_min)

    def filter_nan_user_or_item(self):
        for name in [self._USER, self._TIME]:
            dropped_idx = self.all_data[name].isnull()
            self.all_data = self.all_data[~dropped_idx]

    def filter_by_field_value(self, user_value=None, item_value=None, rating_value=None, time_value=None, **kwargs):
        """
        Args:
            rating_value is a list to filter data between rating_value, such as `rating_value = [3, 5]` or `rating_value = [3, 'inf']`
        """
        print("filtering from field_value...")
        field_value_dict = {self._USER: user_value,
                            self._ITEM: item_value,
                            self._RATING: rating_value,
                            self._TIME: time_value}
        for field, values_list in field_value_dict.items():
            if values_list is not None:
                if isinstance(values_list[0], (int, float)):
                    if values_list[1] == 'inf':
                        values_list[1] = np.inf
                    filtered_idx = (self.all_data[field] >= values_list[0]) & (self.all_data[field] <= values_list[1])
                    self._config[f"{field} value between"] = values_list
                    self.all_data = self.all_data[filtered_idx]

    @typeassert(user_min=int, item_min=int)
    def filter_by_inter_num(self, user_min=0, item_min=0):
        """Filter users and items with a few interactions.
            Lower bound of the interval is also called k-core filtering, which means this method
            will filter loops until all the users and items has at least k interactions.

        Args:
            user_min (int): The users with less interactions than 'user_min' will be filtered.
            item_min (int): The items with less interactions than 'item_min' will be filtered.
        """
        if user_min > 0 or item_min > 0:
            print("filtering users...")
            print("filtering items...")
            while True:
                dropped_users = self.filter_user(user_min)
                dropped_items = self.filter_item(item_min)

                if dropped_users.sum() == 0 and dropped_items.sum() == 0:
                    break

    @typeassert(user_min=int)
    def filter_user(self, user_min=0):
        """Filter users with a few interactions.

        Args:
            user_min (int): The users with less interactions than 'user_min' will be filtered.
        """
        self._config["user_min"] = str(user_min)
        self._user_min = user_min
        if user_min > 0:
            user_count = self.all_data[self._USER].value_counts(sort=False)
            user_count_all = user_count[self.all_data[self._USER].values]
            filtered_idx = user_count_all >= user_min
            self.all_data = self.all_data[filtered_idx.values]
        return ~(filtered_idx.values)

    @typeassert(item_min=int)
    def filter_item(self, item_min=0):
        """Filter items with a few interactions.

        Args:
            item_min (int): The items with less interactions than 'item_min' will be filtered.
        """

        self._config["item_min"] = str(item_min)
        self._item_min = item_min
        if item_min > 0:
            item_count = self.all_data[self._ITEM].value_counts(sort=False)
            item_count_all = item_count[self.all_data[self._ITEM].values]
            filtered_idx = item_count_all >= item_min
            self.all_data = self.all_data[filtered_idx.values]
        return ~(filtered_idx.values)

    def remap_data_id(self):
        """Convert user and item IDs to integers, start from 0.

        """
        self.remap_user_id()
        self.remap_item_id()

    def remap_user_id(self):
        """Convert user IDs to integers, start from 0.

        """
        print("remapping user IDs...")
        self._config["remap_user_id"] = "True"
        unique_user = self.all_data[self._USER].unique()
        self.user2id = pd.Series(data=range(len(unique_user)), index=unique_user)

        self.all_data[self._USER] = self.all_data[self._USER].map(self.user2id)

    def remap_item_id(self):
        """Convert item IDs to integers, start from 0.

        """
        print("remapping item IDs...")
        self._config["remap_item_id"] = "True"
        unique_item = self.all_data[self._ITEM].unique()
        self.item2id = pd.Series(data=range(len(unique_item)), index=unique_item)

        self.all_data[self._ITEM] = self.all_data[self._ITEM].map(self.item2id)

    @typeassert(train=float, valid=float, test=float)
    def split_data_by_ratio(self, train=0.7, valid=0.1, test=0.2, by_time=False):
        """Split dataset by the given ratios.

        The dataset will be split by each user.

        Args:
            train (float): The proportion of training data.
            valid (float): The proportion of validation data.
                '0.0' means no validation set.
            test (float): The proportion of testing data.
            by_time (bool): Splitting data randomly or by time.
        """
        if train <= 0.0:
            raise ValueError("'train' must be a positive value.")

        if not np.isclose(train + valid + test, 1):
            raise ValueError("The sum of 'train', 'valid' and 'test' must equal to 1.0.")
        print("splitting data by ratio...")

        self._config["split_by"] = "ratio"
        self._config["train"] = str(train)
        self._config["valid"] = str(valid)
        self._config["test"] = str(test)
        self._config["by_time"] = str(by_time)

        if by_time is False or self._TIME not in self._column_name:
            sort_key = [self._USER]
        else:
            sort_key = [self._USER, self._TIME]

        self.all_data.sort_values(by=sort_key, inplace=True)

        self._split_manner = "ratio"
        train_data = []
        valid_data = []
        test_data = []

        user_grouped = self.all_data.groupby(by=self._USER)
        for user, u_data in user_grouped:
            u_data_len = len(u_data)
            if not by_time:
                u_data = u_data.sample(frac=1)
            train_end = math.ceil(train * u_data_len)
            train_data.append(u_data.iloc[:train_end])
            if valid != 0:
                test_begin = train_end + math.ceil(valid * u_data_len)
                valid_data.append(u_data.iloc[train_end:test_begin])
            else:
                test_begin = train_end
            test_data.append(u_data.iloc[test_begin:])

        self.train_data = pd.concat(train_data, ignore_index=True)
        if valid != 0:
            self.valid_data = pd.concat(valid_data, ignore_index=True)
        self.test_data = pd.concat(test_data, ignore_index=True)

    # TODO whether align with recbole for sequence data preprocessing
    # see <https://github.com/RUCAIBox/RecBole/issues/1060>
    # now, we use `yelp.train_data.to_truncated_seq_dict(max_len=5, padding='pre', truncating='pre')` to get valid sequence items.
    # and use `(yelp.train_data + yelp.valid_data).to_truncated_seq_dict(max_len=5, padding='pre', truncating='pre')` to get test sequence items.
    @typeassert(valid=int, test=int)
    def split_data_by_leave_out(self, valid=1, test=1, by_time=True):
        """Split dataset by leave out certain number items.

        The dataset will be split by each user.

        Args:
            valid (int): The number of items of validation set for each user.
                Default to 1 and means leave one out.
            test (int): The number of items of test set for each user.
                Default to 1 and means leave one out.
            by_time (bool): Splitting data randomly or by time.
        """

        self._config["split_by"] = "leave_out"
        self._config["valid"] = str(valid)
        self._config["test"] = str(test)
        self._config["by_time"] = str(by_time)

        if by_time is False or self._TIME not in self._column_name:
            sort_key = [self._USER]
        else:
            sort_key = [self._USER, self._TIME]
        print("splitting data by leave out...")

        self.all_data.sort_values(by=sort_key, inplace=True)

        self._split_manner = "leave"
        train_data = []
        valid_data = []
        test_data = []

        user_grouped = self.all_data.groupby(by=self._USER)
        for user, u_data in user_grouped:
            if not by_time:
                u_data = u_data.sample(frac=1)
            train_end = -(valid + test)
            train_data.append(u_data.iloc[:train_end])
            if valid != 0:
                test_begin = train_end + valid
                valid_data.append(u_data.iloc[train_end:test_begin])
            else:
                test_begin = train_end
            test_data.append(u_data.iloc[test_begin:])

        self.train_data = pd.concat(train_data, ignore_index=True)
        if valid != 0:
            self.valid_data = pd.concat(valid_data, ignore_index=True)
        self.test_data = pd.concat(test_data, ignore_index=True)

    def save_data(self, root_dir, base_dir="processed"):
        """Save data to disk.

        Args:
            root_dir (str): The parent dir of processed.
            base_dir (str): default is `processed`
        """
        print("saving data to disk...")
        processed_dir = osp.join(root_dir, base_dir)
        # eg. processed_dir = 'rec_data/processed', self._dataset_name = 'ml-100k'
        # file_name = 'self._dataset_name, self._split_manner, self._user_min, self._item_min'

        file_name = "%s_%s_u%d_i%d" % (self._dataset_name, self._split_manner, self._user_min, self._item_min)
        dir_path = osp.join(processed_dir, self._dataset_name)

        if not osp.exists(dir_path):
            os.makedirs(dir_path)

        if len(os.listdir(dir_path)):
            warnings.warn("there has stored a spilted dataset, will remove them and generate a new spilt dataset.")
            for _file_name in os.listdir(dir_path):
                os.remove(osp.join(dir_path, _file_name))

        # save data
        file_name = osp.join(dir_path, file_name)
        sep = "\t"
        if self.all_data is not None:
            self.all_data.to_csv(file_name + ".all", header=self._column_name, index=False, sep=sep)
        if self.train_data is not None:
            self.train_data.to_csv(file_name + ".train", header=self._column_name, index=False, sep=sep)
        if self.valid_data is not None:
            self.valid_data.to_csv(file_name + ".valid", header=self._column_name, index=False, sep=sep)
        if self.test_data is not None:
            self.test_data.to_csv(file_name + ".test", header=self._column_name, index=False, sep=sep)
        if self.user2id is not None:
            self.user2id.to_csv(file_name + ".user2id", index=True, sep=sep, index_label='user_id', header=['remap_user_id'])
        if self.item2id is not None:
            self.item2id.to_csv(file_name + ".item2id", index=True, sep=sep, index_label='item_id', header=['remap_item_id'])

        # calculate statistics
        user_num = len(self.all_data[self._USER].unique())
        item_num = len(self.all_data[self._ITEM].unique())
        rating_num = len(self.all_data)
        density = 1.0 * rating_num / (user_num * item_num)
        sparsity = 1 - density

        # write log file
        logger = setup_logger(file_name + ".log", color=False, time=False)
        data_info = os.linesep.join(["%s = %s" % (key, value) for key, value in self._config.items()])
        logger.info("---------------------------------")
        logger.info("Config of data:")
        logger.info(data_info)
        logger.info("---------------------------------")
        logger.info("Data statistic:")
        logger.info("The number of users: %d" % user_num)
        logger.info("The number of items: %d" % item_num)
        logger.info("The number of ratings: %d" % rating_num)
        logger.info("Average actions of users: %.2f" % (1.0 * rating_num / user_num))
        logger.info("Average actions of items: %.2f" % (1.0 * rating_num / item_num))
        logger.info("The density of the dataset: %.5f" % (density))
        logger.info("The sparsity of the dataset: %.5f" % (sparsity))
        logger.info("---------------------------------")


# if __name__ == "__main__":
#     # Usage: atom -> processed
#     data = Preprocessor()
#     data.load_data(root_dir="./rec_data", dataset_name="ml-100k", sep="\t", columns="UIRT")
#     data.drop_duplicates()
#     data.filter_data(user_min=5, item_min=5)
#     data.remap_data_id()
#     # data.split_data_by_leave_out(valid=1, test=1)
#     data.split_data_by_ratio(train=0.7, valid=0.0, test=0.3, by_time=True)
#     data.save_data(root_dir="./rec_data")
