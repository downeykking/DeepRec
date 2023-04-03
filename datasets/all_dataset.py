import bz2
import csv
import json
import operator
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from .base_dataset import BaseDataset


class ML100KDataset(BaseDataset):
    def __init__(self, input_path, output_path):
        super(ML100KDataset, self).__init__(input_path, output_path)
        self.dataset_name = 'ml-100k'

        # input file
        self.inter_file = os.path.join(self.input_path, 'u.data')
        self.item_file = os.path.join(self.input_path, 'u.item')
        self.user_file = os.path.join(self.input_path, 'u.user')
        self.item_sep = '|'
        self.user_sep = '|'
        self.inter_sep = '\t'

        # output file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()

        # selected feature fields
        self.inter_fields = {0: 'user_id',
                             1: 'item_id',
                             2: 'rating',
                             3: 'timestamp'}
        self.item_fields = {0: 'item_id:token',
                            1: 'movie_title:token',
                            2: 'release_year:token',
                            3: 'class:token_seq'}
        self.user_fields = {0: 'user_id:token',
                            1: 'age:token',
                            2: 'gender:token',
                            3: 'occupation:token',
                            4: 'zip_code:token'}

    def load_inter_data(self):
        return pd.read_csv(self.inter_file, delimiter=self.inter_sep, header=None, engine='python')

    def load_item_data(self):
        origin_data = pd.read_csv(self.item_file, delimiter=self.item_sep, header=None, engine='python', encoding="ISO-8859-1")
        processed_data = origin_data.iloc[:, 0:4]
        release_year = []
        all_type = ['unkown', 'Action', 'Adventure', 'Animation',
                    'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                    'Thriller', 'War', 'Western']
        genre = []
        for i in range(origin_data.shape[0]):
            type_str = []
            for j in range(5, origin_data.shape[1]):
                if origin_data.iloc[i, j] == 1:
                    type_str.append(all_type[j - 5])
            type_str = ' '.join(type_str)
            genre.append(type_str)
            origin_name = origin_data.iloc[i, 1]
            year_start = origin_name.find('(') + 1
            year_end = origin_name.find(')')
            title_end = year_start - 2
            year = origin_name[year_start:year_end]
            title = origin_name[0: title_end]
            processed_data.iloc[i, 1] = title
            release_year.append(year)
        processed_data.insert(2, 'release_year', pd.Series(release_year))
        processed_data.insert(3, 'class', pd.Series(genre))
        return processed_data

    def load_user_data(self):
        return pd.read_csv(self.user_file, delimiter=self.user_sep, header=None, engine='python')


class ML1MDataset(BaseDataset):
    def __init__(self, input_path, output_path):
        super(ML1MDataset, self).__init__(input_path, output_path)
        self.dataset_name = 'ml-1m'

        # input file
        self.inter_file = os.path.join(self.input_path, 'ratings.dat')
        self.item_file = os.path.join(self.input_path, 'movies.dat')
        self.user_file = os.path.join(self.input_path, 'users.dat')
        self.sep = '::'

        # output file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()

        # selected feature fields
        self.inter_fields = {0: 'user_id',
                             1: 'item_id',
                             2: 'rating',
                             3: 'timestamp'}
        self.item_fields = {0: 'item_id:token',
                            1: 'movie_title:token',
                            2: 'release_year:token',
                            3: 'genre:token_seq'}
        self.user_fields = {0: 'user_id:token',
                            1: 'age:token',
                            2: 'gender:token',
                            3: 'occupation:token',
                            4: 'zip_code:token'}

    def load_inter_data(self):
        return pd.read_csv(self.inter_file, delimiter=self.sep, header=None, engine='python')

    def load_item_data(self):
        origin_data = pd.read_csv(self.item_file, delimiter=self.sep, header=None, engine='python', encoding="ISO-8859-1")
        processed_data = origin_data
        release_year = []
        for i in range(origin_data.shape[0]):
            split_type = origin_data.iloc[i, 2].split('|')
            type_str = ' '.join(split_type)
            processed_data.iloc[i, 2] = type_str
            origin_name = origin_data.iloc[i, 1]
            year_start = origin_name.find('(') + 1
            year_end = origin_name.find(')')
            title_end = year_start - 2
            year = origin_name[year_start:year_end]
            title = origin_name[0: title_end]
            processed_data.iloc[i, 1] = title
            release_year.append(year)
        processed_data.insert(2, 'release_year', pd.Series(release_year))
        return processed_data

    def load_user_data(self):
        return pd.read_csv(self.user_file, delimiter=self.sep, header=None, engine='python')


class GOWALLADataset(BaseDataset):
    def __init__(self, input_path, output_path, duplicate_removal):
        super(GOWALLADataset, self).__init__(input_path, output_path)
        self.dataset_name = 'gowalla'
        self.duplicate_removal = duplicate_removal  # merge repeat interactions if 'repeat' is True

        # input file
        self.inter_file = os.path.join(self.input_path, 'loc-gowalla_totalCheckins.txt')

        self.sep = '\t'

        # output file
        output_files = self.get_output_files()
        self.output_inter_file = output_files[0]

        # selected feature fields
        if self.duplicate_removal == True:
            self.inter_fields = {0: 'user_id',
                                 1: 'item_id',
                                 2: 'timestamp',
                                 3: 'latitude',
                                 4: 'longitude',
                                 5: 'num_repeat'}
        else:
            self.inter_fields = {0: 'user_id:token',
                                 1: 'item_id:token',
                                 2: 'timestamp:float',
                                 3: 'latitude:float',
                                 4: 'longitude:float'}

    def load_inter_data(self):
        if self.duplicate_removal == True:
            processed_data = self.run_duplicate_removal()
        else:
            origin_data = pd.read_csv(self.inter_file, delimiter=self.sep, header=None, engine='python')
            order = [0, 4, 1, 2, 3]
            origin_data = origin_data[order]
            processed_data = origin_data
            processed_data[1] = origin_data[1].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%dT%H:%M:%SZ')))
        return processed_data

    def run_duplicate_removal(self):
        cnt_row = 0
        all_user = {}
        a_user = {}
        pre_userid = '0'
        with open(self.inter_file, 'r') as f:
            line = f.readline()
            while True:
                if not line:
                    for key, value in a_user[pre_userid].items():
                        all_user[cnt_row] = [pre_userid, key, value[0], value[1], value[2], value[3]]
                        cnt_row += 1
                    break
                line = line.strip().split('\t')
                userid, timestamp, lati, longi, itemid = line[0], line[1], line[2], line[3], line[4]
                timestamp = time.mktime(time.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ'))
                if userid not in a_user.keys():
                    a_user[userid] = {}
                if itemid not in a_user[userid].keys():
                    a_user[userid][itemid] = [timestamp, lati, longi, 1]
                else:
                    a_user[userid][itemid][3] += 1

                if userid != pre_userid:
                    for key, value in a_user[pre_userid].items():
                        all_user[cnt_row] = [pre_userid, key, value[0], value[1], value[2], value[3]]
                        cnt_row += 1
                    pre_userid = userid
                line = f.readline()
        order = [0, 1, 2, 3, 4, 5]
        processed_data = pd.DataFrame(all_user).T[order]
        return processed_data


class AmazonBooksDataset(BaseDataset):
    def __init__(self, input_path, output_path):
        super(AmazonBooksDataset, self).__init__(input_path, output_path)
        self.dataset_name = 'Amazon_Books'

        # input file
        self.inter_file = os.path.join(self.input_path, 'ratings_Books.csv')
        self.item_file = os.path.join(self.input_path, 'meta_Books.json')

        self.sep = ','

        # output file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()

        # selected feature fields
        self.inter_fields = {0: 'user_id:token',
                             1: 'item_id:token',
                             2: 'rating:float',
                             3: 'timestamp:float'}

        self.item_fields = {0: 'item_id:token',
                            1: 'sales_type:token',
                            2: 'sales_rank:float',
                            4: 'categories:token_seq',
                            5: 'title:token',
                            7: 'price:float',
                            9: 'brand:token'}

    def count_num(self, data):
        user_set = set()
        item_set = set()
        for i in tqdm(range(data.shape[0])):
            user_id = data.iloc[i, 0]
            item_id = data.iloc[i, 1]
            if user_id not in user_set:
                user_set.add(user_id)

            if item_id not in item_set:
                item_set.add(item_id)
        user_num = len(user_set)
        item_num = len(item_set)
        sparsity = 1 - (data.shape[0] / (user_num * item_num))
        print(user_num, item_num, data.shape[0], sparsity)

    def load_inter_data(self):
        inter_data = pd.read_csv(self.inter_file, delimiter=self.sep, header=None, engine='python')
        return inter_data

    def load_item_data(self):
        origin_data = self.getDF(self.item_file)
        sales_type = []
        sales_rank = []
        new_categories = []
        finished_data = origin_data.drop(columns=['salesRank', 'categories'])
        for i in tqdm(range(origin_data.shape[0])):
            salesRank = origin_data.iloc[i, 1]
            categories = origin_data.iloc[i, 3]
            categories_set = set()
            for j in range(len(categories)):
                for k in range(len(categories[j])):
                    categories_set.add(categories[j][k])
            new_categories.append(str(categories_set)[1:-1])
            if pd.isnull(salesRank):
                sales_type.append(None)
                sales_rank.append(None)
            else:
                for key in salesRank:
                    sales_type.append(key)
                    sales_rank.append(salesRank[key])
        finished_data.insert(1, 'sales_type', pd.Series(sales_type))
        finished_data.insert(2, 'sales_rank', pd.Series(sales_rank))
        finished_data.insert(4, 'categories', pd.Series(new_categories))
        return finished_data

    def convert(self, input_data, selected_fields, output_file):
        output_data = pd.DataFrame()
        for column in selected_fields:
            output_data[selected_fields[column]] = input_data.iloc[:, column]
        output_data.to_csv(output_file, index=0, header=1, sep='\t')

    def convert_item(self):
        try:
            input_item_data = self.load_item_data()
            self.convert(input_item_data, self.item_fields, self.output_item_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to item file\n')

    def convert_inter(self):
        try:
            input_inter_data = self.load_inter_data()
            self.convert(input_inter_data, self.inter_fields, self.output_inter_file)
        except NotImplementedError:
            print('This dataset can\'t be converted to inter file\n')


class TMALLDataset(BaseDataset):
    def __init__(self, input_path, output_path, interaction_type, duplicate_removal):
        super(TMALLDataset, self).__init__(input_path, output_path)
        self.dataset_name = 'tmall'
        self.interaction_type = interaction_type
        assert self.interaction_type in ['click', 'buy'], 'interaction_type must be in [click, buy]'
        self.duplicate_removal = duplicate_removal

        # output file
        interact = '-buy' if self.interaction_type == 'buy' else '-click'
        repeat = ''
        self.dataset_name = self.dataset_name + interact + repeat
        self.output_path = os.path.join(self.output_path, self.dataset_name)
        self.check_output_path()
        self.output_inter_file = os.path.join(self.output_path, self.dataset_name + '.inter')

        # input file
        self.inter_file = os.path.join(self.input_path, 'ijcai2016_taobao.csv')

        self.sep = ','

        # selected feature fields
        if self.duplicate_removal:
            self.inter_fields = {0: 'user_id:token',
                                 1: 'seller_id:token',
                                 2: 'item_id:token',
                                 3: 'category_id:token',
                                 4: 'timestamp:float',
                                 5: 'interactions:float'}
        else:
            self.inter_fields = {0: 'user_id:token',
                                 1: 'seller_id:token',
                                 2: 'item_id:token',
                                 3: 'category_id:token',
                                 4: 'timestamp:float'}

    def load_inter_data(self):
        table = list()
        f = '%Y%m%d'
        with open(self.inter_file, encoding='utf-8') as fp:
            lines = fp.readlines()[1:]
            for line in tqdm(lines):
                words = line.strip().split(self.sep)
                t = int(datetime.strptime(words[5], f).timestamp())
                words[5] = str(t)
                label = words.pop(4)
                if label == '0' and self.interaction_type == 'click':
                    table.append(words)
                elif label == '1' and self.interaction_type == 'buy':
                    table.append(words)
        return table

    def convert_inter(self):
        try:
            inter_table = self.load_inter_data()
            with open(self.output_inter_file, 'w') as fp:
                fp.write('\t'.join([self.inter_fields[i] for i in range(len(self.inter_fields))]) + '\n')
                if self.duplicate_removal:
                    inter_dict = self.merge_duplicate(inter_table)
                    for k, v in tqdm(inter_dict.items()):
                        fp.write('\t'.join([str(item) for item in list(k) + v]) + '\n')
                else:
                    for line in tqdm(inter_table):
                        fp.write('\t'.join(line) + '\n')
        except NotImplementedError:
            print('This dataset can\'t be converted to inter file\n')

    def merge_duplicate(self, inter_table):
        inter_dict = {}
        for line in inter_table:
            key = tuple(line[:-1])
            t = line[-1]
            if key in inter_dict:
                inter_dict[key][0] = t
                inter_dict[key][1] += 1
            else:
                inter_dict[key] = [t, 1]
        return inter_dict


class YELPDataset(BaseDataset):
    def __init__(self, input_path, output_path):
        super(YELPDataset, self).__init__(input_path, output_path)
        self.dataset_name = 'yelp'

        # input file
        self.inter_file = os.path.join(self.input_path, 'yelp_academic_dataset_review.json')
        self.item_file = os.path.join(self.input_path, 'yelp_academic_dataset_business.json')
        self.user_file = os.path.join(self.input_path, 'yelp_academic_dataset_user.json')

        # output_file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()

        # selected feature fields
        self.inter_fields = {0: 'review_id:token',
                             1: 'user_id:token',
                             2: 'business_id:token',
                             3: 'stars:float',
                             4: 'useful:float',
                             5: 'funny:float',
                             6: 'cool:float',
                             8: 'date:float'}

        self.item_fields = {0: 'business_id:token',
                            1: 'item_name:token_seq',
                            2: 'address:token_seq',
                            3: 'city:token_seq',
                            4: 'state:token',
                            5: 'postal_code:token',
                            6: 'latitude:float',
                            7: 'longitude:float',
                            8: 'item_stars:float',
                            9: 'item_review_count:float',
                            10: 'is_open:float',
                            12: 'categories:token_seq'}

        self.user_fields = {0: 'user_id:token',
                            1: 'name:token',
                            2: 'review_count:float',
                            3: 'yelping_since:float',
                            4: 'useful:float',
                            5: 'funny:float',
                            6: 'cool:float',
                            7: 'elite:token',
                            9: 'fans:float',
                            10: 'average_stars:float',
                            11: 'compliment_hot:float',
                            12: 'compliment_more:float',
                            13: 'compliment_profile:float',
                            14: 'compliment_cute:float',
                            15: 'compliment_list:float',
                            16: 'compliment_note:float',
                            17: 'compliment_plain:float',
                            18: 'compliment_cool:float',
                            19: 'compliment_funny:float',
                            20: 'compliment_writer:float',
                            21: 'compliment_photos:float'}
        self.user_head_fields = {0: 'user_id:token',
                                 1: 'user_name:token',
                                 2: 'user_review_count:float',
                                 3: 'yelping_since:float',
                                 4: 'user_useful:float',
                                 5: 'user_funny:float',
                                 6: 'user_cool:float',
                                 7: 'elite:token',
                                 9: 'fans:float',
                                 10: 'average_stars:float',
                                 11: 'compliment_hot:float',
                                 12: 'compliment_more:float',
                                 13: 'compliment_profile:float',
                                 14: 'compliment_cute:float',
                                 15: 'compliment_list:float',
                                 16: 'compliment_note:float',
                                 17: 'compliment_plain:float',
                                 18: 'compliment_cool:float',
                                 19: 'compliment_funny:float',
                                 20: 'compliment_writer:float',
                                 21: 'compliment_photos:float'}

    def load_item_data(self):
        return pd.read_json(self.item_file, lines=True)

    def convert_inter(self):
        fin = open(self.inter_file, "r")
        fout = open(self.output_inter_file, "w")

        lines_count = 0
        for _ in fin:
            lines_count += 1
        fin.seek(0, 0)

        fout.write('\t'.join([self.inter_fields[column] for column in self.inter_fields.keys()]) + '\n')

        for i in tqdm(range(lines_count)):
            line = fin.readline()
            line_dict = json.loads(line)
            line_dict['date'] = int(time.mktime(time.strptime(line_dict['date'], "%Y-%m-%d %H:%M:%S")))
            fout.write('\t'.join([str(line_dict[self.inter_fields[key][0:self.inter_fields[key].find(":")]]) for key in
                                  self.inter_fields.keys()]) + '\n')

        fin.close()
        fout.close()

    def convert_user(self):
        fin = open(self.user_file, "r")
        fout = open(self.output_user_file, "w")

        lines_count = 0
        for _ in fin:
            lines_count += 1
        fin.seek(0, 0)

        fout.write('\t'.join([self.user_head_fields[column] for column in self.user_head_fields.keys()]) + '\n')

        for i in tqdm(range(lines_count)):
            line = fin.readline()
            line_dict = json.loads(line)
            line_dict['yelping_since'] = int(
                time.mktime(time.strptime(line_dict['yelping_since'], "%Y-%m-%d %H:%M:%S")))
            fout.write('\t'.join([str(line_dict[self.user_fields[key][0:self.user_fields[key].find(":")]]) for key in
                                  self.user_fields.keys()]) + '\n')

        fin.close()
        fout.close()
