import argparse
import ast
import os.path as osp
import sys
sys.path.append(".")
from data import Preprocessor
raw_base_dir = "raw"
atom_base_dir = "atom"
processed_base_dir = "processed"

# -------------------- atom to processed ---------------------


def generate_processed_files(root_dir, save_dir, dataset_name, columns, sep,
                             drop_duplicates=True, user_min=5, item_min=5, rating_value=[3, 8],
                             spilt_mode=1, ratio=[0.7, 0.0, 0.3], *args, **kwargs):
    data = Preprocessor()
    data.load_data(root_dir=root_dir, dataset_name=dataset_name, sep=sep, columns=columns)
    if drop_duplicates:
        data.drop_duplicates()
    if rating_value is not None:
        data.filter_by_field_value(rating_value=rating_value)
    data.filter_by_inter_num(user_min=user_min, item_min=item_min)
    data.remap_data_id()
    if spilt_mode == 'leave_out':
        data.split_data_by_leave_out(valid=1, test=1)
    elif spilt_mode == 'ratio':
        train, valid, test = ratio
        data.split_data_by_ratio(train, valid, test, by_time=True)
    else:
        raise TypeError('spilt_mode must in (ratio, leave_out)')
    data.save_data(root_dir=save_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--interaction_type', type=str, default=None)
    parser.add_argument('--sep', type=str, default='\t')
    parser.add_argument('--duplicate_removal', action='store_true')

    parser.add_argument('--user_min', type=int, default=0)
    parser.add_argument('--item_min', type=int, default=0)
    parser.add_argument('--rating_val', type=ast.literal_eval, default=None)
    parser.add_argument('--split_mode', type=str, default='leave_out', help='how to split the data. (ratio, leave_out)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[0.7, 0.0, 0.3])

    args = parser.parse_args()

    assert args.input_path is not None, 'input_path can not be None, please specify the input_path'
    assert args.output_path is not None, 'output_path can not be None, please specify the output_path'

    generate_processed_files(args.input_path, args.output_path, args.dataset, args.interaction_type, args.sep, args.duplicate_removal,
                             args.user_min, args.item_min, args.rating_val, args.split_mode, args.ratio)
