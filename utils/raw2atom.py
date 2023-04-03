import argparse
import importlib
import os.path as osp
import sys
sys.path.append(".")
from utils.meta import dataset2class, multiple_dataset, click_dataset

raw_base_dir = "raw"
atom_base_dir = "atom"
processed_base_dir = "processed"


# -------------------- raw to atom ---------------------
def generate_atom_files(input_path, output_path, dataset, interaction_type, duplicate_removal,
                        convert_inter, convert_item, convert_user):
    input_path = osp.join(input_path, raw_base_dir, dataset)
    output_path = osp.join(output_path, atom_base_dir, dataset)
    input_args = [input_path, output_path]
    dataset_class_name = dataset2class[dataset.lower()]
    dataset_class = getattr(importlib.import_module('.all_dataset', package='datasets'), dataset_class_name)
    if dataset_class_name in multiple_dataset:
        input_args.append(interaction_type)
    if dataset_class_name in click_dataset:
        input_args.append(duplicate_removal)
    datasets = dataset_class(*input_args)

    if convert_inter:
        datasets.convert_inter()
    if convert_item:
        datasets.convert_item()
    if convert_user:
        datasets.convert_user()


# # -------------------- atom to processed ---------------------
# def generate_processed_files(root_dir, dataset_name, sep, columns,
#                              drop_duplicates=True, user_min=5, item_min=5,
#                              spilt_mode=1, ratio=[0.7, 0.0, 0.3],
#                              save_dir='./rec_data', *args, **kwargs):
#     data = Preprocessor()
#     data.load_data(root_dir=root_dir, dataset_name=dataset_name, sep=sep, columns=columns)
#     if drop_duplicates:
#         data.drop_duplicates()
#     data.filter_data(user_min=user_min, item_min=item_min)
#     data.remap_data_id()
#     if spilt_mode:
#         data.split_data_by_leave_out(valid=1, test=1)
#     else:
#         train, valid, test = ratio
#         data.split_data_by_ratio(train, valid, test, by_time=True)
#     data.save_data(root_dir=save_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--interaction_type', type=str, default=None)
    parser.add_argument('--duplicate_removal', action='store_true')

    parser.add_argument('--convert_inter', action='store_true')
    parser.add_argument('--convert_item', action='store_true')
    parser.add_argument('--convert_user', action='store_true')

    args = parser.parse_args()

    assert args.input_path is not None, 'input_path can not be None, please specify the input_path'
    assert args.output_path is not None, 'output_path can not be None, please specify the output_path'

    generate_atom_files(args.input_path, args.output_path, args.dataset, args.interaction_type, args.duplicate_removal,
                        args.convert_inter, args.convert_item, args.convert_user)
