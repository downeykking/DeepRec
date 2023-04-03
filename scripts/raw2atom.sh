# raw -> atom

home_dir=$HOME
repo_dir="myrepo/DeepRec"
data_dir="rec_data"

datasets=("ml-100k" "ml-1m" "gowalla")


# dataset="ml-100k"
# python utils/raw2atom.py --dataset $dataset  --input_path $data_dir --output_path $data_dir \
#                 --convert_inter --convert_item --convert_user

dataset="ml-1m"
python utils/raw2atom.py --dataset $dataset  --input_path $data_dir --output_path $data_dir \
                --convert_inter --convert_item --convert_user

# dataset="gowalla"
# python utils/raw2atom.py --dataset $dataset  --input_path $data_dir --output_path $data_dir \
#                 --convert_inter --convert_item --convert_user --duplicate_removal