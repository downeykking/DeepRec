# atom -> processed

home_dir=$HOME
repo_dir="myrepo/DeepRec"
data_dir="rec_data"

# this is Alibaba-iFashion data, very large.
dataset="Alibaba-iFashion"
python utils/atom2processed.py --dataset $dataset  --input_path $data_dir --output_path $data_dir \
                            --interaction_type "UI" --user_min 0 --item_min 0 --split_mode "ratio" \
                            --ratio 0.8 0.1 0.1