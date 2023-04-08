# atom -> processed

home_dir=$HOME
repo_dir="myrepo/DeepRec"
data_dir="rec_data"

datasets=("ml-100k" "ml-1m" "gowalla" "yelp" "Amazon_Books" "Alibaba-iFashion")



dataset="ml-100k"
python utils/atom2processed.py --dataset $dataset  --input_path $data_dir --output_path $data_dir \
                            --interaction_type "UIRT" --user_min 0 --item_min 0 --split_mode "ratio" \
                            --ratio 0.8 0.1 0.1 --duplicate_removal --rating_val "[0, 'inf']"

# dataset="ml-1m"
# python utils/atom2processed.py --dataset $dataset  --input_path $data_dir --output_path $data_dir \
#                             --interaction_type "UIRT" --user_min 0 --item_min 0 --split_mode "ratio" \
#                             --ratio 0.7 0.0 0.3 --rating_val "[3, 'inf']"

# dataset="gowalla"
# python utils/atom2processed.py --dataset $dataset  --input_path $data_dir --output_path $data_dir \
#                             --interaction_type "UI" --user_min 10 --item_min 10 --split_mode "ratio" \
#                             --ratio 0.8 0.1 0.1

# dataset="yelp"
# python utils/atom2processed.py --dataset $dataset  --input_path $data_dir --output_path $data_dir \
#                             --interaction_type "UIR" --user_min 15 --item_min 15 --split_mode "ratio" \
#                             --ratio 0.8 0.1 0.1 --rating_val "[3, 'inf']"

# dataset="Amazon_Books"
# python utils/atom2processed.py --dataset $dataset  --input_path $data_dir --output_path $data_dir \
#                             --interaction_type "UIR" --user_min 15 --item_min 15 --split_mode "ratio" \
#                             --ratio 0.8 0.1 0.1 --rating_val "[3, 'inf']"

# dataset="Alibaba-iFashion"
# python utils/atom2processed.py --dataset $dataset  --input_path $data_dir --output_path $data_dir \
#                             --interaction_type "UI" --user_min 0 --item_min 0 --split_mode "ratio" \
#                             --ratio 0.8 0.1 0.1

# dataset="lastfm"
# python utils/atom2processed.py --dataset $dataset  --input_path $data_dir --output_path $data_dir \
#                             --interaction_type "UI" --user_min 0 --item_min 0 --split_mode "ratio" \
#                             --ratio 0.8 0.1 0.1 --duplicate_removal 