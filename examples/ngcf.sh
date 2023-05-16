# NGCF
model='ngcf'
epochs=2000
embedding_size=64
reg_weight=1e-5
node_dropout=0.1
message_dropout=0.1
hidden_size_list=(64 64 64)
lr=0.001


python main.py \
        --model $model \
        --epochs $epochs \
        --embedding_size $embedding_size \
        --reg_weight $reg_weight \
        --node_dropout $node_dropout \
        --message_dropout $message_dropout \
        --hidden_size_list ${hidden_size_list[@]} \
        --lr $lr \