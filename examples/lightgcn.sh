# lightgcn
model='lightgcn'
epochs=2000
embedding_size=64
n_layers=3
reg_weight=1e-5
lr=0.001


python main.py \
        --model $model \
        --epochs $epochs \
        --embedding_size $embedding_size \
        --n_layers $n_layers \
        --reg_weight $reg_weight \
        --lr $lr