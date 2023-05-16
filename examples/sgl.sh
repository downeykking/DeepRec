# sgl
model='sgl'
epochs=300
embedding_size=64
n_layers=3
aug_type="ED"
drop_ratio=0.1
ssl_tau=0.5
ssl_weight=0.05
reg_weight=1e-5
lr=0.001


python main.py \
        --model $model \
        --epochs $epochs \
        --embedding_size $embedding_size \
        --n_layers $n_layers \
        --aug_type $aug_type \
        --drop_ratio $drop_ratio \
        --ssl_tau $ssl_tau \
        --reg_weight $reg_weight \
        --ssl_weight $ssl_weight \
        --lr $lr