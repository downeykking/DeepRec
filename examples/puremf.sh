# PureMF
model='puremf'
epochs=200
embedding_size=64
lr=0.001


python main.py \
        --model $model \
        --epochs $epochs \
        --embedding_size $embedding_size \
        --lr $lr