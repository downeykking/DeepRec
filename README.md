# DeepRec
*This is a repository with RecSys Codes based on Pytorch*. 

*DeepRec is mainly based on [RecBole](https://github.com/RUCAIBox/RecBole/tree/master), 
[RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets), [NeuRec](https://github.com/wubinzzu/NeuRec/tree/v3.x),
[SGL](https://github.com/wujcan/SGL-Torch) and adopts some codes from [LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch), [NGCF-PyTorch](https://github.com/huangtinglin/NGCF-PyTorch).
Many thanks to their wonderful work!*

## Pre
1.  `mkdir -p ./rec_data/raw`, `mkdir -p ./rec_data/atom`, `mkdir -p ./rec_data/processed`.
2.  download raw files to `./rec_data/raw`.
3.  transfer raw files to atom files via `scripts/raw2atom.sh`, or you can download atom files directly from https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj. Dataset information see reference https://github.com/RUCAIBox/RecSysDatasets.
4.  transfer atom files to processed files(train, valid, test) via `scripts/ml-100k.sh`. (Configure different dataset by writing scripts/dataset_name.sh)
5.  Now, the files structure will be like `Files structure`.
6.  You can load `train, valid, test` file via dataset.py.

## Files structure
```
DeepRec
├── data
│   ├── __init__.py
│   ├── dataset.py
│   ├── dataloader.py
│   ├── iteraction.py
│   ├── preprocessor.py
│   ├── sampler.py
│   └── README.md
├── datasets
│   ├── __init__.py
│   ├── base_dataset.py
│   └── all_dataset.py
├── losses
│   ├── __init__.py
│   ├── emb_loss.py
│   ├── reg_loss.py
│   └── bpr_loss.py
├── layers
│   ├── __init__.py
│   └── layers.py
├── models
│   ├── __init__.py
│   └── base_model.py
├── rec_data
│   ├── atom
│   │   ├── gowalla
│   │   │   └── gowalla.inter
│   │   ├── ml-1m
│   │   │   ├── ml-1m.inter
│   │   │   ├── ml-1m.item
│   │   │   └── ml-1m.user
│   ├── processed
│   │   ├── gowalla
│   │   │   ├── gowalla_ratio_u10_i10.all
│   │   │   ├── gowalla_ratio_u10_i10.item2id
│   │   │   ├── gowalla_ratio_u10_i10.log
│   │   │   ├── gowalla_ratio_u10_i10.test
│   │   │   ├── gowalla_ratio_u10_i10.train
│   │   │   ├── gowalla_ratio_u10_i10.user2id
│   │   │   └── gowalla_ratio_u10_i10.valid
│   │   ├── ml-1m
│   │   │   ├── ml-1m_ratio_u0_i0.all
│   │   │   ├── ml-1m_ratio_u0_i0.item2id
│   │   │   ├── ml-1m_ratio_u0_i0.log
│   │   │   ├── ml-1m_ratio_u0_i0.test
│   │   │   ├── ml-1m_ratio_u0_i0.train
│   │   │   └── ml-1m_ratio_u0_i0.user2id
│   ├── raw
│   │   ├── gowalla
│   │   │   └── loc-gowalla_totalCheckins.txt
│   │   └── ml-1m
│   │       ├── README
│   │       ├── users.dat
│   │       ├── movies.dat
│   │       └── ratings.dat
├── scripts
│   ├── raw2atom.sh
│   ├── atom2processed.sh
│   └── ml-100k.sh
├── examples
│   └── puremf.sh
├── utils
│   ├── __init__.py
│   ├── meta.py
│   ├── decorators.py
│   ├── logger.py
│   ├── metrics.py
│   ├── metrics_tensor.py
│   ├── util.py
│   ├── raw2atom.py
│   └── atom2processed.py
├── main.py
└── README.md
```
## Run
1.  see `./main.py` to know how to load data iter and design a model training and testing.
2.  write your own models and layers(if need) and losses in `./models/` and `./layers/` and `./losses/`. See `./models/base_model.py`
3.  configure your model arguments and write a bash file under `./examples/`
4.  run ```bash examples/puremf.sh```
