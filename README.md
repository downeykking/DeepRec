# DeepRec
*This is a repository with RecSys Codes based on Pytorch*. 

*DeepRec is mainly based on [RecBole](https://github.com/RUCAIBox/RecBole/tree/master), 
[RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets), [NeuRec](https://github.com/wubinzzu/NeuRec/tree/v3.x),
[SGL](https://github.com/wujcan/SGL-Torch) and adopts some codes from [LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch), [NGCF-PyTorch](https://github.com/huangtinglin/NGCF-PyTorch).
Many thanks to their wonderful work!*

## Pre
1.  `mkdir -p ./rec_data/raw`, `mkdir -p ./rec_data/atom`, `mkdir -p ./rec_data/processed`.
2.  download raw files to `./rec_data/raw`.
3.  transfer raw files to atom files via `scripts/raw2atom.sh`, or you can download atom files directly from https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj.
4.  transfer atom files to processed files via `scripts/atom2processed.sh`.
5.  Now, the files structure will be like `Files structure`.

## Files structure
```
DeepRec
├── data
│   ├── dataloader.py
│   ├── dataset.py
│   ├── __init__.py
│   ├── iteraction.py
│   ├── preprocessor.py
│   ├── README.md
│   └── sampler.py
├── datasets
│   ├── all_dataset.py
│   ├── base_dataset.py
│   └── __init__.py
├── losses
│   ├── bpr_loss.py
│   ├── __init__.py
│   └── reg_loss.py
├── models
│   ├── base_model.py
│   └── __init__.py
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
│   │       ├── movies.dat
│   │       ├── ratings.dat
│   │       ├── README
│   │       └── users.dat
│   └── _recsys_dataset_test.py
├── scripts
│   ├── atom2processed.sh
│   └── raw2atom.sh
├── example.py
├── README.md
└── utils
    ├── atom2processed.py
    ├── decorators.py
    ├── __init__.py
    ├── logger.py
    ├── meta.py
    ├── metrics.py
    ├── raw2atom.py
    └── util.py
```
## Run
1.  see `./example.py` to know how to load data iter.
2.  rewrite your models and losses in `./models/` and `./losses/`
