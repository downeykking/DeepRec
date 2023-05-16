import warnings
warnings.filterwarnings("ignore")
import os
import os.path as osp
import sys
sys.path.append(".")
import torch
from torch.optim import Adam
import numpy as np
from ast import literal_eval
from data import Dataset, PairwiseSampler, PairwiseSamplerV2, DataIterator
from models import PureMF, LightGCN, NGCF, SGL
from utils import setup_logger, Metric, TMetric, print_results

from tqdm import tqdm
from time import time
import datetime

import argparse

parser = argparse.ArgumentParser(description="DeepRec")
parser.add_argument("--model", type=str, default='NGCF', help="the model name.")
parser.add_argument("--dataset", type=str, default="ml-100k", help='dataset')
parser.add_argument("--data_dir", type=str, default="./rec_data/", help='root path of data dir')
parser.add_argument('--train_batch_size', type=int, default=2048, help='train_batch_size')
parser.add_argument('--test_batch_size', type=int, default=4096, help='test_batch_size')
parser.add_argument('--seed', type=int, default=2020, help='random seed for model and dataset. (default: 2022)')
parser.add_argument("--device", type=int, default=0, help="which gpu to use. (default: 0)")

parser.add_argument("--output_model_dir", type=str, default='./saved_model', help="the model name.")

parser.add_argument("--num_neg", type=int, default=1, help="number of negative sampling. (default: 1)")
parser.add_argument("--use_valid", default=False, action='store_true', help="whether use valid data. (default: False)")

parser.add_argument('--embedding_size', type=int, default=64, help='channels of embedding representation. (default: 512)')
parser.add_argument("--n_layers", type=int, default=2, help='layers of encoder. (default: 2)')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training. (default: 0.001)')
parser.add_argument('--reg_weight', type=float, default=1e-6,
                    help='regularization weight on embeddings, use in EmbLoss. (default: 1e-6)')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay for training. (default: 0)')

parser.add_argument('--epochs', type=int, default=200, help='number of training epochs.')
parser.add_argument('--eval_epochs', type=int, default=1, help='which epoch to evaluate.')

# used for ngcf, lightgcn, sgl
parser.add_argument('--hidden_size_list', nargs="+", type=int,
                    default=[64, 64, 64], help='a list of hidden representation channels.')
parser.add_argument('--node_dropout', type=float, default=0.1, help='node dropout rate.')
parser.add_argument('--message_dropout', type=float, default=0.1, help='edge dropout rate.')
parser.add_argument('--require_pow', default=False, action='store_true',
                    help="whether power the embeddings' norm. (default: False)")
parser.add_argument('--aug_type', type=str, default="ED",
                    help='augmentation of contrastive learning, must in ["ND", "ED", "RW"].')
parser.add_argument('--drop_ratio', type=float, default=0, help='drop rate in contrastive learning.')
parser.add_argument('--ssl_tau', type=float, default=0.2, help='temperature in contrastive learning.')
parser.add_argument('--ssl_weight', type=float, default=0.001, help='the weight of contrastive learning loss.')

parser.add_argument('--drop_edge_rate_1', nargs="?", type=float, default=0.2, help='drop_edge_rate_1')
parser.add_argument('--drop_edge_rate_2', nargs="?", type=float, default=0.3, help='drop_edge_rate_2')
parser.add_argument('--drop_feature_rate_1', nargs="?", type=float, default=0.2, help='drop_feature_rate_1')
parser.add_argument('--drop_feature_rate_2', nargs="?", type=float, default=0.1, help='drop_feature_rate_2')

parser.add_argument('--topks', nargs="+", type=int, default=[10, 20, 50], help='top k of evaluating.')
parser.add_argument('--metrics', nargs="+", type=str,
                    default=['recall', 'mrr', 'ndcg', 'hit', 'precision'], help='top k of evaluating.')
parser.add_argument('--valid_metric', type=str, default="recall@10", help='the metric used to judge model pefermence.')
parser.add_argument('--exclude_method', type=str,
                    default=['train', 'valid'], help='when we evaluating model, we only remove train items appeared in model.rating().')
parser.add_argument('--stopping_step_cnt', type=int, default=1000, help='early stopping count.')

args = parser.parse_args()

config = {
    "dataset": args.dataset,
    "device": torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"),
    "n_layers": args.n_layers,
    "hidden_size_list": args.hidden_size_list,
    "node_dropout": args.node_dropout,
    "message_dropout": args.message_dropout,
    "embedding_size": args.embedding_size,
    "reg_weight": args.reg_weight,
    "require_pow": args.require_pow,
    "aug_type": args.aug_type,
    "drop_ratio": args.drop_ratio,
    "ssl_tau": args.ssl_tau,
    "ssl_weight": args.ssl_weight
}


current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logger = setup_logger(f"logs/{args.model.lower()}/{args.model.lower()}_{current_time}" + ".log", color=False, time=False)


# prepare dataset
dataset = Dataset(root_dir=args.data_dir, dataset_name=args.dataset)
train_data_iter = PairwiseSampler(dataset.train_data, num_neg=args.num_neg, batch_size=args.train_batch_size, shuffle=True)
if args.use_valid:
    valid_data_iter = DataIterator(list(dataset.valid_data.to_user_dict().keys()),
                                   batch_size=args.test_batch_size, shuffle=False, drop_last=False)
else:
    valid_data_iter = None
test_data_iter = DataIterator(list(dataset.test_data.to_user_dict().keys()),
                              batch_size=args.test_batch_size, shuffle=False, drop_last=False)


# prepare model
if args.model.lower() == 'puremf':
    Model = PureMF
elif args.model.lower() == 'ngcf':
    Model = NGCF
elif args.model.lower() == 'lightgcn':
    Model = LightGCN
elif args.model.lower() == 'sgl':
    Model = SGL


model = Model(config, dataset=dataset.train_data).to(config['device'])
optimizer = Adam(model.parameters(), lr=args.lr)
logger.info(model)


def train(model, data_iter, optimizer, device=None):
    model.train()
    total_loss = 0
    for data in data_iter:
        optimizer.zero_grad()
        users, pos_items, neg_items = map(lambda x: torch.LongTensor(x).to(device), data)
        loss = model.calculate_loss(users, pos_items, neg_items)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_iter)


@torch.no_grad()
def test(model, dataset, data_iter, metrics=['recall', 'ndcg', 'mrr'], topks=[10], mode='test', exclude_method=['train'], device=None):
    """
        Args:
            mode: which data iter to use. 'valid' or 'test'.
            exclude_method: We will exclude items with `['train']` or `['train', 'valid']`.
                            If set to `['train']`, we will only remove `train pos items` while rating.
                            Otherwise, we will both remove `train pos items` and `valid pos items` while rating.
                            The difference see ref `<https://trello.com/c/Ev5wrgi3/15-performance-on-evaltest-sets>`
    """
    model.eval()
    train_user_pos_dict = dataset.train_data.to_user_dict()
    if mode == 'valid':
        test_user_pos_dict = dataset.valid_data.to_user_dict()
    if mode == 'test':
        test_user_pos_dict = dataset.test_data.to_user_dict()
        if 'valid' in exclude_method:
            valid_user_pos_dict = dataset.valid_data.to_user_dict()
            train_user_pos_dict = {user: np.concatenate([items, valid_user_pos_dict[user]])
                                   for user, items in train_user_pos_dict.items()}

    evaluator = Metric(Ks=topks, used_metrics=metrics)
    all_results = {metric: np.zeros(len(topks)) for metric in metrics}
    # all_results = {metric: torch.zeros(len(topks)) for metric in metrics}

    for users in tqdm(data_iter):
        test_true_data = [test_user_pos_dict[user] for user in users]
        rating = model.rating(users)

        exclude_index = []
        exclude_items = []
        for idx, user in enumerate(users):
            if user in train_user_pos_dict and (len(train_user_pos_dict[user]) > 0):
                exclude_index = idx
                exclude_items = train_user_pos_dict[user]
                rating[exclude_index][exclude_items] = -torch.inf
        _, rating_k = torch.topk(rating, k=max(topks))
        rating_k = rating_k.cpu().numpy()

        result = evaluator(test_true_data, rating_k)
        for metric in metrics:
            all_results[metric] += result[metric]

    for metric in metrics:
        all_results[metric] /= len(test_user_pos_dict.keys())

    return all_results


topks = args.topks
metrics = args.metrics
valid_metric = args.valid_metric
stopping_step_cnt = args.stopping_step_cnt

valid_metric, valid_metric_cnt = valid_metric.split("@")
valid_metric, valid_metric_cnt = valid_metric, topks.index(literal_eval(valid_metric_cnt))

best_value = 0.
best_valid_all_results = None
best_test_all_results = None
stopping_step = 0
best_epoch = 0
tmp_model_dir = osp.join(args.output_model_dir, args.model)
os.makedirs(tmp_model_dir, exist_ok=True)
tmp_model_dir = osp.join(tmp_model_dir, f"{args.model.lower()}_{current_time}.pth")


total_start = time()
# train
for epoch in range(0, args.epochs):
    start = time()
    loss = train(model, train_data_iter, optimizer, config['device'])
    end = time()
    logger.info(f'Train: Epoch {epoch + 1:04d}, Loss: {loss:.4f}, Time: {end-start:.4f}s')

    if (epoch + 1) % args.eval_epochs == 0:
        # compatible if we don't split data to valid
        data_iter, mode = (valid_data_iter, 'valid') if valid_data_iter is not None else (test_data_iter, 'test')
        start = time()
        all_results = test(model, dataset, data_iter, metrics, topks, mode=mode, device=config['device'])
        end = time()
        logger.info(f'Validation: Epoch {epoch + 1:04d}, Time: {end-start:.4f}s, Scores:')
        print_results(all_results, metrics, topks, logger)
        logger.info("")

        cur_value = all_results[valid_metric][valid_metric_cnt]
        if cur_value >= best_value:
            stopping_step = 0
            best_value = cur_value
            best_epoch = epoch
            if args.use_valid:
                best_valid_all_results = all_results
            else:
                best_test_all_results = all_results
            torch.save(model.state_dict(), tmp_model_dir)
        else:
            stopping_step += 1
            if stopping_step >= stopping_step_cnt:
                logger.info(f"Early stopping is trigger. Finished training, best eval result in epoch {best_epoch + 1}")
                break
total_end = time()

# print args
logger.info(f'------------------------------------args-----------------------------------------')
logger.info(f'dataset: {args.dataset}, model: {args.model}')
logger.info(f'num_neg: {args.num_neg}, use_valid: {args.use_valid}')
logger.info(f'topks: {args.topks}, metrics: {args.metrics}, valid_metric: {args.valid_metric}')
logger.info(f'train_batch_size: {args.train_batch_size}, test_batch_size: {args.test_batch_size}')
logger.info(f'training epochs: {args.epochs:4d}, valid epochs: {args.eval_epochs:4d}')
logger.info(f'embedding size: {args.embedding_size:3d}, num layers: {args.n_layers}')
logger.info(f'learning rate: {args.lr:.6f}, weight decay: {args.weight_decay:.6f}, reg_weight: {args.reg_weight:.4f}')

logger.info("-------------------------------------------------------------")
if args.use_valid:
    logger.info(f'Loading from the saved best model at epoch {best_epoch+1} during the training process.')
    model.load_state_dict(torch.load(tmp_model_dir))
    test_results_best = test(model, dataset, test_data_iter, metrics, topks, mode='test',
                             exclude_method=args.exclude_method, device=config['device'])
    logger.info(f"Total time cost: {total_end-total_start}s")
    logger.info(f'Best valid results:')
    print_results(best_valid_all_results, metrics, topks, logger)
    logger.info(f'Best test results:')
    print_results(test_results_best, metrics, topks, logger)
else:
    logger.info(f"Best epoch is {best_epoch+1}, Best test results:")
    print_results(best_test_all_results, metrics, topks, logger)
