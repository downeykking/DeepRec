import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(".")
import torch
from torch.optim import Adam
import numpy as np
from ast import literal_eval

from data import Dataset, PairwiseSamplerV2, DataIterator
from models import NGCF
from losses import BPRLoss, EmbLoss
from utils import setup_logger, Metric, print_results
yelp = Dataset(root_dir="./rec_data/", dataset_name="ml-100k")

train_data_iter = PairwiseSamplerV2(yelp.train_data, num_neg=1, batch_size=2048, shuffle=True)
valid_data_iter = DataIterator(list(yelp.valid_data.to_user_dict().keys()), batch_size=4096,
                               shuffle=False, drop_last=False) if len(yelp.valid_data) else None
test_data_iter = DataIterator(list(yelp.test_data.to_user_dict().keys()), batch_size=4096, shuffle=False, drop_last=False)


device = torch.device("cuda:0")
model = NGCF(embed_dim=64, hidden_size_list=[64, 64, 64], node_dropout=0.0,
             message_dropout=0.1, dataset=yelp.train_data, device=device).to(device)
optimizer = Adam(model.parameters(), lr=0.001)
bpr_loss = BPRLoss()
reg_loss = EmbLoss(1e-5)
print(model)

logger = setup_logger("ngcf" + ".log", color=False, time=False)


def train(epoch, model, data_iter, optimizer, loss_func, device=None):
    model.train()
    t_loss = 0
    for data in data_iter:
        optimizer.zero_grad()
        users, pos_items, neg_items = map(lambda x: torch.LongTensor(x).to(device), data)
        users_emb_ngcf, pos_emb_ngcf, neg_emb_ngcf = model(users, pos_items, neg_items)
        loss = loss_func(users_emb_ngcf, pos_emb_ngcf, neg_emb_ngcf) + reg_loss(users_emb_ngcf, pos_emb_ngcf, neg_emb_ngcf)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()

    return t_loss / len(data_iter)


@torch.no_grad()
def test(model, dataset, data_iter, metrics=['recall', 'ndcg', 'mrr'], topks=[10], mode='test', exclude_method=['train'], device=None):
    """
        Args:
            mode: which data iter to use. 'valid' or 'test'.
            exclude_method: We will exclude items with `['train']` or `['train', 'valid']`.
                            If set to `['train']`, we will only remove `train pos items` while rating.
                            Otherwise, we will both remove `train pos items` and `valid pos items` while rating.
                            The difference see ref `https://trello.com/c/Ev5wrgi3/15-performance-on-evaltest-sets`
    """
    model.eval()

    # prepare
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

    for users in data_iter:
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


topks = [10]
valid_metric = "recall@10"
valid_metric, valid_metric_cnt = valid_metric.split("@")
valid_metric, valid_metric_cnt = valid_metric, topks.index(literal_eval(valid_metric_cnt))
best_value = 0.
best_all_results = None
stopping_step = 0
stopping_step_cnt = 10
best_epoch = 0
val_epoch = 1
tmp_model_dir = './saved_model/ngcf.pkl'
metrics = ['recall', 'ndcg', 'mrr']

# train
from tqdm import tqdm
for epoch in range(0, 1000):
    loss = train(epoch + 1, model, train_data_iter, optimizer, bpr_loss, device)
    logger.info(f'Train: Epoch {epoch + 1:04d}, Loss: {loss:.4f}')

    if (epoch + 1) % val_epoch == 0:
        # compatible if we don't split data to valid
        data_iter, mode = (valid_data_iter, 'valid') if valid_data_iter is not None else (test_data_iter, 'test')
        all_results = test(model, yelp, data_iter, metrics, topks, mode=mode, device=device)
        logger.info(f'Validation: Epoch {epoch + 1:04d}, Scores:')

        print_results(all_results, metrics, topks, logger)

        # early stopping
        cur_value = all_results[valid_metric][valid_metric_cnt]
        if cur_value >= best_value:
            stopping_step = 0
            best_value = cur_value
            best_epoch = epoch
            best_all_results = all_results
        else:
            stopping_step += 1
        if stopping_step >= stopping_step_cnt:
            logger.info(f"Early stopping is trigger. Finished training, best eval result in epoch {best_epoch + 1}")
            torch.save(model.state_dict(), tmp_model_dir)
            break


print_results(best_all_results, metrics, topks, logger)


logger.info('Loading from the saved best model during the training process.')
model.load_state_dict(torch.load(tmp_model_dir))
test_results_best = test(model, yelp, test_data_iter, metrics, topks, mode='test', device=device)
print_results(test_results_best, metrics, topks, logger)
