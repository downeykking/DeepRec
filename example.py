import warnings
warnings.filterwarnings("ignore")
import torch
from torch.optim import Adam
import numpy as np

from data import Dataset, PairwiseSamplerV2, DataIterator
from models import PureMF
from losses import BPRLoss, RegLoss
from utils.metrics import Metric
yelp = Dataset(root_dir="./rec_data/", dataset_name="ml-1m")
print(yelp.test_data)

train_data_iter = PairwiseSamplerV2(yelp.train_data, num_neg=1, batch_size=4096, shuffle=True)
test_data_iter = DataIterator(list(yelp.test_data.to_user_dict().keys()), batch_size=2048, shuffle=False, drop_last=False)
device = torch.device("cuda:0")
model = PureMF(embed_dim=64, dataset=yelp.train_data).to(device)
optimizer = Adam(model.parameters(), lr=0.001)
bpr_loss = BPRLoss()
reg_loss = RegLoss()


def train(epoch, model, data_iter, optimizer, loss_func, device=None):
    model.train()
    t_loss = 0
    for data in data_iter:
        optimizer.zero_grad()
        users, pos_items, neg_items = map(lambda x: torch.LongTensor(x).to(device), data)
        users_emb, pos_emb, neg_emb = model(users, pos_items, neg_items)
        loss = loss_func(users_emb, pos_emb, neg_emb) + reg_loss(users_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()

    print(f'Epoch: {epoch}, Loss: {(t_loss / len(data_iter)):.4f}')
    return t_loss / len(data_iter)


@torch.no_grad()
def test(model, dataset, data_iter, metrics=['recall', 'ndcg'], topks=[5, 10, 20], device=None):
    train_user_pos_dict = dataset.train_data.to_user_dict()
    test_user_pos_dict = dataset.test_data.to_user_dict()
    evaluator = Metric(Ks=topks, used_metrics=metrics)
    all_results = {metric: np.zeros(len(topks)) for metric in metrics}

    for users in data_iter:
        test_true_data = [test_user_pos_dict[user] for user in users]
        users = torch.LongTensor(users).to(device)
        rating = model.rating(users)

        exclude_index = []
        exclude_items = []
        for idx, user in enumerate(users):
            if user in train_user_pos_dict and len(train_user_pos_dict[user]) > 0:
                exclude_index = idx
                exclude_items = train_user_pos_dict[user]
                rating[exclude_index, exclude_items] = -(1 << 10)
        _, rating_k = torch.topk(rating, k=max(topks))
        rating_k = rating_k.cpu().numpy()

        result = evaluator(test_true_data, rating_k)
        for metric in metrics:
            all_results[metric] += result[metric]

    for metric in metrics:
        all_results[metric] /= len(test_user_pos_dict.keys())

    for metric in metrics:
        print(f'Test/{metric.upper()}:', {f"{metric}@{topks[i]}": all_results[metric][i] for i in range(len(topks))})


for epoch in range(100):
    train(epoch + 1, model, train_data_iter, optimizer, bpr_loss, device)
    if epoch + 1 % 10 == 0:
        test(model, yelp, test_data_iter, device=device)
