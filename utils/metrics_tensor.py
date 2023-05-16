import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_score
from sklearn.metrics import auc as sk_auc
import torch
from collections.abc import Iterable

# -------------------------------------metrics------------------------------------------
# ref: <https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/Procedure.py#L135>
# ref: <https://github.com/huangtinglin/NGCF-PyTorch/blob/master/NGCF/utility/metrics.py>


class TMetric(object):
    """
        Args:
            Ks: Ks is a list, such as `[5, 10, 25]`
            used_metrics: use which metric, such as `['recall', 'ndcg']`,
                and must in ['hit', 'precision', 'recall', 'ndcg', 'mrr', 'map']
        Returns:
            dict of every metric@ k in Ks, such as `precision, recall, ndcg, mrr, map, hit (hit_ratio)`.
        Examples:
            ```
            metric = Metric(Ks=[1, 2], used_metrics=["precision", "recall"])
            results = metric(test_true_data, test_pred_data)
            ```
    """

    def __init__(self, Ks, used_metrics):
        self.Ks = Ks
        if used_metrics is None:
            used_metrics = ['hit', 'precision', 'recall', 'ndcg', 'mrr', 'map']
        if isinstance(used_metrics, str):
            used_metrics = [used_metrics]
        self.used_metrics = [metric.lower() for metric in used_metrics]

        # assure metrics in all_metrics
        for metric in self.used_metrics:
            if metric not in ['hit', 'precision', 'recall', 'ndcg', 'mrr', 'map']:
                raise ValueError(f"{metric} is not in all metrics")

    def __call__(self, test_true_data, test_pred_data):
        """
            Args:
            test_true_data: the groundtrue, shape `(test_batch, pos_items_of_every_test_user)`.
                Generated by `test_true_data = [user_dict[u] for u in batch_users]`.
                Test_true_data should be a 2-D array. Because users may have
                different amount of pos items.
            test_pred_data: the predict rating, shape`(test_batch, max_k_items)`. test_pred_data was generated by `_, rating_K = torch.topk(rating, k=max_K)`,
                so metrics@k must less than max_K. (pre-sorted)
        """
        return self.computer(test_true_data, test_pred_data)

    def computer(self, test_true_data, test_pred_data):
        precision, recall, ndcg, mrr, hit, Map = [], [], [], [], [], []
        r = get_relevant(test_true_data, test_pred_data)

        for K in self.Ks:
            if "precision" in self.used_metrics:
                precision.append(precision_at_k(r, K))
            if "recall" in self.used_metrics:
                recall.append(recall_at_k(r, K, test_true_data))
            if "ndcg" in self.used_metrics:
                ndcg.append(ndcg_at_k(r, K, test_true_data))
            if "mrr" in self.used_metrics:
                mrr.append(mrr_at_k(r, K))
            if "hit" in self.used_metrics:
                hit.append(hit_at_k(r, K))
            if "map" in self.used_metrics:
                Map.append(map_at_k(r, K, test_true_data))

        results = {
            'precision': torch.FloatTensor(precision),
            'recall': torch.FloatTensor(recall),
            'ndcg': torch.FloatTensor(ndcg),
            'hit': torch.FloatTensor(hit),
            'mrr': torch.FloatTensor(mrr),
            'map': torch.FloatTensor(Map)
        }
        selected_results = {metric: metric_result_at_k
                            for metric, metric_result_at_k in results.items()
                            if metric in self.used_metrics}
        return selected_results

    def metrics_info(self):
        """Get all metrics information.
        Returns:
            str: A string consist of all metrics information， such as
                `"Precision@10    Precision@20    NDCG@10    NDCG@20"`.
        """
        metrics_show = ['\t'.join([(f"{metric}@" + str(k)) for k in np.sort(self.Ks)])
                        for metric in self.used_metrics]
        metric = '\t'.join(metrics_show)
        return f"metrics:  {metric}"


def get_relevant(test_true_data, test_pred_data):
    """
    transfer pred_data to relevant (eg. if pred_item in pos_item we get 1 otherwise we get 0)
    Args:
        test_true_data: shape `(test_batch, pos_items_of_every_test_user)`.
            Generated by `test_true_data = [user_dict[u] for u in batch_users]`.
            Test_true_data should be a 2-D array. Because users may have
            different amount of pos items.
        test_pred_data: shape`(test_batch, max_k_items)`. test_pred_data was generated by `_, rating_K = torch.topk(rating, k=max_K)`,
            so metrics@k must less than max_K. (pre-sorted)
    Returns:
        r: [b, max_k_items]
    """
    r = torch.zeros_like(test_pred_data)
    for i in range(len(test_true_data)):
        ground_true = torch.from_numpy(test_true_data[i]).to(r.device)
        predict_topk = test_pred_data[i]
        # pred = list(map(lambda x: x in ground_true, predict_topk))
        pred = torch.isin(predict_topk, ground_true)
        r[i] = pred
    return r.bool()


def hit_at_k(r, k):
    """
    hit @ k for batch user
    Args:
        r: means relevant, is binary (nonzero is relevant).
        k: top k, must less than max_K
    Returns:
        total hit @ k for batch user (Hasn't been averaged)
    Raises:
        ValueError: len(r) must be >= k
    """
    pred_data = r[:, :k]
    hit_score = pred_data.sum(1) > 0
    return torch.sum(hit_score)


def precision_at_k(r, k):
    """
    precision @ k for batch user
    Args:
        r: means relevant, is binary (nonzero is relevant).
        k: top k, must less than max_K
    Returns:
        total Precision @ k for batch user (Hasn't been averaged)
    Raises:
        ValueError: len(r) must be >= k
    """
    assert r.shape[1] >= k
    right_pred = r[:, :k].sum(1)
    all_pos_num = k
    batch_total_precision = torch.sum(right_pred / all_pos_num)
    return batch_total_precision


def recall_at_k(r, k, test_true_data):
    """
    recall @ k for batch user
    Args:
        r: means relevant, is binary (nonzero is relevant)
        k: top k, must less than max_K
        test_true_data: shape `(test_batch, pos_items_of_every_test_user)`.
                Generated by `test_true_data = [user_dict[u] for u in batch_users]`.
                Test_true_data should be a 2-D array. Because users may have
                different amount of pos items.
    Returns:
        total Recall @ k for batch user (Hasn't been averaged)
    """
    assert r.shape[1] >= k
    right_pred = r[:, :k].sum(1)
    all_pos_num = torch.FloatTensor([len(test_true_data[i]) for i in range(len(test_true_data))]).to(r.device)
    batch_total_recall = torch.sum(right_pred / all_pos_num)
    return batch_total_recall


def mrr_at_k(r, k):
    """
    Mean Reciprocal Rank @ k for batch user.
    Computes the reciprocal rank of the `first relevant item` found by an algorithm.
    Note that only the rank of the first relevant answer is considered,
    possible further relevant answers are ignored. If users are interested also in further relevant items,
    mean average precision is a potential alternative metric.
    Args:
        r: means relevant, is binary (nonzero is relevant)
        k: top k, must less than max_K
    Returns:
        total Mrr @ k for batch user (Hasn't been averaged)
    """
    assert r.shape[1] >= k
    pred_data = r[:, :k]
    denominator = torch.arange(1, k + 1, device=r.device)
    rr_score = pred_data / denominator
    max_rr_score, _ = rr_score.max(1)
    return torch.sum(max_rr_score)


def dcg_at_k(r, k, method=1):
    """
    Score is discounted cumulative gain (dcg)
    Relevance is positive real values.
    In here, these two formulations of DCG are the same when the relevance values of documents are binary.
    Because `r is binary` here, so we defaultly set `method=1`, otherwise set `method=0`.
    Args:
        r: means relevant, is binary (nonzero is relevant)
        k: top k, must less than max_K
    Returns:
        dcg @ k for batch user, didn't sumed.
    Returns:
        Discounted cumulative gain. shape (test_batch_user, ) (Hasn't been sumed)
    """
    assert r.shape[1] >= k
    pred_data = r[:, :k]
    if method == 0:
        dcg_score = (pred_data / torch.log2(torch.arange(2, k + 2, device=r.device))).sum(1)
        # we don't sum dcg_score for the usage of ndcg
        return dcg_score
    elif method == 1:
        # when `r is binary`, `np.power(2, pred_data) - 1` equals to `pred_data`
        dcg_score = (torch.pow(2, pred_data) - 1) / torch.log2(torch.arange(2, k + 2, device=r.device))
        dcg_score = dcg_score.sum(1)
        # we don't sum dcg_score for the usage of ndcg
        return dcg_score
    else:
        raise ValueError('method must be 0 or 1.')


def ndcg_at_k(r, k, test_true_data):
    """
    Normalized Discounted Cumulative Gain
    Args:
        r: means relevant, is binary (nonzero is relevant)
        k: top k, must less than max_K
        test_true_data: shape `(test_batch, pos_items_of_every_test_user)`.
            Generated by `test_true_data = [user_dict[u] for u in batch_users]`.
            Test_true_data should be a 2-D array. Because users may have
            different amount of pos items.
    Returns:
        total ndcg @ k for batch user (Hasn't been averaged)
    """
    assert len(r) == len(test_true_data)
    assert r.shape[1] >= k
    pred_data = r[:, :k]
    ideal_r = torch.zeros_like(pred_data)
    for i, items in enumerate(test_true_data):
        length = k if k <= len(items) else len(items)
        ideal_r[i, :length] = 1

    dcg = dcg_at_k(r, k, method=1)
    idcg = dcg_at_k(ideal_r, k, method=1)

    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.sum(ndcg)


def map_at_k(r, k, test_true_data):
    """
    Computes the mean average precision at k.
    Ref: <https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#Common-Variations-on-AP-Formula>
    Args:
        r: means relevant, is binary (nonzero is relevant)
        k: top k, must less than max_K
        test_true_data: shape `(test_batch, pos_items_of_every_test_user)`.
            Generated by `test_true_data = [user_dict[u] for u in batch_users]`.
            Test_true_data should be a 2-D array. Because users may have
            different amount of pos items.
    Returns:
        total map @ k for batch user (Hasn't been averaged)
    """
    assert r.shape[1] >= k
    pred_data = r[:, :k]
    ap_rank = torch.where(pred_data, torch.cumsum(pred_data, dim=1), 0)
    denominator = torch.arange(1, k + 1)
    ap_precision = ap_rank / denominator
    test_true_data_len = torch.FloatTensor([len(test_true_data[i]) for i in range(len(test_true_data))]).to(r.device)
    # Minor Variations
    ap_precision_score = ap_precision.sum(1) / torch.minimum(test_true_data_len, torch.full_like(test_true_data_len, k, device=r.device))
    return torch.sum(ap_precision_score)


# CTR Metrics
# NOTE: CTR metrics are different from RecSys metrics. CTR models refer to predict whether click, the model outputs `a num logit`.


def auc(test_true_data, test_pred_data):
    """
        usually the shape of test_true_data is (batch,), the shape of test_pred_data is (batch,)
        Args:
        test_true_data: label.
        test_pred_data: logit of model's output.
    """
    if isinstance(test_true_data, torch.Tensor):
        test_true_data = test_true_data.detach().cpu().numpy()
        test_pred_data = test_pred_data.detach().cpu().numpy()
    return roc_auc_score(test_true_data, test_pred_data)


# TODO ref: https://github.com/RUCAIBox/RecBole/blob/master/recbole/evaluator/metrics.py#L233
def gauc(test_true_data, test_pred_data):
    pass


# Classification Metrics
# NOTE: Classification models refer to output `num of classes logits`, such as [0.233, 0.167, 0.600],
# we should transfer it via `_, y_pred = torch.max(logits, dim=1)` or `y_pred = torch.argmax(logits)`
# to get labels explicitly. And then we can use metrics like `accurary``, precision`, `recall`.
# ref: https://zhuanlan.zhihu.com/p/59862986, https://zhuanlan.zhihu.com/p/383834021
def accurary():
    pass


def precision():
    pass


def recall():
    pass


# used for test our metrics. borrowed from https://zhuanlan.zhihu.com/p/514209681
def topk_metrics(y_true, y_pred, topKs=[3]):
    """choice topk metrics and compute it
    the metrics contains 'ndcg', 'mrr', 'recall' and 'hit'

    Args:
            y_true: list, 2-dim, each row contains the items that the user interacted
            y_pred: list, 2-dim, each row contains the items recommended
            topKs: list or tuple, if you want to get top5 and top10, topKs=(5, 10)

    Return:
            results: list, it contains five metrics, 'ndcg', 'recall', 'mrr', 'hit', 'precision'

    """
    assert len(y_true) == len(y_pred)

    if not isinstance(topKs, (tuple, list)):
        raise ValueError('topKs wrong, it should be tuple or list')

    ndcg_result = []
    mrr_result = []
    hit_result = []
    precision_result = []
    recall_result = []
    for idx in range(len(topKs)):
        ndcgs = 0
        mrrs = 0
        hits = 0
        precisions = 0
        recalls = 0
        for i in range(len(y_true)):
            if len(y_true[i]) != 0:
                mrr_tmp = 0
                mrr_flag = True
                hit_tmp = 0
                dcg_tmp = 0
                idcg_tmp = 0
                hit = 0
                for j in range(topKs[idx]):
                    if y_pred[i][j] in y_true[i]:
                        hit += 1.
                        if mrr_flag:
                            mrr_flag = False
                            mrr_tmp = 1. / (1 + j)
                            hit_tmp = 1.
                        dcg_tmp += 1. / (np.log2(j + 2))
                    idcg_tmp += 1. / (np.log2(j + 2))
                hits += hit_tmp
                mrrs += mrr_tmp
                recalls += hit / len(y_true[i])
                precisions += hit / topKs[idx]
                if idcg_tmp != 0:
                    ndcgs += dcg_tmp / idcg_tmp
        hit_result.append(round(hits / len(y_pred), 4))
        mrr_result.append(round(mrrs / len(y_pred), 4))
        recall_result.append(round(recalls / len(y_pred), 4))
        precision_result.append(round(precisions / len(y_pred), 4))
        ndcg_result.append(round(ndcgs / len(y_pred), 4))

    results = []
    for idx in range(len(topKs)):

        output = f'NDCG@{topKs[idx]}: {ndcg_result[idx]}'
        results.append(output)

        output = f'MRR@{topKs[idx]}: {mrr_result[idx]}'
        results.append(output)

        output = f'Recall@{topKs[idx]}: {recall_result[idx]}'
        results.append(output)
        output = f'Hit@{topKs[idx]}: {hit_result[idx]}'
        results.append(output)
        output = f'Precision@{topKs[idx]}: {precision_result[idx]}'
        results.append(output)
    return results


if __name__ == "__main__":
    y_pred = torch.LongTensor([[0, 1, 4], [0, 1, 6], [2, 3, 7]])
    y_true = [[1, 2, 5, 7], [0, 1, 2, 6, 8], [3, 7, 9]]
    r = get_relevant(y_true, y_pred)
    # print(ndcg_at_k(r, k=3, test_true_data=y_true) / 3)

    # r2 = np.array([[1, 0, 1, 1, 0], [0, 0, 0, 1, 1]])

    # test
    out = topk_metrics(y_true, y_pred, topKs=(2, 3))
    print(out)
    # print(r)
    # r = np.array([[1, 1, 1], [1, 1, 0]])
    # print(map_at_k(r, 3, [[1, 2, 5, 7], [0, 1, 2, 6, 8]]) / 2)
    # print(recall_at_k(r, 2, y_true) / 3)
    # print(hit_at_k(r, 2) / 3)
    # print(precision_at_k(r, 2) / 3)
    # print(dcg_at_k(r, 2, 0))
    # print(ndcg_at_k(r, 1, y_true) / 3)

    metric = TMetric(Ks=[2, 3], used_metrics=["precision", "ndcg", "mrr", "recall", "map"])
    # print(metric.metrics_info())
    results = metric(y_true, y_pred)
    print(results)
