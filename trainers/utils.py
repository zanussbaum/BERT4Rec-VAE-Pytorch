import torch
from sklearn.metrics import dcg_score


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores
    labels = labels

    labels_float = labels.float()
    # returns ranking of pred label (assuming at pos 0)
    # scores = [4, 1, 2, 3]
    # scores.argsort() = [1, 2, 3, 0]
    # scores.argsort().argsort() = [3, 0, 1, 2]
    # can then compare 0th column and see if < k
    # inspired by: https://github.com/FeiSun/BERT4Rec/blob/master/run.py#L176
    rank = (-scores).argsort(dim=1).argsort(dim=1)[:, 0]
    batch_size = labels.size(0)
    for k in sorted(ks, reverse=True):
        valid = rank < k
        valid_count = valid.sum()
        
        # don't normalize as we compute this in the AverageMeter Class
        metrics['Recall@%d' % k] = (valid_count, batch_size)
        # this could also very well be wrong!
        metrics['NDCG@%d' % k] = (dcg_score(labels_float, scores, k=k) * batch_size , batch_size)

    return metrics