import torch
from sklearn.metrics import dcg_score


def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores
    labels = labels

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1).argsort(dim=1)[:, 0]
    batch_size = labels.size(0)
    for k in sorted(ks, reverse=True):
        valid = rank < k
        valid_count = valid.sum()
        
        # don't normalize as we compute this in the AverageMeter Class
        metrics['Recall@%d' % k] = (valid_count, batch_size)
        metrics['NDCG@%d' % k] = (dcg_score(labels_float, scores, k=k) * batch_size , batch_size)

    return metrics