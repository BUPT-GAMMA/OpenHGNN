import torch.nn.functional as F
import torch.nn as nn
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator
import torch
import numpy as np
from sklearn import metrics


@register_task("ktn")
class KTN(BaseTask):
    r"""
    Knowledge Transfer Network (KTN) task.
    
    Attributes
    -----------
    dataset : NodeClassificationDataset
        Task-related dataset

    classifier : Classifier
        custom classifier for node classification task

    Methods
    ---------
    get_graph :
        return a graph
    get_loss_fn :
        return a loss function
    get_loss :
        return a loss value
    evaluate :
        return a evaluation metric value
    """
    def __init__(self, args):
        super(KTN, self).__init__()
        print("starting ktn task")
        self.logger = args.logger
        self.dataset = build_dataset(args.dataset, "ktn", logger=self.logger)
        self.g = self.dataset.g
        self.args = args
        self.task_type = args.task_type
        self.classifier = Classifier(
            args.hid_dim, self.dataset.dims[self.task_type]
        )
        if hasattr(args, "evaluation_metric"):
            self.evaluation_metric = args.evaluation_metric
        else:
            self.evaluation_metric = "ndcg"

    def get_graph(self):
        return self.dataset.g

    def get_split(self, node_type, device="cpu"):
        return self.dataset.get_split(node_type, device=device)

    def get_labels(self):
        return self.dataset.g.ndata[self.task_type]

    def get_loss(self, y_pred, y_true):
        return self.classifier.calc_loss(y_pred, y_true)

    def evaluate(self, y_pred, y_true):
        return self.classifier.calc_acc(y_pred, y_true, self.evaluation_metric)

    def get_loss_fn(self):
        return self.classifier.calc_loss


def mrr(y_pred, y_true):
    sorted_indices = th.argsort(y_pred, dim=1, descending=True)
    sorted_true = th.gather(y_true, 1, sorted_indices)
    first_hit = (sorted_true > 0) & (sorted_true.cumsum(dim=1) == sorted_true)
    ranks = th.nonzero(first_hit, as_tuple=False)[:, 1] + 1
    return th.mean(1.0 / ranks.float()).item()


def ndcg(pred_labels, true_relevance):
    k = true_relevance.shape[1]
    indices = pred_labels.argsort(descending=True, dim=1)
    sorted_true_relevance = torch.gather(true_relevance, 1, indices)
    discounts = torch.log2(torch.arange(k, device=true_relevance.device).float() + 2.0)
    dcg = (sorted_true_relevance[:, :k] / discounts).sum(dim=1)
    true_indices = true_relevance.argsort(descending=True, dim=1)
    ideal_sorted_relevance = torch.gather(true_relevance, 1, true_indices)
    idcg = (ideal_sorted_relevance[:, :k] / discounts).sum(dim=1)
    idcg[idcg == 0] = 1
    ndcg = dcg / idcg

    return ndcg.mean().item()


def f1(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    y_true = torch.argmax(y_true, dim=1).cpu()
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(
        y_true, y_pred, average="macro"
    )


class Classifier(nn.Module):
    def __init__(self, n_in, n_out):
        super(Classifier, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.criterion = torch.nn.KLDivLoss(reduction="batchmean")
        nn.init.xavier_uniform_(self.linear.weight)

    def get_parameters(self):
        ml = list()
        ml.append({"params": self.linear.parameters()})
        return ml

    def forward(self, x):
        y = self.linear(x)
        return torch.log_softmax(y, dim=-1)

    def calc_loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)

    def calc_acc(self, y_pred, y_true, metric):
        if metric not in ["ndcg", "mrr"]:
            raise "metric not supported"
        elif metric == "ndcg":
            return ndcg(y_pred, y_true)
        elif metric == "mrr":
            return mrr(y_pred, y_true)
