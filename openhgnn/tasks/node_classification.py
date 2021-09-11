import torch.nn.functional as F
import torch.nn as nn
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator


@register_task("node_classification")
class NodeClassification(BaseTask):
    r"""
    Node classification tasks.

    Attributes
    -----------
    dataset : NodeClassificationDataset
        Task-related dataset

    evaluator : Evaluator
        offer evaluation metric


    Methods
    ---------
    get_graph :
        return a graph
    get_loss_fn :
        return a loss function
    """
    def __init__(self, args):
        super(NodeClassification, self).__init__()
        self.dataset = build_dataset(args.dataset, 'node_classification')
        # self.evaluator = Evaluator()
        if hasattr(args, 'validation'):
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_idx(args.validation)
        else:
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_idx()
        self.evaluator = Evaluator(args.seed)
        self.labels = self.dataset.get_labels()
        self.multi_label = True if args.dataset == 'HGBn-IMDB' else False

    def get_graph(self):
        return self.dataset.g

    def get_loss_fn(self):
        if self.multi_label:
            return nn.BCEWithLogitsLoss()
        return F.cross_entropy

    def get_evaluator(self, name):
        if name == 'acc':
            return self.evaluator.cal_acc
        elif name == 'f1_lr':
            return self.evaluator.nc_with_LR
        elif name == 'f1':
            return self.evaluator.f1_node_classification

    def evaluate(self, logits, name, mask=None):
        if name == 'acc':
            return self.evaluator.cal_acc(self.labels[mask], logits)
        elif name == 'acc-ogbn-mag':
            from ogb.nodeproppred import Evaluator
            evaluator = Evaluator(name='ogbn-mag')
            logits = logits.unsqueeze(dim=1)
            input_dict = {"y_true": logits, "y_pred": self.labels[self.test_idx]}
            result_dict = evaluator.eval(input_dict)
            return result_dict
        elif name == 'f1_lr':
            return self.evaluator.nc_with_LR(logits, self.labels, self.train_idx, self.test_idx)
        elif name == 'f1':
            return self.evaluator.f1_node_classification(self.labels[mask], logits)
        else:
            raise ValueError('The metric is not supported!')

    def get_idx(self):
        return self.train_idx, self.val_idx, self.test_idx

    def get_labels(self):
        return self.labels
