import torch.nn.functional as F
import torch.nn as nn
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator


@register_task("edge_classification")
class EdgeClassification(BaseTask):
    r"""
    Edge classification tasks.

    Attributes
    -----------
    dataset : EdgeClassificationDataset
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
        super(EdgeClassification, self).__init__()
        self.logger = args.logger
        self.dataset = build_dataset(args.dataset, 'edge_classification', logger=self.logger)
        # self.evaluator = Evaluator()
        self.logger = args.logger
        if hasattr(args, 'validation'):
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_split(args.validation)
        else:
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_split(validation=False)
        self.evaluator = Evaluator(args.seed)
        self.labels = self.dataset.get_labels()
        self.multi_label = self.dataset.multi_label

        if hasattr(args, 'evaluation_metric'):
            self.evaluation_metric = args.evaluation_metric
        else:
            if args.dataset in ['dblp4Mg2vec']:
                self.evaluation_metric = 'acc'
            else:
                self.evaluation_metric = 'f1'

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

    def evaluate(self, logits, mode='test', info=True):
        if mode == 'test':
            mask = self.test_idx
        elif mode == 'valid':
            mask = self.val_idx
        elif mode == 'train':
            mask = self.train_idx

        if self.multi_label:
            pred = (logits[mask].cpu().numpy() > 0).astype(int)
        else:
            pred = logits[mask].argmax(dim=1).to('cpu')

        if self.evaluation_metric == 'acc':
            acc = self.evaluator.cal_acc(self.labels[mask], pred)
            return dict(Accuracy=acc)
        elif self.evaluation_metric == 'f1':
            f1_dict = self.evaluator.f1_node_classification(self.labels[mask], pred)
            return f1_dict
        else:
            raise ValueError('The evaluation metric is not supported!')

    def downstream_evaluate(self, logits, evaluation_metric):
        if evaluation_metric == 'acc_f1':
            micro_f1, macro_f1, acc = self.evaluator.ec_with_SVC(C=100, gamma='auto', emd=logits, labels=self.labels,
                                                                 train_idx=self.train_idx, test_idx=self.test_idx)
            return dict(Macro_f1=macro_f1, Mirco_f1=micro_f1, Accuracy=acc)

    def get_split(self):
        return self.train_idx, self.val_idx, self.test_idx

    def get_labels(self):
        return self.labels
