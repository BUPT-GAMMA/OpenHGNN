import torch.nn.functional as F
import torch.nn as nn
from openhgnn.tasks import BaseTask, register_task
from openhgnn.utils import Evaluator
from dgl.data import DGLDataset
from openhgnn.dataset import build_dataset, CLASS_DATASETS, DATASET_REGISTRY, build_dataset_v2, try_import_task_dataset
import torch


@register_task("hypergraph")
class hypergraph(BaseTask):
    r"""
    Hypergraph tasks.

    Attributes
    -----------
    dataset : HypergraphDataset
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
        super(hypergraph, self).__init__()
        self.logger = args.logger
        self.dataset = build_dataset(
            args.dataset, 'hypergraph_dataset', logger=self.logger)
        # self.evaluator = Evaluator()
        self.logger = args.logger
        self.train_data = self.dataset.get_data(args.dataset, is_train=True)
        self.val_data = self.dataset.get_data(args.dataset, is_train=False)

        self.evaluator = Evaluator(args.seed)

    def get_graph(self):
        return torch.tensor(0)

    def get_loss_fn(self):
        return F.binary_cross_entropy

    def get_evaluator(self, name):
        if name == 'acc':
            return self.evaluator.c
            al_acc
        elif name == 'f1_lr':
            return self.evaluator.nc_with_LR
        elif name == 'f1':
            return self.evaluator.f1_node_classification

    def evaluate(self, logits, mode='test', info=True):
        r"""

        Parameters
        ----------
        logits: th.Tensor
            the logits of batch
        mode: str
            the evaluation mode, train/valid/test
        Returns
        -------

        """
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
        elif self.evaluation_metric == 'acc-ogbn-mag':
            from ogb.nodeproppred import Evaluator
            evaluator = Evaluator(name='ogbn-mag')
            logits = logits.unsqueeze(dim=1)
            input_dict = {"y_true": logits,
                          "y_pred": self.labels[self.test_idx]}
            result_dict = evaluator.eval(input_dict)
            return result_dict
        elif self.evaluation_metric == 'f1':
            f1_dict = self.evaluator.f1_node_classification(
                self.labels[mask], pred)
            return f1_dict
        else:
            raise ValueError('The evaluation metric is not supported!')
