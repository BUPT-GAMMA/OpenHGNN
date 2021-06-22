import torch.nn.functional as F

from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator


@register_task("node_classification")
class NodeClassification(BaseTask):
    """Node classification tasks."""
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

    def get_graph(self):
        return self.dataset.g

    def get_loss_fn(self):
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
            return self.evaluator.cal_acc(logits, self.labels[mask])
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

    def get_idx(self):
        return self.train_idx, self.val_idx, self.test_idx

    def get_labels(self):
        return self.labels

