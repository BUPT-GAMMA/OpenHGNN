import torch.nn.functional as F
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator


@register_task("link_prediction")
class LinkPrediction(BaseTask):
    r"""
    Link prediction tasks.

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
        super(LinkPrediction, self).__init__()
        self.n_dataset = args.dataset
        self.dataset = build_dataset(args.dataset, 'link_prediction')
        # self.evaluator = Evaluator()
        self.train_hg, self.val_hg, self.test_hg = self.dataset.get_idx()
        self.evaluator = Evaluator(args.seed)

    def get_graph(self):
        return self.dataset.g

    def get_loss_fn(self):
        return F.binary_cross_entropy_with_logits

    def get_evaluator(self, name):
        if name == 'acc':
            return self.evaluator.author_link_prediction
        elif name == 'mrr':
            return self.evaluator.mrr_
        elif name == 'academic_lp':
            return self.evaluator.author_link_prediction
        elif name == 'roc_auc':
            return self.evaluator.cal_roc_auc

    def evaluate(self, name, logits):
        r"""

        Parameters
        ----------
        logits : th.Tensor
            the prediction of specific
        name

        Returns
        -------

        """
        if name == 'acc':
            return self.evaluator.author_link_prediction
        elif name == 'mrr':
            return self.evaluator.mrr_
        elif name == 'academic_lp':
            return self.evaluator.author_link_prediction(logits, self.dataset.train_batch, self.dataset.test_batch)

    def get_batch(self):
        return self.dataset.train_batch, self.dataset.test_batch

    def get_idx(self):
        return self.train_hg, self.val_hg, self.test_hg

    def get_labels(self):
        return self.dataset.get_labels()

