import torch
import torch.nn.functional as F

from . import BaseTask, register_task
from ..dataset import build_dataset
from ..layers import to_tensor
from ..utils import Evaluator


@register_task("qgrl")
class qgrl_task(BaseTask):
    """QGRL task."""
    def __init__(self, args):
        super(qgrl_task, self).__init__()
        self.logger = args.logger
        self.dataset = build_dataset(
            args.dataset, 'qgrl_dataset', logger=self.logger)
        self.train_data = self.dataset.get_data(args.dataset, is_train=True)
        self.val_data = self.dataset.get_data(args.dataset, is_train=False)
        self.evaluator = Evaluator(args.seed)

    def get_graph(self):
        return to_tensor(self.val_data.fea)

    def get_loss_fn(self):
        return F.binary_cross_entropy_with_logits

    def evaluate(self, y_true, y_score, name):
        if name == 'ndcg':
            return self.evaluator.ndcg(y_true, y_score)
        return

    def get_hg(self):
        return self.val_data
