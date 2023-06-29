import torch.nn.functional as F

from . import BaseTask, register_task
from ..dataset import build_dataset
from ..dataset.mag_dataset import MagDataset
from ..utils import Evaluator


@register_task("pretrain")
class Pretrain(BaseTask):
    """Pretrain task."""
    def __init__(self, args):
        super(Pretrain, self).__init__()
        self.args = args
        self.n_dataset = args.dataset
        self.logger = args.logger
        self.dataset = build_dataset(args.dataset, 'pretrain', logger=self.logger, train_percent=args.train_percent)
        # self.evaluator = Evaluator()
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

    def evaluate(self, y_true, y_score, name):
        if name == 'ndcg':
            return self.evaluator.ndcg(y_true, y_score)