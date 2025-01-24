import torch.nn.functional as F

from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator


@register_task("demo")
class Demo(BaseTask):
    """Demo task."""
    def __init__(self, args):
        super(Demo, self).__init__()
        self.n_dataset = args.dataset
        self.dataset = build_dataset(args.dataset, 'demo')
        # self.evaluator = Evaluator()
        self.evaluator = Evaluator(args.seed)

    def get_graph(self):
        return getattr(self.dataset, 'g', self.dataset[0])

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