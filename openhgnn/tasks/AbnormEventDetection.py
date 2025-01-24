import torch.nn as nn
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator
from sklearn.metrics import roc_auc_score, average_precision_score

@register_task("abnorm_event_detection")
class AbnormalEventDetection(BaseTask):
    """Abnormal event detection task"""
    def __init__(self, args):
        super(AbnormalEventDetection, self).__init__()
        self.logger = args.logger
        self.n_dataset = args.dataset
        self.dataset = build_dataset(args.dataset+'4'+args.model, 'abnorm_event_detection', logger=self.logger)
        self.logger = args.logger
        self.args = args

    def get_graph(self):
        return self.dataset.g

    def get_loss_fn(self):
        return None

    def get_evaluator(self, name):
        if name == 'AP':
            return average_precision_score
        elif name == 'AUC':
            return roc_auc_score
        else:
            ValueError('evaluator name must be in AP and AUC')

    def evaluate(self, *args, **kwargs):
        pass