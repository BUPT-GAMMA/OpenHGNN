import torch.nn.functional as F
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator


@register_task("Ingram")
class Ingram(BaseTask):
    """Recommendation tasks."""
    def __init__(self, args):
        super().__init__()
        self.logger = args.logger
        self.name_dataset = args.dataset
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = build_dataset(args.dataset, 'Ingram')
    def evaluate(self, y_true, y_score):
        pass

