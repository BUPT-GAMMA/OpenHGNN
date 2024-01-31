import torch.nn.functional as F
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator


@register_task("AdapropI")
class AdapropI(BaseTask):
    """Recommendation tasks."""
    def __init__(self, args):
        super().__init__()
        self.logger = args.logger
        self.dataloader = build_dataset(args, 'AdapropI')
    def evaluate(self, y_true, y_score):
        pass

