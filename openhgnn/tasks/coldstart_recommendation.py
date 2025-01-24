import torch.nn.functional as F
import torch.nn as nn
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator
import torch
import numpy as np
import dgl


@register_task("coldstart_recommendation")
class coldstart_recommendation(BaseTask):
    def __init__(self, args):
        super(coldstart_recommendation, self).__init__()
        # self.logger = args.logger
        self.dataloader = build_dataset(
            args.dataset, "coldstart_recommendation", logger=args.logger, args=args
        )
        self.args = args
        self.evaluator = Evaluator(args.seed)

    def get_graph(self):
        return

    def get_loss_fn(self):
        return F.mse_loss

    def get_evaluator(self):
        return self.evaluator.ndcg

    def evaluate(self, *args, **kwargs):
        return