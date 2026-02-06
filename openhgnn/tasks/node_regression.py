import torch
import torch.nn.functional as F
import torch.nn as nn
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator

@register_task('node_regression')
@register_task('covid_node_regression')
class NodeRegression(BaseTask):
    r"""
    Node regression task for temporal datasets (e.g. COVID).
    """
    def __init__(self, args):
        super(NodeRegression, self).__init__()
        self.logger = args.logger
        
        self.dataset = build_dataset(args.dataset, 'node_regression', 
                                     logger=self.logger, args=args)
        
        self.evaluator = Evaluator(args.seed)
        
        if hasattr(self.dataset, 'train_set'):
            self.train_labels = torch.cat([y for _, y in self.dataset.train_set], dim=0)
            self.val_labels = torch.cat([y for _, y in self.dataset.val_set], dim=0)
            self.test_labels = torch.cat([y for _, y in self.dataset.test_set], dim=0)
        
    def get_graph(self):
        return self.dataset.g

    def get_loss_fn(self):
        return F.l1_loss

    def evaluate(self, logits, mode='test', info=True):
        r"""
        计算回归指标 MAE 和 RMSE
        """
        if mode == 'test':
            labels = self.test_labels
        elif mode == 'valid':
            labels = self.val_labels
        else:
            labels = self.train_labels

        labels = labels.to(logits.device)
        
        mae = F.l1_loss(logits, labels).item()
        rmse = torch.sqrt(F.mse_loss(logits, labels)).item()

        if info:
            print(f"Mode: {mode} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        return {'MAE': mae, 'RMSE': rmse}

    def downstream_predict(self, model, emb_dict):
        logits = emb_dict[self.dataset.category]
        return logits