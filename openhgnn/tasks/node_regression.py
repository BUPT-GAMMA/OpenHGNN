import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from . import BaseTask, register_task
from ..dataset import build_dataset, build_dataset_GB
from ..utils import Evaluator
from sklearn.metrics import mean_squared_error, mean_absolute_error

@register_task("node_regression")
class NodeRegression(BaseTask):
    r"""
    Node regression tasks.

    Attributes
    -----------
    dataset : NodeRegressionDataset
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
        super(NodeRegression, self).__init__()
        self.logger = args.logger
        self.args = args
        self.dataset = build_dataset(args.dataset, 'node_regression', 
                                     logger=self.logger, args=args)
        
        if args.graphbolt:
            self.dataset_GB = build_dataset_GB(args.dataset, 
                                               logger=self.logger, 
                                               args=args)  
        
        self.logger = args.logger

        if hasattr(args, 'validation'):
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_split(args.validation)
        else:
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_split()
            
        self.evaluator = Evaluator(args.seed)
        self.labels = self.dataset.get_labels()
        
        if hasattr(self.dataset, 'multi_label'):
            self.multi_label = self.dataset.multi_label
        else:
            self.multi_label = False

        self.train_hg = self.train_idx
        self.val_hg = self.val_idx
        self.test_hg = self.test_idx
        
        if hasattr(args, 'evaluation_metric'):
            self.evaluation_metric = args.evaluation_metric
        else:
            self.evaluation_metric = 'rmse'

    def get_graph(self):
        return self.dataset.g

    def get_loss_fn(self):
        return nn.MSELoss()

    def get_evaluator(self, name):
        return None 

    def evaluate(self, logits, mode='test', info=True):
        """
        Special handling for SEHTGNN model
        """
        if hasattr(self.args, 'model') and self.args.model == 'SEHTGNN':
            if mode == 'test':
                label_g = self.dataset.test_set[0][1]
            elif mode == 'valid':
                label_g = self.dataset.val_set[0][1]
            elif mode == 'train':
                label_g = self.dataset.train_set[0][1]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            target_ntype = 'item'
            if hasattr(self.dataset, 'category'):
                target_ntype = self.dataset.category

            if target_ntype in label_g.nodes:
                y_true_all = label_g.nodes[target_ntype].data['y']
                if 'mask' in label_g.nodes[target_ntype].data:
                    mask = label_g.nodes[target_ntype].data['mask'].bool()
                else:
                    mask = torch.ones(y_true_all.shape[0], dtype=torch.bool, device=y_true_all.device)
            else:
                raise ValueError(f"Node type {target_ntype} not found in label graph")
            
            y_pred = logits[mask]
            y_true = y_true_all[mask]

            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            
            mse = mean_squared_error(y_true_np, y_pred_np)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_np, y_pred_np)
            
            if info:
                self.logger.info(f"[{mode.upper()}] MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            return {
                'RMSE': rmse,
                'MAE': mae,
                'MSE': mse
            }

        # ==================
        # General
        # ==================
        if mode == 'test':
            mask = self.test_idx
        elif mode == 'valid':
            mask = self.val_idx
        elif mode == 'train':
            mask = self.train_idx

        pred = logits[mask].cpu().detach().numpy()
        true = self.labels[mask].cpu().numpy()
        
        if pred.shape != true.shape:
            pred = pred.reshape(true.shape)

        if self.evaluation_metric == 'rmse':
            score = np.sqrt(mean_squared_error(true, pred))
            return dict(RMSE=score)
        elif self.evaluation_metric == 'mae':
            score = mean_absolute_error(true, pred)
            return dict(MAE=score)
        elif self.evaluation_metric == 'mse':
            score = mean_squared_error(true, pred)
            return dict(MSE=score)
        else:
            raise ValueError(f'The evaluation metric {self.evaluation_metric} is not supported!')

    def downstream_evaluate(self, logits, evaluation_metric):
        pred = logits[self.test_idx].cpu().detach().numpy()
        true = self.labels[self.test_idx].cpu().numpy()
        
        if evaluation_metric == 'rmse':
            return np.sqrt(mean_squared_error(true, pred))
        elif evaluation_metric == 'mae':
            return mean_absolute_error(true, pred)
        return dict()
    
    def get_split(self):
        return self.train_idx, self.val_idx, self.test_idx

    def get_labels(self):
        return self.labels