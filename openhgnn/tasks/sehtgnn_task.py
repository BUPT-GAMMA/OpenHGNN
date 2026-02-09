import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator

@register_task("sehtgnn_task")
class SEHTGNNTask(BaseTask):
    def __init__(self, args):
        super(SEHTGNNTask, self).__init__()
        self.args = args
        self.logger = args.logger
        self.task_type = 'link_prediction'
        
        self.dataset = build_dataset(
            dataset=args.dataset, 
            task=args.dataset,
            raw_dir=getattr(args, 'data_dir', None),
            time_window=getattr(args, 'time_window', 3),
            device=args.device,
            logger=self.logger,
        )

        if hasattr(self.dataset, 'set_args_and_load_feats'):
            self.dataset.set_args_and_load_feats(args)

        if getattr(self.dataset, 'dataset_name', '') == 'sehtgnn_covid':
            self.task_type = 'node_regression'
        
        self.evaluator = Evaluator(args.seed)

    def get_graph(self):
        if hasattr(self.dataset, 'train_set') and len(self.dataset.train_set) > 0:
            return self.dataset.train_set[0][0]
        return self.dataset[0]

    def get_loss_fn(self):
        if self.task_type == 'node_regression':
            return F.l1_loss
        return None

    def get_split(self):
        return self.dataset.get_split()

    def evaluate(self, y_true, y_score, name):
        if self.task_type == 'node_regression':
            y_true = y_true.to(y_score.device)
            
            mae = F.l1_loss(y_score, y_true).item()
            mse = F.mse_loss(y_score, y_true).item()
            rmse = mse ** 0.5
            
            return {'MAE': mae, 'RMSE': rmse}
        else:
            auc = self.evaluator.cal_roc_auc(y_true, y_score)
            
            if hasattr(y_true, 'cpu'): 
                y_true_np = y_true.cpu().detach().numpy()
            else: 
                y_true_np = y_true
                
            if hasattr(y_score, 'cpu'): 
                y_score_np = y_score.cpu().detach().numpy()
            else: 
                y_score_np = y_score
            
            ap = average_precision_score(y_true_np, y_score_np)
            
            return {'AUC': auc, 'AP': ap}