import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import dgl
import random
from tqdm import tqdm
from . import BaseFlow, register_flow
from ..models import build_model 

class MyEarlyStopping:
    """
    早停工具类：当验证集 loss 不再下降时停止训练
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

@register_flow("covid_node_regression")
class COVIDTrainer(BaseFlow):
    def __init__(self, args):
        self.set_seed(0)

        super(COVIDTrainer, self).__init__(args)
        args.dropout = getattr(args, 'dropout', 0.0)
        
        if not hasattr(self, 'output_dir'):
            self.output_dir = getattr(args, 'output_dir', './output')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if isinstance(args.model, str):
            model_name = args.model
            print(f"[COVIDTrainer] Building model {model_name} manually...")
            model_cls = build_model(model_name)
            hg_sample = self.task.dataset.train_set[0][0]
            self.model = model_cls.build_model_from_args(args, hg_sample).to(self.device)
            args.model = self.model
        else:
            self.model = args.model.to(self.device)

        self.loss_fn = F.l1_loss
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=getattr(args, 'lr', 0.005), 
            weight_decay=getattr(args, 'weight_decay', 0.0001)
        )
        
        self.patience = getattr(args, 'patience', 20)
        self.max_epoch = getattr(args, 'max_epoch', 500)
        
        model_name_str = getattr(args, 'model_name', 'SEHTGNN')
        self.checkpoint_path = os.path.join(self.output_dir, f"{model_name_str}_best.pt")
        
        self.stopper = MyEarlyStopping(
            patience=self.patience, 
            verbose=True, 
            path=self.checkpoint_path
        )

        self.max_val = None

    def set_seed(self, seed):
        dgl.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)

    def train(self):
        train_set = self.task.dataset.train_set
        val_set = self.task.dataset.val_set
        
        epoch_iter = tqdm(range(self.max_epoch))
        
        for epoch in epoch_iter:
            self.model.train()
            train_loss_list = []
            
            for g, label in train_set:
                g = g.to(self.device)
                label = label.to(self.device)
                
                emb_dict = self.model(g)
                
                if 'state' in emb_dict:
                    prediction = emb_dict['state']
                else:
                    prediction = list(emb_dict.values())[0]

                if label.dim() == 1:
                    label = label.unsqueeze(1)
                
                # 直接计算 Loss (预测值和标签都是原始大数值)
                loss = self.loss_fn(prediction, label)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss_list.append(loss.item())
            
            avg_train_loss = sum(train_loss_list) / len(train_loss_list)
            
            # 验证集
            val_mae, val_rmse = self._evaluate_manual(val_set)
            
            epoch_iter.set_description(
                f"Epoch {epoch} | Train Loss: {avg_train_loss:.2f} | Val MAE: {val_mae:.2f}"
            )

            self.stopper(val_mae, self.model)
            if self.stopper.early_stop:
                print('Early Stopping!')
                break
        
        print(f"Loading best model from {self.checkpoint_path}...")
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path))
        
        test_mae, test_rmse = self._evaluate_manual(self.task.dataset.test_set)
        print(f"Test Result: MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")

    def _evaluate_manual(self, data_list):
        self.model.eval()
        mae_list = []
        rmse_list = []
        
        with torch.no_grad():
            for g, label in data_list:
                g = g.to(self.device)
                label = label.to(self.device)
                
                emb_dict = self.model(g)
                if 'state' in emb_dict:
                    prediction = emb_dict['state']
                else:
                    prediction = list(emb_dict.values())[0]
                
                if label.dim() == 1: 
                    label = label.unsqueeze(1)
                
                # 直接计算误差
                loss = F.l1_loss(prediction, label)
                rmse = torch.sqrt(F.mse_loss(prediction, label))
                
                mae_list.append(loss.item())
                rmse_list.append(rmse.item())
                
        return sum(mae_list) / len(mae_list), sum(rmse_list) / len(rmse_list)