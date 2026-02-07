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
from ..tasks import build_task

class MyEarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

@register_flow("covid_node_regression")
class COVIDTrainer(BaseFlow):
    def __init__(self, args):
        self.args = args
        self.model_name = args.model
        
        if hasattr(args, 'device'):
            self.device = torch.device(args.device) if isinstance(args.device, str) else args.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.set_seed(0)

        self.task = build_task(args)
        
        self.args.dropout = getattr(args, 'dropout', 0.0)
        self.lr = getattr(args, 'lr', 5e-3) 
        self.weight_decay = getattr(args, 'weight_decay', 1e-4)
        self.patience = getattr(args, 'patience', 5)
        self.max_epoch = getattr(args, 'max_epoch', 500)

        if not hasattr(self, 'output_dir'):
            self.output_dir = getattr(args, 'output_dir', './output')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.train_data = self._to_device(self.task.dataset.train_set)
        self.val_data = self._to_device(self.task.dataset.val_set)
        self.test_data = self._to_device(self.task.dataset.test_set)

        if isinstance(args.model, str):
            print(f"[COVIDTrainer] Building model {self.model_name} manually...")
            self.set_seed(0) 
            model_cls = build_model(self.model_name)
            
            hg_sample, _ = self.train_data[0]
            
            self.model = model_cls.build_model_from_args(args, hg_sample).to(self.device)
            args.model = self.model
        else:
            self.model = args.model.to(self.device)

        self.loss_fn = F.l1_loss
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        model_name_str = getattr(args, 'model_name', 'SEHTGNN')
        self.checkpoint_path = os.path.join(self.output_dir, f"{model_name_str}_best.pt")
        
        self.stopper = MyEarlyStopping(
            patience=self.patience, 
            verbose=True, 
            path=self.checkpoint_path
        )

    def _to_device(self, data_list):
        new_list = []
        for g, label in data_list:
            new_list.append((g.to(self.device), label.to(self.device)))
        return new_list

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
        train_set = self.train_data
        val_set = self.val_data
        
        epoch_iter = tqdm(range(self.max_epoch))
        
        for epoch in epoch_iter:
            self.model.train()
            train_loss_list = []
            train_rmse_list = []
            
            for g, label in train_set:
                emb_dict = self.model(g)
                
                if isinstance(emb_dict, dict):
                    if 'state' in emb_dict:
                        prediction = emb_dict['state']
                    else:
                        prediction = list(emb_dict.values())[0]
                else:
                    prediction = emb_dict

                if label.dim() == 1:
                    label = label.unsqueeze(1)
                
                loss = self.loss_fn(prediction, label)
                rmse = torch.sqrt(F.mse_loss(prediction, label))
                
                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                self.optimizer.step()
                
                train_loss_list.append(loss.item())
                train_rmse_list.append(rmse.item())
            
            avg_train_loss = sum(train_loss_list) / len(train_loss_list)
            
            val_mae, val_rmse = self._evaluate_manual(val_set)
            
            epoch_iter.set_description(
                f"Epoch {epoch} | Train MAE: {avg_train_loss:.2f} | Val MAE: {val_mae:.2f}"
            )

            self.stopper(val_mae, self.model)
            
            if self.stopper.early_stop:
                print('Early Stopping!')
                break
        
        print(f"Loading best model from {self.checkpoint_path}...")
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path))
        
        test_mae, test_rmse = self._evaluate_manual(self.test_data)
        print(f"Test Result: MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")

    def _evaluate_manual(self, data_list):
        self.model.eval()
        mae_list = []
        rmse_list = []
        
        with torch.no_grad():
            for g, label in data_list:
                
                emb_dict = self.model(g)
                if isinstance(emb_dict, dict):
                    if 'state' in emb_dict:
                        prediction = emb_dict['state']
                    else:
                        prediction = list(emb_dict.values())[0]
                else:
                    prediction = emb_dict
                
                if label.dim() == 1: 
                    label = label.unsqueeze(1)
                
                loss = F.l1_loss(prediction, label)
                rmse = torch.sqrt(F.mse_loss(prediction, label))
                
                mae_list.append(loss.item())
                rmse_list.append(rmse.item())
                
        return sum(mae_list) / len(mae_list), sum(rmse_list) / len(rmse_list)