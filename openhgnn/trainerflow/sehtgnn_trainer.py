import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import dgl
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, mean_squared_error, mean_absolute_error
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..models import build_model

class EarlyStopping:
    def __init__(self, patience=10, save_path='checkpoint.pt', mode='max'):
        """
        mode: 'max' for AUC/F1 (higher is better), 'min' for RMSE/Loss (lower is better)
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path
        self.mode = mode

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        else:
            improved = (score > self.best_score) if self.mode == 'max' else (score < self.best_score)
            
            if improved:
                self.best_score = score
                self.save_checkpoint(model)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter % 5 == 0:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    return True
        return False

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)

    def load_checkpoint(self, model):
        if os.path.exists(self.save_path):
            model.load_state_dict(torch.load(self.save_path))
            print(f"Loaded checkpoint from {self.save_path}")
        else:
            print(f"Checkpoint file not found: {self.save_path}, skipping load.")

@register_flow("sehtgnn_trainer")
class SEHTGNNTrainer(BaseFlow):
    def __init__(self, args):
        super(SEHTGNNTrainer, self).__init__(args)
        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.dataset_name = args.dataset
        self.task = build_task(args)
        
        if self.dataset_name == 'sehtgnn_covid':
            self.train_data = self.task.train_hg
            self.val_data = self.task.val_hg
            self.test_data = self.task.test_hg
            self.args.out_dim = 1
            print(f"[Trainer] COVID detected. Setting model out_dim = {self.args.out_dim}")
        elif self.dataset_name == 'sehtgnn_yelp':
            self.train_data = self.task.train_hg
            self.val_data = self.task.val_hg
            self.test_data = self.task.test_hg
            if hasattr(self.task.dataset, 'num_classes'):
                self.args.out_dim = self.task.dataset.num_classes
        else: 
            self.train_data = self.task.train_hg
            self.val_data = self.task.val_hg
            self.test_data = self.task.test_hg

        hg_sample = None
        if isinstance(self.train_data, list) and len(self.train_data) > 0:
            item = self.train_data[0]
            hg_sample = item[0] if isinstance(item, tuple) else item
            hg_sample = hg_sample.to(self.device)
        else:
            g = self.task.get_graph()
            if g is not None: hg_sample = g.to(self.device)

        if hg_sample is None: raise ValueError("Dataset empty.")

        self.model = build_model(self.model_name).build_model_from_args(self.args, hg_sample)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        ckpt_path = os.path.join(self.args.output_dir, 'checkpoint.pt') if hasattr(self.args, 'output_dir') else 'checkpoint.pt'
        es_mode = 'min' if self.dataset_name == 'sehtgnn_covid' else 'max'
        self.stopper = EarlyStopping(patience=args.patience, save_path=ckpt_path, mode=es_mode)

    def _full_train_step(self):
        """
        Full Batch Training Step
        """
        self.model.train()
        total_loss = 0
        
        for i, (bg, target) in enumerate(self.train_data):
            bg = bg.to(self.device)
            embeddings = self.model(bg) 
            
            loss = self.loss_calculation(bg, target, embeddings)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.train_data) if len(self.train_data) > 0 else 0.0

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            train_loss = self._full_train_step()
            
            epoch_iter.set_description(f"Epoch {epoch} | Loss: {train_loss:.4f}")

            if len(self.val_data) > 0:
                val_metrics = self._test_step(split='val')
                early_stop_score = 0
                
                if self.dataset_name == 'sehtgnn_covid':
                    early_stop_score = val_metrics.get('RMSE', float('inf'))
                    print(f"Epoch {epoch} Val MAE: {val_metrics.get('MAE', 0):.4f} | RMSE: {early_stop_score:.4f}")
                elif self.dataset_name == 'sehtgnn_yelp':
                    early_stop_score = val_metrics.get('Macro-F1', 0)
                    print(f"Epoch {epoch} Val Macro-F1: {early_stop_score:.4f}")
                else:
                    early_stop_score = val_metrics.get('AUC', 0)
                    print(f"Epoch {epoch} Val AUC: {early_stop_score:.4f}")
                
                if self.stopper.step(early_stop_score, self.model):
                    print("Early stopping triggered.")
                    break
        
        print("Loading best checkpoint for testing...")
        self.stopper.load_checkpoint(self.model)
        test_metrics = {}
        if len(self.test_data) > 0:
            test_metrics = self._test_step(split='test')
            print(f"Final Test Metrics: {test_metrics}")
        return dict(Final_Test_Metrics=test_metrics)

    def loss_calculation(self, bg, target, embedding):
        target_ntype = self.task.dataset.category 
        emb = embedding[target_ntype]
        
        if self.dataset_name == 'sehtgnn_covid':
            if isinstance(target, tuple): target = target[0]
            label = target.to(self.device)
            if emb.shape != label.shape: label = label.view(emb.shape)
            
            return F.l1_loss(emb, label)
        
        elif self.dataset_name == 'sehtgnn_yelp':
            target = target.to(self.device)
            labels = target.nodes[target_ntype].data['y']
            mask = target.nodes[target_ntype].data['mask'].bool()
            return F.cross_entropy(emb[mask], labels[mask].long())
        else:
            if isinstance(target, tuple):
                pos_g, neg_g = target
                pos_g = pos_g.to(self.device)
                neg_g = neg_g.to(self.device)
                u_pos, v_pos = pos_g.edges()
                pos_score = (emb[u_pos] * emb[v_pos]).sum(dim=1)
                u_neg, v_neg = neg_g.edges()
                neg_score = (emb[u_neg] * emb[v_neg]).sum(dim=1)
                return -torch.mean(F.logsigmoid(pos_score)) - torch.mean(F.logsigmoid(-neg_score))

    def _test_step(self, split=None):
        self.model.eval()
        data_source = self.val_data if split == 'val' else self.test_data
        target_ntype = self.task.dataset.category
        total_metrics = {}
        count = 0
        
        with torch.no_grad():
            for i, (bg, target) in enumerate(data_source):
                bg = bg.to(self.device)
                embeddings = self.model(bg)
                emb = embeddings[target_ntype]

                if self.dataset_name == 'sehtgnn_covid':
                    label = target.to(self.device)
                    if emb.shape != label.shape: label = label.view(emb.shape)
                    
                    y_pred_real = emb.cpu().numpy()
                    y_true_real = label.cpu().numpy()

                    y_pred_real = np.maximum(y_pred_real, 0)
                    
                    mae = mean_absolute_error(y_true_real, y_pred_real)
                    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
                    
                    total_metrics['MAE'] = total_metrics.get('MAE', 0) + mae
                    total_metrics['RMSE'] = total_metrics.get('RMSE', 0) + rmse
                    count += 1

                elif self.dataset_name == 'sehtgnn_yelp':
                    target = target.to(self.device)
                    y_true = target.nodes[target_ntype].data['y']
                    mask = target.nodes[target_ntype].data['mask'].bool()
                    logits = emb[mask]
                    y_true = y_true[mask].cpu().numpy()
                    y_pred = logits.argmax(dim=1).cpu().numpy()
                    
                    total_metrics['Macro-F1'] = total_metrics.get('Macro-F1', 0) + f1_score(y_true, y_pred, average='macro')
                    total_metrics['Recall'] = total_metrics.get('Recall', 0) + recall_score(y_true, y_pred, average='macro')
                    count += 1

                else:
                    if isinstance(target, tuple):
                        pos_g, neg_g = target
                        pos_g = pos_g.to(self.device)
                        neg_g = neg_g.to(self.device)
                        
                        u_pos, v_pos = pos_g.edges()
                        pos_score = (emb[u_pos] * emb[v_pos]).sum(dim=1).cpu().numpy()
                        u_neg, v_neg = neg_g.edges()
                        neg_score = (emb[u_neg] * emb[v_neg]).sum(dim=1).cpu().numpy()
                        
                        labels = [1] * len(pos_score) + [0] * len(neg_score)
                        preds = list(pos_score) + list(neg_score)
                        
                        if len(labels) > 0:
                            try:
                                total_metrics['AUC'] = total_metrics.get('AUC', 0) + roc_auc_score(labels, preds)
                                total_metrics['AP'] = total_metrics.get('AP', 0) + average_precision_score(labels, preds)
                                count += 1
                            except ValueError: pass

        return {k: v / count for k, v in total_metrics.items()} if count > 0 else {}