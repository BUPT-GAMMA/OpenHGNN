import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..models import build_model

class EarlyStopping:
    def __init__(self, patience=10, save_path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return False

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)

    def load_checkpoint(self, model):
        if os.path.exists(self.save_path):
            model.load_state_dict(torch.load(self.save_path))
            print(f"Loaded checkpoint from {self.save_path}")
        else:
            print("Checkpoint file not found, skipping load.")

@register_flow("sehtgnn_trainer")
class SEHTGNNTrainer(BaseFlow):
    """
    Flow for SE-HTGNN model training (Temporal Link Prediction).
    """
    def __init__(self, args):
        super(SEHTGNNTrainer, self).__init__(args)
        
        self.args = args
        self.model_name = args.model
        self.device = args.device
        
        self.task = build_task(args)
        
        self.train_data = self.task.train_hg
        self.val_data = self.task.val_hg
        self.test_data = self.task.test_hg

        if isinstance(self.train_data, list) and len(self.train_data) > 0:
            hg_sample = self.train_data[0][0].to(self.device)
        else:
            hg_sample = self.task.get_graph().to(self.device)

        self.model = build_model(self.model_name).build_model_from_args(self.args, hg_sample)
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        ckpt_path = os.path.join(self.args.output_dir, 'checkpoint.pt') if hasattr(self.args, 'output_dir') else 'checkpoint.pt'
        self.stopper = EarlyStopping(patience=args.patience, save_path=ckpt_path)

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            self.model.train()
            total_loss = 0
            
            for i, (bg, target) in enumerate(self.train_data):
                bg = bg.to(self.device)
                
                embeddings = self.model(bg) 
                loss = self.loss_calculation(bg, target, embeddings)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_data) if len(self.train_data) > 0 else 0.0
            epoch_iter.set_description(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

            if len(self.val_data) > 0:
                val_metrics = self._test_step(split='val')
                
                if isinstance(val_metrics, dict):
                    if 'AUC' in val_metrics:
                        score = val_metrics['AUC']
                        print(f"Epoch {epoch} Val AUC: {score:.4f} | AP: {val_metrics['AP']:.4f}")
                    elif 'MAE' in val_metrics:
                        # 如果是 MAE，取负数
                        score = -val_metrics['MAE'] 
                        print(f"Epoch {epoch} Val MAE: {val_metrics['MAE']:.4f} | RMSE: {val_metrics['RMSE']:.4f}")
                    else:
                        score = 0
                else:
                    score = val_metrics
                    print(f"Epoch {epoch} Val Score: {score:.4f}")
                
                early_stop = self.stopper.step(score, self.model)
                if early_stop:
                    print("Early stopping triggered.")
                    break
        
        self.stopper.load_checkpoint(self.model)
        if len(self.test_data) > 0:
            test_metrics = self._test_step(split='test')
            print(f"Final Test Metrics: {test_metrics}")
        
        return dict(Final_Test_Metrics=test_metrics) if len(self.test_data) > 0 else {}

    def loss_calculation(self, bg, target, embedding):
        target_ntype = self.task.dataset.category 
        emb = embedding[target_ntype]
        
        if torch.is_tensor(target): # Node Regression
            label = target.to(self.device)
            pred = emb
            return F.l1_loss(pred, label)

        else: # Link Prediction
            positive_graph, negative_graph = target
            positive_graph = positive_graph.to(self.device)
            negative_graph = negative_graph.to(self.device)
            
            u_pos, v_pos = positive_graph.edges()
            pos_score = (emb[u_pos] * emb[v_pos]).sum(dim=1)
            
            u_neg, v_neg = negative_graph.edges()
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

                if torch.is_tensor(target):
                    label = target.to(self.device)
                    pred = emb
                    
                    mae = F.l1_loss(pred, label).item()
                    mse = F.mse_loss(pred, label).item()
                    
                    total_metrics['MAE'] = total_metrics.get('MAE', 0) + mae
                    total_metrics['RMSE'] = total_metrics.get('RMSE', 0) + (mse ** 0.5)
                    count += 1

                else:
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
                            auc = roc_auc_score(labels, preds)
                            ap = average_precision_score(labels, preds)
                            
                            total_metrics['AUC'] = total_metrics.get('AUC', 0) + auc
                            total_metrics['AP'] = total_metrics.get('AP', 0) + ap
                            count += 1
                        except ValueError:
                            pass
        
        return {k: v / count for k, v in total_metrics.items()} if count > 0 else {}