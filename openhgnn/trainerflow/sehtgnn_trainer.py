import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import dgl
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score
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
            print(f"Checkpoint file not found: {self.save_path}, skipping load.")

@register_flow("sehtgnn_trainer")
class SEHTGNNTrainer(BaseFlow):
    """
    Flow for SE-HTGNN model training.
    Supports both Link Prediction (Aminer/OGB) and Node Classification (Yelp).
    """
    def __init__(self, args):
        super(SEHTGNNTrainer, self).__init__(args)
        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.dataset_name = args.dataset
        
        # Build Task
        self.task = build_task(args)
        
        # 显式根据数据集名称获取数据
        if self.dataset_name == 'sehtgnn_yelp':
            # Yelp (Node Classification)
            # 在 Task 初始化时我们已经把 list 赋给了这些属性
            self.train_data = self.task.train_hg
            self.val_data = self.task.val_hg
            self.test_data = self.task.test_hg
            
            # 强制设置输出维度为类别数 (3类)
            if hasattr(self.task.dataset, 'num_classes'):
                self.args.out_dim = self.task.dataset.num_classes
                print(f"[Trainer] Yelp detected. Setting model out_dim = {self.args.out_dim}")
                
        else:
            # Aminer / OGB (Link Prediction)
            self.train_data = self.task.train_hg
            self.val_data = self.task.val_hg
            self.test_data = self.task.test_hg

        # 采样图用于模型初始化
        if isinstance(self.train_data, list) and len(self.train_data) > 0:
            hg_sample = self.train_data[0][0].to(self.device)
        else:
            hg_sample = self.task.get_graph().to(self.device)

        # Build Model
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
                
                # 计算 Loss
                loss = self.loss_calculation(bg, target, embeddings)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_data) if len(self.train_data) > 0 else 0.0
            epoch_iter.set_description(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

            # Validation
            if len(self.val_data) > 0:
                val_metrics = self._test_step(split='val')
                
                # 根据数据集打印不同的指标
                if self.dataset_name == 'sehtgnn_yelp':
                    score = val_metrics.get('Macro-F1', 0)
                    print(f"Epoch {epoch} Val Macro-F1: {score:.4f} | Recall: {val_metrics.get('Recall', 0):.4f}")
                else:
                    # Aminer / OGB
                    score = val_metrics.get('AUC', 0)
                    print(f"Epoch {epoch} Val AUC: {score:.4f} | AP: {val_metrics.get('AP', 0):.4f}")
                
                early_stop = self.stopper.step(score, self.model)
                if early_stop:
                    print("Early stopping triggered.")
                    break
        
        # Test
        self.stopper.load_checkpoint(self.model)
        if len(self.test_data) > 0:
            test_metrics = self._test_step(split='test')
            print(f"Final Test Metrics: {test_metrics}")
        
        return dict(Final_Test_Metrics=test_metrics) if len(self.test_data) > 0 else {}

    def loss_calculation(self, bg, target, embedding):
        target_ntype = self.task.dataset.category 
        emb = embedding[target_ntype]
        
        if self.dataset_name == 'sehtgnn_yelp':
            target = target.to(self.device)
            labels = target.nodes[target_ntype].data['y']
            mask = target.nodes[target_ntype].data['mask'].bool()
            
            preds = emb[mask]
            labels = labels[mask]
            
            return F.cross_entropy(preds, labels.long())

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
            else:
                 # Fallback for Node Regression (if any)
                if torch.is_tensor(target):
                    label = target.to(self.device)
                    return F.l1_loss(emb, label)
                else:
                    raise ValueError(f"Unknown target type for dataset {self.dataset_name}: {type(target)}")

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

                if self.dataset_name == 'sehtgnn_yelp':
                    target = target.to(self.device)
                    y_true = target.nodes[target_ntype].data['y']
                    mask = target.nodes[target_ntype].data['mask'].bool()
                    
                    logits = emb[mask] # [N, 3]
                    y_true = y_true[mask].cpu().numpy() # [N]
                    
                    y_pred = logits.argmax(dim=1).cpu().numpy()
                    
                    macro_f1 = f1_score(y_true, y_pred, average='macro')
                    recall = recall_score(y_true, y_pred, average='macro')
                    
                    total_metrics['Macro-F1'] = total_metrics.get('Macro-F1', 0) + macro_f1
                    total_metrics['Recall'] = total_metrics.get('Recall', 0) + recall
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
                                auc = roc_auc_score(labels, preds)
                                ap = average_precision_score(labels, preds)
                                total_metrics['AUC'] = total_metrics.get('AUC', 0) + auc
                                total_metrics['AP'] = total_metrics.get('AP', 0) + ap
                                count += 1
                            except ValueError: pass
                    
                    # Node Regression Fallback
                    elif torch.is_tensor(target):
                        label = target.to(self.device)
                        mae = F.l1_loss(emb, label).item()
                        mse = F.mse_loss(emb, label).item()
                        total_metrics['MAE'] = total_metrics.get('MAE', 0) + mae
                        total_metrics['RMSE'] = total_metrics.get('RMSE', 0) + (mse ** 0.5)
                        count += 1

        return {k: v / count for k, v in total_metrics.items()} if count > 0 else {}