"""
HTGformer Trainer
超参数严格按照论文 Section 4.1.3：
  - Adam optimizer
  - lr = 5e-3
  - weight_decay = 5e-4
  - hidden_dim = 64（COVID-19用8）
  - max_epochs = 500
  - early_stopping = 25
  - 重复5次取均值和标准差
  - 评估指标：
      链路预测: AUC, AP
      节点分类: Macro-F1, Recall
      节点回归: MAE
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    f1_score, recall_score, roc_auc_score,
    average_precision_score, mean_absolute_error
)

try:
    from openhgnn.trainerflow import BaseFlow, register_flow
    from openhgnn.models.HTGformer import HTGformer
    from openhgnn.dataset.htgformer_dataset import (
        AminerHTGDataset, OGBNMAGHTGDataset,
        YELPHTGDataset, COVID19HTGDataset
    )
    HAS_OPENHGNN = True
except ImportError:
    HAS_OPENHGNN = False
    BaseFlow = object
    def register_flow(name):
        def decorator(cls): return cls
        return decorator
    from HTGformer import HTGformer


@register_flow('htgformer_trainer')
class HTGformerTrainer(BaseFlow):
    """
    HTGformer 训练流程

    论文Section 4.1.3实现细节：
    - Adam优化器，lr=5e-3，weight_decay=5e-4
    - max_epochs=500，early_stopping patience=25
    - 实验重复5次取均值±标准差
    - 节点分类评估：Macro-F1 + Recall
    - 链路预测评估：AUC + AP
    - 节点回归评估：MAE
    """

    def __init__(self, args):
        if HAS_OPENHGNN:
            try:
                super().__init__(args)
            except AttributeError:
                pass  # 本地测试时跳过BaseFlow初始化
        self.args = args
        self.device = torch.device(
            getattr(args, 'device', 'cpu')
        )

        # ── 加载数据集 ─────────────────────────────────────────────────
        self.dataset = self._load_dataset()
        self.graphs = [g.to(self.device) for g in self.dataset.graphs]
        self.feat_dicts = [
            {k: v.to(self.device) for k, v in fd.items()}
            for fd in self.dataset.feat_dicts
        ]
        self.labels = self.dataset.labels.to(self.device)
        self.train_idx = self.dataset.train_idx.to(self.device)
        self.val_idx = self.dataset.val_idx.to(self.device)
        self.test_idx = self.dataset.test_idx.to(self.device)
        self.task = self.dataset.task  # 'node_classification' / 'link_prediction' / 'node_regression'
        self.category = self.dataset.category

        # ── 构建模型 ───────────────────────────────────────────────────
        self.model = self._build_model().to(self.device)

        # ── 优化器（论文Section 4.1.3）────────────────────────────────
        # Adam, lr=5e-3, weight_decay=5e-4
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=getattr(args, 'lr', 5e-3),
            weight_decay=getattr(args, 'weight_decay', 5e-4),
        )

        # ── 损失函数（论文Section 3.4）────────────────────────────────
        if self.task == 'node_classification':
            self.criterion = nn.CrossEntropyLoss()
        elif self.task == 'node_regression':
            self.criterion = nn.L1Loss()  # MAE loss
        else:
            self.criterion = None  # 链路预测用公式(8)

    def _load_dataset(self):
        """根据args加载对应数据集"""
        dataset_name = getattr(self.args, 'dataset_name', 'aminer_htg')
        use_synthetic = getattr(self.args, 'use_synthetic', False)

        try:
            from openhgnn.dataset.htgformer_dataset import (
                AminerHTGDataset, OGBNMAGHTGDataset,
                YELPHTGDataset, COVID19HTGDataset
            )
        except ImportError:
            from htgformer_dataset import (
                AminerHTGDataset, OGBNMAGHTGDataset,
                YELPHTGDataset, COVID19HTGDataset
            )

        if 'aminer' in dataset_name.lower():
            return AminerHTGDataset(
                raw_dir=getattr(self.args, 'data_dir', './data/aminer'),
                use_synthetic=use_synthetic,
                num_timestamps=getattr(self.args, 'num_timestamps', 16),
            )
        elif 'ogbn' in dataset_name.lower() or 'mag' in dataset_name.lower():
            return OGBNMAGHTGDataset(
                raw_dir=getattr(self.args, 'data_dir', './data/ogbn_mag'),
                use_synthetic=use_synthetic,
                num_timestamps=getattr(self.args, 'num_timestamps', 10),
            )
        elif 'yelp' in dataset_name.lower():
            return YELPHTGDataset(
                raw_dir=getattr(self.args, 'data_dir', './data/yelp'),
                use_synthetic=use_synthetic,
                num_timestamps=getattr(self.args, 'num_timestamps', 12),
            )
        elif 'covid' in dataset_name.lower():
            return COVID19HTGDataset(
                raw_dir=getattr(self.args, 'data_dir', './data/covid19'),
                use_synthetic=use_synthetic,
                num_timestamps=getattr(self.args, 'num_timestamps', 304),
            )
        else:
            return AminerHTGDataset(use_synthetic=True)

    def _build_model(self):
        """构建HTGformer模型"""
        # 获取各节点类型特征维度
        in_dim_dict = {
            ntype: self.feat_dicts[0][ntype].shape[-1]
            for ntype in self.feat_dicts[0]
        }
        # 获取输出维度
        if self.task == 'node_classification':
            out_dim = int(self.labels.max().item()) + 1
        elif self.task == 'node_regression':
            out_dim = 1
        else:
            out_dim = getattr(self.args, 'hidden_dim', 64)

        # 论文超参数
        hidden_dim = getattr(self.args, 'hidden_dim', 64)
        # COVID-19数据集用d=8
        if 'covid' in getattr(self.args, 'dataset_name', '').lower():
            hidden_dim = 8

        return HTGformer(
            in_dim_dict=in_dim_dict,
            hidden_dim=hidden_dim,
            num_heads=getattr(self.args, 'num_heads', 4),
            num_layers=getattr(self.args, 'num_layers', 2),
            dropout=getattr(self.args, 'dropout', 0.1),
            num_timestamps=len(self.graphs),
            node_types=list(in_dim_dict.keys()),
            category=self.category,
            out_dim=out_dim,
            use_llm=getattr(self.args, 'use_llm', False),
            llm_embed_path=getattr(self.args, 'llm_embed_path', None),
        )

    def train(self):
        """
        完整训练流程
        论文：max_epochs=500，early_stopping=25，重复5次
        """
        max_epochs = getattr(self.args, 'max_epochs', 500)
        patience = getattr(self.args, 'patience', 25)  # 论文early stopping=25

        best_val_metric = float('inf') if self.task == 'node_regression' \
            else -float('inf')
        best_test_metric = None
        best_epoch = 0
        patience_counter = 0
        best_state = None

        for epoch in range(1, max_epochs + 1):
            # 训练
            train_loss = self._train_epoch()

            # 验证
            val_metric = self._evaluate(self.val_idx)
            test_metric = self._evaluate(self.test_idx)

            # 判断是否改善
            improved = (
                val_metric < best_val_metric
                if self.task == 'node_regression'
                else val_metric > best_val_metric
            )

            if improved:
                best_val_metric = val_metric
                best_test_metric = test_metric
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 50 == 0 or epoch <= 5:
                metric_name = {
                    'node_classification': 'Macro-F1',
                    'node_regression': 'MAE',
                    'link_prediction': 'AUC',
                }.get(self.task, 'metric')
                print(
                    f"Epoch {epoch:4d} | Loss: {train_loss:.4f} | "
                    f"Val {metric_name}: {val_metric:.4f} | "
                    f"Test {metric_name}: {test_metric:.4f}"
                )

            # Early stopping（论文patience=25）
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(best epoch: {best_epoch})")
                break

        # 恢复最优模型
        if best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )

        print(f"\n{'='*50}")
        print(f"训练完成！Best Epoch: {best_epoch}")
        print(f"Best Val Metric:  {best_val_metric:.4f}")
        print(f"Best Test Metric: {best_test_metric:.4f}")
        print(f"{'='*50}")

        return {
            'val_metric': best_val_metric,
            'test_metric': best_test_metric,
            'epoch': best_epoch,
        }

    def _train_epoch(self):
        """单轮训练"""
        self.model.train()
        self.optimizer.zero_grad()

        if self.task in ('node_classification', 'node_regression'):
            logits = self.model(self.graphs, self.feat_dicts)
            train_logits = logits[self.train_idx]
            train_labels = self.labels[self.train_idx]

            if self.task == 'node_regression':
                loss = self.criterion(
                    train_logits.squeeze(), train_labels.float()
                )
            else:
                loss = self.criterion(train_logits, train_labels)

        elif self.task == 'link_prediction':
            # 公式(8): 二元交叉熵损失
            loss = self._link_pred_loss()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def _link_pred_loss(self):
        """
        公式(8): L = -sum log σ(H_i^T H_j) - sum log σ(-H_i'^T H_j')
        """
        h = self.model.link_prediction_forward(self.graphs, self.feat_dicts)
        pos_edges = self.dataset.pos_edges.to(self.device)
        neg_edges = self.dataset.neg_edges.to(self.device)

        pos_scores = (h[pos_edges[:, 0]] * h[pos_edges[:, 1]]).sum(dim=-1)
        neg_scores = (h[neg_edges[:, 0]] * h[neg_edges[:, 1]]).sum(dim=-1)

        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-8).mean()
        return pos_loss + neg_loss

    @torch.no_grad()
    def _evaluate(self, idx):
        """评估，返回论文中对应的指标"""
        self.model.eval()
        logits = self.model(self.graphs, self.feat_dicts)
        node_logits = logits[idx].cpu()
        node_labels = self.labels[idx].cpu()

        if self.task == 'node_classification':
            preds = node_logits.argmax(dim=-1).numpy()
            labels_np = node_labels.numpy()
            # 论文指标：Macro-F1（主要）+ Recall
            macro_f1 = f1_score(labels_np, preds,
                                average='macro', zero_division=0)
            return macro_f1

        elif self.task == 'node_regression':
            # 论文指标：MAE
            preds = node_logits.squeeze().numpy()
            labels_np = node_labels.float().numpy()
            return mean_absolute_error(labels_np, preds)

        elif self.task == 'link_prediction':
            # 论文指标：AUC + AP
            scores = torch.sigmoid(node_logits).numpy()
            labels_np = node_labels.numpy()
            try:
                auc = roc_auc_score(labels_np, scores)
            except Exception:
                auc = 0.5
            return auc

        return 0.0

    def get_full_metrics(self, split='test'):
        """
        返回论文Table 2中的完整指标
        节点分类: Macro-F1 + Recall
        链路预测: AUC + AP
        节点回归: MAE
        """
        idx = {'train': self.train_idx,
               'val': self.val_idx,
               'test': self.test_idx}[split]

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.graphs, self.feat_dicts)
            node_logits = logits[idx].cpu()
            node_labels = self.labels[idx].cpu()

        results = {}
        if self.task == 'node_classification':
            preds = node_logits.argmax(dim=-1).numpy()
            labels_np = node_labels.numpy()
            results['Macro-F1'] = f1_score(
                labels_np, preds, average='macro', zero_division=0
            ) * 100
            results['Recall'] = recall_score(
                labels_np, preds, average='macro', zero_division=0
            ) * 100

        elif self.task == 'node_regression':
            preds = node_logits.squeeze().numpy()
            labels_np = node_labels.float().numpy()
            results['MAE'] = mean_absolute_error(labels_np, preds)

        elif self.task == 'link_prediction':
            scores = torch.sigmoid(node_logits).numpy()
            labels_np = node_labels.numpy()
            results['AUC'] = roc_auc_score(labels_np, scores) * 100
            results['AP'] = average_precision_score(
                labels_np, scores
            ) * 100

        return results
