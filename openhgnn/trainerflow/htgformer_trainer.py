"""
HTGformer TrainerFlow
======================
Paper: HTGformer: Heterogeneous Temporal Graph Transformer (SIGIR 2025)

按照 OpenHGNN 规范实现 trainerflow：
  - 继承 BaseFlow
  - 注册为 @register_flow('htgformer_trainer')
  - 实现 __init__, train, _train_step, _test_step

训练流程：
  1. 初始化模型、优化器、数据集
  2. 全图或 mini-batch 训练
  3. 验证集上早停
  4. 测试集上报告最终性能

支持任务：
  - node_classification (分类: Aminer, OGBN-MAG)
  - node_regression     (回归: COVID-19)
"""

import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# OpenHGNN 接口（部署时启用）
# from openhgnn.trainerflow import BaseFlow, register_flow

# 本地导入（开发调试时使用）
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openhgnn.models.HTGformer import HTGformer
from openhgnn.dataset.htgformer_dataset import AminerHTGDataset, COVID19HTGDataset
from openhgnn.sampler.htgformer_sampler import HTGformerDataLoader

logger = logging.getLogger('HTGformer')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# ──────────────────────────────────────────────────────────────────────────────
# Trainerflow
# ──────────────────────────────────────────────────────────────────────────────
# @register_flow('htgformer_trainer')   # 部署到 OpenHGNN 时取消注释
class HTGformerTrainer:
    """
    HTGformer 训练流程。

    与 OpenHGNN BaseFlow 兼容，可直接接入框架。
    主要接口：
      - train(): 完整训练循环，返回最佳验证指标
      - _train_step(batch): 单步训练
      - _test_step(split): 测试/验证步骤

    Args:
        args: 配置参数对象，需包含：
            - dataset_name: 数据集名称
            - hidden_dim, num_heads, num_layers, dropout
            - lr, weight_decay, max_epochs, patience
            - device: 'cuda:0' 或 'cpu'
            - task: 'node_classification' 或 'node_regression'
            - batch_size, full_graph
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            getattr(args, 'device', 'cpu')
        )
        self.task = getattr(args, 'task', 'node_classification')

        # ── 1. 加载数据集 ──────────────────────────────────────────────────
        self.dataset = self._load_dataset()
        self._setup_args_from_dataset()

        # ── 2. 构建模型 ────────────────────────────────────────────────────
        self.model = HTGformer(
            in_dim_dict=args.in_dim_dict,
            hidden_dim=getattr(args, 'hidden_dim', 64),
            num_heads=getattr(args, 'num_heads', 4),
            num_layers=getattr(args, 'num_layers', 2),
            num_gcn_layers=getattr(args, 'num_gcn_layers', 1),
            dropout=getattr(args, 'dropout', 0.1),
            num_timestamps=args.num_timestamps,
            node_types=args.node_types,
            category=args.category,
            out_dim=args.out_dim,
        ).to(self.device)

        logger.info(f"Model parameters: "
                    f"{sum(p.numel() for p in self.model.parameters()):,}")

        # ── 3. 优化器 ──────────────────────────────────────────────────────
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=getattr(args, 'lr', 1e-3),
            weight_decay=getattr(args, 'weight_decay', 1e-4),
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=getattr(args, 'lr_decay_step', 20),
            gamma=getattr(args, 'lr_decay_gamma', 0.5),
        )

        # ── 4. 损失函数 ────────────────────────────────────────────────────
        if self.task == 'node_classification':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()

        # ── 5. DataLoader ──────────────────────────────────────────────────
        full_graph = getattr(args, 'full_graph', True)
        batch_size = getattr(args, 'batch_size', 256)

        self.train_loader = HTGformerDataLoader(
            self.dataset, split='train',
            batch_size=batch_size, full_graph=full_graph, shuffle=True
        )
        self.val_loader = HTGformerDataLoader(
            self.dataset, split='val',
            batch_size=batch_size, full_graph=full_graph, shuffle=False
        )
        self.test_loader = HTGformerDataLoader(
            self.dataset, split='test',
            batch_size=batch_size, full_graph=full_graph, shuffle=False
        )

        # 将图和特征移至设备
        self.graphs = [g.to(self.device) for g in self.dataset.graphs]
        self.feat_dicts = [
            {k: v.to(self.device) for k, v in fd.items()}
            for fd in self.dataset.feat_dicts
        ]
        self.labels = self.dataset.labels.to(self.device)

        # 最优模型保存
        self.best_val_metric = -float('inf')
        self.best_state = None

    def _load_dataset(self):
        """根据 args.dataset_name 加载数据集"""
        name = getattr(self.args, 'dataset_name', 'aminer_htg')
        if name == 'aminer_htg':
            return AminerHTGDataset(
                num_timestamps=getattr(self.args, 'num_timestamps', 10),
                use_synthetic=getattr(self.args, 'use_synthetic', True),
            )
        elif name == 'covid19_htg':
            return COVID19HTGDataset(
                num_timestamps=getattr(self.args, 'num_timestamps', 30),
                use_synthetic=getattr(self.args, 'use_synthetic', True),
            )
        else:
            raise ValueError(f"Unknown dataset: {name}")

    def _setup_args_from_dataset(self):
        """从数据集元信息填充 args"""
        meta = self.dataset.meta
        self.args.in_dim_dict = meta['in_dim_dict']
        self.args.num_timestamps = meta['num_timestamps']
        self.args.node_types = meta['node_types']
        self.args.category = meta['category']
        self.args.out_dim = meta['num_classes']
        self.args.task = meta.get('task', 'node_classification')
        self.task = self.args.task

    # ──────────────────────────────────────────────────────────────────────
    # train(): 完整训练循环
    # ──────────────────────────────────────────────────────────────────────
    def train(self):
        """
        完整训练循环，含早停机制。

        Returns:
            result (dict): {
                'val_metric': best validation metric,
                'test_metric': test metric at best epoch,
                'epoch': best epoch number,
            }
        """
        max_epochs = getattr(self.args, 'max_epochs', 200)
        patience = getattr(self.args, 'patience', 30)
        early_stop_counter = 0

        best_test_metric = 0.0
        best_epoch = 0

        logger.info(f"Start training HTGformer on {self.args.dataset_name}")
        logger.info(f"Task: {self.task}, Category: {self.args.category}")
        logger.info(f"Timestamps: {self.args.num_timestamps}, "
                    f"Node types: {self.args.node_types}")

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()

            # ── Train ──────────────────────────────────────────────────────
            train_loss = self._train_epoch()

            # ── Validate ───────────────────────────────────────────────────
            val_metric = self._test_step('val')

            # ── Learning Rate Decay ─────────────────────────────────────────
            self.scheduler.step()

            # ── Log ────────────────────────────────────────────────────────
            elapsed = time.time() - t0
            metric_name = 'Macro-F1' if self.task == 'node_classification' \
                          else 'MAE'
            logger.info(
                f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
                f"Val {metric_name}: {val_metric:.4f} | "
                f"Time: {elapsed:.2f}s"
            )

            # ── Early Stopping ──────────────────────────────────────────────
            improved = (
                val_metric > self.best_val_metric
                if self.task == 'node_classification'
                else val_metric < self.best_val_metric
            )
            if improved or self.best_val_metric == -float('inf'):
                self.best_val_metric = val_metric
                self.best_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                early_stop_counter = 0
                best_epoch = epoch
                # 同步记录 test 指标
                best_test_metric = self._test_step('test')
                logger.info(
                    f"  ✓ New best! Test {metric_name}: {best_test_metric:.4f}"
                )
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best epoch: {best_epoch}"
                    )
                    break

        # 加载最优模型
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        logger.info(
            f"\n{'='*50}\n"
            f"Training complete.\n"
            f"Best Epoch: {best_epoch}\n"
            f"Best Val {metric_name}: {self.best_val_metric:.4f}\n"
            f"Best Test {metric_name}: {best_test_metric:.4f}\n"
            f"{'='*50}"
        )

        return {
            'val_metric': self.best_val_metric,
            'test_metric': best_test_metric,
            'epoch': best_epoch,
            'metric': best_test_metric,   # OpenHGNN 兼容接口
        }

    # ──────────────────────────────────────────────────────────────────────
    # _train_epoch(): 单轮训练
    # ──────────────────────────────────────────────────────────────────────
    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _train_step(self, batch):
        """
        单步训练。
        对应 OpenHGNN 的 _full_train_step 或 _mini_train_step。
        """
        self.optimizer.zero_grad()

        target_ids = batch['target_ids']
        labels = batch['labels']

        # 前向传播
        logits = self.model(
            self.graphs,
            self.feat_dicts,
            target_node_ids=target_ids,
        )

        # 计算损失
        if self.task == 'node_classification':
            loss = self.loss_fn(logits, labels.long())
        else:
            loss = self.loss_fn(logits.squeeze(-1), labels.float())

        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ──────────────────────────────────────────────────────────────────────
    # _test_step(): 验证/测试
    # ──────────────────────────────────────────────────────────────────────
    def _test_step(self, split='val'):
        """
        评估模型在指定数据集划分上的性能。

        Returns:
            metric (float): 分类返回 Macro-F1；回归返回 -MAE（用于统一比较，越大越好）
        """
        self.model.eval()
        loader = self.val_loader if split == 'val' else self.test_loader

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                target_ids = batch['target_ids']
                labels = batch['labels']

                logits = self.model(
                    self.graphs,
                    self.feat_dicts,
                    target_node_ids=target_ids,
                )

                if self.task == 'node_classification':
                    preds = logits.argmax(dim=-1).cpu().numpy()
                else:
                    preds = logits.squeeze(-1).cpu().numpy()

                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        if self.task == 'node_classification':
            metric = f1_score(all_labels, all_preds,
                              average='macro', zero_division=0)
        else:
            # 回归: MAE，返回负值以便统一"越大越好"
            metric = -float(np.mean(np.abs(all_preds - all_labels)))

        return float(metric)


# ──────────────────────────────────────────────────────────────────────────────
# OpenHGNN 接口适配（与框架集成时使用）
# ──────────────────────────────────────────────────────────────────────────────
class HTGformerFlow:
    """
    与 OpenHGNN BaseFlow 兼容的包装类。
    实现 OpenHGNN 要求的 train() 接口。
    """

    def __init__(self, args):
        self.trainer = HTGformerTrainer(args)

    def train(self):
        return self.trainer.train()


# ──────────────────────────────────────────────────────────────────────────────
# 快速测试入口
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from types import SimpleNamespace

    args = SimpleNamespace(
        dataset_name='aminer_htg',
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        num_gcn_layers=1,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=50,
        patience=20,
        lr_decay_step=15,
        lr_decay_gamma=0.5,
        device='cpu',
        full_graph=True,
        batch_size=256,
        use_synthetic=True,
        num_timestamps=10,
    )

    trainer = HTGformerTrainer(args)
    result = trainer.train()
    print(f"\nFinal Result: {result}")
