"""
HTGformer 快速验证脚本
======================
运行此脚本以验证：
  1. 所有模块能正确导入
  2. 前向传播形状正确
  3. 训练流程可运行（50轮 on 合成数据）
  4. 打印各模块参数量
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import dgl
from types import SimpleNamespace

print("="*60)
print("HTGformer 复现验证脚本")
print("="*60)

# ── 1. 导入验证 ────────────────────────────────────────────────────────────
print("\n[1/4] 导入模块...")
from openhgnn.layers.htgformer_layer import (
    HeteroFeatureProjection,
    GCNAggregation,
    GraphEmbeddingLayer,
    TemporalTypeEmbedding,
    HTGformerEncoderLayer,
)
from openhgnn.models.HTGformer import HTGformer
from openhgnn.dataset.htgformer_dataset import AminerHTGDataset
from openhgnn.trainerflow.htgformer_trainer import HTGformerTrainer

print("  ✓ 所有模块导入成功")

# ── 2. 前向传播测试 ────────────────────────────────────────────────────────
print("\n[2/4] 前向传播形状验证...")

T = 5
num_authors, num_papers, num_venues = 100, 200, 20
feat_dim = {'author': 128, 'paper': 256, 'venue': 64}
hidden_dim = 64
num_classes = 4

# 构建简单测试图
graphs = []
feat_dicts = []
for t in range(T):
    g = dgl.heterograph({
        ('author', 'writes', 'paper'): (
            torch.randint(0, num_authors, (300,)),
            torch.randint(0, num_papers, (300,))
        ),
        ('paper', 'written_by', 'author'): (
            torch.randint(0, num_papers, (300,)),
            torch.randint(0, num_authors, (300,))
        ),
        ('paper', 'published_in', 'venue'): (
            torch.randint(0, num_papers, (100,)),
            torch.randint(0, num_venues, (100,))
        ),
        ('venue', 'publishes', 'paper'): (
            torch.randint(0, num_venues, (100,)),
            torch.randint(0, num_papers, (100,))
        ),
    }, num_nodes_dict={
        'author': num_authors,
        'paper': num_papers,
        'venue': num_venues,
    })
    graphs.append(g)
    feat_dicts.append({
        'author': torch.randn(num_authors, feat_dim['author']),
        'paper': torch.randn(num_papers, feat_dim['paper']),
        'venue': torch.randn(num_venues, feat_dim['venue']),
    })

# 构建模型
model = HTGformer(
    in_dim_dict=feat_dim,
    hidden_dim=hidden_dim,
    num_heads=4,
    num_layers=2,
    num_gcn_layers=1,
    dropout=0.1,
    num_timestamps=T,
    node_types=['author', 'paper', 'venue'],
    category='paper',
    out_dim=num_classes,
)

# 前向传播
model.eval()
with torch.no_grad():
    logits = model(graphs, feat_dicts)

print(f"  输入: T={T} 时间切片，"
      f"author={num_authors}, paper={num_papers}, venue={num_venues}")
print(f"  输出 logits 形状: {logits.shape}")
assert logits.shape == (num_papers, num_classes), \
    f"期望 [{num_papers}, {num_classes}], 得到 {logits.shape}"
print("  ✓ 前向传播形状正确")

# ── 3. 参数量统计 ─────────────────────────────────────────────────────────
print("\n[3/4] 模型参数统计...")
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  总参数量:     {total:>10,}")
print(f"  可训练参数量: {trainable:>10,}")

# 各子模块参数量
for name, module in [
    ('GraphEmbeddingLayer', model.graph_embedding),
    ('TemporalTypeEmb', model.temporal_type_emb),
    ('TransformerEncoder', model.encoder_layers),
    ('Classifier', model.classifier),
]:
    n = sum(p.numel() for p in module.parameters())
    print(f"  {name:25s}: {n:>8,}")

# ── 4. 快速训练测试 ────────────────────────────────────────────────────────
print("\n[4/4] 快速训练测试（20轮）...")
from openhgnn.trainerflow.htgformer_trainer import HTGformerTrainer

args = SimpleNamespace(
    dataset_name='aminer_htg',
    hidden_dim=32,       # 减小以加快测试
    num_heads=2,
    num_layers=1,
    num_gcn_layers=1,
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=20,
    patience=10,
    lr_decay_step=10,
    lr_decay_gamma=0.5,
    device='cpu',
    full_graph=True,
    batch_size=256,
    use_synthetic=True,
    num_timestamps=5,    # 减少时间步
)

trainer = HTGformerTrainer(args)
result = trainer.train()

print(f"\n{'='*60}")
print(f"✓ 所有验证通过！")
print(f"最终测试 Macro-F1: {result['test_metric']:.4f}")
print(f"最佳 Epoch: {result['epoch']}")
print(f"{'='*60}")
print("\n提示：合成数据的指标仅用于验证流程正确性，")
print("真实性能请在 Aminer/COVID-19 等真实数据集上评估。")
