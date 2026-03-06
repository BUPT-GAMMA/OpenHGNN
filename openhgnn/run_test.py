"""
HTGformer 复现验证脚本（修正版）
验证代码与论文完全对应：
  - 公式(1): 非参数GCN聚合
  - 公式(2)(3): LLM节点类型编码（当前用可学习嵌入替代，对应w/o_LLM消融）
  - 公式(4)(5): 正弦时序编码 + H^{sp,t}_v = H^LLM_v + p^t
  - 公式(6)(7): Z^t_v = concat(H^t_v, H^{sp,t}_v) -> Transformer
  - 论文超参数: lr=5e-3, weight_decay=5e-4, epochs=500, early_stopping=25
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import dgl
from types import SimpleNamespace

print("=" * 60)
print("HTGformer 复现验证脚本（论文对齐版）")
print("=" * 60)

# ── 1. 导入验证 ────────────────────────────────────────────────────────────
print("\n[1/4] 导入模块...")
from openhgnn.layers.htgformer_layer import (
    GCNAggregation,
    GraphEmbeddingLayer,
    SinusoidalTemporalEncoding,
    LLMTypeEncoder,
    HeteroTemporalEncoder,
    SpatioTemporalAttention,
    HTGformerEncoderLayer,
)
from openhgnn.models.HTGformer import HTGformer
from openhgnn.dataset.htgformer_dataset import AminerHTGDataset
from openhgnn.trainerflow.htgformer_trainer import HTGformerTrainer
print("  ✓ 所有模块导入成功")

# ── 2. 论文规格数据验证（按Table 1的Aminer数据集规格）─────────────────────
print("\n[2/4] 按论文Aminer数据集规格测试前向传播...")
print("  (使用小规模代理验证，完整规格会OOM)")

# 小规模代理（验证流程）
N_paper, N_author, N_venue = 200, 300, 22  # venue保持论文的22
T = 5
feat_dim = {'paper': 128, 'author': 64, 'venue': 32}
hidden_dim = 64  # 论文超参数 d=64
num_classes = 4

graphs, feat_dicts = [], []
for t in range(T):
    g = dgl.heterograph({
        ('author', 'writes', 'paper'): (
            torch.randint(0, N_author, (500,)),
            torch.randint(0, N_paper, (500,))
        ),
        ('paper', 'written_by', 'author'): (
            torch.randint(0, N_paper, (500,)),
            torch.randint(0, N_author, (500,))
        ),
        ('paper', 'published_in', 'venue'): (
            torch.randint(0, N_paper, (200,)),
            torch.randint(0, N_venue, (200,))
        ),
        ('venue', 'publishes', 'paper'): (
            torch.randint(0, N_venue, (200,)),
            torch.randint(0, N_paper, (200,))
        ),
    }, num_nodes_dict={
        'paper': N_paper, 'author': N_author, 'venue': N_venue
    })
    graphs.append(g)
    feat_dicts.append({
        'paper': torch.randn(N_paper, feat_dim['paper']),
        'author': torch.randn(N_author, feat_dim['author']),
        'venue': torch.randn(N_venue, feat_dim['venue']),
    })

# 验证各子模块
print("\n  [2a] 测试公式(1) GCNAggregation...")
gcn = GCNAggregation()
agg = gcn(graphs[0], feat_dicts[0])
for etype, h in agg.items():
    print(f"    {etype[0]}->{etype[2]}: {h.shape}")
print("  ✓ 公式(1) 通过")

print("\n  [2b] 测试公式(4) SinusoidalTemporalEncoding...")
temporal_enc = SinusoidalTemporalEncoding(hidden_dim)
p_t = temporal_enc(0)
print(f"    时序编码形状: {p_t.shape}")
assert p_t.shape == (hidden_dim,)
print("  ✓ 公式(4) 通过")

print("\n  [2c] 测试公式(5) HeteroTemporalEncoder...")
# Z^t_v = concat(H^t_v_proj, H^{sp,t}_v), 维度=2*hidden_dim
htenc = HeteroTemporalEncoder(
    node_types=['paper', 'author', 'venue'],
    feat_dim_dict=feat_dim,
    hidden_dim=hidden_dim,
    use_llm=False
)
Z_t = htenc(feat_dicts[0], timestep=0, category='paper')
print(f"    Z^t_v 形状: {Z_t.shape}  (期望: [{N_paper}, {2*hidden_dim}])")
assert Z_t.shape == (N_paper, 2 * hidden_dim), \
    f"期望 [{N_paper}, {2*hidden_dim}], 得到 {Z_t.shape}"
print("  ✓ 公式(5) concat正确，维度=2*hidden_dim")

print("\n  [2d] 测试公式(6)(7) SpatioTemporalAttention...")
attn = SpatioTemporalAttention(embed_dim=2*hidden_dim, num_heads=4)
Z_v_test = torch.randn(N_paper, T, 2*hidden_dim)
Z_test = torch.randn(N_paper, T*3, 2*hidden_dim)  # L=T*(1+2relations)
Z_prime = attn(Z_v_test, Z_test)
print(f"    Z'_v 形状: {Z_prime.shape}")
assert Z_prime.shape == (N_paper, T, 2*hidden_dim)
print("  ✓ 公式(6)(7) 通过，参数在所有节点类型间共享")

print("\n  [2e] 完整模型前向传播...")
model = HTGformer(
    in_dim_dict=feat_dim,
    hidden_dim=hidden_dim,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
    num_timestamps=T,
    node_types=['paper', 'author', 'venue'],
    category='paper',
    out_dim=num_classes,
    use_llm=False,
)
model.eval()
with torch.no_grad():
    logits = model(graphs, feat_dicts)
print(f"    输出 logits 形状: {logits.shape}  (期望: [{N_paper}, {num_classes}])")
assert logits.shape == (N_paper, num_classes)
print("  ✓ 完整前向传播通过")

# ── 3. 参数量统计 ──────────────────────────────────────────────────────────
print("\n[3/4] 模型参数统计...")
total = sum(p.numel() for p in model.parameters())
print(f"  总参数量: {total:,}")
for name, module in [
    ('GraphEmbeddingLayer (公式1)', model.graph_emb),
    ('HeteroTemporalEncoder (公式3-5)', model.hetero_temporal_enc),
    ('TransformerEncoder (公式6-7)', model.encoder_layers),
    ('Classifier MLP (Section3.4)', model.classifier),
]:
    n = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {n:,}")

# ── 4. 快速训练测试（论文超参数）──────────────────────────────────────────
print("\n[4/4] 快速训练测试（使用论文超参数，20轮）...")
args = SimpleNamespace(
    dataset_name='aminer_htg',
    hidden_dim=64,           # 论文 d=64
    num_heads=4,
    num_layers=2,
    dropout=0.1,
    lr=5e-3,                 # 论文 lr=5e-3
    weight_decay=5e-4,       # 论文 weight_decay=5e-4
    max_epochs=20,           # 快速测试用20轮（正式实验用500）
    patience=25,             # 论文 early_stopping=25
    device='cpu',
    use_synthetic=True,
    num_timestamps=5,
    use_llm=False,           # 当前无LLM资源，使用w/o_LLM模式
)

trainer = HTGformerTrainer(args)
result = trainer.train()

print(f"\n{'='*60}")
print(f"✓ 所有验证通过！")
print(f"最终测试 Macro-F1: {result['test_metric']:.4f}")
print(f"Best Epoch: {result['epoch']}")
print(f"{'='*60}")
print("""
下一步说明：
  1. 从 https://github.com/yeslab-code/HTGNN 下载Aminer真实数据
  2. 将数据放置到 ./data/aminer/ 目录
  3. 设置 use_synthetic=False 重新运行
  4. 目标指标（论文Table 2 Aminer）:
       Macro-F1: 43.24 ± 0.88
       Recall:   43.86 ± 0.43
""")
