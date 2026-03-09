"""
HTGformer 完整模型
严格对应论文 HTGformer (SIGIR 2025) Section 3

整体流程（对应论文Figure 1(a)）：
  输入: T个时间切片异质图 + 节点特征
    ↓
  [Graph Embedding Layer] Section 3.1
    对每个时间步t，对目标节点v：
    - 公式(1): H^t_{v,r} = A^t_r H^t_{N^t_r(v)}  （各关系邻居聚合）
    - 收集 {H^t_{v,r}} ∪ {H^t_v} 作为多视角特征
    ↓
  [Hetero-Temporal Encoder] Section 3.2
    - 公式(3): H^LLM_v = LLM(Prompt(v))          （LLM类型编码）
    - 公式(4): p^t_i = sin/cos(...)               （正弦时序编码）
    - 公式(5): H^{sp,t}_v = H^LLM_v + p^t        （时空编码）
    - Z^t_v = concat(H^t_v, H^{sp,t}_v)          （增强表示）
    ↓
  [Spatio-Temporal Attention] Section 3.3
    - 公式(6)(7): Z'_v = Softmax(QK^T/√d_K)V     （共享参数注意力）
    ↓
  [Optimization] Section 3.4
    - MLP分类头 -> 损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from openhgnn.layers.htgformer_layer import (
    GraphEmbeddingLayer,
    HeteroTemporalEncoder,
    HTGformerEncoderLayer,
)

try:
    from openhgnn.models import BaseModel, register_model
    HAS_OPENHGNN = True
except ImportError:
    HAS_OPENHGNN = False
    BaseModel = nn.Module
    def register_model(name):
        def decorator(cls): return cls
        return decorator


@register_model('HTGformer')
class HTGformer(BaseModel):
    """
    HTGformer: Heterogeneous Temporal Graph Transformer
    
    论文超参数（Section 4.1.3）：
      hidden_dim = 64（COVID-19用8）
      num_heads = 4（推断，论文未明确说明头数）
      num_layers = 2（推断）
      lr = 5e-3
      weight_decay = 5e-4
      max_epochs = 500
      early_stopping = 25
      spatio-temporal encoding dim = 64
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(
            in_dim_dict={ntype: hg.nodes[ntype].data['h'].shape[-1]
                         for ntype in hg.ntypes
                         if 'h' in hg.nodes[ntype].data},
            hidden_dim=getattr(args, 'hidden_dim', 64),
            num_heads=getattr(args, 'num_heads', 4),
            num_layers=getattr(args, 'num_layers', 2),
            dropout=getattr(args, 'dropout', 0.1),
            num_timestamps=getattr(args, 'num_timestamps', 10),
            node_types=hg.ntypes,
            category=args.category,
            out_dim=args.out_dim,
            use_llm=getattr(args, 'use_llm', False),
            llm_embed_path=getattr(args, 'llm_embed_path', None),
        )

    def __init__(
        self,
        in_dim_dict,        # {ntype: int}，各节点类型原始特征维度
        hidden_dim,         # int，隐层维度d（论文d=64）
        num_heads,          # int，注意力头数
        num_layers,         # int，Transformer层数
        dropout,            # float
        num_timestamps,     # int，时间步数T
        node_types,         # List[str]
        category,           # str，目标节点类型
        out_dim,            # int，输出类别数
        use_llm=False,      # bool，是否使用LLM编码
        llm_embed_path=None,# str，预计算LLM嵌入路径
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_timestamps = num_timestamps
        self.node_types = node_types
        self.category = category

        # embed_dim = 2 * hidden_dim
        # 因为 Z^t_v = concat(H^t_v, H^{sp,t}_v)，各hidden_dim维
        self.embed_dim = 2 * hidden_dim

        # ── Section 3.1: Graph Embedding Layer ──────────────────────────
        # 公式(1): 非参数图卷积
        self.graph_emb = GraphEmbeddingLayer()

        # ── Section 3.2: Hetero-Temporal Encoder ────────────────────────
        # 公式(2)(3)(4)(5): LLM类型编码 + 正弦时序编码 + concat
        self.hetero_temporal_enc = HeteroTemporalEncoder(
            node_types=node_types,
            feat_dim_dict=in_dim_dict,
            hidden_dim=hidden_dim,
            use_llm=use_llm,
            llm_embed_path=llm_embed_path,
        )

        # ── Section 3.3: Spatio-Temporal Attention ──────────────────────
        # 公式(6)(7): 共享参数的多头自注意力，堆叠num_layers层
        self.encoder_layers = nn.ModuleList([
            HTGformerEncoderLayer(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Section 3.4: Optimization ────────────────────────────────────
        # 两层MLP（论文原文："two layer multilayer perceptron (MLP)"）
        # 输入：flatten后的Z'_v（T个时间步的预测表示）
        # 注意：论文取T+1时刻的表示进行预测：H_v = MLP(Z^{T+1}_v)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, graphs, feat_dicts, target_node_ids=None):
        """
        完整前向传播

        Args:
            graphs: List[DGLGraph]，T个时间步的异质图
            feat_dicts: List[dict]，T个时间步的节点特征 {ntype: tensor}
            target_node_ids: Optional tensor，目标节点索引（None则取全部）

        Returns:
            logits: tensor [N_target, out_dim]
        """
        T = len(graphs)
        device = feat_dicts[0][self.category].device

        # ── Step 1: Graph Embedding Layer (公式1) ──────────────────────
        # seq_list[t]: {'self': [N,d], rel1: [N,d], ...}
        seq_list = self.graph_emb(graphs, feat_dicts, self.category)

        # ── Step 2: Hetero-Temporal Encoder (公式3,4,5) ─────────────────
        # 对每个时间步，对每种视角（自身+各关系）生成Z^t_v
        # 收集成序列 Z: [N, L, embed_dim]
        # L = T * (1 + num_relations)
        all_tokens = []
        for t in range(T):
            t_repr = seq_list[t]
            # 对目标节点自身特征
            self_feat = t_repr['self']  # [N, feat_dim]
            # 构造临时feat_dict用于hetero_temporal_enc
            t_feat_dict = {self.category: self_feat}
            Z_t_self = self.hetero_temporal_enc(
                t_feat_dict, t, self.category
            )  # [N, 2*d]
            all_tokens.append(Z_t_self)

            # 对每种关系的邻居聚合
            for rel, h_agg in t_repr.items():
                if rel == 'self':
                    continue
                t_feat_dict_rel = {self.category: h_agg}
                Z_t_rel = self.hetero_temporal_enc(
                    t_feat_dict_rel, t, self.category
                )  # [N, 2*d]
                all_tokens.append(Z_t_rel)

        # Z: [N, L, embed_dim]
        Z = torch.stack(all_tokens, dim=1)  # [N, L, 2*d]

        # Z_v: 目标节点在各时间步的自身表示（作为Query）
        # 取前T个token（对应T个时间步的self表示）
        Z_v = Z[:, :T, :]  # [N, T, 2*d]

        # ── Step 3: Spatio-Temporal Attention (公式6,7) ─────────────────
        for layer in self.encoder_layers:
            Z_v = layer(Z_v, Z)  # [N, T, 2*d]

        # ── Step 4: Optimization (Section 3.4) ──────────────────────────
        # 取最后一个时间步（T时刻）的表示做预测
        # 论文："Take the T+1 as an example: H_v = MLP(Z^{T+1}_v)"
        # 实际上取Z_v的最后一个时间步（对应预测T+1时刻）
        h_v = Z_v[:, -1, :]  # [N, 2*d]

        if target_node_ids is not None:
            h_v = h_v[target_node_ids]

        logits = self.classifier(h_v)  # [N_target, out_dim]
        return logits

    def link_prediction_forward(self, graphs, feat_dicts):
        """
        链路预测任务的前向传播
        公式(8): L = -sum log σ(H_i^T H_j) - sum log σ(-H_i'^T H_j')
        """
        T = len(graphs)
        seq_list = self.graph_emb(graphs, feat_dicts, self.category)

        all_tokens = []
        for t in range(T):
            t_repr = seq_list[t]
            self_feat = t_repr['self']
            t_feat_dict = {self.category: self_feat}
            Z_t = self.hetero_temporal_enc(t_feat_dict, t, self.category)
            all_tokens.append(Z_t)
            for rel, h_agg in t_repr.items():
                if rel == 'self':
                    continue
                t_feat_dict_rel = {self.category: h_agg}
                Z_t_rel = self.hetero_temporal_enc(
                    t_feat_dict_rel, t, self.category
                )
                all_tokens.append(Z_t_rel)

        Z = torch.stack(all_tokens, dim=1)
        Z_v = Z[:, :T, :]

        for layer in self.encoder_layers:
            Z_v = layer(Z_v, Z)

        h_v = Z_v[:, -1, :]
        # 通过MLP得到节点嵌入
        h_v = self.classifier(h_v)  # 完整MLP，对应论文公式(8)
        return h_v
