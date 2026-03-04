"""
HTGformer Layer Implementation
================================
Paper: HTGformer: Heterogeneous Temporal Graph Transformer (SIGIR 2025)
Paper URL: https://dl.acm.org/doi/10.1145/3726302.3730209

架构概述
--------
HTGformer 由两大模块组成：
  1. Graph Embedding Layer（图嵌入层）：
       - Heterogeneous Feature Projection（异质特征投影）：公式(1)
         h_v = W_{τ(v)} * x_v + b_{τ(v)}    ∀v ∈ V
       - Non-parametric GCN Aggregation（无参数GCN聚合）：公式(2)
         Z^t_r = σ( D̃^{-1/2} Ã D̃^{-1/2} H^t_r )   ∀r ∈ R

  2. Heterogeneous-Temporal Encoder（异质时序编码器）：
       - 将所有时间步的所有节点类型的特征序列拼成 token 序列
       - 标准 Multi-Head Self-Attention，但配有：
           * 节点类型嵌入 (type embedding)
           * 时间步嵌入 (temporal embedding)
         联合注入 → 公式(3)(4)(5)

公式对应关系
-----------
公式(1):  HeteroFeatureProjection.forward()
公式(2):  GCNAggregation.forward()
公式(3):  TemporalTypeEmbedding.forward() — 类型嵌入 e_τ(v)
公式(4):  TemporalTypeEmbedding.forward() — 时间嵌入 e_t
公式(5):  HTGformerEncoderLayer.forward() — MHA over token sequence
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.ops import edge_softmax


# ──────────────────────────────────────────────────────────────────────────────
# Module 1: Heterogeneous Feature Projection
# 公式(1): h_v = W_{τ(v)} · x_v + b_{τ(v)}
# ──────────────────────────────────────────────────────────────────────────────
class HeteroFeatureProjection(nn.Module):
    """
    将各类型节点的原始特征投影到统一的隐层空间。
    对应论文 Section 3.1 "Heterogeneous Feature Projection"。

    Args:
        in_dim_dict  (dict): {node_type: input_feature_dim}
        out_dim      (int):  统一的隐层维度 d
    """

    def __init__(self, in_dim_dict: dict, out_dim: int):
        super().__init__()
        self.projectors = nn.ModuleDict()
        for ntype, in_dim in in_dim_dict.items():
            self.projectors[ntype] = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, feat_dict: dict) -> dict:
        """
        Args:
            feat_dict: {node_type: Tensor[N_v, in_dim_v]}
        Returns:
            proj_dict: {node_type: Tensor[N_v, out_dim]}
        """
        out = {}
        for ntype, feat in feat_dict.items():
            out[ntype] = self.projectors[ntype](feat)   # 公式(1)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Module 2: GCN-based Neighbor Aggregation (per relation, per time slice)
# 公式(2): Z^t_r = σ( D̃^{-1/2} Ã D̃^{-1/2} H^t_r )
# ──────────────────────────────────────────────────────────────────────────────
class GCNAggregation(nn.Module):
    """
    无参数 GCN 聚合，对每个关系类型独立执行。
    使用对称归一化：D̃^{-1/2} Ã D̃^{-1/2}
    对应论文 Section 3.1 "Simplified Neighbor Aggregation"。

    Args:
        activation: 激活函数，默认 ELU（论文使用 ELU）
    """

    def __init__(self, activation=None):
        super().__init__()
        self.activation = activation or nn.ELU()

    def forward(self, graph: dgl.DGLGraph, feat_dict: dict) -> dict:
        """
        Args:
            graph:     DGLHeteroGraph，包含当前时间切片的边
            feat_dict: {node_type: Tensor[N_v, d]}
        Returns:
            agg_dict:  {node_type: Tensor[N_v, d]}，聚合后表示
        """
        with graph.local_scope():
            # 写入节点特征
            for ntype, feat in feat_dict.items():
                graph.nodes[ntype].data['h'] = feat

            # 对每条边类型做标准化消息传递
            for etype in graph.canonical_etypes:
                src_type, rel_type, dst_type = etype
                sub_g = graph[etype]

                # 计算度的倒数平方根（对称归一化）
                src_deg = sub_g.out_degrees().float().clamp(min=1)
                dst_deg = sub_g.in_degrees().float().clamp(min=1)
                src_norm = src_deg.pow(-0.5).unsqueeze(1)   # [N_src, 1]
                dst_norm = dst_deg.pow(-0.5).unsqueeze(1)   # [N_dst, 1]

                sub_g.nodes[src_type].data['_norm_h'] = (
                    graph.nodes[src_type].data['h'] * src_norm
                )

                sub_g.update_all(
                    fn.copy_u('_norm_h', 'm'),
                    fn.sum('m', '_agg')
                )

                # 目标节点再乘以 dst_norm，加上自环
                src_self = graph.nodes[dst_type].data['h'] * dst_norm
                agg = sub_g.nodes[dst_type].data['_agg'] * dst_norm
                graph.nodes[dst_type].data['h'] = graph.nodes[dst_type].data['h'] + agg

            # 激活
            result = {}
            for ntype in feat_dict.keys():
                result[ntype] = self.activation(graph.nodes[ntype].data['h'])
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Module 3: Temporal & Type Embedding
# 公式(3): token_input = h_v + e_{τ(v)} + e_t
# ──────────────────────────────────────────────────────────────────────────────
class TemporalTypeEmbedding(nn.Module):
    """
    为每个 token（节点-时间步对）添加：
      - 节点类型嵌入 e_{τ(v)}：可学习，形状 [num_node_types, d]
      - 时间步嵌入 e_t：可学习，形状 [num_timestamps, d]
    对应论文 Section 3.2 公式(3)(4)。

    Args:
        num_node_types  (int): 节点类型数目
        num_timestamps  (int): 时间切片数目 T
        hidden_dim      (int): 隐层维度 d
    """

    def __init__(self, num_node_types: int, num_timestamps: int, hidden_dim: int):
        super().__init__()
        self.type_emb = nn.Embedding(num_node_types, hidden_dim)     # e_{τ(v)}
        self.time_emb = nn.Embedding(num_timestamps, hidden_dim)     # e_t
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.type_emb.weight)
        nn.init.xavier_uniform_(self.time_emb.weight)

    def forward(self, token_feat: torch.Tensor,
                type_ids: torch.Tensor,
                time_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_feat: [L, d]  L = sum_t sum_v 1  (序列中所有token特征)
            type_ids:   [L]     每个token的节点类型id
            time_ids:   [L]     每个token对应的时间步id
        Returns:
            out:        [L, d]  注入位置/类型信息后的 token
        """
        return token_feat + self.type_emb(type_ids) + self.time_emb(time_ids)


# ──────────────────────────────────────────────────────────────────────────────
# Module 4: Standard Transformer Encoder Layer (Multi-Head Self-Attention)
# 公式(5): Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
# ──────────────────────────────────────────────────────────────────────────────
class HTGformerEncoderLayer(nn.Module):
    """
    单层 Transformer Encoder，作用于拼接后的跨时间跨类型 token 序列。
    对应论文 Section 3.2 "Heterogeneous-Temporal Encoder"。

    与标准 Transformer 的区别：
      - 无位置编码（由 TemporalTypeEmbedding 提供时间/类型感知）
      - 支持可选的 mask（用于 padding）

    Args:
        hidden_dim  (int): d
        num_heads   (int): 注意力头数 H
        ffn_dim     (int): FFN 隐层维度，通常 4*d
        dropout     (float): dropout 率
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 ffn_dim: int = None, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        ffn_dim = ffn_dim or 4 * hidden_dim

        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True        # [B, L, d]
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Pre-LayerNorm 变体（训练更稳定）。
        公式(5): Z = x + MHA(LN(x))
                 out = Z + FFN(LN(Z))

        Args:
            x:                 [B, L, d]
            attn_mask:         [L, L] 或 None
            key_padding_mask:  [B, L] bool，True 表示被 mask 的位置
        Returns:
            out: [B, L, d]
        """
        # Multi-Head Self-Attention with residual
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)

        # FFN with residual
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Module 5: Complete Graph Embedding Layer
# 整合 HeteroFeatureProjection + GCNAggregation，处理所有时间切片
# ──────────────────────────────────────────────────────────────────────────────
class GraphEmbeddingLayer(nn.Module):
    """
    论文 Section 3.1 完整实现：Graph Embedding Layer。

    对每个时间切片 t ∈ {1, ..., T}：
      1) 将各类型节点特征投影到统一维度  → 公式(1)
      2) 对每个关系类型做 GCN 聚合       → 公式(2)

    Args:
        in_dim_dict   (dict): {node_type: raw_feature_dim}
        hidden_dim    (int):  统一隐层维度 d
        num_layers    (int):  GCN 层数，默认 1（论文用单层）
        dropout       (float)
    """

    def __init__(self, in_dim_dict: dict, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.projection = HeteroFeatureProjection(in_dim_dict, hidden_dim)
        self.gcn_layers = nn.ModuleList([
            GCNAggregation() for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, graphs: list, feat_dicts: list) -> list:
        """
        Args:
            graphs:     List[DGLHeteroGraph], 长度 T
            feat_dicts: List[dict], 每个 dict 是 {node_type: Tensor[N, d_raw]}
        Returns:
            emb_list: List[dict{node_type: Tensor[N, d]}], 长度 T
        """
        emb_list = []
        for g, feats in zip(graphs, feat_dicts):
            # 公式(1): 异质特征投影
            proj = self.projection(feats)
            proj = {k: self.dropout(v) for k, v in proj.items()}

            # 公式(2): GCN 聚合（可堆叠多层）
            h = proj
            for gcn in self.gcn_layers:
                h = gcn(g, h)

            # LayerNorm
            h = {k: self.norm(v) for k, v in h.items()}
            emb_list.append(h)
        return emb_list
