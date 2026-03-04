"""
HTGformer Model Implementation
================================
Paper: HTGformer: Heterogeneous Temporal Graph Transformer (SIGIR 2025)
Paper URL: https://dl.acm.org/doi/10.1145/3726302.3730209

完整模型架构：
  输入: T 个时间切片的异质图序列 + 各时间步各类型节点特征
  流程:
    Step 1 → GraphEmbeddingLayer
             ├─ HeteroFeatureProjection (公式1)
             └─ GCNAggregation per relation (公式2)
    Step 2 → Token Sequence Construction
             将所有时间步、所有类型的节点嵌入拼成 [B, L, d]
    Step 3 → TemporalTypeEmbedding injection (公式3,4)
    Step 4 → Stacked HTGformerEncoderLayer (公式5)
    Step 5 → Readout：取目标节点的最后时间步 token 做分类/回归
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from openhgnn.models import BaseModel, register_model

# 引入自定义层（在 OpenHGNN 中路径为 openhgnn/layers/htgformer_layer.py）
# 这里使用相对路径，部署时改为 openhgnn.layers.htgformer_layer
from openhgnn.layers.htgformer_layer import (
    GraphEmbeddingLayer,
    TemporalTypeEmbedding,
    HTGformerEncoderLayer,
)


@register_model('HTGformer')
class HTGformer(BaseModel):
    """
    HTGformer: Heterogeneous Temporal Graph Transformer

    核心思想（区别于 HTGNN 等现有方法）：
      * 现有方法：空间建模 → 时序建模 （串行，参数独立）
      * HTGformer：将所有时空信息 flatten 成 token 序列，
                   用单一 Transformer 统一建模，避免参数爆炸和
                   优化困难，实现最多 6× 加速。

    Args（通过 args 传入）:
        in_dim_dict    (dict):  {node_type: raw_input_dim}
        hidden_dim     (int):   统一隐层维度 d，默认 64
        num_heads      (int):   MHA 头数，默认 4
        num_layers     (int):   Transformer 层数，默认 2
        num_gcn_layers (int):   GCN 聚合层数，默认 1
        dropout        (float): dropout，默认 0.1
        num_timestamps (int):   时间切片数 T
        node_types     (list):  所有节点类型列表
        category       (str):   目标节点类型（用于预测）
        out_dim        (int):   输出维度（类别数）
    """

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
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
        )

    def __init__(
        self,
        in_dim_dict: dict,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_gcn_layers: int,
        dropout: float,
        num_timestamps: int,
        node_types: list,
        category: str,
        out_dim: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_timestamps = num_timestamps
        self.node_types = node_types
        self.category = category
        self.out_dim = out_dim

        # 节点类型到 ID 的映射
        self.ntype2id = {ntype: i for i, ntype in enumerate(node_types)}
        num_node_types = len(node_types)

        # ── Step 1: Graph Embedding Layer ──────────────────────────────────
        # 公式(1)(2)
        self.graph_embedding = GraphEmbeddingLayer(
            in_dim_dict=in_dim_dict,
            hidden_dim=hidden_dim,
            num_layers=num_gcn_layers,
            dropout=dropout,
        )

        # ── Step 2: Temporal & Type Embedding ─────────────────────────────
        # 公式(3)(4)
        self.temporal_type_emb = TemporalTypeEmbedding(
            num_node_types=num_node_types,
            num_timestamps=num_timestamps,
            hidden_dim=hidden_dim,
        )

        # ── Step 3: Stacked Transformer Encoder ───────────────────────────
        # 公式(5)
        self.encoder_layers = nn.ModuleList([
            HTGformerEncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=4 * hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Step 4: Output Head ────────────────────────────────────────────
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    # ──────────────────────────────────────────────────────────────────────
    # Forward Pass
    # ──────────────────────────────────────────────────────────────────────
    def forward(self, graphs: list, feat_dicts: list,
                target_node_ids: dict = None):
        """
        Args:
            graphs:          List[DGLHeteroGraph], 长度 T，每个是一个时间切片
            feat_dicts:      List[dict], 每个 dict: {node_type: Tensor[N, d_raw]}
            target_node_ids: {node_type: Tensor[N_target]}，目标节点ID（可选）
                             若 None，使用所有目标类型节点

        Returns:
            logits: Tensor[N_target, out_dim]
        """
        T = len(graphs)
        assert T == self.num_timestamps, \
            f"Expected {self.num_timestamps} time steps, got {T}"

        # ── Step 1: Graph Embedding for all time slices ────────────────────
        # emb_list[t] = {ntype: Tensor[N_v, d]}
        emb_list = self.graph_embedding(graphs, feat_dicts)

        # ── Step 2: Build Token Sequence ───────────────────────────────────
        # 对每个时间步 t，对每个节点类型，将所有节点视为 token
        # 最终拼成 [1, L, d]  (batch_size=1 for full-graph training)
        # L = T * sum_v |V_v|

        tokens = []      # List of [N_v, d]
        type_ids = []    # List of [N_v]  (节点类型id)
        time_ids = []    # List of [N_v]  (时间步id)

        # 记录目标节点在序列中的位置（用于 readout）
        target_positions = []
        running_offset = 0

        for t in range(T):
            emb_t = emb_list[t]
            for ntype in self.node_types:
                if ntype not in emb_t:
                    continue
                feat = emb_t[ntype]         # [N_v, d]
                N_v = feat.shape[0]

                tokens.append(feat)
                type_ids.append(
                    torch.full((N_v,), self.ntype2id[ntype],
                               dtype=torch.long, device=feat.device)
                )
                time_ids.append(
                    torch.full((N_v,), t,
                               dtype=torch.long, device=feat.device)
                )

                # 如果是目标类型且是最后一个时间步，记录 offset
                if ntype == self.category and t == T - 1:
                    if target_node_ids is not None:
                        tgt_ids = target_node_ids.get(ntype, None)
                        if tgt_ids is not None:
                            target_positions.append(
                                running_offset + tgt_ids
                            )
                        else:
                            target_positions.append(
                                torch.arange(running_offset,
                                             running_offset + N_v,
                                             device=feat.device)
                            )
                    else:
                        target_positions.append(
                            torch.arange(running_offset,
                                         running_offset + N_v,
                                         device=feat.device)
                        )
                running_offset += N_v

        # [L, d], [L], [L]
        token_seq = torch.cat(tokens, dim=0)        # [L, d]
        type_id_seq = torch.cat(type_ids, dim=0)    # [L]
        time_id_seq = torch.cat(time_ids, dim=0)    # [L]

        # ── Step 3: Inject Temporal & Type Embeddings (公式3,4) ────────────
        token_seq = self.temporal_type_emb(token_seq, type_id_seq, time_id_seq)

        # Reshape to [1, L, d] for Transformer (batch=1)
        token_seq = token_seq.unsqueeze(0)           # [1, L, d]

        # ── Step 4: Transformer Encoder Layers (公式5) ─────────────────────
        for layer in self.encoder_layers:
            token_seq = layer(token_seq)

        token_seq = token_seq.squeeze(0)             # [L, d]

        # ── Step 5: Readout ────────────────────────────────────────────────
        # 取目标节点在最后时间步对应的 token
        tgt_pos = target_positions[0] if target_positions else \
                  torch.arange(token_seq.shape[0], device=token_seq.device)

        tgt_emb = token_seq[tgt_pos]                 # [N_target, d]
        tgt_emb = self.output_norm(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        logits = self.classifier(tgt_emb)            # [N_target, out_dim]

        return logits

    def get_embeddings(self, graphs: list, feat_dicts: list) -> dict:
        """
        返回所有节点在最后时间步的嵌入，用于下游任务或可视化。

        Returns:
            emb_dict: {node_type: Tensor[N, d]}
        """
        T = len(graphs)
        emb_list = self.graph_embedding(graphs, feat_dicts)

        tokens_by_type = {ntype: [] for ntype in self.node_types}
        type_ids_all, time_ids_all = [], []
        offset_map = {}    # {(ntype, t): (start, end)}
        running = 0

        for t in range(T):
            emb_t = emb_list[t]
            for ntype in self.node_types:
                if ntype not in emb_t:
                    continue
                feat = emb_t[ntype]
                N_v = feat.shape[0]
                tokens_by_type[ntype].append(feat)
                type_ids_all.append(
                    torch.full((N_v,), self.ntype2id[ntype],
                               dtype=torch.long, device=feat.device)
                )
                time_ids_all.append(
                    torch.full((N_v,), t,
                               dtype=torch.long, device=feat.device)
                )
                offset_map[(ntype, t)] = (running, running + N_v)
                running += N_v

        token_all = torch.cat(
            [feat for nt in self.node_types for feat in tokens_by_type[nt]
             if tokens_by_type[nt]], dim=0
        )
        type_id_all = torch.cat(type_ids_all, dim=0)
        time_id_all = torch.cat(time_ids_all, dim=0)

        token_all = self.temporal_type_emb(token_all, type_id_all, time_id_all)
        token_all = token_all.unsqueeze(0)
        for layer in self.encoder_layers:
            token_all = layer(token_all)
        token_all = token_all.squeeze(0)

        result = {}
        for ntype in self.node_types:
            start, end = offset_map.get((ntype, T - 1), (0, 0))
            if end > start:
                result[ntype] = self.output_norm(token_all[start:end])
        return result
