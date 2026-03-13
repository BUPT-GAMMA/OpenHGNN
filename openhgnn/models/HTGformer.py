"""
The complete model of HTGformer
Strictly corresponds to the paper HTGformer (SIGIR 2025) Section 3

Overall process (corresponding to Figure 1(a) in the paper):
  Input: T time slices of heterogeneous graphs + node features
    ↓
  [Graph Embedding Layer] Section 3.1
    For each time step t, for the target node v：
    - Formula(1): H^t_{v,r} = A^t_r H^t_{N^t_r(v)}  (All neighboring relations aggregation)
    - 收集 {H^t_{v,r}} ∪ {H^t_v} As a multi-perspective characteristic
    ↓
  [Hetero-Temporal Encoder] Section 3.2
    - Formula(3): H^LLM_v = LLM(Prompt(v))          (LLM Type Code)
    - Formula(4): p^t_i = sin/cos(...)               (Sine Wave Time-division Coding)
    - Formula(5): H^{sp,t}_v = H^LLM_v + p^t        (Space-time encoding)
    - Z^t_v = concat(H^t_v, H^{sp,t}_v)          (Enhanced expression)
    ↓
  [Spatio-Temporal Attention] Section 3.3
    - Formula(6)(7): Z'_v = Softmax(QK^T/√d_K)V     （Shared Parameter Attention）
    ↓
  [Optimization] Section 3.4
    - MLP classification head -> loss function
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
    
    Paper hyperparameters（Section 4.1.3）：
      hidden_dim = 64（COVID-19 use 8）
      num_heads = 4（It can be inferred that the paper did not explicitly state the number of heads）
      num_layers = 2（inferred）
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
        in_dim_dict,        # {ntype: int}，The original feature dimensions of each node type
        hidden_dim,         # int，Hidden layer dimension d (in the paper, d = 64)
        num_heads,          # int，Number of attention heads
        num_layers,         # int， the number of Transformer layers
        dropout,            # float
        num_timestamps,     # int，Time step T
        node_types,         # List[str]
        category,           # str，Target node type
        out_dim,            # int，Number of output categories
        use_llm=False,      # bool，Whether to use LLM encoding
        llm_embed_path=None,# str，Pre-computed LLM embedding path
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_timestamps = num_timestamps
        self.node_types = node_types
        self.category = category

        # embed_dim = 2 * hidden_dim
        # Because Z^t_v = concat(H^t_v, H^{sp,t}_v), where each of the dimensions is hidden_dim.
        self.embed_dim = 2 * hidden_dim

        # ── Section 3.1: Graph Embedding Layer ──────────────────────────
        # Formula(1):Non-parametric graph convolution
        self.graph_emb = GraphEmbeddingLayer()

        # ── Section 3.2: Hetero-Temporal Encoder ────────────────────────
        # Formula (2)(3)(4)(5): LLM type encoding + sine time sequence encoding + concat
        self.hetero_temporal_enc = HeteroTemporalEncoder(
            node_types=node_types,
            feat_dim_dict=in_dim_dict,
            hidden_dim=hidden_dim,
            use_llm=use_llm,
            llm_embed_path=llm_embed_path,
        )

        # ── Section 3.3: Spatio-Temporal Attention ──────────────────────
        # Formula (6)(7): Shared parameter multi-head self-attention, with num_layers layers stacked together
        self.encoder_layers = nn.ModuleList([
            HTGformerEncoderLayer(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Section 3.4: Optimization ────────────────────────────────────
        # Two-layer MLP (Original paper: "two layer multilayer perceptron (MLP)" )
        # Input: Flattened Z'_v (prediction representation for T time steps)
        # Note: The paper predicts based on the representation at time step T+1: H_v = MLP(Z^{T+1}_v)
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
        Complete forward propagation

        Args:
            graphs: List[DGLGraph]，Heterogeneous graph over T time steps
            feat_dicts: List[dict]，The node features over T time steps: {ntype: tensor}
            target_node_ids: Optional tensor，Target node index (all will be taken if None is specified)

        Returns:
            logits: tensor [N_target, out_dim]
        """
        T = len(graphs)
        device = feat_dicts[0][self.category].device

        # ── Step 1: Graph Embedding Layer (Formula 1) ──────────────────────
        # seq_list[t]: {'self': [N,d], rel1: [N,d], ...}
        seq_list = self.graph_emb(graphs, feat_dicts, self.category)

        # ── Step 2: Hetero-Temporal Encoder (Formula 3,4,5) ─────────────────
        # For each time step, generate Z^t_v for each perspective (self + all relationships)
        # Converge into a sequence Z: [N, L, embed_dim]
        # L = T * (1 + num_relations)
        all_tokens = []
        for t in range(T):
            t_repr = seq_list[t]
            # egarding the characteristics of the target node itself
            self_feat = t_repr['self']  # [N, feat_dim]
            # Construct a temporary feat_dict for the hetero_temporal_enc operation
            t_feat_dict = {self.category: self_feat}
            Z_t_self = self.hetero_temporal_enc(
                t_feat_dict, t, self.category
            )  # [N, 2*d]
            all_tokens.append(Z_t_self)

            # Aggregation of neighbors for each type of relationship
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

        # Z_v: The self-representation of the target node at each time step (as Query)
        # # Select the first T tokens (corresponding to the self-representation at T time steps)
        Z_v = Z[:, :T, :]  # [N, T, 2*d]

        # ── Step 3: Spatio-Temporal Attention (Formula 6,7) ─────────────────
        for layer in self.encoder_layers:
            Z_v = layer(Z_v, Z)  # [N, T, 2*d]

        # ── Step 4: Optimization (Section 3.4) ──────────────────────────
        # Make a prediction based on the representation of the last time step (at time T)
        # Paper："Take the T+1 as an example: H_v = MLP(Z^{T+1}_v)"
        # In fact, take the last time step of Z_v (corresponding to the prediction of the (T + 1)th moment)
        h_v = Z_v[:, -1, :]  # [N, 2*d]

        if target_node_ids is not None:
            h_v = h_v[target_node_ids]

        logits = self.classifier(h_v)  # [N_target, out_dim]
        return logits

    def link_prediction_forward(self, graphs, feat_dicts):
        """
        Forward propagation of the link prediction task
        Formula (8): L = -sum log σ(H_i^T H_j) - sum log σ(-H_i'^T H_j')
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
        # Obtain node embeddings through MLP
        h_v = self.classifier(h_v)  # Complete MLP, corresponding to formula (8) in the paper
        return h_v
