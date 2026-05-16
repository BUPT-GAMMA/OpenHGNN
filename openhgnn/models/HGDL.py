"""HGDL: Heterogeneous Graph Label Distribution Learning.

OpenHGNN-compatible reimplementation of the model from the paper
"Heterogeneous Graph Label Distribution Learning" (NeurIPS 2024).

Numerical alignment with the upstream reference
(https://github.com/Listener-Watcher/HGDL) is the hard constraint of this
port: with seed 0 on DBLP, this model must reach KL ~= 0.0705 on the test
set. Any change that touches the layer declaration order, the unused
"dead" layers, the initialisation, or the forward computation graph will
shift the RNG state and break alignment. Comments throughout this file
flag the spots where the upstream code is deliberately reproduced even
though it looks redundant.

Adaptation notes (versus the upstream model.py):
  * forward signature is the OpenHGNN-standard ``(hg, h_dict)`` instead of
    ``(adj_list, x)``. The trainerflow attaches ``adj_list`` to the model
    instance after construction (``model.adj_list = ...``); inside forward
    we read it back from ``self``. Features come from
    ``h_dict[self.category]``.
  * Output is wrapped as ``{category: probs}`` to match the OpenHGNN
    convention of returning ``dict[ntype, Tensor]``. The per-metapath
    attention matrix ``nj`` is stashed on ``self.last_attn_nj`` so the
    trainerflow can read it for the Omega regulariser.
  * ``build_model_from_args`` reads scalar hyperparameters from ``args``
    and derives ``in_dim`` and ``num_nodes`` from the task graph if not
    overridden. ``adj_list`` is *not* taken from args; the trainerflow
    sets it on the model after build.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BaseModel, register_model


# ---------------------------------------------------------------------------
# Minimal GCN layer, matching layers.py:GraphConvolution in the upstream repo.
# This layer does NOT normalise adj internally. Normalisation is done outside
# the model (in the dataset, once per metapath, via gcn_norm).
# ---------------------------------------------------------------------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Uniform init exactly as in the upstream layers.py. Do not switch
        # to xavier_uniform_ etc. -- it would consume a different number of
        # RNG draws and shift every subsequent layer.
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # adj may be sparse (preferred, matches the upstream spmm path) or
        # dense. We handle both.
        support = torch.mm(x, self.weight)
        if adj.is_sparse:
            out = torch.sparse.mm(adj, support)
        else:
            out = torch.mm(adj, support)
        if self.bias is not None:
            out = out + self.bias
        return out


def _gcn_norm_dense(adj):
    """Symmetric normalisation D^{-1/2} (A + I) D^{-1/2} on a dense tensor.

    Identical to utils_.gcn_norm in the upstream repo. We need it inside
    forward() (after the attention softmax) too, exactly mirroring the
    upstream forward computation.
    """
    n = adj.shape[0]
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
    adj = adj + eye
    deg = torch.sum(adj, dim=1).pow(-0.5)
    # Guard against isolated nodes producing inf.
    deg = torch.where(torch.isinf(deg), torch.zeros_like(deg), deg)
    D = torch.diag(deg)
    return D @ adj @ D


@register_model('HGDL')
class HGDL(BaseModel):
    r"""Heterogeneous Graph Label Distribution Learning.

    Parameters
    ----------
    in_dim : int
        Input feature dimension. For DBLP this is the size of the
        term vocabulary (8920).
    hidden_dim : int
        Hidden / message-passing dimension. Default 64 in the paper.
    out_dim : int
        Output dimension == number of label-distribution components.
        For DBLP this is 4.
    attention_dim : int
        Per-metapath projection dimension used by ``atten_list``.
        Default 5 in the paper.
    num_heads : int
        Number of metapaths (k). For DBLP this is 2 (APCPA, APTPA).
    num_nodes : int
        Number of target-type nodes (n). Used to size the first dimension
        of each per-metapath ``W^0_i`` projection (R^n -> R^attention_dim).
    category : str
        Target node type, e.g. 'author'. Used by forward() to pluck the
        right tensor out of ``h_dict``.
    dropout : float
        Currently unused inside forward but kept for parity with the
        upstream constructor and for future extension.

    Notes
    -----
    The trainerflow is responsible for setting ``model.adj_list`` after
    construction. ``adj_list`` is a list of k dense, already-gcn_norm-ed
    (n, n) tensors. Forward reads it from ``self.adj_list`` -- it is
    *not* a forward argument because the OpenHGNN convention pins
    ``forward(hg, h_dict)``.
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        # Derive missing dims from the graph if the caller did not set them
        # on args. The trainerflow is expected to set
        # ``args.in_dim`` from the dataset features and ``args.category``.
        category = getattr(args, 'category', 'author')
        num_nodes = hg.num_nodes(category) if hg is not None \
            else int(args.num_nodes)
        in_dim = int(getattr(args, 'in_dim'))
        out_dim = int(getattr(args, 'out_dim'))
        hidden_dim = int(getattr(args, 'hidden_dim', 64))
        attention_dim = int(getattr(args, 'attention_dim', 5))
        num_heads = int(getattr(args, 'num_heads', 2))
        dropout = float(getattr(args, 'dropout', 0.3))

        return cls(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            num_nodes=num_nodes,
            category=category,
            dropout=dropout,
        )

    def __init__(self, in_dim, hidden_dim, out_dim, attention_dim,
                 num_heads, num_nodes, category='author', dropout=0.3):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.category = category
        self.dropout = dropout

        # ============================================================
        # IMPORTANT: layer declaration order is FROZEN.
        # ============================================================
        # The upstream GCN_attention_v2.__init__ declares Q, K, V, gcn1,
        # gcn2, layer_norm, atten_list[k], agg, gcn -- in that exact order.
        # ``layer_norm`` and the final ``gcn`` are declared but NEVER used
        # in forward. They look like dead code but they are not safe to
        # remove: every nn.Parameter creation advances the global RNG
        # state, so omitting them shifts the initial values of every
        # subsequent layer and breaks numerical alignment. KL goes from
        # 0.0705 to roughly 0.1284. Do not touch.
        self.Q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.K = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.gcn1 = GraphConvolution(in_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, out_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)        # declared, unused
        self.atten_list = nn.ModuleList([
            nn.Linear(num_nodes, attention_dim) for _ in range(num_heads)
        ])
        self.agg = nn.Linear(attention_dim * num_heads, num_heads)
        self.gcn = GraphConvolution(hidden_dim, hidden_dim)  # declared, unused

        # Set by the trainerflow after construction.
        # Type: list[torch.Tensor], length k, each (n, n) dense, gcn_norm-ed.
        self.adj_list = None
        # Cache for trainerflow access to the per-metapath attention
        # weights (used by the Omega regulariser).
        self.last_attn_nj = None

    def forward(self, hg, h_dict):
        """OpenHGNN-standard signature.

        Parameters
        ----------
        hg : DGLHeteroGraph
            Passed for interface compatibility; HGDL does not use it in
            forward because the metapath adjacencies are precomputed and
            attached via ``self.adj_list``.
        h_dict : dict[str, Tensor]
            Node features keyed by ntype. We read ``h_dict[self.category]``
            as the (n, in_dim) target-type feature matrix.

        Returns
        -------
        dict[str, Tensor]
            ``{category: probs}`` where probs is (n, out_dim) row-stochastic
            (already softmax'd; do not re-softmax outside the model).
        """
        if self.adj_list is None:
            raise RuntimeError(
                "HGDL.adj_list is not set. The trainerflow must do "
                "`model.adj_list = task.dataset.adj_list` after "
                "build_model_from_args.")
        adj_list = self.adj_list
        x = h_dict[self.category]

        # ---- Module 1: active topology homogenisation ----
        # For each metapath i, treat the i-th adjacency as a (n, n) matrix
        # and project each row (a node's neighbourhood signature) to
        # R^attention_dim.
        z_list = [self.atten_list[i](adj_list[i]) for i in range(self.num_heads)]
        # Concat -> (n, k * attn) -> linear -> (n, k) -> softmax over k.
        z4 = self.agg(torch.cat(z_list, dim=1))
        nz = torch.softmax(z4, dim=1)

        # Weighted sum of the k adjacency matrices, weights are *per-node*.
        # ``nz[:, i:i+1]`` is (n, 1); broadcasting against (n, n) multiplies
        # each row of ``adj_list[i]`` by the corresponding weight. This
        # matches the upstream pattern
        # ``nz[:,0]*adj_list[0] + nz[:,1]*adj_list[1] + ...``.
        adj = nz[:, 0:1] * adj_list[0]
        for i in range(1, self.num_heads):
            adj = adj + nz[:, i:i + 1] * adj_list[i]

        # Convert to sparse for the two GCN calls (matches the upstream
        # ``adj = adj.to_sparse()``).
        adj_sparse = adj.to_sparse()

        # ---- Module 2: topology-feature consistency transformer ----
        h = F.relu(self.gcn1(x, adj_sparse))
        Q_h = self.Q(h)
        K_h = self.K(h).transpose(0, 1)
        V_h = self.V(h)

        # Hadamard-mask the QK^T similarities by the (dense) topology.
        # Densify here because elementwise multiply needs same density.
        sim = torch.matmul(Q_h, K_h)
        A_tilde = torch.mul(adj, sim)
        attention = F.softmax(A_tilde, dim=1)

        # Important: the upstream code applies gcn_norm to the softmax
        # output before multiplying V. Counter-intuitive but reproduced.
        attention_norm = _gcn_norm_dense(attention)
        X_tilde = torch.matmul(attention_norm, V_h)
        X_tilde = F.leaky_relu(X_tilde, negative_slope=0.9)

        # ---- Prediction head ----
        # Uses the ORIGINAL (un-attention-masked) topology adj_sparse, not
        # the attention-masked one. This is a documented deviation between
        # the paper and the reference implementation; we follow the code.
        z = self.gcn2(X_tilde, adj_sparse)
        probs = F.softmax(z, dim=1)

        # Stash per-metapath attention for the trainerflow.
        self.last_attn_nj = nz

        return {self.category: probs}
