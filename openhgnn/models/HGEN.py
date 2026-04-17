import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from . import BaseModel, register_model


@register_model('HGEN')
class HGEN(BaseModel):
    r"""
    HGEN (Heterogeneous Graph Ensemble Networks), IJCAI 2025.
    See paper: https://www.ijcai.org/proceedings/2025/

    For each meta-path, HGEN runs multiple independent GCNs, uses an attention
    mechanism to fuse their outputs (with residual connections), and linearly decodes
    each path. Final prediction is the sum over all meta-paths.
    The loss includes an L1 regularization term on the path embedding correlation matrix.

    Parameters
    ----------
    meta_paths_dict : dict[str, list[tuple]]
        Dict from meta-path name to meta-path,
        e.g., {"APA": [("author", "paper", "author")], ...}.
    category : str
        Target node type for classification.
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer dimension.
    out_dim : int
        Output dimension (number of classes).
    num_gcn : int
        Number of GCN models per meta-path (default: 3).
    num_layers : int
        Number of GCN layers (default: 2).
    attention_dim : int
        Attention projection dimension (default: 8).
    dropout : float
        Dropout probability.
    lambda_cov : float
        Regularization coefficient for the Gram matrix (default: 0).
    """

    @classmethod
    def build_model_from_args(cls, args, hg, in_dim=None):
        meta_paths_dict = args.meta_paths_dict
        category = args.category
        hidden_dim = args.hidden_dim
        if in_dim is None:
            in_dim = hidden_dim
        return cls(
            meta_paths_dict=meta_paths_dict,
            category=category,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=args.out_dim,
            num_gcn=args.num_gcn,
            num_layers=args.num_layers,
            attention_dim=args.attention_dim,
            dropout=args.dropout,
            lambda_cov=args.lambda_cov
        )

    def __init__(self, meta_paths_dict, category, in_dim, hidden_dim, out_dim,
                 num_gcn, num_layers, attention_dim, dropout, lambda_cov):
        super(HGEN, self).__init__()
        self.meta_paths_dict = meta_paths_dict
        self.category = category
        self.num_gcn = num_gcn
        self.lambda_cov = lambda_cov
        self.path_encoders = nn.ModuleDict()
        self._cached_graphs = {}

        for mp_name, mp in meta_paths_dict.items():
            self.path_encoders[mp_name] = _PathEncoder(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_gcn=num_gcn,
                num_layers=num_layers,
                attention_dim=attention_dim,
                dropout=dropout
            )

    def _get_meta_path_graph(self, hg, mp):
        key = str(mp)
        if key not in self._cached_graphs:
            sub_g = dgl.metapath_reachable_graph(hg, mp)
            sub_g = dgl.add_self_loop(sub_g)
            self._cached_graphs[key] = sub_g
        return self._cached_graphs[key]

    def forward(self, hg, h_dict):
        r"""
        Parameters
        ----------
        hg : dgl.DGLHeteroGraph
            Input heterogeneous graph.
        h_dict : dict[str, torch.Tensor]
            Input node features for each node type.

        Returns
        -------
        out_dict : dict[str, torch.Tensor]
            {category: logits} with shape (num_category_nodes, num_classes).
        """
        h_category = h_dict[self.category]
        all_logits = []
        all_path_embeds = []

        for mp_name, mp in self.meta_paths_dict.items():
            encoder = self.path_encoders[mp_name]
            sub_g = self._get_meta_path_graph(hg, mp)
            gcn_outputs = encoder.gcn_embed(h_category, sub_g)
            fused_embed, _ = encoder.attention(gcn_outputs)
            logits = encoder.decoder(fused_embed)
            all_logits.append(logits)
            all_path_embeds.append(fused_embed.mean(dim=0))

        final_logits = torch.stack(all_logits).sum(dim=0)
        self._saved_path_embeds = torch.stack(all_path_embeds)
        return {self.category: final_logits}

    def extra_loss(self):
        r"""
        L1 regularization on the Gram matrix of path-level embeddings.
        S = H_tilde @ H_tilde.T, where H_tilde is the mean-pooled embedding per path.
        """
        if not hasattr(self, '_saved_path_embeds') or self._saved_path_embeds is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        H_tilde = self._saved_path_embeds
        S = H_tilde @ H_tilde.T
        return self.lambda_cov * (torch.norm(S, p=1) ** 2)


class _PathEncoder(nn.Module):
    """Per-meta-path encoder: multiple GCNs + attention aggregation + linear decoder."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_gcn, num_layers, attention_dim, dropout):
        super(_PathEncoder, self).__init__()
        self.num_gcn = num_gcn
        self.gcn_list = nn.ModuleList([
            GCN_embed(in_dim, hidden_dim, hidden_dim, dropout, num_layers)
            for _ in range(num_gcn)
        ])
        self.attention = AttentionH(hidden_dim, attention_dim, num_gcn)
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def gcn_embed(self, features, sub_g):
        return [gcn(features, sub_g) for gcn in self.gcn_list]


class GCN_embed(nn.Module):
    """Single GCN encoder: Linear projection + multiple GraphConv layers."""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout, layer_s):
        super(GCN_embed, self).__init__()
        self.enc = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            dgl.nn.pytorch.conv.GraphConv(hidden_dim, hidden_dim)  # no activation, matching official GCNConv
            for _ in range(layer_s)
        ])
        self.dropout = dropout

    def forward(self, x, g):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.enc(x), 0.1)
        for conv in self.layers:
            x = conv(g, x)
            x = F.leaky_relu(x, 0.1)
        return x


class AttentionH(nn.Module):
    """
    Attention-based aggregation of multiple GCN outputs with residual connections.
    All k GCNs receive a residual weight of 1/k (per paper Eq.6).
    """

    def __init__(self, feat_dim, attention_dim, num_gcn):
        super(AttentionH, self).__init__()
        self.num_gcn = num_gcn
        self.attention_list = nn.ModuleList([
            nn.Linear(feat_dim, attention_dim)
            for _ in range(num_gcn)
        ])
        self.agg = nn.Linear(num_gcn * attention_dim, num_gcn)

    def forward(self, embed_list):
        # Official model.py: no activation on attention projections
        gcn_attention = [
            self.attention_list[i](embed_list[i])
            for i in range(self.num_gcn)
        ]
        attention = self.agg(torch.cat(gcn_attention, dim=1))
        attention = attention - attention.mean(dim=1, keepdim=True)
        atten_min = attention.min(dim=1, keepdim=True).values
        atten_max = attention.max(dim=1, keepdim=True).values
        attention = (attention - atten_min) / (atten_max - atten_min + 1e-8)

        # Official model.py: first GCN has NO residual, others have 1/k residual
        final_embed = attention[:, 0].unsqueeze(1) * embed_list[0]
        for i in range(1, self.num_gcn):
            final_embed += (attention[:, i].unsqueeze(1) + (1.0 / self.num_gcn)) * embed_list[i]
        return final_embed, attention
