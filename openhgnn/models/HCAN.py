"""
Heterogeneous Convolution-oriented Attention Network (HCAN).

This module implements Vanilla HCAN (V-HCAN) and Decoupled HCAN (D-HCAN) from
"Effective and Scalable Heterogeneous Graph Neural Network Framework
with Convolution-oriented Attention".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.ops import edge_softmax

from . import BaseModel, register_model


@register_model("HCAN")
class HCAN(BaseModel):
    r"""Vanilla HCAN.

    The model follows Algorithm 1 in the paper. For node classification,
    ``forward`` returns decoded logits. For representation-learning tasks
    such as link prediction, it returns node embeddings and lets the
    trainerflow provide the task-specific decoder.
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        task = getattr(args, "task", None)
        use_decoder = getattr(args, "use_decoder", task == "node_classification")
        hidden_dim = getattr(args, "hidden_dim", 64)
        out_dim = getattr(args, "out_dim", hidden_dim)
        if not use_decoder:
            out_dim = hidden_dim
            args.out_dim = hidden_dim
        return cls(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=getattr(args, "num_layers", 1),
            max_hop=getattr(args, "max_hop", 2),
            num_heads=getattr(args, "num_heads", 4),
            ntypes=hg.ntypes,
            etypes=hg.canonical_etypes,
            dropout=getattr(args, "dropout", 0.5),
            use_decoder=use_decoder,
            attn_activation=getattr(args, "attn_activation", "identity"),
        )

    def __init__(
        self,
        hidden_dim,
        out_dim,
        num_layers,
        max_hop,
        num_heads,
        ntypes,
        etypes,
        dropout,
        use_decoder=True,
        attn_activation="identity",
    ):
        super(HCAN, self).__init__()
        if hidden_dim % (2 * num_heads) != 0:
            raise ValueError(
                "hidden_dim ({}) must be divisible by twice num_heads ({})".format(
                    hidden_dim, 2 * num_heads
                )
            )

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim if use_decoder else hidden_dim
        self.emb_dim = hidden_dim
        self.num_layers = num_layers
        self.max_hop = max_hop
        self.num_heads = num_heads
        self.ntypes = list(ntypes)
        self.etypes = list(etypes)
        self.use_decoder = use_decoder

        self.layers = nn.ModuleList(
            [
                VHCANLayer(
                    hidden_dim=hidden_dim,
                    max_hop=max_hop,
                    num_heads=num_heads,
                    ntypes=self.ntypes,
                    etypes=self.etypes,
                    dropout=dropout,
                    attn_activation=attn_activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = (
            nn.Linear(hidden_dim, out_dim, bias=False) if use_decoder else None
        )

    def encode(self, hg, h_dict):
        if not hasattr(hg, "ntypes"):
            raise NotImplementedError(
                "HCAN currently supports full-batch heterographs."
            )
        for layer in self.layers:
            h_dict = layer(hg, h_dict)
        return h_dict

    def forward(self, hg, h_dict, return_emb=False):
        h_dict = self.encode(hg, h_dict)
        if return_emb or self.decoder is None:
            return h_dict
        return {ntype: self.decoder(h) for ntype, h in h_dict.items()}

    def get_emb(self, hg, h_dict):
        return {
            ntype: h.detach().cpu().numpy()
            for ntype, h in self.encode(hg, h_dict).items()
        }


class VHCANLayer(nn.Module):
    r"""One V-HCAN layer with convolution-oriented attention."""

    def __init__(
        self,
        hidden_dim,
        max_hop,
        num_heads,
        ntypes,
        etypes,
        dropout,
        attn_activation="identity",
    ):
        super(VHCANLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_hop = max_hop
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.value_head_dim = hidden_dim // (2 * num_heads)
        self.ntypes = list(ntypes)
        self.etypes = list(etypes)
        self.etype_keys = {etype: str(idx) for idx, etype in enumerate(self.etypes)}
        self.enhanced_dim = hidden_dim * (max_hop + 1)
        self.attn_activation = attn_activation.lower()
        if self.attn_activation not in {"identity", "leaky_relu", "relu"}:
            raise ValueError(
                "Unsupported HCAN attention activation: {}".format(attn_activation)
            )

        self.proj_mlp = nn.ModuleDict(
            {
                ntype: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.PReLU())
                for ntype in self.ntypes
            }
        )
        self.hop_mlp = nn.ModuleDict(
            {
                ntype: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.PReLU()
                )
                for ntype in self.ntypes
            }
        )

        self.hat_norm = nn.ModuleDict(
            {ntype: nn.LayerNorm(self.enhanced_dim) for ntype in self.ntypes}
        )
        self.hat_act = nn.PReLU()
        self.attn_Q = nn.ParameterDict(
            {
                ntype: nn.Parameter(
                    torch.empty(num_heads, self.enhanced_dim, self.head_dim)
                )
                for ntype in self.ntypes
            }
        )
        self.attn_K = nn.ParameterDict(
            {
                ntype: nn.Parameter(
                    torch.empty(num_heads, self.enhanced_dim, self.head_dim)
                )
                for ntype in self.ntypes
            }
        )
        self.attn_a = nn.ParameterDict(
            {
                key: nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
                for key in self.etype_keys.values()
            }
        )
        self.value_V = nn.ParameterDict(
            {
                key: nn.Parameter(
                    torch.empty(num_heads, self.enhanced_dim, self.value_head_dim)
                )
                for key in self.etype_keys.values()
            }
        )
        self.residual_W = nn.ModuleDict(
            {
                ntype: nn.Linear(self.enhanced_dim, hidden_dim, bias=True)
                for ntype in self.ntypes
            }
        )
        self.act = nn.PReLU()
        self.norm = nn.ModuleDict(
            {ntype: nn.LayerNorm(hidden_dim) for ntype in self.ntypes}
        )
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for ntype in self.ntypes:
            for head in range(self.num_heads):
                nn.init.xavier_uniform_(self.attn_Q[ntype][head])
                nn.init.xavier_uniform_(self.attn_K[ntype][head])
        for key in self.etype_keys.values():
            nn.init.xavier_uniform_(self.attn_a[key])
            for head in range(self.num_heads):
                nn.init.xavier_uniform_(self.value_V[key][head])

    def _activate_attention(self, score):
        if self.attn_activation == "identity":
            return score
        if self.attn_activation == "relu":
            return F.relu(score)
        return F.leaky_relu(score, negative_slope=0.2)

    def _etype_key(self, etype):
        return self.etype_keys[etype]

    def _compute_attn_score(self, edges, etype):
        src_t, _, dst_t = etype
        q = torch.einsum("ei,hid->ehd", edges.dst["h_hat"], self.attn_Q[dst_t])
        k = torch.einsum("ei,hid->ehd", edges.src["h_hat"], self.attn_K[src_t])
        qk = torch.cat([q, k], dim=-1)
        etype_key = self._etype_key(etype)
        score = (qk * self.attn_a[etype_key].unsqueeze(0)).sum(dim=-1)
        return {"attn_score": self._activate_attention(score).unsqueeze(-1)}

    def _all_relation_mean_propagation(self, hg, h_dict):
        with hg.local_scope():
            for ntype in self.ntypes:
                hg.nodes[ntype].data["h"] = h_dict[ntype]

            hg.multi_update_all(
                {
                    etype: (fn.copy_u("h", "m"), fn.sum("m", "h_sum"))
                    for etype in self.etypes
                },
                cross_reducer="sum",
            )

            next_h = {}
            for ntype in self.ntypes:
                if "h_sum" in hg.nodes[ntype].data:
                    h_sum = hg.nodes[ntype].data["h_sum"]
                else:
                    h_sum = torch.zeros_like(h_dict[ntype])

                degree = torch.zeros(
                    hg.num_nodes(ntype), device=h_sum.device, dtype=h_sum.dtype
                )
                for etype in self.etypes:
                    if etype[2] == ntype:
                        degree = degree + hg.in_degrees(etype=etype).to(
                            device=h_sum.device, dtype=h_sum.dtype
                        )
                next_h[ntype] = h_sum / degree.clamp(min=1).unsqueeze(-1)
            return next_h

    def _edge_softmax_by_head(self, g, scores):
        gamma = []
        for head in range(self.num_heads):
            gamma.append(edge_softmax(g, scores[:, head, :]))
        return torch.stack(gamma, dim=1)

    def forward(self, hg, h_dict):
        h_proj = {
            ntype: self.proj_mlp[ntype](h_dict[ntype])
            for ntype in self.ntypes
        }

        raw_tokens = [h_proj]
        current = h_proj
        for _ in range(self.max_hop):
            current = self._all_relation_mean_propagation(hg, current)
            raw_tokens.append(current)

        refined_tokens = [h_proj]
        for hop in range(1, self.max_hop + 1):
            refined_tokens.append(
                {
                    ntype: self.hop_mlp[ntype](raw_tokens[hop][ntype])
                    for ntype in self.ntypes
                }
            )

        h_hat = {}
        for ntype in self.ntypes:
            h_hat[ntype] = torch.cat(
                [refined_tokens[hop][ntype] for hop in range(self.max_hop + 1)],
                dim=-1,
            )
            h_hat[ntype] = self.hat_act(
                self.dropout(self.hat_norm[ntype](h_hat[ntype]))
            )

        with hg.local_scope():
            for ntype in self.ntypes:
                hg.nodes[ntype].data["h_hat"] = h_hat[ntype]

            h_high = {
                ntype: h_hat[ntype].new_zeros(
                    hg.num_nodes(ntype), self.hidden_dim // 2
                )
                for ntype in self.ntypes
            }
            h_low = {
                ntype: h_hat[ntype].new_zeros(
                    hg.num_nodes(ntype), self.hidden_dim // 2
                )
                for ntype in self.ntypes
            }

            for etype in self.etypes:
                if hg.num_edges(etype) == 0:
                    continue

                src_t, _, dst_t = etype
                etype_key = self._etype_key(etype)
                hg[etype].apply_edges(
                    lambda edges, e=etype: self._compute_attn_score(edges, e)
                )

                gamma = self._edge_softmax_by_head(
                    hg[etype], hg[etype].edata["attn_score"]
                )
                gamma_tilde = 1.0 - gamma

                value = torch.einsum(
                    "ni,hid->nhd",
                    hg.nodes[src_t].data["h_hat"],
                    self.value_V[etype_key],
                )
                hg.nodes[src_t].data["v"] = value

                hg[etype].edata["gamma"] = gamma
                hg.update_all(
                    fn.u_mul_e("v", "gamma", "m_high"),
                    fn.sum("m_high", "agg_high"),
                    etype=etype,
                )
                hg[etype].edata["gamma_tilde"] = gamma_tilde
                hg.update_all(
                    fn.u_mul_e("v", "gamma_tilde", "m_low"),
                    fn.sum("m_low", "agg_low"),
                    etype=etype,
                )

                h_high[dst_t] = h_high[dst_t] + hg.nodes[dst_t].data[
                    "agg_high"
                ].reshape(hg.num_nodes(dst_t), -1)
                h_low[dst_t] = h_low[dst_t] + hg.nodes[dst_t].data[
                    "agg_low"
                ].reshape(hg.num_nodes(dst_t), -1)

            h_out = {}
            for ntype in self.ntypes:
                combined = torch.cat([h_high[ntype], h_low[ntype]], dim=-1)
                out = self.act(combined + self.residual_W[ntype](h_hat[ntype]))
                h_out[ntype] = self.norm[ntype](out)
            return h_out


@register_model("D-HCAN")
@register_model("DHCAN")
class DHCAN(BaseModel):
    r"""Decoupled HCAN.

    D-HCAN decouples feature projection from K-hop subgraph convolution.
    It first generates edge-type-specific K-hop tokens without trainable
    parameters, then computes non-parametric attention weights for each
    token channel and fuses the resulting embeddings with feed-forward
    networks.
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        task = getattr(args, "task", None)
        use_decoder = getattr(args, "use_decoder", task == "node_classification")
        hidden_dim = getattr(args, "hidden_dim", 64)
        out_dim = getattr(args, "out_dim", hidden_dim)
        if not use_decoder:
            out_dim = hidden_dim
            args.out_dim = hidden_dim
        return cls(
            hidden_dim=hidden_dim,
            token_dim=getattr(args, "dhcan_input_dim", hidden_dim),
            out_dim=out_dim,
            num_layers=getattr(args, "num_layers", 1),
            max_hop=getattr(args, "max_hop", 2),
            ntypes=hg.ntypes,
            etypes=hg.canonical_etypes,
            dropout=getattr(args, "dropout", 0.5),
            use_decoder=use_decoder,
            cache_device=getattr(args, "cache_device", "cpu"),
        )

    def __init__(
        self,
        hidden_dim,
        token_dim,
        out_dim,
        num_layers,
        max_hop,
        ntypes,
        etypes,
        dropout,
        use_decoder=True,
        cache_device="cpu",
    ):
        super(DHCAN, self).__init__()
        if num_layers != 1:
            raise ValueError(
                "D-HCAN uses one decoupled propagation stage; num_layers must be 1."
            )
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.out_dim = out_dim if use_decoder else hidden_dim
        self.emb_dim = hidden_dim
        self.num_layers = num_layers
        self.max_hop = max_hop
        self.ntypes = list(ntypes)
        self.etypes = list(etypes)
        self.use_decoder = use_decoder
        self.cache_device = torch.device(cache_device)
        self.layers = nn.ModuleList(
            [
                DHCANLayer(
                    hidden_dim=hidden_dim,
                    token_dim=token_dim,
                    max_hop=max_hop,
                    ntypes=self.ntypes,
                    etypes=self.etypes,
                    dropout=dropout,
                    cache_device=self.cache_device,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = (
            nn.Linear(hidden_dim, out_dim, bias=False) if use_decoder else None
        )

    def encode(self, hg, h_dict):
        if not hasattr(hg, "ntypes"):
            raise NotImplementedError(
                "DHCAN currently supports full-batch heterographs."
            )
        for layer in self.layers:
            h_dict = layer(hg, h_dict)
        return h_dict

    def clear_cache(self):
        """Discard non-parametric propagation results before using new inputs."""
        for layer in self.layers:
            layer.clear_cache()

    def precompute(self, hg, h_dict, target_ntype=None):
        """Precompute parameter-free channels, optionally for one node type."""
        return self.layers[0].precompute(hg, h_dict, target_ntype)

    def move_cached_type(self, ntype, device):
        """Move only one output node type's cached channels to ``device``."""
        self.layers[0].move_cached_type(ntype, device)

    def forward_cached(self, ntype, node_idx=None):
        """Decode cached representations for one node type or node batch."""
        device = next(self.parameters()).device
        embedding = self.layers[0].fuse_cached(ntype, node_idx, device)
        return self.decoder(embedding) if self.decoder is not None else embedding

    def forward(self, hg, h_dict, return_emb=False):
        h_dict = self.encode(hg, h_dict)
        if return_emb or self.decoder is None:
            return h_dict
        return {ntype: self.decoder(h) for ntype, h in h_dict.items()}

    def get_emb(self, hg, h_dict):
        return {
            ntype: h.detach().cpu().numpy()
            for ntype, h in self.encode(hg, h_dict).items()
        }


class DHCANLayer(nn.Module):
    r"""One D-HCAN layer following Equations 13-15."""

    def __init__(
        self,
        hidden_dim,
        token_dim,
        max_hop,
        ntypes,
        etypes,
        dropout,
        cache_device="cpu",
    ):
        super(DHCANLayer, self).__init__()
        if max_hop < 1:
            raise ValueError("D-HCAN max_hop must be at least 1.")
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.max_hop = max_hop
        self.ntypes = list(ntypes)
        self.etypes = list(etypes)
        self.cache_device = torch.device(cache_device)
        self.relation_paths = self._enumerate_relation_paths()
        self.num_channels = len(self.relation_paths)
        if self.num_channels == 0:
            raise ValueError(
                "D-HCAN found no valid relation path for max_hop={}.".format(
                    max_hop
                )
            )
        self._cached_tokens = None

        self.channel_proj = nn.ModuleDict(
            {
                ntype: nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(token_dim, hidden_dim),
                            nn.PReLU(),
                            nn.Dropout(dropout),
                        )
                        for _ in range(self.num_channels)
                    ]
                )
                for ntype in self.ntypes
            }
        )
        self.semantic_fusion = nn.ModuleDict(
            {
                ntype: nn.Sequential(
                    nn.Linear(self.num_channels * hidden_dim, hidden_dim),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                )
                for ntype in self.ntypes
            }
        )

    def _enumerate_relation_paths(self):
        paths = [(etype,) for etype in self.etypes]
        for _ in range(1, self.max_hop):
            paths = [
                path + (etype,)
                for path in paths
                for etype in self.etypes
                if path[-1][2] == etype[0]
            ]
        return paths

    def _relation_propagate(self, hg, features, etype):
        src_t, _, dst_t = etype
        if hg.num_edges(etype) == 0:
            return features.new_zeros(hg.num_nodes(dst_t), features.shape[-1])

        with hg.local_scope():
            hg.nodes[src_t].data["h"] = features
            hg.update_all(
                fn.copy_u("h", "m"),
                fn.mean("m", "h_mean"),
                etype=etype,
            )
            return hg.nodes[dst_t].data["h_mean"]

    def _generate_decoupled_tokens(self, hg, h_dict):
        def expand(path, features):
            if len(path) == self.max_hop:
                yield path[-1][2], features
                return

            for etype in self.etypes:
                if path[-1][2] != etype[0]:
                    continue
                next_features = self._relation_propagate(hg, features, etype)
                yield from expand(path + (etype,), next_features)

        for etype in self.etypes:
            features = self._relation_propagate(hg, h_dict[etype[0]], etype)
            yield from expand((etype,), features)

    def _safe_divide(self, numerator, denominator):
        valid = denominator.abs() > 1e-12
        safe_denominator = torch.where(
            valid, denominator, torch.ones_like(denominator)
        )
        return torch.where(
            valid, numerator / safe_denominator, torch.zeros_like(numerator)
        )

    def _non_parametric_attention(self, hg, h_dict, token, target_ntype=None):
        token_type, token_features = token
        reachable_types = {token_type}
        reachable_types.update(
            etype[2]
            for etype in self.etypes
            if etype[0] == token_type and hg.num_edges(etype) > 0
        )
        if target_ntype is not None:
            if target_ntype not in reachable_types:
                return {}
            output_types = {target_ntype}
        else:
            output_types = reachable_types
        numerator = {
            ntype: h_dict[ntype].new_zeros(hg.num_nodes(ntype), self.token_dim)
            for ntype in output_types
        }
        denominator = {
            ntype: h_dict[ntype].new_zeros(hg.num_nodes(ntype), 1)
            for ntype in output_types
        }
        with hg.local_scope():
            for ntype in self.ntypes:
                hg.nodes[ntype].data["base"] = h_dict[ntype]
            hg.nodes[token_type].data["token"] = token_features

            if token_type in output_types:
                self_score = (h_dict[token_type] * token_features).sum(
                    dim=-1, keepdim=True
                )
                denominator[token_type] = denominator[token_type] + self_score
                numerator[token_type] = (
                    numerator[token_type] + token_features * self_score
                )

            for etype in self.etypes:
                if (
                    etype[0] != token_type
                    or etype[2] not in output_types
                    or hg.num_edges(etype) == 0
                ):
                    continue
                _, _, dst_t = etype
                hg[etype].apply_edges(
                    lambda edges: {
                        "score": (edges.dst["base"] * edges.src["token"]).sum(
                            dim=-1, keepdim=True
                        )
                    }
                )
                hg.update_all(
                    fn.copy_e("score", "m_score"),
                    fn.sum("m_score", "score_sum"),
                    etype=etype,
                )
                hg.update_all(
                    fn.u_mul_e("token", "score", "m_token"),
                    fn.sum("m_token", "token_sum"),
                    etype=etype,
                )
                denominator[dst_t] = denominator[dst_t] + hg.nodes[dst_t].data[
                    "score_sum"
                ]
                numerator[dst_t] = numerator[dst_t] + hg.nodes[dst_t].data[
                    "token_sum"
                ]

        return {
            ntype: self._safe_divide(numerator[ntype], denominator[ntype])
            for ntype in output_types
        }

    def clear_cache(self):
        self._cached_tokens = None

    def move_cached_type(self, ntype, device):
        if self._cached_tokens is None:
            raise RuntimeError(
                "D-HCAN channels must be precomputed before moving them."
            )
        device = torch.device(device)
        for token in self._cached_tokens:
            if ntype in token:
                token[ntype] = token[ntype].to(device)

    @torch.no_grad()
    def precompute(self, hg, h_dict, target_ntype=None):
        """Run Equations 13-15 once and cache parameter-free outputs."""
        if target_ntype is not None and target_ntype not in self.ntypes:
            raise ValueError("Unknown D-HCAN target node type: {}".format(target_ntype))
        self._cached_tokens = [
            {
                ntype: value.detach().to(self.cache_device)
                for ntype, value in self._non_parametric_attention(
                    hg, h_dict, token, target_ntype
                ).items()
            }
            for token in self._generate_decoupled_tokens(hg, h_dict)
        ]
        return self._cached_tokens

    def fuse_cached(self, ntype, node_idx, device):
        if self._cached_tokens is None:
            raise RuntimeError("D-HCAN channels have not been precomputed.")
        fusion_linear = self.semantic_fusion[ntype][0]
        if node_idx is None:
            num_nodes = next(
                token[ntype].shape[0]
                for token in self._cached_tokens
                if ntype in token
            )
        else:
            num_nodes = node_idx.shape[0]
        fused = fusion_linear.bias.unsqueeze(0).expand(num_nodes, -1)
        for channel, token in enumerate(self._cached_tokens):
            if ntype not in token:
                continue
            token_features = token[ntype]
            if node_idx is not None:
                token_features = token_features.index_select(
                    0, node_idx.to(token_features.device)
                )
            token_features = token_features.to(device)
            mask = token_features.abs().sum(dim=-1, keepdim=True) > 0
            projected_token = self.channel_proj[ntype][channel](token_features)
            projected_token = projected_token * mask.to(projected_token.dtype)
            start = channel * self.hidden_dim
            stop = start + self.hidden_dim
            fused = fused + F.linear(
                projected_token, fusion_linear.weight[:, start:stop]
            )
        fused = self.semantic_fusion[ntype][1](fused)
        return self.semantic_fusion[ntype][2](fused)

    def forward(self, hg, h_dict):
        if self._cached_tokens is None:
            self.precompute(hg, h_dict)

        out_dict = {}
        for ntype in self.ntypes:
            out_dict[ntype] = self.fuse_cached(
                ntype=ntype,
                node_idx=None,
                device=h_dict[ntype].device,
            )
        return out_dict
