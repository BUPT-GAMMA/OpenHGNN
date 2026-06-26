"""
Relational Graph Transformer (RelGT)
Paper: Relational Graph Transformer (https://arxiv.org/abs/2505.10960)

Merged from relgt/{codebook.py, local_module.py, encoders.py, model.py}.
Algorithm and architecture are identical to the original implementation.
"""
import math
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BaseModel, register_model

# ---------- optional heavy dependencies: imported lazily ----------
try:
    from torch_geometric.nn.dense.linear import Linear as PyGLinear
    from torch_geometric.nn import MLP as PyGMLP
    from torch_geometric.nn import GINConv, PositionalEncoding
    import torch_geometric.transforms as T
    from torch_geometric.data import Data as PyGData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    import torch_frame
    from torch_frame.data.stats import StatType
    from torch_frame.nn.models import ResNet as TFResNet
    HAS_TORCH_FRAME = True
except ImportError:
    HAS_TORCH_FRAME = False

try:
    from einops import rearrange
    HAS_EINOPS = True
except ImportError:
    HAS_EINOPS = False


# =============================================================================
# VectorQuantizerEMA  (codebook.py)
# =============================================================================

class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average (EMA) for the codebook.
    Adapted from https://github.com/devnkong/GOAT
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float = 0.99):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._decay = decay

        self.register_buffer("_embedding",
                             torch.randn(self._num_embeddings, self._embedding_dim))
        self.register_buffer("_embedding_output",
                             torch.randn(self._num_embeddings, self._embedding_dim))
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_w",
                             torch.randn(self._num_embeddings, self._embedding_dim))

        self.bn = nn.BatchNorm1d(self._embedding_dim, affine=False)

    def reset_parameters(self):
        nn.init.normal_(self._embedding, mean=0.0, std=1.0)
        nn.init.normal_(self._embedding_output, mean=0.0, std=1.0)
        nn.init.zeros_(self._ema_cluster_size)
        nn.init.normal_(self._ema_w, mean=0.0, std=1.0)
        self.bn.reset_parameters()

    def get_k(self) -> torch.Tensor:
        return self._embedding_output

    def get_v(self) -> torch.Tensor:
        return self._embedding_output[:, : self._embedding_dim]

    def update(self, x: torch.Tensor) -> torch.Tensor:
        inputs_normalized = self.bn(x)
        embedding_normalized = self._embedding

        distances = (
            torch.sum(inputs_normalized ** 2, dim=1, keepdim=True)
            + torch.sum(embedding_normalized ** 2, dim=1)
            - 2 * torch.matmul(inputs_normalized, embedding_normalized.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=x.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        if self.training:
            self._ema_cluster_size.data = (
                self._ema_cluster_size * self._decay
                + (1 - self._decay) * torch.sum(encodings, 0)
            )
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size.data = (
                (self._ema_cluster_size + 1e-5)
                / (n + self._num_embeddings * 1e-5) * n
            )
            dw = torch.matmul(encodings.t(), inputs_normalized)
            self._ema_w.data = self._ema_w * self._decay + (1 - self._decay) * dw
            self._embedding.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)

            running_std = torch.sqrt(self.bn.running_var + 1e-5).unsqueeze(0)
            running_mean = self.bn.running_mean.unsqueeze(0)
            self._embedding_output.data = self._embedding * running_std + running_mean

        return encoding_indices


# =============================================================================
# LocalModule  (local_module.py)
# =============================================================================

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, dropout_rate: float):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(hidden_size)
        self.bn_out = nn.BatchNorm1d(hidden_size)
        self.ffn_net = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout_rate),
        )

    def reset_parameters(self):
        for layer in self.ffn_net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]  —  BN operates on the feature dim
        x = x.permute(0, 2, 1)   # [B, D, L]
        x = self.bn_in(x)
        x = x.permute(0, 2, 1)   # [B, L, D]
        x = self.ffn_net(x)
        x = x.permute(0, 2, 1)
        x = self.bn_out(x)
        x = x.permute(0, 2, 1)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int,
                 dropout_rate: float, attention_dropout_rate: float, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout_rate = attention_dropout_rate

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)

    def reset_parameters(self):
        self.self_attention_norm.reset_parameters()
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        residual = x
        x_norm = self.self_attention_norm(x)

        Q = self.q_proj(x_norm)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)
        B, L, D = Q.shape
        head_dim = D // self.num_heads

        Q = Q.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_bias,
            dropout_p=self.attention_dropout_rate if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        attn_output = self.out_proj(attn_output)
        attn_output = self.self_attention_dropout(attn_output)

        x = residual + attn_output

        residual = x
        x_norm = self.ffn_norm(x)
        ffn_output = self.ffn(x_norm)
        x = residual + ffn_output
        return x


class LocalModule(nn.Module):
    def __init__(self, seq_len: int, input_dim: int,
                 node_only_readout: bool = False, n_layers: int = 1,
                 num_heads: int = 8, hidden_dim: int = 64,
                 dropout_rate: float = 0.3, attention_dropout_rate: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.node_only_readout = node_only_readout
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.att_embeddings_nope = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, self.ffn_dim,
                         dropout_rate, attention_dropout_rate, num_heads)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.attn_layer = nn.Linear(2 * hidden_dim, 1)

    def reset_parameters(self):
        self.att_embeddings_nope.reset_parameters()
        self.attn_layer.reset_parameters()
        self.final_ln.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, batched_data: torch.Tensor,
                pretrain_token: bool = False) -> torch.Tensor:
        # batched_data: [B, K, input_dim]
        tensor = self.att_embeddings_nope(batched_data)   # [B, K, hidden_dim]

        for enc_layer in self.layers:
            tensor = enc_layer(tensor)

        output = self.final_ln(tensor)   # [B, K, hidden_dim]

        if pretrain_token:
            return output

        _target = output[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        split_tensor = torch.split(output, [1, self.seq_len - 1], dim=1)
        node_tensor = split_tensor[0]        # [B, 1, hidden_dim]
        _neighbor_tensor = split_tensor[1]   # [B, K-1, hidden_dim]

        if self.node_only_readout:
            indices = torch.arange(1, self.seq_len, 1)
            neighbor_tensor = _neighbor_tensor[:, indices]
            target = _target[:, indices]
        else:
            target = _target
            neighbor_tensor = _neighbor_tensor

        layer_atten = self.attn_layer(
            torch.cat((target, neighbor_tensor), dim=2))  # [B, K-1, 1]
        layer_atten = F.softmax(layer_atten, dim=1)

        neighbor_tensor = neighbor_tensor * layer_atten
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)  # [B, 1, H]

        output = (node_tensor + neighbor_tensor).squeeze(1)  # [B, H]
        return output


# =============================================================================
# Encoders  (encoders.py)
# =============================================================================

class NeighborNodeTypeEncoder(nn.Module):
    """Embeds integer node-type indices into dense vectors."""

    def __init__(self, node_type_map: Dict[str, int], embedding_dim: int):
        super().__init__()
        num_types = max(node_type_map.values()) + 1
        self.embedding = nn.Embedding(num_embeddings=num_types + 1,
                                      embedding_dim=embedding_dim)

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, type_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(type_indices)


class NeighborHopEncoder(nn.Module):
    """Embeds hop-distance integers into dense vectors."""

    def __init__(self, max_neighbor_hop: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=max_neighbor_hop + 2,
                                      embedding_dim=embedding_dim)

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, hop_distances: torch.Tensor) -> torch.Tensor:
        return self.embedding(hop_distances + 1)


class NeighborTimeEncoder(nn.Module):
    """Two-stage time encoder: positional encoding → linear projection."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required for NeighborTimeEncoder. "
                "Install it with: pip install torch_geometric"
            )
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.mask_vector = nn.Parameter(torch.zeros(embedding_dim))

    def reset_parameters(self):
        self.linear.reset_parameters()
        nn.init.normal_(self.mask_vector, mean=0.0, std=0.02)

    def forward(self, rel_time: torch.Tensor) -> torch.Tensor:
        # rel_time: [B, K]
        B, K = rel_time.shape
        flattened_time = rel_time.view(-1)
        pos_encoded = self.pos_encoder(flattened_time)
        linear_out = self.linear(pos_encoded).view(B, K, -1)

        mask = (rel_time < 0).unsqueeze(-1).float()
        mask_vector = self.mask_vector.unsqueeze(0).unsqueeze(0).expand(B, K, -1)
        return (1 - mask) * linear_out + mask * mask_vector


class NeighborTfsEncoder(nn.Module):
    """
    Encoder for TorchFrame tabular features.
    Requires torch_frame to be installed.
    """

    def __init__(
        self,
        channels: int,
        node_type_map: Dict[str, int],
        col_names_dict: Dict,
        col_stats_dict: Dict,
        torch_frame_model_cls=None,
        torch_frame_model_kwargs: Dict[str, Any] = None,
        default_stype_encoder_cls_kwargs: Dict = None,
    ):
        super().__init__()
        if not HAS_TORCH_FRAME:
            raise ImportError(
                "torch_frame is required for NeighborTfsEncoder. "
                "Install it with: pip install torch_frame[full]"
            )

        if torch_frame_model_cls is None:
            torch_frame_model_cls = TFResNet
        if torch_frame_model_kwargs is None:
            torch_frame_model_kwargs = {"channels": 128, "num_layers": 4}
        if default_stype_encoder_cls_kwargs is None:
            default_stype_encoder_cls_kwargs = {
                torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
                torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
                torch_frame.multicategorical: (
                    torch_frame.nn.MultiCategoricalEmbeddingEncoder, {}),
                torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
                torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
            }

        self.node_type_map = node_type_map
        self.inv_node_type_map = {idx: nt for nt, idx in node_type_map.items()}
        self.channels = channels
        self.encoders = nn.ModuleDict()

        for node_type, stype_dict in col_names_dict.items():
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1])
                for stype in stype_dict.keys()
                if stype in default_stype_encoder_cls_kwargs
            }
            self.encoders[node_type] = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=channels,
                col_stats=col_stats_dict[node_type],
                col_names_dict=stype_dict,
                stype_encoder_dict=stype_encoder_dict,
            )

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(self, batch_dict: Dict, neighbor_types: torch.Tensor) -> torch.Tensor:
        grouped_tfs = batch_dict["grouped_tfs"]
        grouped_indices = batch_dict["grouped_indices"]
        flat_batch_idx = batch_dict["flat_batch_idx"]
        flat_nbr_idx = batch_dict["flat_nbr_idx"]

        B, K = neighbor_types.shape
        N = len(flat_batch_idx)
        device = neighbor_types.device

        encoded_flat_tensor = torch.zeros((N, self.channels), device=device)

        for t_int, big_tf in grouped_tfs.items():
            node_type_str = self.inv_node_type_map[t_int]
            encoder = self.encoders[node_type_str]
            big_tf = big_tf.to(device=device)

            for stype_key, tensor in big_tf.feat_dict.items():
                if isinstance(tensor, torch.Tensor):
                    big_tf.feat_dict[stype_key] = torch.nan_to_num(
                        tensor, nan=0.0, posinf=1e6, neginf=-1e6)

            out_t = encoder(big_tf)
            if out_t.dim() == 3 and out_t.shape[1] == 1:
                out_t = out_t.squeeze(1)

            idx_list = grouped_indices[t_int]
            idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
            encoded_flat_tensor[idx_tensor] = out_t

        output = torch.zeros((B, K, self.channels), device=device)
        indices_i = torch.tensor(flat_batch_idx, dtype=torch.long, device=device)
        indices_j = torch.tensor(flat_nbr_idx, dtype=torch.long, device=device)
        output[indices_i, indices_j] = encoded_flat_tensor
        return output


class GNNPEEncoder(nn.Module):
    """
    GNN-based positional encoder using GIN convolutions on local subgraphs.
    Requires torch_geometric.
    """

    def __init__(self, embedding_dim: int, num_layers: int = 4,
                 pooling: str = "none", pe_dim: int = 0):
        super().__init__()
        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required for GNNPEEncoder."
            )
        self.pooling = pooling.lower()
        self.num_layers = num_layers
        self.layer_embedding_dim = embedding_dim // 4
        self.pe_dim = pe_dim

        if self.pe_dim > 0:
            self.input_proj = nn.Linear(self.pe_dim, self.layer_embedding_dim)
        else:
            self.input_proj = nn.Linear(1, self.layer_embedding_dim)

        self.conv = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.layer_embedding_dim, self.layer_embedding_dim * 2),
                nn.BatchNorm1d(self.layer_embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(self.layer_embedding_dim * 2, self.layer_embedding_dim),
            )
            self.conv.append(GINConv(mlp, train_eps=True))

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.layer_embedding_dim) for _ in range(num_layers)
        ])

        if self.pooling == "cat":
            final_input_dim = self.layer_embedding_dim * num_layers
        elif self.pooling in ["none", "mean", "max"]:
            final_input_dim = self.layer_embedding_dim
        else:
            raise ValueError("pooling must be one of 'none','cat','mean','max'")

        self.final_transform = nn.Linear(final_input_dim, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        for conv in self.conv:
            for layer in conv.nn:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        nn.init.xavier_uniform_(self.final_transform.weight)
        if self.final_transform.bias is not None:
            nn.init.zeros_(self.final_transform.bias)

    def forward(self, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        device = edge_index.device
        total_nodes = batch.size(0)

        if self.pe_dim > 0:
            data = PyGData(edge_index=edge_index, num_nodes=total_nodes)
            transform = T.AddLaplacianEigenvectorPE(k=self.pe_dim)
            data = transform(data)
            x_input = data.laplacian_eigenvector_pe.to(device)
        else:
            x_input = torch.randn(total_nodes, 1, device=device)

        x = self.input_proj(x_input)

        outputs = []
        for i, conv in enumerate(self.conv):
            x_res = x
            x_new = conv(x, edge_index)
            x_new = self.bns[i](x_new)
            x_new = F.relu(x_new)
            x = x_new + x_res
            outputs.append(x)

        if self.pooling == "none":
            x_final = outputs[-1]
        elif self.pooling == "cat":
            x_final = torch.cat(outputs, dim=-1)
        elif self.pooling == "mean":
            x_final = torch.stack(outputs, dim=-1).mean(dim=-1)
        elif self.pooling == "max":
            x_final = torch.stack(outputs, dim=-1).max(dim=-1)[0]

        x = self.final_transform(x_final)

        B = batch.max().item() + 1
        K = total_nodes // B
        return x.view(B, K, -1)


# =============================================================================
# RelGTLayer  (model.py)
# =============================================================================

class RelGTLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        local_num_layers: int,
        global_dim: int,
        num_nodes: int,
        heads: int = 1,
        concat: bool = True,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        edge_dim=None,
        conv_type: str = "local",
        num_centroids=None,
        sample_node_len: int = 100,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.local_num_layers = local_num_layers
        self.heads = heads
        self.concat = concat
        self.ff_dropout = ff_dropout
        self.attn_dropout = attn_dropout
        self.edge_dim = edge_dim
        self.conv_type = conv_type
        self.num_centroids = num_centroids
        self._alpha = None
        self.sample_node_len = sample_node_len

        self.local_module = LocalModule(
            seq_len=self.sample_node_len,
            input_dim=in_channels,
            n_layers=local_num_layers,
            num_heads=heads,
            hidden_dim=out_channels,
            dropout_rate=ff_dropout,
            attention_dropout_rate=attn_dropout,
        )
        self.layer_norm_local = nn.LayerNorm(out_channels)

        if self.conv_type != "local":
            if not HAS_PYG:
                raise ImportError(
                    "torch_geometric is required for global/full conv_type."
                )
            if not HAS_EINOPS:
                raise ImportError(
                    "einops is required for global/full conv_type. "
                    "Install with: pip install einops"
                )
            self.vq = VectorQuantizerEMA(num_centroids, global_dim, decay=0.99)
            c = torch.randint(0, num_centroids, (num_nodes,), dtype=torch.long)
            self.register_buffer("c_idx", c)
            self.attn_fn = F.softmax

            attn_channels = out_channels // heads

            self.lin_proj_g = PyGLinear(in_channels, global_dim)
            self.lin_key_g = PyGLinear(global_dim, heads * attn_channels)
            self.lin_query_g = PyGLinear(global_dim, heads * attn_channels)
            self.lin_value_g = PyGLinear(global_dim, heads * attn_channels)
            self.layer_norm_global = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_type != "local":
            self.lin_proj_g.reset_parameters()
            self.lin_key_g.reset_parameters()
            self.lin_query_g.reset_parameters()
            self.lin_value_g.reset_parameters()
            if hasattr(self, "vq"):
                self.vq.reset_parameters()
        if hasattr(self.local_module, "reset_parameters"):
            self.local_module.reset_parameters()

    def forward(self, x_set: torch.Tensor, x: torch.Tensor,
                node_indices: torch.Tensor) -> torch.Tensor:
        if self.conv_type == "local":
            out = self.local_forward(x_set)
            return self.layer_norm_local(out)
        elif self.conv_type == "global":
            out = self.global_forward(x, node_indices)
            return self.layer_norm_global(out)
        elif self.conv_type == "full":
            out_local = self.layer_norm_local(self.local_forward(x_set))
            out_global = self.layer_norm_global(self.global_forward(x, node_indices))
            return torch.cat([out_local, out_global], dim=1)
        else:
            raise NotImplementedError(f"conv_type '{self.conv_type}' is not supported.")

    def global_forward(self, x: torch.Tensor,
                       batch_idx: torch.Tensor) -> torch.Tensor:
        d, h = self.out_channels, self.heads
        scale = 1.0 / math.sqrt(d)

        q_x = self.lin_proj_g(x)

        k_x = self.vq.get_k().detach().clone()
        v_x = self.vq.get_v().detach().clone()

        q = self.lin_query_g(q_x)
        k = self.lin_key_g(k_x)
        v = self.lin_value_g(v_x)

        q, k, v = map(lambda t: rearrange(t, "n (h d) -> h n d", h=h), (q, k, v))
        dots = torch.einsum("h i d, h j d -> h i j", q, k) * scale

        c, c_count = self.c_idx.unique(return_counts=True)
        centroid_count = torch.zeros(self.num_centroids, dtype=torch.long,
                                     device=x.device)
        centroid_count[c.to(torch.long)] = c_count
        dots = dots + torch.log(centroid_count.view(1, 1, -1).float() + 1e-8)

        attn = self.attn_fn(dots, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)

        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h n d -> n (h d)")

        if self.training:
            x_idx = self.vq.update(q_x)
            self.c_idx[batch_idx] = x_idx.squeeze().to(torch.long)

        return out

    def local_forward(self, x_set: torch.Tensor,
                      pretrain_token: bool = False) -> torch.Tensor:
        return self.local_module(x_set, pretrain_token)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads}, "
            f"local_num_layers={self.local_num_layers})"
        )


# =============================================================================
# RelGT  (model.py)  — wrapped with BaseModel for OpenHGNN registry
# =============================================================================

@register_model("RelGT")
class RelGT(BaseModel):
    r"""Relational Graph Transformer (RelGT).

    Designed for heterogeneous temporal graphs derived from multi-table
    relational data (RelBench benchmark).  Each node is tokenized into five
    components: tabular features, node type, hop distance, relative time, and
    local structural position (GNN-PE).  A local Transformer attends over the
    sampled neighbourhood sequence, and an optional global module attends over
    a learned VQ codebook.

    Paper: https://arxiv.org/abs/2505.10960

    Parameters
    ----------
    num_nodes : int
        Total number of nodes across all types.
    max_neighbor_hop : int
        Maximum hop distance used during neighbour sampling.
    node_type_map : Dict[str, int]
        Mapping from node-type string to integer index.
    col_names_dict : Dict
        Per-node-type TorchFrame column-name dictionaries.
    col_stats_dict : Dict
        Per-node-type TorchFrame column statistics.
    local_num_layers : int
        Number of Transformer encoder layers in LocalModule.
    channels : int
        Hidden channel width.
    out_channels : int
        Output dimension (e.g. 1 for binary classification / regression).
    global_dim : int
        Projection dimension for the global VQ attention.
    heads : int
        Number of attention heads.
    ff_dropout : float
        Dropout applied inside feed-forward and attention layers.
    attn_dropout : float
        Dropout on attention weights.
    conv_type : str
        One of ``"local"``, ``"global"``, or ``"full"``.
    ablate : str
        Name of the tokenisation component to ablate (``"none"`` = no ablation).
    gnn_pe_dim : int
        Laplacian PE dimension for GNNPEEncoder; 0 disables it (random input).
    num_centroids : int
        Codebook size for VQ global attention.
    sample_node_len : int
        Number of tokens K per sample (seed + K-1 neighbours).
    args : Any
        Extra args (unused; kept for compatibility with ``main_node_ddp.py``).
    """

    @classmethod
    def build_model_from_args(cls, args, data_info: Dict[str, Any]) -> "RelGT":
        """
        Build a RelGT instance from OpenHGNN-style args and a data_info dict.

        Parameters
        ----------
        args : argparse.Namespace or similar
            Must contain: channels, out_channels, num_layers, num_heads,
            ff_dropout, attn_dropout, gt_conv_type, num_centroids,
            num_neighbors, gnn_pe_dim, ablate (optional).
        data_info : dict
            Keys: num_nodes, max_neighbor_hop, node_type_map,
                  col_names_dict, col_stats_dict.
        """
        return cls(
            num_nodes=data_info["num_nodes"],
            max_neighbor_hop=data_info["max_neighbor_hop"],
            node_type_map=data_info["node_type_map"],
            col_names_dict=data_info["col_names_dict"],
            col_stats_dict=data_info["col_stats_dict"],
            local_num_layers=getattr(args, "num_layers", 1),
            channels=getattr(args, "channels", 128),
            out_channels=getattr(args, "out_channels", 1),
            global_dim=getattr(args, "channels", 128) // 2,
            heads=getattr(args, "num_heads", 4),
            ff_dropout=getattr(args, "ff_dropout", 0.1),
            attn_dropout=getattr(args, "attn_dropout", 0.1),
            conv_type=getattr(args, "gt_conv_type", "full"),
            ablate=getattr(args, "ablate", "none"),
            gnn_pe_dim=getattr(args, "gnn_pe_dim", 0),
            num_centroids=getattr(args, "num_centroids", 4096),
            sample_node_len=getattr(args, "num_neighbors", 100),
        )

    def __init__(
        self,
        num_nodes: int,
        max_neighbor_hop: int,
        node_type_map: Dict[str, int],
        col_names_dict: Dict,
        col_stats_dict: Dict,
        local_num_layers: int,
        channels: int,
        out_channels: int,
        global_dim: int,
        heads: int = 4,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        conv_type: str = "full",
        ablate: str = "none",
        gnn_pe_dim: int = 0,
        num_centroids: int = 4096,
        sample_node_len: int = 100,
        args: Any = None,
    ):
        super(RelGT, self).__init__()

        self.max_neighbor_hop = max_neighbor_hop
        self.node_type_map = node_type_map

        # --- five tokenisation encoders ---
        self.type_encoder = NeighborNodeTypeEncoder(
            embedding_dim=channels, node_type_map=node_type_map)
        self.hop_encoder = NeighborHopEncoder(
            embedding_dim=channels, max_neighbor_hop=max_neighbor_hop)
        self.time_encoder = NeighborTimeEncoder(embedding_dim=channels)
        self.tfs_encoder = NeighborTfsEncoder(
            channels=channels, node_type_map=node_type_map,
            col_names_dict=col_names_dict, col_stats_dict=col_stats_dict)
        self.pe_encoder = GNNPEEncoder(embedding_dim=channels, pe_dim=gnn_pe_dim)

        self.layer_norm_type = nn.LayerNorm(channels)
        self.layer_norm_hop = nn.LayerNorm(channels)
        self.layer_norm_time = nn.LayerNorm(channels)
        self.layer_norm_tfs = nn.LayerNorm(channels)
        self.layer_norm_pe = nn.LayerNorm(channels)

        hidden_channels = channels

        ablate_key_dict = {"type": 0, "hop": 1, "time": 2, "tfs": 3, "gnn": 4}
        self.ablate_idx = ablate_key_dict.get(ablate, None)
        channel_mult = 5 if self.ablate_idx is None else 4

        self.in_mixture = nn.Sequential(
            nn.Linear(channel_mult * channels, 2 * channels),
            nn.ReLU(),
            nn.Linear(2 * channels, channels),
        )

        self.convs = nn.ModuleList()
        self.ffs = nn.ModuleList()

        _overall_num_layers = 1
        for _ in range(_overall_num_layers):
            self.convs.append(
                RelGTLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    local_num_layers=local_num_layers,
                    global_dim=global_dim,
                    num_nodes=num_nodes,
                    heads=heads,
                    ff_dropout=ff_dropout,
                    attn_dropout=attn_dropout,
                    conv_type=conv_type,
                    num_centroids=num_centroids,
                    sample_node_len=sample_node_len,
                )
            )
            h_times = 2 if conv_type == "full" else 1
            self.ffs.append(
                nn.Sequential(
                    nn.BatchNorm1d(hidden_channels * h_times),
                    nn.Linear(h_times * hidden_channels, hidden_channels * 2),
                    nn.GELU(),
                    nn.Dropout(ff_dropout),
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.Dropout(ff_dropout),
                    nn.BatchNorm1d(hidden_channels),
                )
            )

        if not HAS_PYG:
            raise ImportError("torch_geometric is required for RelGT (MLP head).")
        self.head = PyGMLP(
            channels,
            hidden_channels=channels,
            out_channels=out_channels,
            num_layers=2,
        )

    def reset_parameters(self):
        self.type_encoder.reset_parameters()
        self.hop_encoder.reset_parameters()
        self.time_encoder.reset_parameters()
        self.tfs_encoder.reset_parameters()
        self.pe_encoder.reset_parameters()
        for layer in self.in_mixture:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.head.reset_parameters()

    def forward(
        self,
        neighbor_types: torch.Tensor,    # [B, K]
        node_indices: torch.Tensor,       # [B]
        neighbor_hops: torch.Tensor,      # [B, K]
        neighbor_times: torch.Tensor,     # [B, K]
        grouped_tf_dict: Dict,
        edge_index: Optional[torch.Tensor] = None,  # [2, E]
        batch: Optional[torch.Tensor] = None,        # [B*K]
    ) -> torch.Tensor:
        neighbor_tfs = self.layer_norm_tfs(
            self.tfs_encoder(grouped_tf_dict, neighbor_types))
        neighbor_types_emb = self.layer_norm_type(
            self.type_encoder(neighbor_types.long()))
        neighbor_hops_emb = self.layer_norm_hop(
            self.hop_encoder(neighbor_hops.long()))
        neighbor_times_emb = self.layer_norm_time(
            self.time_encoder(neighbor_times.float()))
        neighbor_subgraph_pe = self.layer_norm_pe(
            self.pe_encoder(edge_index, batch))

        cat_list = [
            neighbor_types_emb, neighbor_hops_emb,
            neighbor_times_emb, neighbor_tfs, neighbor_subgraph_pe,
        ]
        if self.ablate_idx is not None:
            cat_list.pop(self.ablate_idx)

        x_set = torch.cat(cat_list, dim=-1)    # [B, K, channel_mult * channels]
        x_set = self.in_mixture(x_set)          # [B, K, channels]

        x = x_set[:, 0, :]                      # seed-node representation [B, channels]
        for i, conv in enumerate(self.convs):
            x_set = conv(x_set, x, node_indices)
            x_set = self.ffs[i](x_set)

        return self.head(x_set)                  # [B, out_channels]
