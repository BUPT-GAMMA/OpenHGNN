"""HTHGN model adapted from the authors' public implementation."""

from __future__ import annotations

import math
from typing import Dict, List

import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BaseModel, register_model


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, time_window: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

        pe = self._positional_encoding(hidden_dim, time_window)
        self.register_buffer(
            "position_encoding",
            torch.tensor(pe, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _positional_encoding(dim: int, length: int) -> np.ndarray:
        pe = np.zeros((length, dim), dtype=np.float32)
        for t in range(length):
            for k in range(0, dim, 2):
                div_term = math.exp(k * -math.log(100000.0) / dim)
                pe[t, k] = math.sin((t + 1) * div_term)
                if k + 1 < dim:
                    pe[t, k + 1] = math.cos((t + 1) * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)
        h = self.proj(x) + self.position_encoding
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)
        score = F.softmax(torch.matmul(q, k.permute(0, 2, 1)), dim=-1)
        return F.relu(self.fc(torch.matmul(score, v)))


class HeterogeneousAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        relation_names: List[str],
        node_types: List[str],
        num_heads: int = 4,
        activation=None,
        self_loop: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError(
                f"output_dim={output_dim} must be divisible by num_heads={num_heads}"
            )
        self.node_types = node_types
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        head_dim = output_dim // num_heads

        self.conv = dglnn.HeteroGraphConv(
            {
                relation: dglnn.GATv2Conv(
                    input_dim,
                    head_dim,
                    num_heads=num_heads,
                )
                for relation in relation_names
            },
            aggregate=self._relation_aggregate,
        )

        self.semantic_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 1, bias=False),
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter("bias", None)

        if self_loop:
            self.loop_weight = nn.Parameter(torch.empty(input_dim, output_dim))
            nn.init.xavier_uniform_(
                self.loop_weight,
                gain=nn.init.calculate_gain("relu"),
            )
        else:
            self.register_parameter("loop_weight", None)

        gain = nn.init.calculate_gain("relu")
        for layer in self.semantic_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=gain)

    def _relation_aggregate(
        self,
        tensors: List[torch.Tensor],
        destination_type: str,
    ) -> torch.Tensor:
        if not tensors:
            raise RuntimeError(
                f"No incoming relation tensor for node type {destination_type!r}"
            )
        h = torch.stack(tensors, dim=1)
        h = h.flatten(-2)
        weights = self.semantic_projection(h).mean(0)
        beta = torch.softmax(weights, dim=0)
        beta = beta.expand((h.shape[0],) + beta.shape)
        return F.relu((beta * h).sum(1))

    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        graph = graph.local_var()
        if graph.is_block:
            inputs_src = inputs
            inputs_dst = {
                ntype: value[: graph.number_of_dst_nodes(ntype)]
                for ntype, value in inputs.items()
            }
        else:
            inputs_src = inputs_dst = inputs

        outputs = self.conv(graph, inputs_src)
        result: Dict[str, torch.Tensor] = {}
        for ntype, h in outputs.items():
            if self.loop_weight is not None:
                h = h + torch.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias is not None:
                h = h + self.bias
            if self.activation is not None:
                h = self.activation(h)
            result[ntype] = self.dropout(h)
        return result


class HTHGNLayer(nn.Module):
    def __init__(
        self,
        graph: dgl.DGLHeteroGraph,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        timeframe: List[str],
        norm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.timeframe = timeframe
        self.node_types = list(graph.ntypes)
        relation_names = sorted(set(graph.etypes))

        self.heterogeneous_attention = HeterogeneousAttention(
            hidden_dim,
            hidden_dim,
            relation_names,
            self.node_types,
            num_heads=num_heads,
            activation=F.relu,
            self_loop=False,
            dropout=dropout,
        )
        self.temporal_attention = nn.ModuleDict(
            {
                ntype: TemporalAttention(hidden_dim, hidden_dim, len(timeframe))
                for ntype in self.node_types
            }
        )
        self.residual_fc = nn.ModuleDict(
            {
                ntype: nn.Linear(input_dim, hidden_dim)
                for ntype in self.node_types
            }
        )
        self.residual_weight = nn.ParameterDict(
            {
                ntype: nn.Parameter(torch.randn(1))
                for ntype in self.node_types
            }
        )
        self.norm = norm
        if norm:
            self.norm_layer = nn.ModuleDict(
                {ntype: nn.LayerNorm(hidden_dim) for ntype in self.node_types}
            )

        gain = nn.init.calculate_gain("relu")
        for layer in self.residual_fc.values():
            nn.init.xavier_normal_(layer.weight, gain=gain)

    def forward(
        self,
        graph_list: List[dgl.DGLHeteroGraph],
        features: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        intermediate = {ntype: {} for ntype in self.node_types}

        for t, graph in enumerate(graph_list):
            timestamp = str(t)
            current = {
                ntype: features[ntype][timestamp]
                for ntype in graph.ntypes
            }
            aggregated = self.heterogeneous_attention(graph, current)
            for ntype in graph.ntypes:
                intermediate[ntype][timestamp] = aggregated[ntype]

        temporal_outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for ntype, by_time in intermediate.items():
            stacked = torch.stack(
                [by_time[timestamp] for timestamp in self.timeframe],
                dim=0,
            )
            h = self.temporal_attention[ntype](stacked).permute(1, 0, 2)
            temporal_outputs[ntype] = {
                timestamp: h[index]
                for index, timestamp in enumerate(self.timeframe)
            }

        outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for ntype, by_time in temporal_outputs.items():
            outputs[ntype] = {}
            alpha = torch.sigmoid(self.residual_weight[ntype])
            for timestamp in self.timeframe:
                value = (
                    by_time[timestamp] * alpha
                    + self.residual_fc[ntype](features[ntype][timestamp])
                    * (1.0 - alpha)
                )
                if self.norm:
                    value = self.norm_layer[ntype](value)
                outputs[ntype][timestamp] = value
        return outputs


@register_model("HTHGN")
class HTHGN(BaseModel):
    @classmethod
    def build_model_from_args(
        cls,
        args,
        graph: dgl.DGLHeteroGraph,
    ) -> "HTHGN":
        return cls(
            graph=graph,
            input_dim=int(args.input_dim),
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            num_heads=int(args.num_heads),
            time_window=int(args.time_window),
            norm=bool(getattr(args, "norm", False)),
            dropout=float(args.dropout),
        )

    def __init__(
        self,
        graph: dgl.DGLHeteroGraph,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        time_window: int,
        norm: bool,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.timeframe = [str(i) for i in range(time_window)]
        self.node_types = list(graph.ntypes)

        self.adaptation = nn.ModuleDict(
            {
                ntype: nn.Linear(input_dim, hidden_dim)
                for ntype in self.node_types
            }
        )
        self.layers = nn.ModuleList(
            [
                HTHGNLayer(
                    graph,
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    self.timeframe,
                    norm,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        graph_list: List[dgl.DGLHeteroGraph],
        feature_list: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        if len(graph_list) != len(feature_list):
            raise ValueError("graph_list and feature_list must have equal length")
        if len(graph_list) != len(self.timeframe):
            raise ValueError(
                f"Expected {len(self.timeframe)} snapshots, got {len(graph_list)}"
            )

        device = next(self.parameters()).device
        inputs: Dict[str, Dict[str, torch.Tensor]] = {
            ntype: {} for ntype in self.node_types
        }
        for t, (graph, snapshot_features) in enumerate(
            zip(graph_list, feature_list)
        ):
            timestamp = str(t)
            for ntype in graph.ntypes:
                if ntype in snapshot_features:
                    inputs[ntype][timestamp] = self.adaptation[ntype](
                        snapshot_features[ntype]
                    )
                else:
                    inputs[ntype][timestamp] = torch.zeros(
                        graph.num_nodes(ntype),
                        self.hidden_dim,
                        device=device,
                    )

        for layer in self.layers:
            inputs = layer(graph_list, inputs)
        return inputs


class LinkPredictor(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2, input_dim)
        self.fc2 = nn.Linear(input_dim, num_classes)

    def _apply_edges(self, edges):
        x = torch.cat([edges.src["h"], edges.dst["h"]], dim=1)
        return {"score": self.fc2(F.relu(self.fc1(x)))}

    def forward(
        self,
        graph: dgl.DGLHeteroGraph,
        node_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        with graph.local_scope():
            graph.ndata["h"] = {
                ntype: node_features[ntype]
                for ntype in graph.ntypes
            }
            scores = []
            for canonical_etype in graph.canonical_etypes:
                graph.apply_edges(self._apply_edges, etype=canonical_etype)
                scores.append(graph.edges[canonical_etype].data["score"])
            if not scores:
                return torch.empty(
                    0,
                    device=next(self.parameters()).device,
                )
            return torch.cat(scores).flatten()
