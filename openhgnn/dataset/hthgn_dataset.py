"""HTHGN temporal graph dataset and public-protocol sampling utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import os
import pickle

import dgl
import numpy as np
import torch

from . import BaseDataset, register_dataset


def _safe_index(tensor: torch.Tensor, index) -> torch.Tensor:
    return tensor[
        torch.as_tensor(index, dtype=torch.long, device=tensor.device)
    ]



def build_bidirectional_label_graphs(
    graphs: List[dgl.DGLHeteroGraph],
) -> List[dgl.DGLHeteroGraph]:
    """Build the bidirectional glabel graphs used by the public repository."""
    label_graphs = []

    for graph in graphs:
        num_nodes = {
            ntype: graph.num_nodes(ntype)
            for ntype in graph.ntypes
        }
        edge_data = {}

        for canonical_etype in graph.canonical_etypes:
            source_type, relation, destination_type = canonical_etype
            src, dst = graph.edges(etype=canonical_etype)

            # Original direction.
            edge_data[canonical_etype] = (src, dst)

            # Public repository load_detection() behavior.
            reverse_etype = (
                destination_type,
                f"{relation}_r",
                source_type,
            )
            edge_data[reverse_etype] = (dst, src)

        label_graphs.append(
            dgl.heterograph(
                edge_data,
                num_nodes_dict=num_nodes,
                device=graph.device,
            )
        )

    return label_graphs


def build_public_prediction_graphs(
    graphs: List[dgl.DGLHeteroGraph],
    negative_ratio: float = 0.3,
) -> Tuple[List[dgl.DGLHeteroGraph], List[dgl.DGLHeteroGraph]]:
    positive_graphs = []
    negative_graphs = []
    for graph in graphs:
        num_nodes = {
            ntype: graph.num_nodes(ntype)
            for ntype in graph.ntypes
        }
        negative_edges = {}
        for canonical_etype in graph.canonical_etypes:
            edge_count = graph.num_edges(etype=canonical_etype)
            src_neg, dst_neg = dgl.sampling.global_uniform_negative_sampling(
                graph,
                edge_count,
                etype=canonical_etype,
            )
            # DGL may return fewer negative edges than requested.
            # Sample only from the candidates that were actually returned.
            available = min(
                int(src_neg.shape[0]),
                int(dst_neg.shape[0]),
            )
            sample_count = min(
                int(negative_ratio * edge_count),
                available,
            )
            selected = np.random.permutation(available)[:sample_count]
            negative_edges[canonical_etype] = (
                _safe_index(src_neg, selected),
                _safe_index(dst_neg, selected),
            )
        positive_graphs.append(graph)
        negative_graphs.append(
            dgl.heterograph(
                negative_edges,
                num_nodes_dict=num_nodes,
                device=graph.device,
            )
        )
    return positive_graphs, negative_graphs


def build_public_new_link_graphs(
    graphs: List[dgl.DGLHeteroGraph],
) -> Tuple[List[dgl.DGLHeteroGraph], List[dgl.DGLHeteroGraph]]:
    positive_graphs = []
    negative_graphs = []

    first = graphs[0]
    first_num_nodes = {
        ntype: first.num_nodes(ntype)
        for ntype in first.ntypes
    }
    first_negative = {}
    for canonical_etype in first.canonical_etypes:
        edge_count = first.num_edges(etype=canonical_etype)
        first_negative[canonical_etype] = (
            dgl.sampling.global_uniform_negative_sampling(
                first,
                edge_count,
                etype=canonical_etype,
            )
        )
    positive_graphs.append(first)
    negative_graphs.append(
        dgl.heterograph(
            first_negative,
            num_nodes_dict=first_num_nodes,
            device=first.device,
        )
    )

    for index in range(1, len(graphs)):
        current = graphs[index]
        previous = graphs[index - 1]
        num_nodes = {
            ntype: current.num_nodes(ntype)
            for ntype in current.ntypes
        }
        positive_edges = {}
        negative_edges = {}

        for canonical_etype in current.canonical_etypes:
            current_adj = current[canonical_etype].adj().to_dense()
            previous_adj = previous[canonical_etype].adj().to_dense()
            new_edges = np.transpose(
                np.asarray(
                    np.where(
                        (current_adj - previous_adj).cpu().numpy() > 0
                    )
                )
            )
            edge_count = int(new_edges.shape[0])
            if edge_count:
                positive_edges[canonical_etype] = (
                    torch.as_tensor(
                        new_edges[:, 0],
                        dtype=current.idtype,
                        device=current.device,
                    ),
                    torch.as_tensor(
                        new_edges[:, 1],
                        dtype=current.idtype,
                        device=current.device,
                    ),
                )
                negative_edges[canonical_etype] = (
                    dgl.sampling.global_uniform_negative_sampling(
                        current,
                        edge_count,
                        etype=canonical_etype,
                    )
                )
            else:
                empty = torch.empty(
                    0,
                    dtype=current.idtype,
                    device=current.device,
                )
                positive_edges[canonical_etype] = (empty, empty)
                negative_edges[canonical_etype] = (empty, empty)

        positive_graphs.append(
            dgl.heterograph(
                positive_edges,
                num_nodes_dict=num_nodes,
                device=current.device,
            )
        )
        negative_graphs.append(
            dgl.heterograph(
                negative_edges,
                num_nodes_dict=num_nodes,
                device=current.device,
            )
        )
    return positive_graphs, negative_graphs


def construct_hypergraph(
    graph: dgl.DGLHeteroGraph,
    k: int = 3,
    p: int = 100,
    keep_original: bool = True,
    hypernode_type: str = "_all",
    hyperedge_type: str = "ring",
) -> dgl.DGLHeteroGraph:
    num_nodes = {
        ntype: graph.num_nodes(ntype)
        for ntype in graph.ntypes
    }
    if hypernode_type == "_all":
        for ntype in graph.ntypes:
            num_nodes[f"hyper_{ntype}"] = graph.num_nodes(ntype)
    else:
        if hypernode_type not in graph.ntypes:
            raise ValueError(
                f"Unknown hypernode_type={hypernode_type!r}; "
                f"available types are {graph.ntypes}"
            )
        num_nodes[f"hyper_{hypernode_type}"] = graph.num_nodes(
            hypernode_type
        )

    homogeneous = dgl.to_homogeneous(graph)
    if hyperedge_type == "hop":
        expanded = dgl.to_simple(
            dgl.merge(
                [
                    dgl.khop_graph(homogeneous, distance)
                    for distance in range(1, k + 1)
                ]
            )
        )
    elif hyperedge_type == "ring":
        expanded = dgl.to_simple(dgl.khop_graph(homogeneous, k))
    else:
        raise ValueError(
            f"hyperedge_type must be 'hop' or 'ring', got {hyperedge_type!r}"
        )

    expanded.edata["_ID"] = torch.arange(
        expanded.num_edges(),
        dtype=homogeneous.idtype,
    )
    expanded.edata["_TYPE"] = torch.full(
        (expanded.num_edges(),),
        fill_value=len(graph.etypes),
        dtype=homogeneous.idtype,
    )
    heterogeneous_expanded = dgl.to_heterogeneous(
        expanded,
        ntypes=list(graph.ntypes),
        etypes=list(graph.etypes) + ["hyperedge"],
    )

    if p > 0:
        heterogeneous_expanded = dgl.sampling.sample_neighbors(
            heterogeneous_expanded,
            {
                ntype: heterogeneous_expanded.nodes(ntype)
                for ntype in heterogeneous_expanded.ntypes
            },
            p,
        )

    graph_data = (
        {
            canonical_etype: graph.edges(etype=canonical_etype)
            for canonical_etype in graph.canonical_etypes
        }
        if keep_original
        else {}
    )

    for source_type, relation, destination_type in (
        heterogeneous_expanded.canonical_etypes
    ):
        if hypernode_type == "_all" or source_type == hypernode_type:
            src, dst = heterogeneous_expanded.edges(
                etype=(source_type, relation, destination_type)
            )
            graph_data[
                (f"hyper_{source_type}", relation, destination_type)
            ] = (src, dst)
            graph_data[
                (
                    destination_type,
                    f"{relation}_r",
                    f"hyper_{source_type}",
                )
            ] = (dst, src)
        elif destination_type == hypernode_type:
            src, dst = heterogeneous_expanded.edges(
                etype=(source_type, relation, destination_type)
            )
            graph_data[
                (source_type, relation, f"hyper_{destination_type}")
            ] = (src, dst)
            graph_data[
                (
                    f"hyper_{destination_type}",
                    f"{relation}_r",
                    source_type,
                )
            ] = (dst, src)

    return dgl.heterograph(graph_data, num_nodes_dict=num_nodes)


def load_or_construct_hypergraphs(
    graphs: List[dgl.DGLHeteroGraph],
    cache_dir: str,
    dataset_key: str,
    k: int,
    p: int,
    keep_original: bool,
    hypernode_type: str,
    hyperedge_type: str,
) -> List[dgl.DGLHeteroGraph]:
    cache_root = Path(os.path.expanduser(cache_dir)).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / (
        f"Hyper_{dataset_key}_{k}_{hypernode_type}_"
        f"{hyperedge_type}_{p}_{keep_original}.pkl"
    )

    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    hypergraphs = []
    for index, graph in enumerate([g.to("cpu") for g in graphs]):
        print(f"Constructing snapshot {index} HTHG on CPU...")
        hypergraphs.append(
            construct_hypergraph(
                graph,
                k=k,
                p=p,
                keep_original=keep_original,
                hypernode_type=hypernode_type,
                hyperedge_type=hyperedge_type,
            )
        )

    with cache_path.open("wb") as handle:
        pickle.dump(hypergraphs, handle)
    return hypergraphs


@register_dataset("hthgn_dataset")
class HTHGNDataset(BaseDataset):
    SUPPORTED = {
        "yelp4HTHGN": "yelp",
    }

    def __init__(self, dataset_name: str, logger=None, args=None) -> None:
        super().__init__(logger=logger)
        if dataset_name not in self.SUPPORTED:
            raise ValueError(f"Unsupported HTHGN dataset {dataset_name!r}")

        self.dataset_name = dataset_name
        self.dataset_key = self.SUPPORTED[dataset_name]
        self.args = args
        self.data_dir = Path(
            os.path.expanduser(
                str(getattr(args, "data_dir", "./openhgnn/dataset/HTHGN"))
            )
        ).resolve()

        graph_path = self.data_dir / f"{self.dataset_key}.bin"
        feature_path = (
            self.data_dir
            / "mp2vec"
            / f"{self.dataset_key}_mpfeat.bin"
        )
        missing = [
            str(path)
            for path in (graph_path, feature_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "HTHGN preprocessed data is missing:\n  "
                + "\n  ".join(missing)
                + "\nPass data_dir=... to Experiment."
            )

        self.graphs, _ = dgl.load_graphs(str(graph_path))
        feature_graphs, _ = dgl.load_graphs(str(feature_path))
        if len(self.graphs) != len(feature_graphs):
            raise ValueError(
                "Graph snapshot count does not match feature snapshot count: "
                f"{len(self.graphs)} vs {len(feature_graphs)}"
            )

        self.features: List[Dict[str, torch.Tensor]] = [
            feature_graph.ndata["mp_feat"]
            for feature_graph in feature_graphs
        ]
        for index, (graph, features) in enumerate(
            zip(self.graphs, self.features)
        ):
            missing_types = set(graph.ntypes) - set(features)
            if missing_types:
                raise ValueError(
                    f"Snapshot {index} is missing features for "
                    f"{sorted(missing_types)}"
                )

        self.label_graphs = build_bidirectional_label_graphs(self.graphs)

        self.positive_graphs, self.negative_graphs = (
            build_public_prediction_graphs(
                self.graphs,
                negative_ratio=float(
                    getattr(args, "eval_negative_ratio", 0.3)
                ),
            )
        )
        (
            self.new_positive_graphs,
            self.new_negative_graphs,
        ) = build_public_new_link_graphs(self.graphs)
        self.input_dim = int(
            next(iter(self.features[0].values())).shape[1]
        )
