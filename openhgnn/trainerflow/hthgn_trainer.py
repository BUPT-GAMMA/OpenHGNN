"""OpenHGNN trainerflow reproducing the HTHGN public repository protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import statistics

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from openhgnn.dataset import build_dataset
from openhgnn.dataset.hthgn_dataset import load_or_construct_hypergraphs
from openhgnn.models import build_model
from openhgnn.models.HTHGN import LinkPredictor
from openhgnn.trainerflow import BaseFlow, register_flow


def _safe_index(tensor: torch.Tensor, index) -> torch.Tensor:
    return tensor[
        torch.as_tensor(index, dtype=torch.long, device=tensor.device)
    ]


def construct_positive_graph(
    graph: dgl.DGLHeteroGraph,
    ratio: float,
) -> dgl.DGLHeteroGraph:
    edge_data = {}
    num_nodes = {
        ntype: graph.num_nodes(ntype)
        for ntype in graph.ntypes
    }
    for canonical_etype in graph.canonical_etypes:
        # Match the authors' public implementation: hyperedge relations
        # participate in message passing, but not in link-prediction loss.
        if "hyper" in canonical_etype[1]:
            continue

        edge_count = graph.num_edges(etype=canonical_etype)
        src, dst = graph.edges(etype=canonical_etype)
        selected = np.random.permutation(edge_count)[
            : int(ratio * edge_count)
        ]
        edge_data[canonical_etype] = (
            _safe_index(src, selected),
            _safe_index(dst, selected),
        )
    return dgl.heterograph(
        edge_data,
        num_nodes_dict=num_nodes,
        device=graph.device,
    )


def construct_negative_graph(
    graph: dgl.DGLHeteroGraph,
    negative_count: int,
) -> dgl.DGLHeteroGraph:
    edge_data = {}
    num_nodes = {
        ntype: graph.num_nodes(ntype)
        for ntype in graph.ntypes
    }
    for canonical_etype in graph.canonical_etypes:
        if "hyper" in canonical_etype[1]:
            continue
        edge_count = graph.num_edges(etype=canonical_etype)
        edge_data[canonical_etype] = (
            dgl.sampling.global_uniform_negative_sampling(
                graph,
                edge_count * negative_count,
                etype=canonical_etype,
            )
        )
    return dgl.heterograph(
        edge_data,
        num_nodes_dict=num_nodes,
        device=graph.device,
    )


def binary_cross_entropy_loss(
    predictor: LinkPredictor,
    graph: dgl.DGLHeteroGraph,
    features: Dict[str, torch.Tensor],
    positive_ratio: float,
    negative_count: int,
) -> torch.Tensor:
    positive_graph = construct_positive_graph(graph, positive_ratio)
    negative_graph = construct_negative_graph(graph, negative_count)
    positive_score = predictor(positive_graph, features)
    negative_score = predictor(negative_graph, features)
    prediction = torch.cat((positive_score, negative_score), dim=0)
    labels = torch.cat(
        (
            torch.ones(positive_score.shape[0], device=graph.device),
            torch.zeros(negative_score.shape[0], device=graph.device),
        ),
        dim=0,
    )
    return F.binary_cross_entropy_with_logits(prediction, labels)


@torch.no_grad()
def evaluate_graph_lists(
    predictor: LinkPredictor,
    positive_graphs: List[dgl.DGLHeteroGraph],
    negative_graphs: List[dgl.DGLHeteroGraph],
    embeddings: List[Dict[str, torch.Tensor]],
) -> Tuple[List[float], List[float]]:
    predictor.eval()
    auc_scores = []
    ap_scores = []
    for positive_graph, negative_graph, embedding in zip(
        positive_graphs,
        negative_graphs,
        embeddings,
    ):
        positive_score = (
            predictor(positive_graph, embedding)
            .sigmoid()
            .detach()
            .cpu()
            .numpy()
        )
        negative_score = (
            predictor(negative_graph, embedding)
            .sigmoid()
            .detach()
            .cpu()
            .numpy()
        )
        predictions = np.hstack([positive_score, negative_score])
        labels = np.hstack(
            [
                np.ones(len(positive_score)),
                np.zeros(len(negative_score)),
            ]
        )
        auc_scores.append(float(roc_auc_score(labels, predictions)))
        ap_scores.append(
            float(average_precision_score(labels, predictions))
        )
    return auc_scores, ap_scores


class _EarlyStopping:
    def __init__(self, patience: int, checkpoint_path: Path) -> None:
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = -1

    def step(self, loss_value: float, model: nn.Module, epoch: int) -> None:
        score = -float(loss_value)
        if self.best_score is None or score >= self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
            return
        self.counter += 1
        print(
            f"EarlyStopping counter: {self.counter} "
            f"out of {self.patience}"
        )
        if self.counter >= self.patience:
            self.early_stop = True


@register_flow("hthgn_trainer")
class HTHGNTrainer(BaseFlow):
    def __init__(self, args) -> None:
        self.args = args
        self.logger = getattr(args, "logger", None)
        gpu = int(getattr(args, "gpu", -1))
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu}")
        else:
            self.device = torch.device("cpu")
        self.args.device = self.device
        self.dataset = build_dataset(
            getattr(args, "dataset_name", args.dataset),
            "link_prediction",
            logger=self.logger,
            args=args,
        )

    def _build_model(
        self,
        hypergraphs: List[dgl.DGLHeteroGraph],
    ) -> nn.Module:
        self.args.input_dim = self.dataset.input_dim
        encoder_class = build_model("HTHGN")
        encoder = encoder_class.build_model_from_args(
            self.args,
            hypergraphs[0],
        )
        predictor = LinkPredictor(int(self.args.hidden_dim))
        return nn.ModuleDict(
            {"encoder": encoder, "predictor": predictor}
        ).to(self.device)

    def _window_embeddings(
        self,
        model: nn.ModuleDict,
        hypergraphs: List[dgl.DGLHeteroGraph],
        features: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        time_window = int(self.args.time_window)
        embeddings = []
        for left in range(len(hypergraphs) - time_window):
            output = model["encoder"](
                hypergraphs[left : left + time_window],
                features[left : left + time_window],
            )
            embeddings.append(
                {
                    ntype: sum(
                        output[ntype][str(t)]
                        for t in range(time_window)
                    )
                    for ntype in hypergraphs[0].ntypes
                }
            )
        return embeddings

    def _train_one_repeat(
        self,
        repeat_index: int,
    ) -> Tuple[float, float, float, float, int]:
        features = [
            {
                ntype: value.to(self.device)
                for ntype, value in snapshot.items()
            }
            for snapshot in self.dataset.features
        ]
        cache_dir = str(
            getattr(
                self.args,
                "cache_dir",
                Path(self.args.openhgnn_dir)
                / "dataset"
                / "HTHGN"
                / "cache",
            )
        )
        hypergraphs = load_or_construct_hypergraphs(
            graphs=self.dataset.label_graphs,
            cache_dir=cache_dir,
            dataset_key=self.dataset.dataset_key,
            k=int(self.args.k),
            p=int(self.args.p),
            keep_original=bool(self.args.keep_original),
            hypernode_type=str(self.args.hypernode_type),
            hyperedge_type=str(self.args.hyperedge_type),
        )
        hypergraphs = [graph.to(self.device) for graph in hypergraphs]

        model = self._build_model(hypergraphs)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.args.lr),
            weight_decay=float(self.args.weight_decay),
        )
        checkpoint_path = (
            Path(self.args.output_dir)
            / (
                f"HTHGN_{self.dataset.dataset_key}_"
                f"seed{int(self.args.seed)}_repeat{repeat_index}.pt"
            )
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        early_stopping = _EarlyStopping(
            int(self.args.patience),
            checkpoint_path,
        )

        sequence_length = len(hypergraphs)
        prediction_length = int(self.args.prediction_length)
        sequence_end = sequence_length - prediction_length
        time_window = int(self.args.time_window)

        for epoch in range(int(self.args.max_epoch)):
            model.train()
            epoch_loss = 0.0
            last_loss = None

            # Public repository behavior: test-target snapshots also
            # participate in training.
            for left in range(sequence_length - time_window):
                output = model["encoder"](
                    hypergraphs[left : left + time_window],
                    features[left : left + time_window],
                )
                collapsed = {
                    ntype: sum(
                        output[ntype][str(t)]
                        for t in range(time_window)
                    )
                    for ntype in hypergraphs[0].ntypes
                }
                loss = binary_cross_entropy_loss(
                    model["predictor"],
                    hypergraphs[time_window + left],
                    collapsed,
                    positive_ratio=float(self.args.edge_ratio),
                    negative_count=int(self.args.negative_count),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = loss
                epoch_loss += float(loss.item())

            if last_loss is None:
                raise RuntimeError("No HTHGN training window was generated")

            print(
                "Epoch: {:03d} | Loss: {:.4f} | "
                "Lr: {:.6f} | Count: {:02d}".format(
                    epoch,
                    epoch_loss,
                    optimizer.param_groups[0]["lr"],
                    early_stopping.counter,
                )
            )
            # Public repository behavior: checkpoint by final-window loss.
            early_stopping.step(
                float(last_loss.item()),
                model,
                epoch,
            )
            if early_stopping.early_stop:
                break

        model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        model.eval()
        with torch.no_grad():
            embedding_list = self._window_embeddings(
                model,
                hypergraphs,
                features,
            )

        positive = [
            graph.to(self.device)
            for graph in self.dataset.positive_graphs[sequence_end:]
        ]
        negative = [
            graph.to(self.device)
            for graph in self.dataset.negative_graphs[sequence_end:]
        ]
        new_positive = [
            graph.to(self.device)
            for graph in self.dataset.new_positive_graphs[sequence_end:]
        ]
        new_negative = [
            graph.to(self.device)
            for graph in self.dataset.new_negative_graphs[sequence_end:]
        ]
        test_embeddings = embedding_list[
            sequence_end - time_window :
        ]

        auc_values, ap_values = evaluate_graph_lists(
            model["predictor"],
            positive,
            negative,
            test_embeddings,
        )
        new_auc_values, new_ap_values = evaluate_graph_lists(
            model["predictor"],
            new_positive,
            new_negative,
            test_embeddings,
        )

        auc = statistics.mean(auc_values)
        ap = statistics.mean(ap_values)
        new_auc = statistics.mean(new_auc_values)
        new_ap = statistics.mean(new_ap_values)

        print(
            "Prediction AUC/AP: {:.4f}/{:.4f} | "
            "Prediction New AUC/AP: {:.4f}/{:.4f}".format(
                auc,
                ap,
                new_auc,
                new_ap,
            )
        )
        return (
            auc,
            ap,
            new_auc,
            new_ap,
            early_stopping.best_epoch,
        )

    def train(self):
        repeats = int(getattr(self.args, "num_repeats", 1))
        results = [
            self._train_one_repeat(index)
            for index in range(repeats)
        ]
        scores = np.asarray([row[:4] for row in results], dtype=float)
        mean = scores.mean(axis=0)
        std = scores.std(axis=0)
        minimum = scores.min(axis=0)
        maximum = scores.max(axis=0)

        names = ("AUC", "AP", "New AUC", "New AP")
        print("\n===== HTHGN public-protocol summary =====")
        for name, avg, deviation in zip(names, mean, std):
            print(
                f"{name}: {avg * 100:.2f} ± "
                f"{deviation * 100:.2f}"
            )

        return {
            "metric": {
                "test": {
                    "AUC": float(mean[0]),
                    "AP": float(mean[1]),
                    "New AUC": float(mean[2]),
                    "New AP": float(mean[3]),
                },
                "std": {
                    "AUC": float(std[0]),
                    "AP": float(std[1]),
                    "New AUC": float(std[2]),
                    "New AP": float(std[3]),
                },
                "min": {
                    "AUC": float(minimum[0]),
                    "AP": float(minimum[1]),
                    "New AUC": float(minimum[2]),
                    "New AP": float(minimum[3]),
                },
                "max": {
                    "AUC": float(maximum[0]),
                    "AP": float(maximum[1]),
                    "New AUC": float(maximum[2]),
                    "New AP": float(maximum[3]),
                },
            },
            "epoch": int(results[-1][4]),
        }
