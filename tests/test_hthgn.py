"""Smoke tests for the OpenHGNN HTHGN integration."""

import dgl
import numpy as np
import torch

from openhgnn.dataset import DATASET_REGISTRY
from openhgnn.dataset.hthgn_dataset import (
    build_public_prediction_graphs,
)
from openhgnn.models import MODEL_REGISTRY
from openhgnn.models.HTHGN import HTHGN, LinkPredictor
from openhgnn.trainerflow import FLOW_REGISTRY


def _tiny_hypergraph():
    index = torch.tensor([0, 1], dtype=torch.int64)

    return dgl.heterograph(
        {
            ("u", "ub", "b"): (index, index),
            ("b", "ub_r", "u"): (index, index),
            ("hyper_u", "hyperedge", "b"): (index, index),
            ("b", "hyperedge_r", "hyper_u"): (index, index),
        },
        num_nodes_dict={
            "u": 2,
            "b": 2,
            "hyper_u": 2,
        },
    )


def test_hthgn_registration():
    assert "HTHGN" in MODEL_REGISTRY
    assert "hthgn_dataset" in DATASET_REGISTRY
    assert "hthgn_trainer" in FLOW_REGISTRY


def test_public_negative_sampling_handles_shortfall():
    # A nearly complete 3x3 bipartite graph has only one valid
    # negative edge, although eight candidates are requested.
    src = []
    dst = []

    for source in range(3):
        for destination in range(3):
            if (source, destination) != (2, 2):
                src.append(source)
                dst.append(destination)

    graph = dgl.heterograph(
        {
            ("u", "ub", "b"): (
                torch.tensor(src, dtype=torch.int64),
                torch.tensor(dst, dtype=torch.int64),
            )
        },
        num_nodes_dict={
            "u": 3,
            "b": 3,
        },
    )

    np.random.seed(4)
    torch.manual_seed(4)
    dgl.seed(4)

    positive, negative = build_public_prediction_graphs(
        [graph],
        negative_ratio=0.3,
    )

    assert positive[0].num_edges() == 8
    assert negative[0].num_edges() <= 1


def test_hthgn_forward_backward():
    torch.manual_seed(0)

    graph = _tiny_hypergraph()
    graph_list = [graph, graph, graph]

    feature_list = [
        {
            "u": torch.randn(2, 4),
            "b": torch.randn(2, 4),
            # hyper_u intentionally has no input feature.
        }
        for _ in range(3)
    ]

    model = HTHGN(
        graph=graph,
        input_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        time_window=3,
        norm=False,
        dropout=0.0,
    )

    predictor = LinkPredictor(input_dim=8)

    output = model(graph_list, feature_list)

    assert set(output) == set(graph.ntypes)

    for ntype in graph.ntypes:
        assert set(output[ntype]) == {"0", "1", "2"}

        for timestamp in ("0", "1", "2"):
            assert output[ntype][timestamp].shape == (
                graph.num_nodes(ntype),
                8,
            )
            assert torch.isfinite(
                output[ntype][timestamp]
            ).all()

    collapsed = {
        ntype: sum(
            output[ntype][str(timestamp)]
            for timestamp in range(3)
        )
        for ntype in graph.ntypes
    }

    scores = predictor(graph, collapsed)

    assert scores.numel() == graph.num_edges()
    assert torch.isfinite(scores).all()

    loss = scores.square().mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in list(model.parameters())
        + list(predictor.parameters())
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    assert gradients
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )
