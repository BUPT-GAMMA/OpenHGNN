"""
Tests for HGSketch model component.
Covers: registration, config, core algorithm, and linearization.
"""
import os
import sys
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import dgl
import torch


# ============================================================
# 1. Registration Tests
# ============================================================

def test_model_registered():
    """HGSketch should be in MODEL_REGISTRY after import."""
    from openhgnn.models import MODEL_REGISTRY
    from openhgnn.models.HGSketch import HGSketch  # trigger registration
    assert 'HGSketch' in MODEL_REGISTRY, "HGSketch not found in MODEL_REGISTRY"


def test_flow_registered():
    """HGSketch_trainer should be in FLOW_REGISTRY after import."""
    from openhgnn.trainerflow import FLOW_REGISTRY
    from openhgnn.trainerflow.HGSketch_trainer import HGSketchTrainer  # trigger registration
    assert 'HGSketch_trainer' in FLOW_REGISTRY, "HGSketch_trainer not found in FLOW_REGISTRY"


def test_experiment_binding():
    """HGSketch should be mapped in Experiment.specific_trainerflow."""
    from openhgnn.experiment import Experiment
    assert 'HGSketch' in Experiment.specific_trainerflow
    assert Experiment.specific_trainerflow['HGSketch'] == 'HGSketch_trainer'


def test_supported_models_entry():
    """HGSketch should be in SUPPORTED_MODELS dict."""
    from openhgnn.models import SUPPORTED_MODELS
    assert 'HGSketch' in SUPPORTED_MODELS
    assert SUPPORTED_MODELS['HGSketch'] == 'openhgnn.models.HGSketch'


def test_supported_flows_entry():
    """HGSketch_trainer should be in SUPPORTED_FLOWS dict."""
    from openhgnn.trainerflow import SUPPORTED_FLOWS
    assert 'HGSketch_trainer' in SUPPORTED_FLOWS


# ============================================================
# 2. Config Tests
# ============================================================

def test_config_reads_hgsketch_params():
    """Config should correctly read HGSketch params from config.ini."""
    from openhgnn.config import Config
    conf_path = os.path.join(os.path.dirname(__file__), '..', 'openhgnn', 'config.ini')
    config = Config(file_path=conf_path, model='HGSketch', dataset='test_ds', task='graph_classification', gpu=-1)
    assert config.K == 2
    assert config.R == 3
    assert config.D == 128
    assert config.seed == 0
    assert config.max_epoch == 1  # non-parametric


# ============================================================
# 3. Helper: create a small heterogeneous graph
# ============================================================

def _make_small_hg():
    """Create a small heterogeneous graph with 2 node types and 2 edge types."""
    # 4 'user' nodes, 3 'item' nodes
    # edges: user->item (likes), item->user (liked_by)
    data = {
        ('user', 'likes', 'item'): (torch.tensor([0, 1, 2, 0, 3]), torch.tensor([0, 1, 2, 1, 0])),
        ('item', 'liked_by', 'user'): (torch.tensor([0, 1, 2, 1, 0]), torch.tensor([0, 1, 2, 0, 3])),
    }
    hg = dgl.heterograph(data)
    return hg


def _make_triangle_hg():
    """Create a graph that contains a triangle (3-clique) to test 2-simplices."""
    # 3 nodes of type 'a', fully connected -> forms a triangle
    data = {
        ('a', 'e1', 'a'): (torch.tensor([0, 1, 0, 2, 1, 2]), torch.tensor([1, 0, 2, 0, 2, 1])),
    }
    hg = dgl.heterograph(data)
    return hg


# ============================================================
# 4. Core Algorithm Tests
# ============================================================

def test_hg_to_nx_conversion():
    """Test heterogeneous graph to NetworkX conversion."""
    from openhgnn.models.HGSketch import HGSketch
    model = HGSketch(K=1, R=1, D=8, num_node_types=2, seed=42)
    hg = _make_small_hg()
    nx_g = model._hg_to_nx(hg)
    # Total nodes: 4 users + 3 items = 7
    assert nx_g.number_of_nodes() == 7
    # Should have edges (undirected, no self-loops)
    assert nx_g.number_of_edges() > 0


def test_node_type_map():
    """Test node type mapping."""
    from openhgnn.models.HGSketch import HGSketch
    model = HGSketch(K=1, R=1, D=8, num_node_types=2, seed=42)
    hg = _make_small_hg()
    type_map = model._build_node_type_map(hg)
    # DGL sorts ntypes alphabetically: ['item', 'user']
    # First 3 nodes (item) should be type 0, next 4 (user) should be type 1
    assert type_map[0] == 0
    assert type_map[2] == 0
    assert type_map[3] == 1
    assert type_map[6] == 1


def test_simplex_extraction():
    """Test simplex extraction from a triangle graph."""
    from openhgnn.models.HGSketch import HGSketch
    model = HGSketch(K=2, R=1, D=8, num_node_types=1, seed=42)
    hg = _make_triangle_hg()
    nx_g = model._hg_to_nx(hg)
    simplices = model._extract_simplices(nx_g, K=2)

    # 0-simplices: 3 nodes
    assert len(simplices[0]) == 3
    # 1-simplices: 3 edges
    assert len(simplices[1]) == 3
    # 2-simplices: 1 triangle
    assert len(simplices[2]) == 1


def test_compute_sketch_output_shape():
    """Test that compute_sketch returns a non-empty array with values in {-1, 1}."""
    from openhgnn.models.HGSketch import HGSketch
    model = HGSketch(K=1, R=2, D=16, num_node_types=2, seed=42)
    hg = _make_small_hg()
    x_g = model.compute_sketch(hg)

    assert isinstance(x_g, np.ndarray)
    assert x_g.size > 0
    # All values should be -1 or 1
    unique_vals = set(np.unique(x_g))
    assert unique_vals.issubset({-1.0, 1.0}), f"Unexpected values: {unique_vals}"


def test_compute_sketch_with_triangle():
    """Test compute_sketch on a graph with higher-order simplices (K=2)."""
    from openhgnn.models.HGSketch import HGSketch
    model = HGSketch(K=2, R=2, D=16, num_node_types=1, seed=42)
    hg = _make_triangle_hg()
    x_g = model.compute_sketch(hg)

    assert isinstance(x_g, np.ndarray)
    assert x_g.size > 0
    unique_vals = set(np.unique(x_g))
    assert unique_vals.issubset({-1.0, 1.0})


def test_compute_sketch_deterministic():
    """Same graph + same seed should produce identical sketches."""
    from openhgnn.models.HGSketch import HGSketch
    model = HGSketch(K=1, R=2, D=16, num_node_types=2, seed=123)
    hg = _make_small_hg()
    x1 = model.compute_sketch(hg)
    x2 = model.compute_sketch(hg)
    np.testing.assert_array_equal(x1, x2)


# ============================================================
# 5. Linearization Tests
# ============================================================

def test_linearize_output_length():
    """Linearized vector should have length 2L."""
    from openhgnn.models.HGSketch import HGSketch
    model = HGSketch(K=1, R=1, D=8, num_node_types=2, seed=42)
    hg = _make_small_hg()
    x_g = model.compute_sketch(hg)
    L = len(x_g)
    x_lin = model.linearize(x_g)
    assert len(x_lin) == 2 * L


def test_linearize_values():
    """Linearized values should be 0 or 1/sqrt(L)."""
    from openhgnn.models.HGSketch import HGSketch
    model = HGSketch(K=1, R=1, D=8, num_node_types=2, seed=42)
    hg = _make_small_hg()
    x_g = model.compute_sketch(hg)
    L = len(x_g)
    x_lin = model.linearize(x_g)

    expected_nonzero = 1.0 / np.sqrt(L)
    for v in np.unique(x_lin):
        assert np.isclose(v, 0.0, atol=1e-8) or np.isclose(v, expected_nonzero, atol=1e-8), \
            f"Unexpected value: {v}, expected 0 or {expected_nonzero}"


def test_linearize_kernel_property():
    """
    Verify the Hamming kernel linearization property:
    <linearize(x_i), linearize(x_j)> should equal the normalized Hamming agreement.
    """
    from openhgnn.models.HGSketch import HGSketch
    model = HGSketch(K=1, R=2, D=16, num_node_types=1, seed=42)

    hg1 = _make_triangle_hg()
    hg2 = _make_small_hg()
    # Use same num_node_types for both
    model2 = HGSketch(K=1, R=2, D=16, num_node_types=2, seed=42)

    x1 = model.compute_sketch(hg1)
    x2 = model.compute_sketch(hg1)  # same graph

    x1_lin = model.linearize(x1)
    x2_lin = model.linearize(x2)

    # For identical graphs, inner product should equal 1.0 (perfect agreement)
    L = len(x1)
    if L > 0:
        inner = np.dot(x1_lin, x2_lin)
        # Since x1 == x2, all bits agree, so kernel = L/L = 1.0
        np.testing.assert_almost_equal(inner, 1.0, decimal=5)


# ============================================================
# 6. Hodge Laplacian Tests
# ============================================================

def test_hodge_laplacian_shape():
    """L_k should be square with size = number of k-simplices."""
    from openhgnn.models.HGSketch import HGSketch
    model = HGSketch(K=2, R=1, D=8, num_node_types=1, seed=42)
    hg = _make_triangle_hg()
    nx_g = model._hg_to_nx(hg)
    simplices = model._extract_simplices(nx_g, K=2)

    for k in range(3):
        n_k = len(simplices[k])
        L_k = model._build_hodge_laplacian(simplices, k)
        assert L_k.shape == (n_k, n_k), f"L_{k} shape mismatch: {L_k.shape} vs ({n_k}, {n_k})"


def test_hodge_laplacian_symmetric():
    """Hodge Laplacian should be symmetric."""
    from openhgnn.models.HGSketch import HGSketch
    from scipy import sparse as sp
    model = HGSketch(K=2, R=1, D=8, num_node_types=1, seed=42)
    hg = _make_triangle_hg()
    nx_g = model._hg_to_nx(hg)
    simplices = model._extract_simplices(nx_g, K=2)

    for k in range(3):
        L_k = model._build_hodge_laplacian(simplices, k)
        if L_k.shape[0] > 0:
            diff = L_k - L_k.T
            assert abs(diff).max() < 1e-10, f"L_{k} is not symmetric"


# ============================================================
# 7. Build Model from Args Test
# ============================================================

def test_build_model_from_args():
    """Test build_model_from_args class method."""
    from openhgnn.models.HGSketch import HGSketch

    class MockArgs:
        K = 2
        R = 3
        D = 64
        seed = 42

    hg = _make_small_hg()
    model = HGSketch.build_model_from_args(MockArgs(), hg)
    assert model.K == 2
    assert model.R == 3
    assert model.D == 64
    assert model.num_node_types == 2  # user, item
    assert model.seed == 42


# ============================================================
# 8. Edge Cases
# ============================================================

def test_empty_graph():
    """Model should handle a graph with no edges gracefully."""
    from openhgnn.models.HGSketch import HGSketch
    model = HGSketch(K=1, R=1, D=8, num_node_types=1, seed=42)
    # Graph with 3 nodes, no edges
    hg = dgl.heterograph({('a', 'e', 'a'): (torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))},
                         num_nodes_dict={'a': 3})
    x_g = model.compute_sketch(hg)
    assert isinstance(x_g, np.ndarray)
    # Should still produce output (at least 0-simplex features)
    assert x_g.size > 0


def test_single_node_graph():
    """Model should handle a single-node graph."""
    from openhgnn.models.HGSketch import HGSketch
    model = HGSketch(K=1, R=1, D=8, num_node_types=1, seed=42)
    hg = dgl.heterograph({('a', 'e', 'a'): (torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))},
                         num_nodes_dict={'a': 1})
    x_g = model.compute_sketch(hg)
    assert isinstance(x_g, np.ndarray)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
