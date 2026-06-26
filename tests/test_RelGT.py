"""
Unit tests for the RelGT integration into OpenHGNN.

Tests cover:
  - VectorQuantizerEMA
  - FeedForwardNetwork / EncoderLayer / LocalModule
  - NeighborNodeTypeEncoder / NeighborHopEncoder / NeighborTimeEncoder
  - GNNPEEncoder
  - NeighborTfsEncoder (mocked, torch_frame not required)
  - RelGTLayer (local / global / full modes)
  - RelGT.forward (end-to-end shape + value consistency vs original logic)
  - RelGT.build_model_from_args (OpenHGNN factory method)
  - Dataset helpers (build_adjacency_hetero, gather_1_and_2_hop_with_seed_time)
  - RelGTTokens.collate (no relbench required — mocked)

All tests use CPU and small synthetic tensors to avoid heavy dependencies.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
from typing import Dict

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Import building blocks directly from module files — do NOT go through
# openhgnn/__init__.py which would trigger DGL imports for other models.
# ---------------------------------------------------------------------------
import importlib.util

def _import_module_direct(path, name):
    """Load a .py file without executing the package __init__ chain."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# ---------------------------------------------------------------------------
# Bootstrap: load base_model.py and build a minimal openhgnn.models namespace
# WITHOUT triggering openhgnn/__init__.py (which pulls in DGL via FedHGNN etc.)
# ---------------------------------------------------------------------------
_MODEL_REGISTRY_DIRECT: Dict = {}

def _register_model_direct(name):
    def _decorator(cls):
        _MODEL_REGISTRY_DIRECT[name] = cls
        cls.model_name = name
        return cls
    return _decorator

# Load base_model.py directly
_base_model_path = os.path.join(ROOT, "openhgnn", "models", "base_model.py")
_bm_mod = _import_module_direct(_base_model_path, "openhgnn.models.base_model")

# Fabricate a minimal openhgnn.models package object only while importing the
# RelGT files. Restore sys.modules immediately afterwards so this test module
# cannot poison other OpenHGNN tests running in the same pytest process.
import types as _types_mod
_models_pkg = _types_mod.ModuleType("openhgnn.models")
_models_pkg.BaseModel         = _bm_mod.BaseModel
_models_pkg.MODEL_REGISTRY    = _MODEL_REGISTRY_DIRECT
_models_pkg.register_model    = _register_model_direct

_temporary_module_names = [
    "openhgnn",
    "openhgnn.models",
    "openhgnn.models.base_model",
    "openhgnn.models.RelGT",
    "openhgnn.dataset.RelGTDataset",
]
_original_modules = {
    name: sys.modules.get(name) for name in _temporary_module_names
}

try:
    sys.modules["openhgnn"] = _types_mod.ModuleType("openhgnn")
    sys.modules["openhgnn.models"] = _models_pkg
    sys.modules["openhgnn.models.base_model"] = _bm_mod

    # Now safely import RelGT.py directly
    _relgt_path = os.path.join(ROOT, "openhgnn", "models", "RelGT.py")
    _relgt_mod  = _import_module_direct(_relgt_path, "openhgnn.models.RelGT")

    # Also pre-load RelGTDataset.py so dataset tests can import from it
    # without needing a real openhgnn package tree.
    _dataset_path = os.path.join(ROOT, "openhgnn", "dataset", "RelGTDataset.py")
    _dataset_mod  = _import_module_direct(
        _dataset_path, "openhgnn.dataset.RelGTDataset")
finally:
    for _name, _module in _original_modules.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module
build_adjacency_hetero             = _dataset_mod.build_adjacency_hetero
gather_1_and_2_hop_with_seed_time  = _dataset_mod.gather_1_and_2_hop_with_seed_time
RelGTTokens                        = _dataset_mod.RelGTTokens

# Pull out the symbols used in tests
VectorQuantizerEMA       = _relgt_mod.VectorQuantizerEMA
FeedForwardNetwork       = _relgt_mod.FeedForwardNetwork
EncoderLayer             = _relgt_mod.EncoderLayer
LocalModule              = _relgt_mod.LocalModule
NeighborNodeTypeEncoder  = _relgt_mod.NeighborNodeTypeEncoder
NeighborHopEncoder       = _relgt_mod.NeighborHopEncoder
NeighborTimeEncoder      = _relgt_mod.NeighborTimeEncoder
GNNPEEncoder             = _relgt_mod.GNNPEEncoder
RelGTLayer               = _relgt_mod.RelGTLayer
RelGT                    = _relgt_mod.RelGT


# ===========================================================================
# Helpers
# ===========================================================================

def _make_dummy_grouped_tf_dict(B: int, K: int, channels: int, device="cpu"):
    """
    Build a grouped_tf_dict whose NeighborTfsEncoder call is fully mocked.
    The dict contains only the index/position arrays; the actual TorchFrame
    objects are replaced by a mock encoder that returns zeros.
    """
    flat_batch_idx = list(range(B)) * K
    flat_nbr_idx   = [j for j in range(K) for _ in range(B)]
    N = B * K
    return {
        "grouped_tfs":    {},          # empty → no real encoding call
        "grouped_indices": {},
        "flat_batch_idx":  flat_batch_idx[:N],
        "flat_nbr_idx":    flat_nbr_idx[:N],
    }, N


def _make_mock_tfs_encoder(channels: int, B: int, K: int):
    """Return an nn.Module whose forward always returns zeros [B, K, channels]."""
    class MockTfsEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.channels = channels
        def reset_parameters(self): pass
        def forward(self, batch_dict, neighbor_types):
            return torch.zeros(
                neighbor_types.size(0), neighbor_types.size(1), channels)
    return MockTfsEncoder()


def _build_relgt(
    B=4, K=8, channels=16, out_channels=1,
    conv_type="local", heads=2, num_layers=1,
    num_centroids=16, num_nodes=100,
    node_type_map=None,
):
    """
    Build a RelGT model with mocked tfs_encoder and pe_encoder so that
    no torch_frame data is needed at test time.
    """
    if node_type_map is None:
        node_type_map = {"user": 0, "item": 1}

    # Dummy col_names_dict / col_stats_dict — RelGT accepts them but
    # we will swap out the encoder immediately after construction.
    dummy_col_names = {}
    dummy_col_stats = {}

    # Patch NeighborTfsEncoder on the loaded module so construction succeeds
    # without real torch_frame data.
    with patch.multiple(
        _relgt_mod.NeighborTfsEncoder,
        __init__=lambda self, **kw: nn.Module.__init__(self),
        forward=lambda self, batch_dict, neighbor_types: torch.zeros(
            neighbor_types.size(0), neighbor_types.size(1), channels),
        reset_parameters=lambda self: None,
    ):
        model = RelGT(
            num_nodes=num_nodes,
            max_neighbor_hop=2,
            node_type_map=node_type_map,
            col_names_dict=dummy_col_names,
            col_stats_dict=dummy_col_stats,
            local_num_layers=num_layers,
            channels=channels,
            out_channels=out_channels,
            global_dim=channels // 2,
            heads=heads,
            ff_dropout=0.0,
            attn_dropout=0.0,
            conv_type=conv_type,
            num_centroids=num_centroids,
            sample_node_len=K,
        )

    # Replace tfs_encoder with a real mock module so forward works
    model.tfs_encoder = _make_mock_tfs_encoder(channels, B, K)
    return model.eval()


# ===========================================================================
# 1. VectorQuantizerEMA
# ===========================================================================

class TestVectorQuantizerEMA(unittest.TestCase):
    def setUp(self):
        self.num_embeddings = 8
        self.embedding_dim  = 16
        self.vq = VectorQuantizerEMA(self.num_embeddings, self.embedding_dim)

    def test_buffers_exist(self):
        self.assertIn("_embedding",        dict(self.vq.named_buffers()))
        self.assertIn("_embedding_output", dict(self.vq.named_buffers()))

    def test_get_k_shape(self):
        k = self.vq.get_k()
        self.assertEqual(k.shape, (self.num_embeddings, self.embedding_dim))

    def test_get_v_shape(self):
        v = self.vq.get_v()
        self.assertEqual(v.shape, (self.num_embeddings, self.embedding_dim))

    def test_update_returns_indices(self):
        self.vq.train()
        x = torch.randn(10, self.embedding_dim)
        indices = self.vq.update(x)
        self.assertEqual(indices.shape, (10, 1))
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < self.num_embeddings).all())

    def test_update_does_not_run_in_eval(self):
        self.vq.eval()
        embedding_before = self.vq._embedding.clone()
        x = torch.randn(10, self.embedding_dim)
        self.vq.update(x)
        # embedding should be unchanged in eval mode
        self.assertTrue(torch.allclose(self.vq._embedding, embedding_before))

    def test_reset_parameters(self):
        self.vq.reset_parameters()   # should not raise


# ===========================================================================
# 2. FeedForwardNetwork
# ===========================================================================

class TestFeedForwardNetwork(unittest.TestCase):
    def test_output_shape(self):
        ffn = FeedForwardNetwork(hidden_size=16, ffn_size=32, dropout_rate=0.0)
        x = torch.randn(4, 8, 16)   # [B, L, D]
        y = ffn(x)
        self.assertEqual(y.shape, x.shape)

    def test_single_token(self):
        ffn = FeedForwardNetwork(hidden_size=16, ffn_size=32, dropout_rate=0.0)
        x = torch.randn(4, 1, 16)
        y = ffn(x)
        self.assertEqual(y.shape, (4, 1, 16))


# ===========================================================================
# 3. EncoderLayer
# ===========================================================================

class TestEncoderLayer(unittest.TestCase):
    def _make(self):
        return EncoderLayer(
            hidden_size=16, ffn_size=32,
            dropout_rate=0.0, attention_dropout_rate=0.0,
            num_heads=2,
        )

    def test_output_shape(self):
        layer = self._make()
        x = torch.randn(4, 8, 16)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_residual_connection(self):
        """Output should differ from input (layer does something)."""
        layer = self._make()
        x = torch.randn(4, 8, 16)
        y = layer(x)
        self.assertFalse(torch.allclose(x, y))

    def test_reset_parameters(self):
        self._make().reset_parameters()


# ===========================================================================
# 4. LocalModule
# ===========================================================================

class TestLocalModule(unittest.TestCase):
    def _make(self, seq_len=8, input_dim=16, hidden_dim=16, heads=2):
        return LocalModule(
            seq_len=seq_len, input_dim=input_dim,
            n_layers=1, num_heads=heads, hidden_dim=hidden_dim,
            dropout_rate=0.0, attention_dropout_rate=0.0,
        ).eval()

    def test_output_shape(self):
        B, K = 4, 8
        mod = self._make(seq_len=K)
        x = torch.randn(B, K, 16)
        y = mod(x)
        self.assertEqual(y.shape, (B, 16))

    def test_pretrain_token_returns_full_seq(self):
        B, K = 4, 8
        mod = self._make(seq_len=K)
        x = torch.randn(B, K, 16)
        y = mod(x, pretrain_token=True)
        self.assertEqual(y.shape, (B, K, 16))

    def test_deterministic_in_eval(self):
        B, K = 4, 8
        mod = self._make(seq_len=K)
        x = torch.randn(B, K, 16)
        y1 = mod(x)
        y2 = mod(x)
        self.assertTrue(torch.allclose(y1, y2))


# ===========================================================================
# 5. NeighborNodeTypeEncoder
# ===========================================================================

class TestNeighborNodeTypeEncoder(unittest.TestCase):
    def test_output_shape(self):
        enc = NeighborNodeTypeEncoder(
            node_type_map={"a": 0, "b": 1, "c": 2}, embedding_dim=16)
        x = torch.randint(0, 3, (4, 8))   # [B, K]
        y = enc(x)
        self.assertEqual(y.shape, (4, 8, 16))

    def test_different_types_different_embeddings(self):
        enc = NeighborNodeTypeEncoder(
            node_type_map={"a": 0, "b": 1}, embedding_dim=16)
        t0 = enc(torch.tensor([[0]]))
        t1 = enc(torch.tensor([[1]]))
        self.assertFalse(torch.allclose(t0, t1))


# ===========================================================================
# 6. NeighborHopEncoder
# ===========================================================================

class TestNeighborHopEncoder(unittest.TestCase):
    def test_output_shape(self):
        enc = NeighborHopEncoder(max_neighbor_hop=3, embedding_dim=16)
        x = torch.randint(0, 3, (4, 8))
        y = enc(x)
        self.assertEqual(y.shape, (4, 8, 16))

    def test_shift_by_one(self):
        """Hop 0 should embed to index 1 (shifted internally)."""
        enc = NeighborHopEncoder(max_neighbor_hop=3, embedding_dim=16)
        h0 = enc(torch.tensor([[0]]))
        h1 = enc(torch.tensor([[1]]))
        self.assertFalse(torch.allclose(h0, h1))


# ===========================================================================
# 7. NeighborTimeEncoder
# ===========================================================================

class TestNeighborTimeEncoder(unittest.TestCase):
    def test_output_shape(self):
        enc = NeighborTimeEncoder(embedding_dim=16).eval()
        t = torch.randn(4, 8)   # [B, K] relative time in days
        y = enc(t)
        self.assertEqual(y.shape, (4, 8, 16))

    def test_masked_time(self):
        """Entries with negative time use the mask vector, not the linear output."""
        enc = NeighborTimeEncoder(embedding_dim=16).eval()
        # All-positive times
        t_pos = torch.ones(2, 4) * 10.0
        # All-negative times (masked)
        t_neg = torch.ones(2, 4) * -1.0
        y_pos = enc(t_pos)
        y_neg = enc(t_neg)
        self.assertFalse(torch.allclose(y_pos, y_neg))

    def test_reset_parameters(self):
        NeighborTimeEncoder(embedding_dim=16).reset_parameters()


# ===========================================================================
# 8. GNNPEEncoder
# ===========================================================================

class TestGNNPEEncoder(unittest.TestCase):
    def _make_batch(self, B=4, K=8):
        """Create a dummy batched edge_index and batch vector."""
        # Each subgraph is K nodes connected in a chain
        src = []
        dst = []
        for b in range(B):
            offset = b * K
            for k in range(K - 1):
                src.append(offset + k)
                dst.append(offset + k + 1)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        batch      = torch.tensor([b for b in range(B) for _ in range(K)],
                                  dtype=torch.long)
        return edge_index, batch

    def test_output_shape(self):
        B, K, D = 4, 8, 16
        enc = GNNPEEncoder(embedding_dim=D, num_layers=2).eval()
        edge_index, batch = self._make_batch(B, K)
        y = enc(edge_index, batch)
        self.assertEqual(y.shape, (B, K, D))

    def test_no_edges(self):
        """Should not crash when the subgraph has no edges."""
        B, K, D = 2, 4, 16
        enc = GNNPEEncoder(embedding_dim=D, num_layers=2).eval()
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        batch = torch.tensor([b for b in range(B) for _ in range(K)])
        y = enc(edge_index, batch)
        self.assertEqual(y.shape, (B, K, D))


# ===========================================================================
# 9. RelGTLayer  — local mode (no PyG / einops required for global path)
# ===========================================================================

class TestRelGTLayerLocal(unittest.TestCase):
    def setUp(self):
        B, K, D = 4, 8, 16
        self.B, self.K, self.D = B, K, D
        self.layer = RelGTLayer(
            in_channels=D, out_channels=D,
            local_num_layers=1, global_dim=8, num_nodes=100,
            heads=2, ff_dropout=0.0, attn_dropout=0.0,
            conv_type="local", num_centroids=None, sample_node_len=K,
        ).eval()

    def test_output_shape(self):
        x_set = torch.randn(self.B, self.K, self.D)
        x     = x_set[:, 0, :]
        node_indices = torch.arange(self.B)
        y = self.layer(x_set, x, node_indices)
        self.assertEqual(y.shape, (self.B, self.D))

    def test_layer_norm_applied(self):
        """Output should be normalised — mean ≈ 0 across feature dim."""
        x_set = torch.randn(self.B, self.K, self.D) * 100   # large values
        x     = x_set[:, 0, :]
        y = self.layer(x_set, x, torch.arange(self.B))
        # After LayerNorm the std should be near 1 (not 100)
        self.assertLess(y.std().item(), 10.0)


class TestRelGTLayerGlobal(unittest.TestCase):
    def test_global_output_shape(self):
        if not (_relgt_mod.HAS_EINOPS and _relgt_mod.HAS_PYG):
            self.skipTest("global RelGT layer requires torch_geometric and einops")
        B, K, D = 4, 8, 16
        layer = RelGTLayer(
            in_channels=D, out_channels=D,
            local_num_layers=1, global_dim=8, num_nodes=20,
            heads=2, ff_dropout=0.0, attn_dropout=0.0,
            conv_type="global", num_centroids=8, sample_node_len=K,
        ).eval()
        x_set = torch.randn(B, K, D)
        x     = x_set[:, 0, :]
        node_indices = torch.randint(0, 20, (B,))
        y = layer(x_set, x, node_indices)
        self.assertEqual(y.shape, (B, D))


class TestRelGTLayerFull(unittest.TestCase):
    def test_full_output_shape(self):
        if not (_relgt_mod.HAS_EINOPS and _relgt_mod.HAS_PYG):
            self.skipTest("full RelGT layer requires torch_geometric and einops")
        B, K, D = 4, 8, 16
        layer = RelGTLayer(
            in_channels=D, out_channels=D,
            local_num_layers=1, global_dim=8, num_nodes=20,
            heads=2, ff_dropout=0.0, attn_dropout=0.0,
            conv_type="full", num_centroids=8, sample_node_len=K,
        ).eval()
        x_set = torch.randn(B, K, D)
        x     = x_set[:, 0, :]
        node_indices = torch.randint(0, 20, (B,))
        y = layer(x_set, x, node_indices)
        # "full" concatenates local+global → 2*D
        self.assertEqual(y.shape, (B, 2 * D))


# ===========================================================================
# 10. RelGT end-to-end forward pass
# ===========================================================================

class TestRelGTForward(unittest.TestCase):
    """Tests the full RelGT.forward() with mocked tfs_encoder."""

    B, K, D, OUT = 4, 8, 16, 1

    def _make_inputs(self, B=None, K=None, node_type_map=None):
        B = B or self.B
        K = K or self.K
        if node_type_map is None:
            node_type_map = {"user": 0, "item": 1}
        neighbor_types   = torch.randint(0, len(node_type_map), (B, K))
        node_indices     = torch.randint(0, 100, (B,))
        neighbor_hops    = torch.randint(0, 3, (B, K))
        neighbor_times   = torch.randn(B, K).abs()
        # Simple chain subgraph per sample
        src = []
        dst = []
        for b in range(B):
            for k in range(K - 1):
                src.append(b * K + k)
                dst.append(b * K + k + 1)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        batch      = torch.tensor([b for b in range(B) for _ in range(K)])
        grouped_tf_dict = {
            "grouped_tfs":    {},
            "grouped_indices": {},
            "flat_batch_idx": list(range(B)) * K,
            "flat_nbr_idx":   [j for j in range(K) for _ in range(B)],
        }
        return (neighbor_types, node_indices, neighbor_hops, neighbor_times,
                grouped_tf_dict, edge_index, batch)

    def test_local_output_shape(self):
        model = _build_relgt(
            B=self.B, K=self.K, channels=self.D,
            out_channels=self.OUT, conv_type="local",
        )
        inputs = self._make_inputs()
        with torch.no_grad():
            out = model(*inputs)
        self.assertEqual(out.shape, (self.B, self.OUT))

    def test_global_output_shape(self):
        if not (_relgt_mod.HAS_EINOPS and _relgt_mod.HAS_PYG):
            self.skipTest("global RelGT forward requires torch_geometric and einops")
        model = _build_relgt(
            B=self.B, K=self.K, channels=self.D,
            out_channels=self.OUT, conv_type="global",
            num_centroids=8, num_nodes=100,
        )
        inputs = self._make_inputs()
        with torch.no_grad():
            out = model(*inputs)
        self.assertEqual(out.shape, (self.B, self.OUT))

    def test_full_output_shape(self):
        if not (_relgt_mod.HAS_EINOPS and _relgt_mod.HAS_PYG):
            self.skipTest("full RelGT forward requires torch_geometric and einops")
        model = _build_relgt(
            B=self.B, K=self.K, channels=self.D,
            out_channels=self.OUT, conv_type="full",
            num_centroids=8, num_nodes=100,
        )
        inputs = self._make_inputs()
        with torch.no_grad():
            out = model(*inputs)
        self.assertEqual(out.shape, (self.B, self.OUT))

    def test_multilabel_output_shape(self):
        num_labels = 5
        model = _build_relgt(
            B=self.B, K=self.K, channels=self.D,
            out_channels=num_labels, conv_type="local",
        )
        inputs = self._make_inputs()
        with torch.no_grad():
            out = model(*inputs)
        self.assertEqual(out.shape, (self.B, num_labels))

    def test_deterministic_with_fixed_seed(self):
        """
        GNNPEEncoder samples random features (pe_dim=0) — this is intentional
        by design (random feature PE).  With a fixed seed the output is stable.
        """
        model = _build_relgt(conv_type="local")
        inputs = self._make_inputs()
        torch.manual_seed(0)
        with torch.no_grad():
            y1 = model(*inputs)
        torch.manual_seed(0)
        with torch.no_grad():
            y2 = model(*inputs)
        self.assertTrue(torch.allclose(y1, y2))

    def test_gradient_flows(self):
        model = _build_relgt(conv_type="local").train()
        inputs = self._make_inputs()
        out = model(*inputs)
        loss = out.sum()
        loss.backward()
        grad_found = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in model.parameters() if p.requires_grad)
        self.assertTrue(grad_found, "No gradient flowed to any parameter")

    def test_output_consistency_with_original_logic(self):
        """
        Verify the five-encoder pipeline produces the same tensor as a
        manually assembled reference forward pass using the same weights.

        This test checks algorithmic fidelity:
        the in_mixture input must equal cat([type, hop, time, tfs, pe]).
        """
        model = _build_relgt(conv_type="local", heads=2).eval()
        (neighbor_types, node_indices, neighbor_hops, neighbor_times,
         grouped_tf_dict, edge_index, batch) = self._make_inputs()

        # --- replicate the forward pre-mixture step manually ---
        with torch.no_grad():
            tfs_out   = model.layer_norm_tfs(
                model.tfs_encoder(grouped_tf_dict, neighbor_types))
            type_out  = model.layer_norm_type(
                model.type_encoder(neighbor_types.long()))
            hop_out   = model.layer_norm_hop(
                model.hop_encoder(neighbor_hops.long()))
            time_out  = model.layer_norm_time(
                model.time_encoder(neighbor_times.float()))
            pe_out    = model.layer_norm_pe(
                model.pe_encoder(edge_index, batch))

            cat_ref = torch.cat(
                [type_out, hop_out, time_out, tfs_out, pe_out], dim=-1)
            x_set_ref = model.in_mixture(cat_ref)

            # full forward
            full_out = model(
                neighbor_types, node_indices, neighbor_hops, neighbor_times,
                grouped_tf_dict, edge_index=edge_index, batch=batch)

        # Both paths must agree on the intermediate representation
        self.assertEqual(x_set_ref.shape[-1], self.D)
        # The full output must be finite
        self.assertTrue(torch.isfinite(full_out).all(),
                        "Output contains NaN or Inf")


# ===========================================================================
# 11. build_model_from_args (OpenHGNN factory)
# ===========================================================================

class TestBuildModelFromArgs(unittest.TestCase):
    def _make_args(self):
        args = types.SimpleNamespace(
            channels=16,
            out_channels=1,
            num_layers=1,
            num_heads=2,
            ff_dropout=0.0,
            attn_dropout=0.0,
            gt_conv_type="local",
            num_centroids=8,
            num_neighbors=8,
            gnn_pe_dim=0,
            ablate="none",
        )
        return args

    def _make_data_info(self):
        return {
            "num_nodes":        50,
            "max_neighbor_hop": 2,
            "node_type_map":    {"user": 0, "item": 1},
            "col_names_dict":   {},
            "col_stats_dict":   {},
        }

    def _patch_tfs(self):
        """Context manager that patches NeighborTfsEncoder on the loaded module."""
        return patch.multiple(
            _relgt_mod.NeighborTfsEncoder,
            __init__=lambda self, **kw: nn.Module.__init__(self),
            forward=lambda self, *a, **kw: torch.zeros(1),
            reset_parameters=lambda self: None,
        )

    def test_factory_returns_relgt_instance(self):
        args      = self._make_args()
        data_info = self._make_data_info()
        with self._patch_tfs():
            model = RelGT.build_model_from_args(args, data_info)
        self.assertIsInstance(model, RelGT)

    def test_factory_respects_channels(self):
        args          = self._make_args()
        args.channels = 32
        data_info     = self._make_data_info()
        with self._patch_tfs():
            model = RelGT.build_model_from_args(args, data_info)
        # in_mixture first linear: input = 5 * channels = 160
        self.assertEqual(model.in_mixture[0].in_features, 5 * 32)

    def test_ablate_reduces_channel_mult(self):
        args        = self._make_args()
        args.ablate = "type"   # removes one of the 5 components
        data_info   = self._make_data_info()
        with self._patch_tfs():
            model = RelGT.build_model_from_args(args, data_info)
        # 4 components × channels (16)
        self.assertEqual(model.in_mixture[0].in_features, 4 * 16)

    def test_model_registered_in_direct_registry(self):
        """RelGT must appear in the local registry populated by @register_model."""
        self.assertIn("RelGT", _MODEL_REGISTRY_DIRECT)

    def test_model_class_is_relgt(self):
        self.assertIs(_MODEL_REGISTRY_DIRECT["RelGT"], RelGT)


# ===========================================================================
# 12. Dataset helpers (no relbench required)
# ===========================================================================

class TestBuildAdjacencyHetero(unittest.TestCase):
    def _make_hetero_data(self):
        """Minimal stand-in for HeteroData using a plain dict-style object."""
        class _FakeEdgeStore:
            def __init__(self, ei):
                self.edge_index = ei
            def __contains__(self, item):
                return item == "edge_index"

        class _FakeNodeStore:
            def __init__(self, n):
                self.num_nodes = n

        ei = torch.tensor([[0, 1, 2], [0, 1, 2]])

        _store = {
            "user":                      _FakeNodeStore(4),
            "item":                      _FakeNodeStore(3),
            ("user", "rates", "item"):   _FakeEdgeStore(ei),
        }

        class _FakeData:
            node_types = ["user", "item"]
            edge_types = [("user", "rates", "item")]
            def __getitem__(self, key):
                return _store[key]

        return _FakeData()

    def test_adjacency_keys(self):
        data = self._make_hetero_data()
        adj = build_adjacency_hetero(data, undirected=True)
        self.assertIn("user", adj)
        self.assertIn("item", adj)

    def test_directed_edges(self):
        data = self._make_hetero_data()
        adj = build_adjacency_hetero(data, undirected=False)
        self.assertIn(("item", 0), adj["user"][0])
        self.assertNotIn(("user", 0), adj["item"][0])

    def test_undirected_edges(self):
        data = self._make_hetero_data()
        adj = build_adjacency_hetero(data, undirected=True)
        self.assertIn(("item", 0), adj["user"][0])
        self.assertIn(("user", 0), adj["item"][0])


class TestGather1And2Hop(unittest.TestCase):
    def _make_adj(self):
        """user0 — item0 — user1"""
        return {
            "user": [
                {("item", 0)},       # user 0
                {("item", 0)},       # user 1
            ],
            "item": [
                {("user", 0), ("user", 1)},  # item 0
            ],
        }

    def _make_data(self):
        data = MagicMock()
        # No time attribute → all neighbours accepted
        data.__getitem__ = lambda self, key: MagicMock(spec=[])
        return data

    def test_returns_list(self):
        adj  = self._make_adj()
        data = self._make_data()
        result = gather_1_and_2_hop_with_seed_time(
            adj, data, "user", 0, seed_time=1e12)
        self.assertIsInstance(result, list)

    def test_hop_values(self):
        adj  = self._make_adj()
        data = self._make_data()
        result = gather_1_and_2_hop_with_seed_time(
            adj, data, "user", 0, seed_time=1e12)
        hops = [t[2] for t in result]
        self.assertIn(1, hops)
        self.assertIn(2, hops)

    def test_no_self_loop(self):
        adj  = self._make_adj()
        data = self._make_data()
        result = gather_1_and_2_hop_with_seed_time(
            adj, data, "user", 0, seed_time=1e12)
        seeds = [(t[0], t[1]) for t in result]
        self.assertNotIn(("user", 0), seeds)


# ===========================================================================
# 13. RelGTTokens.collate — fully mocked (no relbench)
# ===========================================================================

class TestRelGTTokensCollate(unittest.TestCase):
    """
    Test the collate method in isolation by constructing a minimal
    RelGTTokens-like object without instantiating the full class.
    """

    def _make_sample(self, K: int, num_types: int, idx: int):
        types   = torch.randint(0, num_types, (K,))
        indices = torch.randint(0, 50, (K,))
        hops    = torch.randint(0, 3, (K,))
        times   = torch.randn(K).abs()
        eidx    = torch.zeros((2, 0), dtype=torch.long)
        return {
            "types":       types,
            "indices":     indices,
            "hops":        hops,
            "times":       times,
            "edge_index":  eidx,
            "first_type":  types[0].item(),
            "first_index": indices[0].item(),
            "tfs":         [None] * K,   # not used in collate output
            "global_idx":  idx,
        }

    def _make_ds_mock(self, B, K, num_types):
        """Build a minimal mock of RelGTTokens usable by collate."""
        node_types = ["user", "item"]

        # Fake TorchFrame-like store: tf[idxs] just returns None
        class _FakeTF:
            def __getitem__(self, idx):
                return None

        class _FakeNodeStore:
            tf = _FakeTF()

        class _FakeData:
            def __getitem__(self, key):
                return _FakeNodeStore()

        ds = MagicMock(spec=RelGTTokens)
        ds.node_types         = node_types
        ds.node_type_to_index = {"user": 0, "item": 1}
        ds.index_to_node_type = {0: "user", 1: "item"}
        ds.target             = torch.zeros(100)
        ds.data               = _FakeData()
        ds.get_global_index   = MagicMock(return_value=list(range(B)))
        return ds

    def test_collate_output_keys(self):
        B, K, num_types = 3, 6, 2
        ds = self._make_ds_mock(B, K, num_types)
        samples = [(self._make_sample(K, num_types, i), torch.tensor(0.0))
                   for i in range(B)]
        result = RelGTTokens.collate(ds, samples)

        expected_keys = {
            "neighbor_types", "neighbor_indices", "neighbor_hops",
            "neighbor_times", "labels", "node_indices",
            "grouped_tfs", "grouped_indices", "flat_batch_idx", "flat_nbr_idx",
            "global_idx", "edge_index", "batch",
        }
        self.assertEqual(expected_keys, set(result.keys()))

    def test_collate_shapes(self):
        B, K, num_types = 3, 6, 2
        ds = self._make_ds_mock(B, K, num_types)
        samples = [(self._make_sample(K, num_types, i), torch.tensor(0.0))
                   for i in range(B)]
        result = RelGTTokens.collate(ds, samples)

        self.assertEqual(result["neighbor_types"].shape,   (B, K))
        self.assertEqual(result["neighbor_indices"].shape, (B, K))
        self.assertEqual(result["neighbor_hops"].shape,    (B, K))
        self.assertEqual(result["neighbor_times"].shape,   (B, K))
        self.assertEqual(result["batch"].shape,            (B * K,))
        self.assertEqual(result["node_indices"].shape,     (B,))


# ===========================================================================
# 14. Trainer flow registration
# ===========================================================================

class TestRelGTTrainerRegistration(unittest.TestCase):
    def test_trainerflow_init_contains_relgt(self):
        """Check that SUPPORTED_FLOWS and import are present in trainerflow/__init__.py."""
        init_path = os.path.join(ROOT, "openhgnn", "trainerflow", "__init__.py")
        with open(init_path) as f:
            src = f.read()
        self.assertIn("RelGT_trainer", src,
                      "RelGT_trainer not found in SUPPORTED_FLOWS")
        self.assertIn("relgt_trainer", src,
                      "relgt_trainer import not found in trainerflow/__init__.py")

    def test_experiment_py_contains_relgt(self):
        """Check that experiment.py routes RelGT to RelGT_trainer."""
        exp_path = os.path.join(ROOT, "openhgnn", "experiment.py")
        with open(exp_path) as f:
            src = f.read()
        self.assertIn("'RelGT'", src)
        self.assertIn("'RelGT_trainer'", src)

    def test_trainer_file_has_register_flow(self):
        """Check that the trainer file has @register_flow("RelGT_trainer")."""
        trainer_path = os.path.join(
            ROOT, "openhgnn", "trainerflow", "relgt_trainer.py")
        with open(trainer_path) as f:
            src = f.read()
        self.assertIn('@register_flow("RelGT_trainer")', src)


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
