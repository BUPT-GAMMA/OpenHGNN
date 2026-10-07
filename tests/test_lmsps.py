"""
Tests for the LMSPS model component.
Covers: registration, config, progressive sampling (Eq. 2-4), sampling
evaluation (Eq. 6) and the target-net (Eq. 7).

The tests are self-contained: they build tiny synthetic meta-path features
instead of downloading ogbn-mag, so they run on CPU in a few seconds.
"""
import math
import os
import random
import sys

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch


# ============================================================
# 1. Registration Tests
# ============================================================

def test_model_registered():
    """LMSPS should be in MODEL_REGISTRY after import."""
    from openhgnn.models import MODEL_REGISTRY
    from openhgnn.models.LMSPS import LMSPS  # trigger registration
    assert 'LMSPS' in MODEL_REGISTRY, "LMSPS not found in MODEL_REGISTRY"


def test_flow_registered():
    """LMSPS_trainer should be in FLOW_REGISTRY after import."""
    from openhgnn.trainerflow import FLOW_REGISTRY
    from openhgnn.trainerflow.LMSPS_trainer import LMSPStrainer  # trigger registration
    assert 'LMSPS_trainer' in FLOW_REGISTRY, "LMSPS_trainer not found in FLOW_REGISTRY"


def test_experiment_binding():
    """LMSPS should be mapped in Experiment.specific_trainerflow."""
    from openhgnn.experiment import Experiment
    assert 'LMSPS' in Experiment.specific_trainerflow
    assert Experiment.specific_trainerflow['LMSPS'] == 'LMSPS_trainer'


def test_supported_models_entry():
    """LMSPS should be in SUPPORTED_MODELS dict."""
    from openhgnn.models import SUPPORTED_MODELS
    assert 'LMSPS' in SUPPORTED_MODELS
    assert SUPPORTED_MODELS['LMSPS'] == 'openhgnn.models.LMSPS'


def test_supported_flows_entry():
    """LMSPS_trainer should be in SUPPORTED_FLOWS dict."""
    from openhgnn.trainerflow import SUPPORTED_FLOWS
    assert 'LMSPS_trainer' in SUPPORTED_FLOWS


# ============================================================
# 2. Config Tests
# ============================================================

def test_config_reads_lmsps_params():
    """Config should correctly read LMSPS params from config.ini."""
    from openhgnn.config import Config
    conf_path = os.path.join(os.path.dirname(__file__), '..', 'openhgnn', 'config.ini')
    config = Config(file_path=conf_path, model='LMSPS', dataset='ogbn-mag',
                    task='node_classification', gpu=-1)
    # search stage
    assert config.num_final == 60
    assert config.ns == 30
    assert config.tau_max == 8
    assert config.tau_min == 4
    assert config.warmup_epochs == 20
    assert config.repeat == 200
    assert config.alr == 0.001
    assert config.search_dropout == 0.0
    assert config.search_input_drop == 0.0
    # target-net stage
    assert config.num_hops == 6
    assert config.embed_size == 256
    assert config.hidden == 512
    assert config.lr == 0.003
    assert config.weight_decay == 0
    assert config.batch_size == 10000
    assert config.patience == 50
    assert config.seed == 1
    assert config.meta_paths == ''


# ============================================================
# 3. Helpers: tiny synthetic meta-path features
# ============================================================

# meta-path keys follow the convention of the reference implementation: the
# first character is the destination node type ('P' = paper, the target type),
# so every key below describes a path ending at a paper.
FEAT_KEYS_12 = ['P', 'PA', 'PF', 'PI', 'PAP', 'PFP', 'PIP', 'PAPP', 'PAPAP',
                'PAPFP', 'PFPAP', 'PFPPP']
TGT_KEY = 'P'


def _make_search_net(feat_keys=FEAT_KEYS_12, nfeat=8, hidden=16, nclass=3,
                     tgt_key=TGT_KEY, num_final=5, num_sampled=3, residual=True,
                     dims=None):
    from openhgnn.models.LMSPS import LMSPS_SearchNet
    if dims is None:
        data_size = {k: nfeat for k in list(feat_keys) + [tgt_key]}
    else:
        data_size = {k: dims.get(k, nfeat) for k in list(feat_keys) + [tgt_key]}
    return LMSPS_SearchNet(feat_keys=feat_keys, data_size=data_size, hidden=hidden,
                           nclass=nclass, tgt_key=tgt_key, dropout=0.5,
                           input_drop=0.1, num_final=num_final,
                           num_sampled=num_sampled, residual=residual)


def _make_feats(feat_keys, num_nodes=4, nfeat=8, dims=None):
    return {k: torch.randn(num_nodes, dims[k] if dims else nfeat)
            for k in feat_keys}


class MockArgs:
    """Arguments for LMSPS.build_model_from_args."""
    def __init__(self, feat_keys=FEAT_KEYS_12, data_size=None, nfeat=8,
                 hidden=16, nclass=3, tgt_key=TGT_KEY, dropout=0.5,
                 input_drop=0.1, residual=True, bns=True, n_layers_2=2):
        self.feat_keys = feat_keys
        self.data_size = data_size or {k: nfeat for k in feat_keys}
        self.nfeat = nfeat
        self.hidden = hidden
        self.nclass = nclass
        self.tgt_key = tgt_key
        self.dropout = dropout
        self.input_drop = input_drop
        self.residual = residual
        self.bns = bns
        self.n_layers_2 = n_layers_2


# ============================================================
# 4. Progressive Sampling Tests (paper Eq. 2 - Eq. 4)
# ============================================================

def test_gumbel_prob_is_a_distribution():
    """Eq. 2 should produce a probability distribution over all meta-paths."""
    net = _make_search_net()
    q = net._get_gumbel_prob(net.alpha.detach())
    assert q.shape == (len(FEAT_KEYS_12),)
    assert torch.all(q >= 0)
    assert torch.isclose(q.sum(), torch.tensor(1.0), atol=1e-5)


def test_gumbel_temperature_sharpens():
    """Annealing tau from tau_max to tau_min should sharpen the distribution."""
    torch.manual_seed(0)
    net = _make_search_net()
    net.set_tau(8.0)
    q_high = net._get_gumbel_prob(net.alpha.detach())
    torch.manual_seed(0)
    net.set_tau(4.0)
    q_low = net._get_gumbel_prob(net.alpha.detach())
    assert q_low.max() > q_high.max()


@pytest.mark.parametrize("lam,expected_c", [
    (1.0, 12),    # start of the search: the whole space is kept
    (0.75, 11),   # 12 - 5 = 7, ceil(0.75 * 7) = 6 -> 6 + 5 = 11
    (0.5, 9),     # ceil(3.5) = 4 -> 4 + 5 = 9
    (0.25, 7),    # ceil(1.75) = 2 -> 2 + 5 = 7
    (0.0, 5),     # end of the search: only the compact space S_V is kept
])
def test_dynamic_search_space_size(lam, expected_c):
    """Eq. 3: |S_C| should be exactly ceil(lam * (K - V)) + V."""
    net = _make_search_net()
    torch.manual_seed(42)
    space, c = net.dynamic_search_space(lam)
    assert c == expected_c
    assert space.numel() == expected_c
    assert expected_c == int(math.ceil(lam * (len(FEAT_KEYS_12) - 5))) + 5
    # indexes are unique and within range
    assert space.unique().numel() == space.numel()
    assert space.min() >= 0 and space.max() < len(FEAT_KEYS_12)


def test_dynamic_search_space_on_real_ogbn_mag_shape():
    """K = 226 and V = 60 are the ogbn-mag settings of the paper (Table 2)."""
    feat_keys = ['P'] + ['PAP' * i for i in range(1, 226)]
    net = _make_search_net(feat_keys=feat_keys, num_final=60, num_sampled=30)
    torch.manual_seed(0)
    space, c = net.dynamic_search_space(1.0)
    assert c == 226 and space.numel() == 226
    torch.manual_seed(0)
    space, c = net.dynamic_search_space(0.5)
    assert c == int(math.ceil(0.5 * (226 - 60))) + 60 == 143
    assert space.numel() == 143


def test_epoch_sample_activates_num_sampled_paths():
    """Eq. 4: S = UniformSample(S_C, M) should activate M meta-paths."""
    net = _make_search_net()
    torch.manual_seed(0)
    sampled = net.epoch_sample(1.0)
    assert len(sampled) == 3
    assert len(set(sampled)) == 3                      # without replacement
    assert all(0 <= i < len(FEAT_KEYS_12) for i in sampled)
    assert sampled == sorted(sampled)                  # stable order


def test_epoch_sample_is_uniform_over_the_dynamic_space():
    """S should be a subset of the dynamic search space S_C."""
    lam = 0.5
    net = _make_search_net()
    # the gumbel draw of S_C is random, so replay the same draw for both calls
    torch.manual_seed(7)
    random.seed(7)
    space, _ = net.dynamic_search_space(lam)
    torch.manual_seed(7)
    random.seed(7)
    sampled = net.epoch_sample(lam)
    assert set(sampled).issubset(set(space.tolist()))


def test_epoch_sample_shrinks_to_compact_space():
    """At the end of the search |S_C| = V < M is possible, then |S| = V."""
    net = _make_search_net(num_final=2, num_sampled=3)
    torch.manual_seed(0)
    sampled = net.epoch_sample(0.0)
    assert len(sampled) == 2


# ============================================================
# 5. Super-net Forward Tests (paper Eq. 4)
# ============================================================

def test_search_net_forward_shape():
    """Eq. 4 should map the activated meta-path features to class logits."""
    torch.manual_seed(0)
    net = _make_search_net()
    net.train()
    feats = _make_feats(FEAT_KEYS_12)
    torch.manual_seed(1)
    sampled = net.epoch_sample(1.0)
    out = net(sampled, feats)
    assert out.shape == (4, 3)
    assert torch.isfinite(out).all()


def test_search_net_only_activates_sampled_paths():
    """Only the activated meta-paths (and the residual) should get gradients."""
    torch.manual_seed(0)
    net = _make_search_net()
    net.train()
    feats = _make_feats(FEAT_KEYS_12, dims={k: 8 for k in FEAT_KEYS_12})
    sampled = [0, 2, 4]
    out = net(sampled, feats)
    out.sum().backward()

    active = {FEAT_KEYS_12[i] for i in sampled}
    for k in FEAT_KEYS_12:
        grad = net.feat_proj_l1[k].grad
        if k in active:
            assert grad is not None and grad.abs().sum() > 0, f"{k} should be trained"
        else:
            assert grad is None or grad.abs().sum() == 0, f"{k} should be frozen"
    # the architecture parameter is differentiable as well
    assert net.alpha.grad is not None
    assert net.alpha.grad[list(sampled)].abs().sum() > 0


def test_search_net_handles_mixed_feature_dims():
    """Meta-path features may come from node types with different raw dims."""
    dims = {k: (8 if i % 2 == 0 else 16) for i, k in enumerate(FEAT_KEYS_12)}
    net = _make_search_net(dims=dims)
    net.train()
    feats = _make_feats(FEAT_KEYS_12, dims=dims)
    out = net([0, 1, 2], feats)
    assert out.shape == (4, 3)
    assert torch.isfinite(out).all()


def test_search_net_without_residual():
    """The residual connection is optional."""
    net = _make_search_net(residual=False)
    net.train()
    assert not hasattr(net, 'res_fc')
    out = net([0, 1, 2], _make_feats(FEAT_KEYS_12))
    assert out.shape == (4, 3)


def test_search_net_forward_without_target_key_in_feat_keys():
    """The target key feeds the residual only and may be absent from S."""
    torch.manual_seed(0)
    net = _make_search_net()
    net.train()
    feats = _make_feats(FEAT_KEYS_12)
    out = net([1, 2, 3], feats)     # index 0 ('P') not activated
    assert out.shape == (4, 3)


# ============================================================
# 6. Sampling Evaluation Tests (paper Eq. 6)
# ============================================================

def test_sample_eval_cardinality():
    """Eq. 6: S_bar = DiscreteSample(S_V, M, Q_bar), without replacement."""
    net = _make_search_net()
    torch.manual_seed(0)
    sampled = net.sample_eval()
    assert len(sampled) == 3
    assert len(set(sampled)) == 3
    assert sampled == sorted(sampled)


def test_sample_eval_draws_from_the_compact_space():
    """Every returned meta-path should belong to the compact space S_V."""
    net = _make_search_net()
    torch.manual_seed(0)
    compact = set(net.compact_search_space().tolist())
    assert len(compact) == 5
    for _ in range(20):
        assert set(net.sample_eval()).issubset(compact)


def test_sample_eval_follows_path_strength():
    """Raising the strength of one path should raise its inclusion rate."""
    net = _make_search_net(num_final=5, num_sampled=1)
    with torch.no_grad():
        net.alpha.fill_(-10.0)
        net.alpha[3] = 10.0
    torch.manual_seed(0)
    hits = sum(1 for _ in range(50) if net.sample_eval() == [3])
    # with M = 1 the strongest path should be drawn almost always
    assert hits >= 40, f"strongest path drawn only {hits}/50 times"


def test_sample_eval_is_weighted_not_uniform():
    """The draw should follow the path strength instead of being uniform."""
    net = _make_search_net(num_final=5, num_sampled=1)
    with torch.no_grad():
        net.alpha.fill_(0.0)
        net.alpha[0] = 2.0     # the strongest path of the compact space
        net.alpha[1] = 1.0
    torch.manual_seed(0)
    counts = {k: 0 for k in range(len(FEAT_KEYS_12))}
    for _ in range(200):
        counts[net.sample_eval()[0]] += 1
    # a uniform draw over the 5 candidates would give ~40 hits each
    assert counts[0] > counts[1]
    assert counts[0] > 60, f"weighted draw looks uniform: {counts}"


# ============================================================
# 7. Target-net Tests (paper Eq. 7)
# ============================================================

def test_target_net_forward_shape():
    """Eq. 7 concatenates the projections of the selected meta-paths."""
    from openhgnn.models.LMSPS import LMSPS
    torch.manual_seed(0)
    model = LMSPS(MockArgs())
    model.train()
    out = model({'0': _make_feats(FEAT_KEYS_12)})
    assert out.shape == (4, 3)
    assert torch.isfinite(out).all()


def test_target_net_concatenates_every_feat_key():
    """Eq. 7 concatenates one projection per key of ``feat_keys``."""
    from openhgnn.models.LMSPS import LMSPS
    keys = [k for k in FEAT_KEYS_12 if k != TGT_KEY]
    model = LMSPS(MockArgs(feat_keys=keys))
    assert model.num_feats == 11
    assert model.concat_project_layer.in_features == 11 * model.hidden
    # the target key feeds the residual through its own linear layer, so it is
    # passed in ``feat_keys`` by the flow only when it was actually selected
    assert model.res_fc.weight.shape == (model.hidden, model.nfeat)


def test_target_net_accepts_target_key_outside_feat_keys():
    """The flow keeps the target key in the dict even when it is not selected."""
    from openhgnn.models.LMSPS import LMSPS
    keys = [k for k in FEAT_KEYS_12 if k != TGT_KEY]
    model = LMSPS(MockArgs(feat_keys=keys))
    model.train()
    out = model({'0': _make_feats(FEAT_KEYS_12)})   # dict still has 'P'
    assert out.shape == (4, 3)


def test_target_net_embeddings_only_for_foreign_dims():
    """A projection is created only for meta-paths of a different raw dim."""
    from openhgnn.models.LMSPS import LMSPS
    # all raw dims equal nfeat -> no embedding at all
    model = LMSPS(MockArgs())
    assert model.embedings is None
    # one foreign dim (256-d features for a 128-d target) -> exactly one
    dims = {k: 8 for k in FEAT_KEYS_12}
    dims['PA'] = 16
    model = LMSPS(MockArgs(data_size=dims))
    assert model.embedings is not None
    assert list(model.embedings.keys()) == ['PA']
    model.train()
    out = model({'0': _make_feats(FEAT_KEYS_12, dims=dims)})
    assert out.shape == (4, 3)


def test_target_net_without_residual():
    """The residual connection is optional."""
    from openhgnn.models.LMSPS import LMSPS
    model = LMSPS(MockArgs(residual=False))
    model.train()
    assert not hasattr(model, 'res_fc')
    out = model({'0': _make_feats(FEAT_KEYS_12)})
    assert out.shape == (4, 3)


def test_build_model_from_args():
    """Test build_model_from_args class method."""
    from openhgnn.models.LMSPS import LMSPS
    args = MockArgs(feat_keys=['PA', 'PF', 'PAP'], hidden=32, n_layers_2=3)
    model = LMSPS.build_model_from_args(args)
    assert model.hidden == 32
    assert model.num_feats == 3
    assert model.tgt_key == TGT_KEY


def test_target_net_is_a_pure_mlp_architecture():
    """The target-net has no attention at all (paper Section 5.3)."""
    from openhgnn.models.LMSPS import LMSPS
    model = LMSPS(MockArgs())
    assert not hasattr(model, 'semantic_aggr_layers')
    assert not any(isinstance(m, torch.nn.MultiheadAttention)
                   for m in model.modules())
    n_params = sum(p.numel() for p in model.parameters())
    # one projection per meta-path plus a small MLP head; no attention matrices
    assert n_params < 20000, f"unexpectedly large target-net: {n_params} params"


# ============================================================
# 8. Edge Cases
# ============================================================

def test_single_meta_path():
    """A search space of one meta-path should not crash the sampler."""
    net = _make_search_net(feat_keys=['PA'], num_final=1, num_sampled=1)
    torch.manual_seed(0)
    assert net.epoch_sample(1.0) == [0]
    assert net.sample_eval() == [0]
    net.train()
    # the target key is still needed by the residual connection
    out = net([0], {'PA': torch.randn(4, 8), TGT_KEY: torch.randn(4, 8)})
    assert out.shape == (4, 3)


def test_num_final_equal_to_num_paths():
    """V = K means the dynamic search space never shrinks."""
    net = _make_search_net(num_final=len(FEAT_KEYS_12), num_sampled=2)
    for lam in (1.0, 0.5, 0.0):
        torch.manual_seed(0)
        space, c = net.dynamic_search_space(lam)
        assert c == len(FEAT_KEYS_12)
        assert space.numel() == len(FEAT_KEYS_12)


def test_num_sampled_larger_than_search_space():
    """M > |S_C| should be capped instead of raising."""
    net = _make_search_net(num_final=2, num_sampled=10)
    torch.manual_seed(0)
    sampled = net.epoch_sample(0.0)
    assert len(sampled) == 2
    torch.manual_seed(0)
    assert len(net.sample_eval()) == 2


def test_set_tau():
    """The temperature is set explicitly by the flow."""
    net = _make_search_net()
    net.set_tau(4.0)
    assert net.tau == 4.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
