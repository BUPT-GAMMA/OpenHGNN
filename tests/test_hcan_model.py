from types import SimpleNamespace

import pytest
import torch

dgl = pytest.importorskip("dgl")

from openhgnn.models import build_model


def _toy_graph():
    return dgl.heterograph(
        {
            ("author", "writes", "paper"): (
                torch.tensor([0, 1, 1]),
                torch.tensor([0, 1, 2]),
            ),
            ("paper", "written-by", "author"): (
                torch.tensor([0, 1, 2]),
                torch.tensor([0, 1, 1]),
            ),
        },
        num_nodes_dict={"author": 2, "paper": 3},
    )


def _args(task, out_dim):
    return SimpleNamespace(
        hidden_dim=8,
        out_dim=out_dim,
        num_layers=1,
        max_hop=2,
        num_heads=2,
        dropout=0.0,
        attn_activation="identity",
        task=task,
    )


def _features():
    return {
        "author": torch.randn(2, 8),
        "paper": torch.randn(3, 8),
    }


def test_hcan_node_classification_outputs_logits():
    hg = _toy_graph()
    model = build_model("HCAN").build_model_from_args(
        _args(task="node_classification", out_dim=4), hg
    )

    out = model(hg, _features())

    assert out["author"].shape == (2, 4)
    assert out["paper"].shape == (3, 4)


def test_hcan_counterweight_payload_matches_equation_11_dimensions():
    hg = _toy_graph()
    model = build_model("HCAN").build_model_from_args(
        _args(task="node_classification", out_dim=4), hg
    )
    layer = model.layers[0]

    assert layer.value_head_dim * layer.num_heads == layer.hidden_dim // 2
    assert not hasattr(layer, "out_linear")


def test_hcan_reuses_type_projection_across_hop_tokens():
    hg = _toy_graph()
    model = build_model("HCAN").build_model_from_args(
        _args(task="node_classification", out_dim=4), hg
    )
    layer = model.layers[0]

    assert isinstance(layer.hop_mlp["author"], torch.nn.Sequential)
    assert isinstance(layer.hop_mlp["paper"], torch.nn.Sequential)


def test_hcan_relation_mean_matches_equation_5():
    hg = _toy_graph()
    model = build_model("HCAN").build_model_from_args(
        _args(task="node_classification", out_dim=4), hg
    )
    layer = model.layers[0]
    features = {
        "author": torch.tensor([[2.0] * 8, [4.0] * 8]),
        "paper": torch.tensor([[1.0] * 8, [3.0] * 8, [5.0] * 8]),
    }

    propagated = layer._all_relation_mean_propagation(hg, features)

    assert torch.allclose(
        propagated["paper"],
        torch.tensor([[2.0] * 8, [4.0] * 8, [4.0] * 8]),
    )
    assert torch.allclose(
        propagated["author"],
        torch.tensor([[1.0] * 8, [4.0] * 8]),
    )


def test_hcan_requires_half_width_per_attention_payload():
    hg = _toy_graph()
    args = _args(task="node_classification", out_dim=4)
    args.hidden_dim = 10

    with pytest.raises(ValueError, match="twice num_heads"):
        build_model("HCAN").build_model_from_args(args, hg)


def test_hcan_link_prediction_outputs_embeddings():
    hg = _toy_graph()
    args = _args(task="link_prediction", out_dim=4)
    model = build_model("HCAN").build_model_from_args(args, hg)

    out = model(hg, _features())

    assert args.out_dim == 8
    assert out["author"].shape == (2, 8)
    assert out["paper"].shape == (3, 8)


def test_dhcan_node_classification_outputs_logits():
    hg = _toy_graph()
    model = build_model("DHCAN").build_model_from_args(
        _args(task="node_classification", out_dim=4), hg
    )

    out = model(hg, _features())

    assert out["author"].shape == (2, 4)
    assert out["paper"].shape == (3, 4)


def test_dhcan_link_prediction_outputs_embeddings():
    hg = _toy_graph()
    args = _args(task="link_prediction", out_dim=4)
    model = build_model("DHCAN").build_model_from_args(args, hg)

    out = model(hg, _features())

    assert args.out_dim == 8
    assert out["author"].shape == (2, 8)
    assert out["paper"].shape == (3, 8)


def test_dhcan_paper_name_alias():
    assert build_model("D-HCAN") is build_model("DHCAN")


def test_dhcan_precomputes_only_valid_paths_and_reuses_cache():
    hg = _toy_graph()
    features = _features()
    features["author"].requires_grad_()
    features["paper"].requires_grad_()
    model = build_model("DHCAN").build_model_from_args(
        _args(task="node_classification", out_dim=4), hg
    )

    first = model(hg, features)
    layer = model.layers[0]
    cached_ids = [id(token) for token in layer._cached_tokens]
    second = model(hg, {key: value + 10 for key, value in features.items()})
    sum(value.sum() for value in first.values()).backward()

    assert layer.num_channels == 2
    assert (
        layer.channel_proj["paper"][0][0].weight.data_ptr()
        != layer.channel_proj["paper"][1][0].weight.data_ptr()
    )
    assert cached_ids == [id(token) for token in layer._cached_tokens]
    assert all(torch.allclose(first[key], second[key]) for key in first)
    assert features["author"].grad is None
    assert features["paper"].grad is None
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_dhcan_reuses_relation_path_prefixes():
    hg = dgl.heterograph(
        {
            ("node", "left", "node"): (torch.tensor([0]), torch.tensor([1])),
            ("node", "right", "node"): (torch.tensor([1]), torch.tensor([0])),
        },
        num_nodes_dict={"node": 2},
    )
    model = build_model("DHCAN").build_model_from_args(
        _args(task="node_classification", out_dim=2), hg
    )
    layer = model.layers[0]
    relation_propagate = layer._relation_propagate
    call_count = 0

    def counted_propagation(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return relation_propagate(*args, **kwargs)

    layer._relation_propagate = counted_propagation
    tokens = list(layer._generate_decoupled_tokens(hg, {"node": _features()["author"]}))

    assert len(tokens) == 4
    assert call_count == 6


def test_dhcan_clear_cache_recomputes_features():
    hg = _toy_graph()
    model = build_model("DHCAN").build_model_from_args(
        _args(task="node_classification", out_dim=4), hg
    )
    model.eval()
    first = model(hg, _features())
    model.clear_cache()
    second = model(hg, {key: value + 10 for key, value in _features().items()})

    assert any(not torch.allclose(first[key], second[key]) for key in first)


def test_dhcan_streaming_fusion_matches_concatenation():
    hg = _toy_graph()
    features = _features()
    model = build_model("DHCAN").build_model_from_args(
        _args(task="node_classification", out_dim=4), hg
    )
    model.eval()

    actual = model(hg, features, return_emb=True)
    layer = model.layers[0]
    for ntype in hg.ntypes:
        projected = []
        for token in layer._cached_tokens:
            if ntype in token:
                token_features = token[ntype].to(features[ntype].device)
                mask = token_features.abs().sum(dim=-1, keepdim=True) > 0
                projected.append(
                    layer.channel_proj[ntype][len(projected)](token_features)
                    * mask.to(torch.float32)
                )
            else:
                projected.append(
                    features[ntype].new_zeros(
                        hg.num_nodes(ntype), layer.hidden_dim
                    )
                )
        expected = layer.semantic_fusion[ntype](torch.cat(projected, dim=-1))

        assert torch.allclose(actual[ntype], expected, atol=1e-6)


def test_dhcan_attention_includes_center_node():
    hg = dgl.heterograph(
        {
            ("node", "link", "node"): (
                torch.tensor([0, 1]),
                torch.tensor([1, 0]),
            )
        },
        num_nodes_dict={"node": 2},
    )
    args = _args(task="node_classification", out_dim=2)
    args.max_hop = 1
    model = build_model("DHCAN").build_model_from_args(args, hg)
    layer = model.layers[0]
    base = torch.zeros(2, layer.token_dim)
    base[:, 0] = torch.tensor([1.0, 2.0])
    token = next(layer._generate_decoupled_tokens(hg, {"node": base}))

    attended = layer._non_parametric_attention(hg, {"node": base}, token)

    expected = torch.zeros_like(base)
    expected[:, 0] = 5.0 / 3.0
    assert torch.allclose(attended["node"], expected)


def test_dhcan_decodes_cached_node_batch():
    hg = _toy_graph()
    features = _features()
    model = build_model("DHCAN").build_model_from_args(
        _args(task="node_classification", out_dim=4), hg
    )
    model.eval()
    model.precompute(hg, features)

    node_idx = torch.tensor([2, 0])
    batch_logits = model.forward_cached("paper", node_idx)
    full_logits = model(hg, features)["paper"]

    assert torch.allclose(batch_logits, full_logits[node_idx], atol=1e-6)


def test_dhcan_target_cache_matches_full_cache():
    hg = _toy_graph()
    features = _features()
    model = build_model("DHCAN").build_model_from_args(
        _args(task="node_classification", out_dim=4), hg
    )
    model.eval()

    model.precompute(hg, features)
    full_logits = model.forward_cached("paper")
    model.clear_cache()
    target_cache = model.precompute(hg, features, target_ntype="paper")
    target_logits = model.forward_cached("paper")

    assert all(set(token).issubset({"paper"}) for token in target_cache)
    assert any("paper" in token for token in target_cache)
    assert torch.allclose(target_logits, full_logits, atol=1e-6)
