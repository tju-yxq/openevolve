from dataclasses import replace
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
pytest.importorskip("e3nn")
from e3nn import o3

from scripts.audit_equiformer_v1_graph_attention_contract import _load_official_module
from equivariant_nas.dsl import Compiler, DSLValidationError, core_registry, reference_motif_registry
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.reference_programs import (
    EQUIFORMER_V1_GRAPH_ATTENTION_PARAMETER_MAPPING,
    EQUIFORMER_V1_TRANSBLOCK_PARAMETER_MAPPING,
    equiformer_v1_graph_attention_program,
    equiformer_v1_transblock_program,
)


HIDDEN_IRREPS = "4x0e+2x1e+1x2e"
EDGE_IRREPS = "1x0e+1x1e+1x2e"


def _official(module):
    return module.GraphAttention(
        irreps_node_input=o3.Irreps(HIDDEN_IRREPS),
        irreps_node_attr=o3.Irreps("1x0e"),
        irreps_edge_attr=o3.Irreps(EDGE_IRREPS),
        irreps_node_output=o3.Irreps(HIDDEN_IRREPS),
        fc_neurons=[6, 8],
        irreps_head=o3.Irreps("2x0e+1x1e+1x2e"),
        num_heads=2,
        irreps_pre_attn=None,
        rescale_degree=True,
        nonlinear_message=False,
        alpha_drop=0.0,
        proj_drop=0.0,
    ).double().eval()


def _scales(official):
    result = torch.ones(30, dtype=torch.float64)
    for output_slice, scale in official.sep.dtp.slices_sqrt_k.values():
        result[output_slice] *= float(scale)
    return result.tolist()


def _inputs(node_input, edge_src, edge_dst, edge_attr, edge_scalars):
    return {
        "node_input": node_input,
        "edge_attr": edge_attr,
        "edge_scalars": edge_scalars,
        "source_index": {"indices": edge_src, "target_size": edge_src.numel()},
        "target_index": {"indices": edge_dst, "target_size": edge_dst.numel()},
        "segment_index": {"indices": edge_dst, "target_size": node_input.shape[0]},
    }


def _segment_softmax(values, indices, count):
    expanded = indices.reshape(-1, 1).expand_as(values)
    maximum = values.new_full((count, values.shape[1]), float("-inf"))
    maximum.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
    numerator = torch.exp(values - maximum.index_select(0, indices))
    denominator = values.new_zeros((count, values.shape[1]))
    denominator.index_add_(0, indices, numerator)
    return numerator / denominator.index_select(0, indices)


def _official_intermediates(official, node_input, edge_src, edge_dst, edge_attr, edge_scalars):
    merge_src = official.merge_src(node_input)
    merge_dst = official.merge_dst(node_input)
    message = merge_src.index_select(0, edge_src) + merge_dst.index_select(0, edge_dst)
    radial = official.sep.dtp_rad(edge_scalars)
    tp = official.sep.dtp(message, edge_attr, radial)
    post_tp = official.sep.lin(tp)
    heads = official.vec2heads(post_tp)
    alpha_channels = heads.narrow(2, 0, official.mul_alpha_head)
    value = heads.narrow(2, official.mul_alpha_head, heads.shape[-1] - official.mul_alpha_head)
    alpha_activated = official.alpha_act(alpha_channels)
    logits = torch.einsum("bik,aik->bi", alpha_activated, official.alpha_dot)
    softmax = _segment_softmax(logits, edge_dst, node_input.shape[0])
    weighted = value * softmax.unsqueeze(-1)
    aggregate = weighted.new_zeros((node_input.shape[0],) + weighted.shape[1:])
    aggregate.index_add_(0, edge_dst, weighted)
    degree = torch.bincount(edge_dst, minlength=node_input.shape[0]).to(node_input.dtype)
    aggregate = aggregate * degree.reshape(-1, 1, 1)
    merged = official.heads2vec(aggregate)
    return {
        "merge_src": merge_src,
        "merge_dst": merge_dst,
        "message": message,
        "radial": radial,
        "tp": tp,
        "post_tp": post_tp,
        "heads": heads,
        "alpha_channels": alpha_channels,
        "value": value,
        "alpha_activated": alpha_activated,
        "logits": logits,
        "softmax": softmax,
        "weighted": weighted,
        "aggregate": aggregate,
        "merged": merged,
        "out": official.proj(merged),
    }


def test_v1_degree_rescale_matches_official_forward_gradients_and_so3_without_hidden_topology():
    root = Path(__file__).resolve().parents[2] / "equiformer"
    official_module, _source = _load_official_module(root, torch)
    torch.manual_seed(20260731)
    official = _official(official_module)
    program = equiformer_v1_graph_attention_program(
        _scales(official),
        rescale_degree=True,
    )
    aggregate_node = next(node for node in program.nodes if node.id == "aggregate")
    assert aggregate_node.attrs == {"reduce": "sum", "normalization": "target_cardinality"}
    registry = core_registry()
    artifact = Compiler(registry, reference_motif_registry()).analyze(program)
    torch.manual_seed(20260731)
    model = E3NNGraphBackend(registry).build(artifact.expanded_program, artifact.inference).double().eval()

    mapping = dict(EQUIFORMER_V1_GRAPH_ATTENTION_PARAMETER_MAPPING)
    official_parameters = dict(official.named_parameters())
    dsl_parameters = dict(model.named_parameters())
    assert set(dsl_parameters) == set(mapping)
    for dsl_name, official_name in mapping.items():
        assert torch.equal(dsl_parameters[dsl_name], official_parameters[official_name])

    edge_src = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    edge_dst = torch.tensor([1, 1, 3, 3, 0, 4, 4, 2], dtype=torch.long)
    torch.manual_seed(20260732)
    node_input = torch.randn(5, 15, dtype=torch.float64, requires_grad=True)
    edge_attr = torch.randn(8, 9, dtype=torch.float64, requires_grad=True)
    edge_scalars = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
    official_node = node_input.detach().clone().requires_grad_(True)
    official_edge = edge_attr.detach().clone().requires_grad_(True)
    official_radial = edge_scalars.detach().clone().requires_grad_(True)

    actual = model(_inputs(node_input, edge_src, edge_dst, edge_attr, edge_scalars), {})
    expected = _official_intermediates(
        official,
        official_node,
        edge_src,
        edge_dst,
        official_edge,
        official_radial,
    )
    assert set(actual) == set(expected)
    for name in expected:
        assert torch.allclose(actual[name], expected[name], atol=1.0e-12, rtol=1.0e-12), name

    direct = official(
        official_node.detach(),
        None,
        edge_src,
        edge_dst,
        official_edge.detach(),
        official_radial.detach(),
        None,
    )
    assert torch.allclose(direct, expected["out"].detach(), atol=1.0e-12, rtol=1.0e-12)

    torch.manual_seed(20260733)
    probe = torch.randn_like(actual["out"])
    (actual["out"] * probe).sum().backward()
    (expected["out"] * probe).sum().backward()
    for actual_gradient, expected_gradient in (
        (node_input.grad, official_node.grad),
        (edge_attr.grad, official_edge.grad),
        (edge_scalars.grad, official_radial.grad),
    ):
        assert torch.allclose(actual_gradient, expected_gradient, atol=1.0e-11, rtol=1.0e-11)
    for dsl_name, official_name in mapping.items():
        assert torch.allclose(
            dsl_parameters[dsl_name].grad,
            official_parameters[official_name].grad,
            atol=1.0e-11,
            rtol=1.0e-11,
        )

    with torch.no_grad():
        rotation = o3.rand_matrix(dtype=torch.float64)
        node_action = o3.Irreps(HIDDEN_IRREPS).D_from_matrix(rotation)
        edge_action = o3.Irreps(EDGE_IRREPS).D_from_matrix(rotation)
        transformed = model(
            _inputs(
                node_input.detach() @ node_action.transpose(0, 1),
                edge_src,
                edge_dst,
                edge_attr.detach() @ edge_action.transpose(0, 1),
                edge_scalars.detach(),
            ),
            {},
        )["out"]
        expected_transformed = actual["out"].detach() @ node_action.transpose(0, 1)
        relative_error = (
            (transformed - expected_transformed).norm()
            / expected_transformed.norm().clamp_min(1.0e-12)
        )
        assert float(relative_error) < 1.0e-7


def test_target_cardinality_rejects_mean_reduction():
    program = equiformer_v1_graph_attention_program([1.0] * 30, rescale_degree=True)
    nodes = tuple(
        replace(node, attrs={"reduce": "mean", "normalization": "target_cardinality"})
        if node.id == "aggregate"
        else node
        for node in program.nodes
    )
    with pytest.raises(DSLValidationError) as captured:
        Compiler(core_registry(), reference_motif_registry()).analyze(replace(program, nodes=nodes))
    assert any(item.code == "E_ATTR_008" for item in captured.value.diagnostics)


def test_v1_transblock_degree_rescale_matches_official_forward_and_gradients():
    root = Path(__file__).resolve().parents[2] / "equiformer"
    official_module, _source = _load_official_module(root, torch)
    torch.manual_seed(20260741)
    official = official_module.TransBlock(
        irreps_node_input=o3.Irreps(HIDDEN_IRREPS),
        irreps_node_attr=o3.Irreps("1x0e"),
        irreps_edge_attr=o3.Irreps(EDGE_IRREPS),
        irreps_node_output=o3.Irreps(HIDDEN_IRREPS),
        fc_neurons=[6, 8],
        irreps_head=o3.Irreps("2x0e+1x1e+1x2e"),
        num_heads=2,
        irreps_pre_attn=None,
        rescale_degree=True,
        nonlinear_message=False,
        alpha_drop=0.0,
        proj_drop=0.0,
        drop_path_rate=0.0,
        irreps_mlp_mid=o3.Irreps(HIDDEN_IRREPS),
        norm_layer="layer",
    ).double().eval()
    scales = torch.ones(30, dtype=torch.float64)
    for output_slice, scale in official.ga.sep.dtp.slices_sqrt_k.values():
        scales[output_slice] *= float(scale)
    program = equiformer_v1_transblock_program(scales.tolist(), rescale_degree=True)
    registry = core_registry()
    artifact = Compiler(registry, reference_motif_registry()).analyze(program)
    torch.manual_seed(20260741)
    model = E3NNGraphBackend(registry).build(artifact.expanded_program, artifact.inference).double().eval()

    mapping = dict(EQUIFORMER_V1_TRANSBLOCK_PARAMETER_MAPPING)
    official_parameters = dict(official.named_parameters())
    dsl_parameters = dict(model.named_parameters())
    assert set(dsl_parameters) == set(mapping)
    for dsl_name, official_name in mapping.items():
        assert torch.equal(dsl_parameters[dsl_name], official_parameters[official_name])

    edge_src = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    edge_dst = torch.tensor([1, 1, 3, 3, 0, 4, 4, 2], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    torch.manual_seed(20260742)
    node_input = torch.randn(5, 15, dtype=torch.float64, requires_grad=True)
    node_attr = torch.randn(5, 1, dtype=torch.float64, requires_grad=True)
    edge_attr = torch.randn(8, 9, dtype=torch.float64, requires_grad=True)
    edge_scalars = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
    official_node = node_input.detach().clone().requires_grad_(True)
    official_node_attr = node_attr.detach().clone().requires_grad_(True)
    official_edge = edge_attr.detach().clone().requires_grad_(True)
    official_radial = edge_scalars.detach().clone().requires_grad_(True)

    actual = model(
        {
            **_inputs(node_input, edge_src, edge_dst, edge_attr, edge_scalars),
            "node_attr": node_attr,
        },
        {},
    )["out"]
    expected = official(
        official_node,
        official_node_attr,
        edge_src,
        edge_dst,
        official_edge,
        official_radial,
        batch,
    )
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)

    torch.manual_seed(20260743)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    for actual_gradient, expected_gradient in (
        (node_input.grad, official_node.grad),
        (node_attr.grad, official_node_attr.grad),
        (edge_attr.grad, official_edge.grad),
        (edge_scalars.grad, official_radial.grad),
    ):
        assert torch.allclose(actual_gradient, expected_gradient, atol=1.0e-11, rtol=1.0e-11)
    for dsl_name, official_name in mapping.items():
        assert torch.allclose(
            dsl_parameters[dsl_name].grad,
            official_parameters[official_name].grad,
            atol=1.0e-11,
            rtol=1.0e-11,
        )
