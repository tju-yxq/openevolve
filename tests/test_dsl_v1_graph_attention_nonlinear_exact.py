# ruff: noqa: E402

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
pytest.importorskip("e3nn")
from e3nn import o3

from scripts.audit_equiformer_v1_graph_attention_contract import (
    _load_official_module,
    _segment_reduce,
    _segment_softmax,
)
from equivariant_nas.dsl import Compiler, core_registry, reference_motif_registry
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.reference_programs import (
    EQUIFORMER_V1_GRAPH_ATTENTION_NONLINEAR_PARAMETER_MAPPING,
    equiformer_v1_graph_attention_program,
)


HIDDEN_IRREPS = "4x0e+2x1e+1x2e"
EDGE_IRREPS = "1x0e+1x1e+1x2e"
HEAD_IRREPS = "2x0e+1x1e+1x2e"


def _official_graph_attention(module):
    return module.GraphAttention(
        irreps_node_input=o3.Irreps(HIDDEN_IRREPS),
        irreps_node_attr=o3.Irreps("1x0e"),
        irreps_edge_attr=o3.Irreps(EDGE_IRREPS),
        irreps_node_output=o3.Irreps(HIDDEN_IRREPS),
        fc_neurons=[6, 8],
        irreps_head=o3.Irreps(HEAD_IRREPS),
        num_heads=2,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=True,
        alpha_drop=0.0,
        proj_drop=0.0,
    ).double().eval()


def _radial_output_scales(official):
    scales = torch.ones(30, dtype=torch.float64)
    for output_slice, scale in official.sep_act.dtp.slices_sqrt_k.values():
        scales[output_slice] *= float(scale)
    return scales.tolist()


def _dsl_inputs(node_input, edge_src, edge_dst, edge_attr, edge_scalars):
    return {
        "node_input": node_input,
        "edge_attr": edge_attr,
        "edge_scalars": edge_scalars,
        "source_index": {"indices": edge_src, "target_size": edge_src.numel()},
        "target_index": {"indices": edge_dst, "target_size": edge_dst.numel()},
        "segment_index": {"indices": edge_dst, "target_size": node_input.shape[0]},
    }


def _official_intermediates(official, node_input, edge_src, edge_dst, edge_attr, edge_scalars):
    merge_src = official.merge_src(node_input)
    merge_dst = official.merge_dst(node_input)
    message = merge_src.index_select(0, edge_src) + merge_dst.index_select(0, edge_dst)
    radial = official.sep_act.dtp_rad(edge_scalars)
    tp = official.sep_act.dtp(message, edge_attr, radial)

    message_linear = official.sep_act.lin(tp)
    message_scalars = message_linear[..., :4]
    message_gates = message_linear[..., 4:7]
    message_gated_values = message_linear[..., 7:]
    message_activated_scalars = official.sep_act.gate.act_scalars(message_scalars)
    message_activated_gates = official.sep_act.gate.act_gates(message_gates)
    message_gated = official.sep_act.gate.mul(message_gated_values, message_activated_gates)
    message_gate_output = torch.cat([message_activated_scalars, message_gated], dim=-1)

    alpha_projection = official.sep_alpha(tp)
    alpha_channels = official.vec2heads_alpha(alpha_projection)
    value_tp = official.sep_value.dtp(message_gate_output, edge_attr)
    value_projection = official.sep_value.lin(value_tp)
    value = official.vec2heads_value(value_projection)

    alpha_activated = official.alpha_act(alpha_channels)
    logits = torch.einsum("bik,aik->bi", alpha_activated, official.alpha_dot)
    softmax = _segment_softmax(logits, edge_dst, torch)
    weighted = value * softmax.unsqueeze(-1)
    aggregate = _segment_reduce(weighted, edge_dst, node_input.shape[0], torch, reduce="sum")
    merged = official.heads2vec(aggregate)
    out = official.proj(merged)
    return {
        "merge_src": merge_src,
        "merge_dst": merge_dst,
        "message": message,
        "radial": radial,
        "tp": tp,
        "message_linear": message_linear,
        "message_scalars": message_scalars,
        "message_gates": message_gates,
        "message_gated_values": message_gated_values,
        "message_activated_scalars": message_activated_scalars,
        "message_activated_gates": message_activated_gates,
        "message_gated": message_gated,
        "message_gate_output": message_gate_output,
        "alpha_projection": alpha_projection,
        "value_tp": value_tp,
        "value_projection": value_projection,
        "alpha_channels": alpha_channels,
        "value": value,
        "alpha_activated": alpha_activated,
        "logits": logits,
        "softmax": softmax,
        "weighted": weighted,
        "aggregate": aggregate,
        "merged": merged,
        "out": out,
    }


def test_v1_nonlinear_graph_attention_matches_official_19_parameters_forward_gradients_and_symmetries():
    root = Path(__file__).resolve().parents[2] / "equiformer"
    official_module, _source = _load_official_module(root, torch)
    torch.manual_seed(20260761)
    official = _official_graph_attention(official_module)

    registry = core_registry()
    program = equiformer_v1_graph_attention_program(
        _radial_output_scales(official),
        nonlinear_message=True,
    )
    artifact = Compiler(registry, reference_motif_registry()).analyze(program)
    torch.manual_seed(20260761)
    model = E3NNGraphBackend(registry).build(
        artifact.expanded_program,
        artifact.inference,
    ).double().eval()

    mapping = dict(EQUIFORMER_V1_GRAPH_ATTENTION_NONLINEAR_PARAMETER_MAPPING)
    official_parameters = dict(official.named_parameters())
    dsl_parameters = dict(model.named_parameters())
    assert len(mapping) == 19
    assert set(dsl_parameters) == set(mapping)
    assert set(mapping.values()) == set(official_parameters)
    assert sum(parameter.numel() for parameter in dsl_parameters.values()) == 649
    for dsl_name, official_name in mapping.items():
        assert tuple(dsl_parameters[dsl_name].shape) == tuple(official_parameters[official_name].shape)
        assert torch.equal(dsl_parameters[dsl_name], official_parameters[official_name]), (
            dsl_name,
            official_name,
            float((dsl_parameters[dsl_name] - official_parameters[official_name]).abs().max()),
        )

    assert program.annotations["official_constructor_used_for_execution"] is False
    assert all("official" not in node.op.lower() for node in artifact.expanded_program.nodes)

    node_count = 5
    edge_src = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    edge_dst = torch.tensor([1, 1, 3, 3, 0, 4, 4, 2], dtype=torch.long)
    node_input = torch.randn(node_count, 15, dtype=torch.float64, requires_grad=True)
    edge_attr = torch.randn(edge_src.numel(), 9, dtype=torch.float64, requires_grad=True)
    edge_scalars = torch.randn(edge_src.numel(), 6, dtype=torch.float64, requires_grad=True)
    official_node = node_input.detach().clone().requires_grad_(True)
    official_edge = edge_attr.detach().clone().requires_grad_(True)
    official_radial = edge_scalars.detach().clone().requires_grad_(True)

    actual = model(_dsl_inputs(node_input, edge_src, edge_dst, edge_attr, edge_scalars), {})
    expected = _official_intermediates(
        official,
        official_node,
        edge_src,
        edge_dst,
        official_edge,
        official_radial,
    )
    assert set(actual) == set(expected)
    for name, expected_value in expected.items():
        assert actual[name].shape == expected_value.shape, name
        assert torch.allclose(actual[name], expected_value, atol=1.0e-12, rtol=1.0e-12), (
            name,
            float((actual[name] - expected_value).detach().abs().max()),
        )

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
        ), (dsl_name, official_name)

    with torch.no_grad():
        for matrix in (o3.rand_matrix(dtype=torch.float64), -torch.eye(3, dtype=torch.float64)):
            hidden_action = o3.Irreps(HIDDEN_IRREPS).D_from_matrix(matrix)
            edge_action = o3.Irreps(EDGE_IRREPS).D_from_matrix(matrix)
            tp_action = official.sep_act.dtp.irreps_out.D_from_matrix(matrix)
            message_linear_action = official.sep_act.lin.irreps_out.D_from_matrix(matrix)
            gated_value_action = official.sep_act.gate.irreps_gated.D_from_matrix(matrix)
            value_projection_action = official.sep_value.lin.irreps_out.D_from_matrix(matrix)
            head_action = o3.Irreps(HEAD_IRREPS).D_from_matrix(matrix)
            transformed = model(
                _dsl_inputs(
                    node_input.detach() @ hidden_action.transpose(0, 1),
                    edge_src,
                    edge_dst,
                    edge_attr.detach() @ edge_action.transpose(0, 1),
                    edge_scalars.detach(),
                ),
                {},
            )
            equivariant_actions = {
                "merge_src": hidden_action,
                "merge_dst": hidden_action,
                "message": hidden_action,
                "tp": tp_action,
                "message_linear": message_linear_action,
                "message_gated_values": gated_value_action,
                "message_gated": gated_value_action,
                "message_gate_output": hidden_action,
                "value_tp": tp_action,
                "value_projection": value_projection_action,
                "value": head_action,
                "weighted": head_action,
                "aggregate": head_action,
                "merged": value_projection_action,
                "out": hidden_action,
            }
            for name, action in equivariant_actions.items():
                transformed_expected = actual[name] @ action.transpose(0, 1)
                relative_error = (
                    (transformed[name] - transformed_expected).norm()
                    / transformed_expected.norm().clamp_min(1.0e-12)
                )
                assert float(relative_error) < 1.0e-7, (name, float(relative_error))
            for name in (
                "radial",
                "message_scalars",
                "message_gates",
                "message_activated_scalars",
                "message_activated_gates",
                "alpha_projection",
                "alpha_channels",
                "alpha_activated",
                "logits",
                "softmax",
            ):
                assert torch.allclose(transformed[name], actual[name], atol=1.0e-9, rtol=1.0e-9), name

        edge_permutation = torch.tensor([5, 0, 7, 3, 1, 6, 2, 4], dtype=torch.long)
        edge_permuted = model(
            _dsl_inputs(
                node_input.detach(),
                edge_src.index_select(0, edge_permutation),
                edge_dst.index_select(0, edge_permutation),
                edge_attr.detach().index_select(0, edge_permutation),
                edge_scalars.detach().index_select(0, edge_permutation),
            ),
            {},
        )
        assert torch.allclose(
            edge_permuted["softmax"],
            actual["softmax"].index_select(0, edge_permutation),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert torch.allclose(edge_permuted["out"], actual["out"], atol=1.0e-12, rtol=1.0e-12)

        node_permutation = torch.tensor([3, 0, 4, 1, 2], dtype=torch.long)
        inverse = torch.empty_like(node_permutation)
        inverse[node_permutation] = torch.arange(node_count)
        node_permuted = model(
            _dsl_inputs(
                node_input.detach().index_select(0, node_permutation),
                inverse.index_select(0, edge_src),
                inverse.index_select(0, edge_dst),
                edge_attr.detach(),
                edge_scalars.detach(),
            ),
            {},
        )
        assert torch.allclose(
            node_permuted["out"],
            actual["out"].index_select(0, node_permutation),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
