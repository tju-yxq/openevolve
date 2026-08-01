# ruff: noqa: E402

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
pytest.importorskip("e3nn")
from e3nn import o3

from scripts.audit_equiformer_v1_graph_attention_contract import _load_official_module
from equivariant_nas.dsl import Compiler, core_registry, reference_motif_registry
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.reference_programs import (
    EQUIFORMER_V1_TRANSBLOCK_NONLINEAR_SHORTCUT_PARAMETER_MAPPING,
    EQUIFORMER_V1_TRANSBLOCK_SHORTCUT_PARAMETER_MAPPING,
    equiformer_v1_transblock_program,
)


HIDDEN_IRREPS = "4x0e+2x1e+1x2e"
OUTPUT_IRREPS = "3x0e+1x1e+1x2e"
EDGE_IRREPS = "1x0e+1x1e+1x2e"


def _official_transblock(module, *, nonlinear_message):
    return module.TransBlock(
        irreps_node_input=o3.Irreps(HIDDEN_IRREPS),
        irreps_node_attr=o3.Irreps("1x0e"),
        irreps_edge_attr=o3.Irreps(EDGE_IRREPS),
        irreps_node_output=o3.Irreps(OUTPUT_IRREPS),
        fc_neurons=[6, 8],
        irreps_head=o3.Irreps("2x0e+1x1e+1x2e"),
        num_heads=2,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=bool(nonlinear_message),
        alpha_drop=0.0,
        proj_drop=0.0,
        drop_path_rate=0.0,
        irreps_mlp_mid=o3.Irreps(HIDDEN_IRREPS),
        norm_layer="layer",
    ).double().eval()


def _radial_output_scales(official):
    scales = torch.ones(30, dtype=torch.float64)
    depthwise = official.ga.sep_act.dtp if official.ga.nonlinear_message else official.ga.sep.dtp
    for output_slice, scale in depthwise.slices_sqrt_k.values():
        scales[output_slice] *= float(scale)
    return scales.tolist()


def _dsl_inputs(node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars):
    return {
        "node_input": node_input,
        "node_attr": node_attr,
        "edge_attr": edge_attr,
        "edge_scalars": edge_scalars,
        "source_index": {"indices": edge_src, "target_size": edge_src.numel()},
        "target_index": {"indices": edge_dst, "target_size": edge_dst.numel()},
        "segment_index": {"indices": edge_dst, "target_size": node_input.shape[0]},
    }


def _official_intermediates(
    official,
    node_input,
    node_attr,
    edge_src,
    edge_dst,
    edge_attr,
    edge_scalars,
    batch,
):
    norm_1 = official.norm_1(node_input, batch=batch)
    attention = official.ga(
        node_input=norm_1,
        node_attr=node_attr,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_attr=edge_attr,
        edge_scalars=edge_scalars,
        batch=batch,
    )
    attention_residual = node_input + attention
    norm_2 = official.norm_2(attention_residual, batch=batch)
    ffn = official.ffn(norm_2, node_attr)
    ffn_shortcut = official.ffn_shortcut(attention_residual, node_attr)
    return {
        "norm_1": norm_1,
        "attention": attention,
        "attention_residual": attention_residual,
        "norm_2": norm_2,
        "ffn": ffn,
        "ffn_shortcut": ffn_shortcut,
        "out": ffn_shortcut + ffn,
    }


@pytest.mark.parametrize(
    ("nonlinear_message", "parameter_mapping", "parameter_count"),
    (
        (False, EQUIFORMER_V1_TRANSBLOCK_SHORTCUT_PARAMETER_MAPPING, 626),
        (True, EQUIFORMER_V1_TRANSBLOCK_NONLINEAR_SHORTCUT_PARAMETER_MAPPING, 747),
    ),
)
def test_v1_transblock_different_output_irreps_uses_exact_typed_shortcut(
    nonlinear_message,
    parameter_mapping,
    parameter_count,
):
    root = Path(__file__).resolve().parents[2] / "equiformer"
    official_module, _source = _load_official_module(root, torch)
    seed = 20260763 + int(nonlinear_message)
    torch.manual_seed(seed)
    official = _official_transblock(
        official_module,
        nonlinear_message=nonlinear_message,
    )
    program = equiformer_v1_transblock_program(
        _radial_output_scales(official),
        node_output_irreps=OUTPUT_IRREPS,
        nonlinear_message=nonlinear_message,
    )
    registry = core_registry()
    artifact = Compiler(registry, reference_motif_registry()).analyze(program)
    torch.manual_seed(seed)
    model = E3NNGraphBackend(registry).build(
        artifact.expanded_program,
        artifact.inference,
    ).double().eval()

    mapping = dict(parameter_mapping)
    official_parameters = dict(official.named_parameters())
    dsl_parameters = dict(model.named_parameters())
    assert len(mapping) == 24 + 5 * int(nonlinear_message)
    assert set(dsl_parameters) == set(mapping)
    assert set(mapping.values()) == set(official_parameters)
    assert sum(parameter.numel() for parameter in dsl_parameters.values()) == parameter_count
    for dsl_name, official_name in mapping.items():
        assert tuple(dsl_parameters[dsl_name].shape) == tuple(official_parameters[official_name].shape)
        assert torch.equal(dsl_parameters[dsl_name], official_parameters[official_name]), (
            dsl_name,
            official_name,
            float((dsl_parameters[dsl_name] - official_parameters[official_name]).abs().max()),
        )

    shortcut_nodes = [node for node in artifact.expanded_program.nodes if node.id == "ffn_shortcut"]
    assert len(shortcut_nodes) == 1
    assert shortcut_nodes[0].op == "core.tensor_product@4"
    assert all("official" not in node.op.lower() for node in artifact.expanded_program.nodes)

    node_count = 5
    edge_src = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    edge_dst = torch.tensor([1, 1, 3, 3, 0, 4, 4, 2], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    node_input = torch.randn(node_count, 15, dtype=torch.float64, requires_grad=True)
    node_attr = torch.randn(node_count, 1, dtype=torch.float64, requires_grad=True)
    edge_attr = torch.randn(edge_src.numel(), 9, dtype=torch.float64, requires_grad=True)
    edge_scalars = torch.randn(edge_src.numel(), 6, dtype=torch.float64, requires_grad=True)
    official_node = node_input.detach().clone().requires_grad_(True)
    official_node_attr = node_attr.detach().clone().requires_grad_(True)
    official_edge = edge_attr.detach().clone().requires_grad_(True)
    official_radial = edge_scalars.detach().clone().requires_grad_(True)

    actual = model(
        _dsl_inputs(node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars),
        {},
    )
    expected = _official_intermediates(
        official,
        official_node,
        official_node_attr,
        edge_src,
        edge_dst,
        official_edge,
        official_radial,
        batch,
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
        ), (dsl_name, official_name)

    with torch.no_grad():
        for matrix in (o3.rand_matrix(dtype=torch.float64), -torch.eye(3, dtype=torch.float64)):
            hidden_action = o3.Irreps(HIDDEN_IRREPS).D_from_matrix(matrix)
            output_action = o3.Irreps(OUTPUT_IRREPS).D_from_matrix(matrix)
            edge_action = o3.Irreps(EDGE_IRREPS).D_from_matrix(matrix)
            transformed = model(
                _dsl_inputs(
                    node_input.detach() @ hidden_action.transpose(0, 1),
                    node_attr.detach(),
                    edge_src,
                    edge_dst,
                    edge_attr.detach() @ edge_action.transpose(0, 1),
                    edge_scalars.detach(),
                ),
                {},
            )
            for name in ("norm_1", "attention", "attention_residual", "norm_2"):
                transformed_expected = actual[name] @ hidden_action.transpose(0, 1)
                relative_error = (
                    (transformed[name] - transformed_expected).norm()
                    / transformed_expected.norm().clamp_min(1.0e-12)
                )
                assert float(relative_error) < 1.0e-7, (name, float(relative_error))
            for name in ("ffn", "ffn_shortcut", "out"):
                transformed_expected = actual[name] @ output_action.transpose(0, 1)
                relative_error = (
                    (transformed[name] - transformed_expected).norm()
                    / transformed_expected.norm().clamp_min(1.0e-12)
                )
                assert float(relative_error) < 1.0e-7, (name, float(relative_error))

        edge_permutation = torch.tensor([5, 0, 7, 3, 1, 6, 2, 4], dtype=torch.long)
        edge_permuted = model(
            _dsl_inputs(
                node_input.detach(),
                node_attr.detach(),
                edge_src.index_select(0, edge_permutation),
                edge_dst.index_select(0, edge_permutation),
                edge_attr.detach().index_select(0, edge_permutation),
                edge_scalars.detach().index_select(0, edge_permutation),
            ),
            {},
        )
        assert torch.allclose(edge_permuted["out"], actual["out"], atol=1.0e-12, rtol=1.0e-12)

        node_permutation = torch.tensor([3, 0, 4, 1, 2], dtype=torch.long)
        inverse = torch.empty_like(node_permutation)
        inverse[node_permutation] = torch.arange(node_count)
        node_permuted = model(
            _dsl_inputs(
                node_input.detach().index_select(0, node_permutation),
                node_attr.detach().index_select(0, node_permutation),
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
