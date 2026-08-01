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
    EQUIFORMER_V1_TRANSBLOCK_PARAMETER_MAPPING,
    equiformer_v1_transblock_program,
)


HIDDEN_IRREPS = "4x0e+2x1e+1x2e"
EDGE_IRREPS = "1x0e+1x1e+1x2e"
ALPHA_DROP = 0.2
PROJ_DROP = 0.15
DROP_PATH = 0.25


def _official(
    module,
    *,
    alpha_drop=ALPHA_DROP,
    proj_drop=PROJ_DROP,
    drop_path=DROP_PATH,
):
    return module.TransBlock(
        irreps_node_input=o3.Irreps(HIDDEN_IRREPS),
        irreps_node_attr=o3.Irreps("1x0e"),
        irreps_edge_attr=o3.Irreps(EDGE_IRREPS),
        irreps_node_output=o3.Irreps(HIDDEN_IRREPS),
        fc_neurons=[6, 8],
        irreps_head=o3.Irreps("2x0e+1x1e+1x2e"),
        num_heads=2,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=float(alpha_drop),
        proj_drop=float(proj_drop),
        drop_path_rate=float(drop_path),
        irreps_mlp_mid=o3.Irreps(HIDDEN_IRREPS),
        norm_layer="layer",
    ).double()


def _scales(official):
    result = torch.ones(30, dtype=torch.float64)
    for output_slice, scale in official.ga.sep.dtp.slices_sqrt_k.values():
        result[output_slice] *= float(scale)
    return result.tolist()


def _dsl_inputs(
    node_input,
    node_attr,
    edge_src,
    edge_dst,
    edge_attr,
    edge_scalars,
    batch,
    *,
    graph_count=2,
):
    return {
        "node_input": node_input,
        "node_attr": node_attr,
        "edge_attr": edge_attr,
        "edge_scalars": edge_scalars,
        "source_index": {"indices": edge_src, "target_size": edge_src.numel()},
        "target_index": {"indices": edge_dst, "target_size": edge_dst.numel()},
        "segment_index": {"indices": edge_dst, "target_size": node_input.shape[0]},
        "batch_index": {"indices": batch, "target_size": int(graph_count)},
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
    attention_drop_path = official.drop_path(attention, batch)
    attention_residual = node_input + attention_drop_path
    norm_2 = official.norm_2(attention_residual, batch=batch)
    ffn_pre_dropout = official.ffn.fctp_2(
        official.ffn.fctp_1(norm_2, node_attr),
        node_attr,
    )
    ffn = official.ffn.proj_drop(ffn_pre_dropout)
    ffn_drop_path = official.drop_path(ffn, batch)
    return {
        "norm_1": norm_1,
        "attention": attention,
        "attention_drop_path": attention_drop_path,
        "attention_residual": attention_residual,
        "norm_2": norm_2,
        "ffn_pre_dropout": ffn_pre_dropout,
        "ffn": ffn,
        "ffn_drop_path": ffn_drop_path,
        "out": attention_residual + ffn_drop_path,
    }


def test_v1_stochastic_transblock_matches_official_rng_forward_gradients_and_o3():
    root = Path(__file__).resolve().parents[2] / "equiformer"
    official_module, _source = _load_official_module(root, torch)
    torch.manual_seed(20260731)
    official = _official(official_module).train()
    program = equiformer_v1_transblock_program(
        _scales(official),
        graph_count=2,
        alpha_drop=ALPHA_DROP,
        proj_drop=PROJ_DROP,
        drop_path=DROP_PATH,
    )
    registry = core_registry()
    artifact = Compiler(registry, reference_motif_registry()).analyze(program)
    torch.manual_seed(20260731)
    model = E3NNGraphBackend(registry).build(artifact.expanded_program, artifact.inference).double().train()

    mapping = dict(EQUIFORMER_V1_TRANSBLOCK_PARAMETER_MAPPING)
    official_parameters = dict(official.named_parameters())
    dsl_parameters = dict(model.named_parameters())
    assert set(dsl_parameters) == set(mapping)
    for dsl_name, official_name in mapping.items():
        assert torch.equal(dsl_parameters[dsl_name], official_parameters[official_name])

    assert [node.op for node in artifact.expanded_program.nodes].count("core.graph_stochastic_depth@1") == 2
    assert [node.op for node in artifact.expanded_program.nodes].count("core.equivariant_dropout@1") == 2
    assert [node.op for node in artifact.expanded_program.nodes].count("core.scalar_dropout@1") == 1

    node_count = 5
    edge_src = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    edge_dst = torch.tensor([1, 1, 3, 3, 0, 4, 4, 2], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    torch.manual_seed(20260732)
    node_input = torch.randn(node_count, 15, dtype=torch.float64, requires_grad=True)
    node_attr = torch.randn(node_count, 1, dtype=torch.float64, requires_grad=True)
    edge_attr = torch.randn(edge_src.numel(), 9, dtype=torch.float64, requires_grad=True)
    edge_scalars = torch.randn(edge_src.numel(), 6, dtype=torch.float64, requires_grad=True)
    official_node = node_input.detach().clone().requires_grad_(True)
    official_node_attr = node_attr.detach().clone().requires_grad_(True)
    official_edge = edge_attr.detach().clone().requires_grad_(True)
    official_radial = edge_scalars.detach().clone().requires_grad_(True)

    torch.manual_seed(20260733)
    actual = model(
        _dsl_inputs(node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, batch),
        {},
    )
    actual_rng_state = torch.get_rng_state()
    torch.manual_seed(20260733)
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
    expected_rng_state = torch.get_rng_state()
    assert torch.equal(actual_rng_state, expected_rng_state)
    assert set(actual) == set(expected)
    for name in expected:
        assert torch.allclose(actual[name], expected[name], atol=1.0e-12, rtol=1.0e-12), (
            name,
            float((actual[name] - expected[name]).detach().abs().max()),
        )
    for name in ("attention", "attention_drop_path", "ffn", "ffn_drop_path"):
        assert torch.equal(actual[name] == 0, expected[name] == 0), name

    torch.manual_seed(20260733)
    direct_official = official(
        node_input.detach(),
        node_attr.detach(),
        edge_src,
        edge_dst,
        edge_attr.detach(),
        edge_scalars.detach(),
        batch,
    )
    assert torch.allclose(direct_official, expected["out"].detach(), atol=1.0e-12, rtol=1.0e-12)

    torch.manual_seed(20260734)
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
        )

    with torch.no_grad():
        rotation = o3.rand_matrix(dtype=torch.float64)
        node_action = o3.Irreps(HIDDEN_IRREPS).D_from_matrix(rotation)
        edge_action = o3.Irreps(EDGE_IRREPS).D_from_matrix(rotation)
        torch.manual_seed(20260735)
        reference = model(
            _dsl_inputs(
                node_input.detach(),
                node_attr.detach(),
                edge_src,
                edge_dst,
                edge_attr.detach(),
                edge_scalars.detach(),
                batch,
            ),
            {},
        )["out"]
        torch.manual_seed(20260735)
        transformed = model(
            _dsl_inputs(
                node_input.detach() @ node_action.transpose(0, 1),
                node_attr.detach(),
                edge_src,
                edge_dst,
                edge_attr.detach() @ edge_action.transpose(0, 1),
                edge_scalars.detach(),
                batch,
            ),
            {},
        )["out"]
        expected_transformed = reference @ node_action.transpose(0, 1)
        relative_error = (
            (transformed - expected_transformed).norm()
            / expected_transformed.norm().clamp_min(1.0e-12)
        )
        assert float(relative_error) < 1.0e-7

    official.eval()
    model.eval()
    actual_eval = model(
        _dsl_inputs(
            node_input.detach(),
            node_attr.detach(),
            edge_src,
            edge_dst,
            edge_attr.detach(),
            edge_scalars.detach(),
            batch,
        ),
        {},
    )["out"]
    expected_eval = official(
        node_input.detach(),
        node_attr.detach(),
        edge_src,
        edge_dst,
        edge_attr.detach(),
        edge_scalars.detach(),
        batch,
    )
    assert torch.allclose(actual_eval, expected_eval, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.parametrize(
    ("graph_count", "batch", "alpha_drop", "proj_drop", "drop_path"),
    (
        (1, torch.tensor([0, 0, 0, 0, 0], dtype=torch.long), 1.0e-6, 1.0e-6, 1.0e-6),
        (3, torch.tensor([0, 1, 1, 2, 2], dtype=torch.long), 0.95, 0.9, 0.95),
    ),
)
def test_v1_stochastic_transblock_boundary_matrix_preserves_official_rng_and_forward(
    graph_count,
    batch,
    alpha_drop,
    proj_drop,
    drop_path,
):
    root = Path(__file__).resolve().parents[2] / "equiformer"
    official_module, _source = _load_official_module(root, torch)
    seed = 20260780 + graph_count
    torch.manual_seed(seed)
    official = _official(
        official_module,
        alpha_drop=alpha_drop,
        proj_drop=proj_drop,
        drop_path=drop_path,
    ).train()
    program = equiformer_v1_transblock_program(
        _scales(official),
        graph_count=graph_count,
        alpha_drop=alpha_drop,
        proj_drop=proj_drop,
        drop_path=drop_path,
    )
    registry = core_registry()
    artifact = Compiler(registry, reference_motif_registry()).analyze(program)
    torch.manual_seed(seed)
    model = E3NNGraphBackend(registry).build(
        artifact.expanded_program,
        artifact.inference,
    ).double().train()

    mapping = dict(EQUIFORMER_V1_TRANSBLOCK_PARAMETER_MAPPING)
    official_parameters = dict(official.named_parameters())
    dsl_parameters = dict(model.named_parameters())
    assert set(dsl_parameters) == set(mapping)
    for dsl_name, official_name in mapping.items():
        assert torch.equal(dsl_parameters[dsl_name], official_parameters[official_name])

    edge_src = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    edge_dst = torch.tensor([1, 1, 3, 3, 0, 4, 4, 2], dtype=torch.long)
    torch.manual_seed(seed + 1)
    node_input = torch.randn(5, 15, dtype=torch.float64)
    node_attr = torch.randn(5, 1, dtype=torch.float64)
    edge_attr = torch.randn(edge_src.numel(), 9, dtype=torch.float64)
    edge_scalars = torch.randn(edge_src.numel(), 6, dtype=torch.float64)

    torch.manual_seed(seed + 2)
    actual = model(
        _dsl_inputs(
            node_input,
            node_attr,
            edge_src,
            edge_dst,
            edge_attr,
            edge_scalars,
            batch,
            graph_count=graph_count,
        ),
        {},
    )
    actual_rng_state = torch.get_rng_state()
    torch.manual_seed(seed + 2)
    expected = _official_intermediates(
        official,
        node_input,
        node_attr,
        edge_src,
        edge_dst,
        edge_attr,
        edge_scalars,
        batch,
    )
    expected_rng_state = torch.get_rng_state()
    assert torch.equal(actual_rng_state, expected_rng_state)
    assert set(actual) == set(expected)
    for name, expected_value in expected.items():
        assert torch.allclose(actual[name], expected_value, atol=1.0e-12, rtol=1.0e-12), (
            name,
            float((actual[name] - expected_value).abs().max()),
        )

    model.eval()
    official.eval()
    actual_eval = model(
        _dsl_inputs(
            node_input,
            node_attr,
            edge_src,
            edge_dst,
            edge_attr,
            edge_scalars,
            batch,
            graph_count=graph_count,
        ),
        {},
    )["out"]
    expected_eval = official(
        node_input,
        node_attr,
        edge_src,
        edge_dst,
        edge_attr,
        edge_scalars,
        batch,
    )
    assert torch.allclose(actual_eval, expected_eval, atol=1.0e-12, rtol=1.0e-12)


def test_v1_transblock_rejects_nonpositive_graph_count_when_drop_path_is_enabled():
    with pytest.raises(ValueError, match="graph_count must be positive"):
        equiformer_v1_transblock_program(
            [1.0] * 30,
            graph_count=0,
            drop_path=0.25,
        )
