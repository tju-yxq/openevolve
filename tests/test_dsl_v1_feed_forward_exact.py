from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
pytest.importorskip("e3nn")
from e3nn import o3

from scripts.audit_equiformer_v1_graph_attention_contract import _load_official_module
from equivariant_nas.dsl import (
    ArchitectureProgram,
    Carrier,
    Compiler,
    EquivariantTensorType,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    core_registry,
    reference_motif_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend


HIDDEN_IRREPS = "4x0e+2x1e+1x2e"
PRE_GATE_IRREPS = "7x0e+2x1e+1x2e"


def _type(irreps):
    group = GroupSpec.o3()
    return EquivariantTensorType(
        group,
        Carrier.NODE,
        Irreps.parse(irreps, group.family),
        dtype="float64",
    )


def _program():
    hidden = _type(HIDDEN_IRREPS)
    node_attr = _type("1x0e")
    return ArchitectureProgram(
        "2.10.0",
        "official-v1-feed-forward",
        (InputPort("x", hidden), InputPort("node_attr", node_attr)),
        (
            Node(
                "ffn",
                "motif.v1_feed_forward@1",
                {"x": ("input:x",), "node_attr": ("input:node_attr",)},
                {
                    "pre_gate_irreps": PRE_GATE_IRREPS,
                    "scalar_multiplicity": 4,
                    "gate_multiplicity": 3,
                    "gated_selections": [
                        {"irrep": "1e", "start": 0, "multiplicity": 2},
                        {"irrep": "2e", "start": 0, "multiplicity": 1},
                    ],
                    "out_irreps": HIDDEN_IRREPS,
                },
            ),
        ),
        (OutputPort("out", "ffn", hidden),),
        program_id="official-v1-feed-forward@1",
    )


def _official():
    root = Path(__file__).resolve().parents[2] / "equiformer"
    module, _source = _load_official_module(root, torch)
    return module.FeedForwardNetwork(
        irreps_node_input=o3.Irreps(HIDDEN_IRREPS),
        irreps_node_attr=o3.Irreps("1x0e"),
        irreps_node_output=o3.Irreps(HIDDEN_IRREPS),
        irreps_mlp_mid=o3.Irreps(HIDDEN_IRREPS),
        proj_drop=0.0,
    ).double().eval()


def test_v1_feed_forward_motif_matches_official_parameters_forward_gradients_and_o3():
    root = Path(__file__).resolve().parents[2] / "equiformer"
    _load_official_module(root, torch)
    torch.manual_seed(20260731)
    official = _official()
    torch.manual_seed(20260731)
    registry = core_registry()
    motifs = reference_motif_registry()
    artifact = Compiler(registry, motifs).analyze(_program())
    model = E3NNGraphBackend(registry).build(artifact.expanded_program, artifact.inference).double().eval()
    mapping = {
        "node_modules.ffn__fctp_1.tp.weight": "fctp_1.tp.weight",
        "node_modules.ffn__fctp_1.bias.0": "fctp_1.bias.0",
        "node_modules.ffn__fctp_2.tp.weight": "fctp_2.tp.weight",
        "node_modules.ffn__fctp_2.bias.0": "fctp_2.bias.0",
    }
    dsl_parameters = dict(model.named_parameters())
    official_parameters = dict(official.named_parameters())
    assert set(dsl_parameters) == set(mapping)
    for dsl_name, official_name in mapping.items():
        assert tuple(dsl_parameters[dsl_name].shape) == tuple(official_parameters[official_name].shape)
        assert torch.equal(dsl_parameters[dsl_name], official_parameters[official_name])

    node_input = torch.randn(7, o3.Irreps(HIDDEN_IRREPS).dim, dtype=torch.float64, requires_grad=True)
    node_attr = torch.randn(7, 1, dtype=torch.float64, requires_grad=True)
    official_input = node_input.detach().clone().requires_grad_(True)
    official_attr = node_attr.detach().clone().requires_grad_(True)
    actual = model({"x": node_input, "node_attr": node_attr}, {})["out"]
    expected = official(official_input, official_attr)
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)

    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert torch.allclose(node_input.grad, official_input.grad, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(node_attr.grad, official_attr.grad, atol=1.0e-12, rtol=1.0e-12)
    for dsl_name, official_name in mapping.items():
        assert torch.allclose(
            dsl_parameters[dsl_name].grad,
            official_parameters[official_name].grad,
            atol=1.0e-12,
            rtol=1.0e-12,
        )

    with torch.no_grad():
        reference = actual.detach()
        for matrix in (o3.rand_matrix(dtype=torch.float64), -torch.eye(3, dtype=torch.float64)):
            action = o3.Irreps(HIDDEN_IRREPS).D_from_matrix(matrix)
            transformed = model(
                {
                    "x": node_input.detach() @ action.transpose(0, 1),
                    "node_attr": node_attr.detach(),
                },
                {},
            )["out"]
            expected_transformed = reference @ action.transpose(0, 1)
            relative_error = (
                (transformed - expected_transformed).norm()
                / expected_transformed.norm().clamp_min(1.0e-12)
            )
            assert float(relative_error) < 1.0e-7

    assert [node.op for node in artifact.expanded_program.nodes] == [
        "core.tensor_product@4",
        "core.irrep_select@2",
        "core.irrep_select@2",
        "core.irrep_select@2",
        "core.scalar_activation@1",
        "core.scalar_activation@1",
        "core.gate@1",
        "core.irrep_concat@1",
        "core.tensor_product@4",
    ]
