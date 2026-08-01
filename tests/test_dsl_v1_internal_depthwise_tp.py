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
    DSLValidationError,
    EquivariantTensorType,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend


HIDDEN_IRREPS = "4x0e+2x1e+1x2e"
EDGE_IRREPS = "1x0e+1x1e+1x2e"


def _attrs():
    blocks = (
        (4, "0e"), (2, "0e"), (1, "0e"),
        (4, "1e"), (2, "1e"), (2, "1e"), (2, "1e"), (1, "1e"), (1, "1e"),
        (4, "2e"), (2, "2e"), (2, "2e"), (1, "2e"), (1, "2e"), (1, "2e"),
    )
    indices = (
        (0, 0, 0), (0, 1, 3), (0, 2, 9),
        (1, 0, 4), (1, 1, 1), (1, 1, 5), (1, 1, 10), (1, 2, 6), (1, 2, 11),
        (2, 0, 12), (2, 1, 7), (2, 1, 13), (2, 2, 2), (2, 2, 8), (2, 2, 14),
    )
    return {
        "path_blocks": [
            {"multiplicity": multiplicity, "irrep": irrep}
            for multiplicity, irrep in blocks
        ],
        "instructions": [
            {
                "left": left,
                "right": right,
                "out": output,
                "mode": "uvu",
                "has_weight": True,
                "path_weight": 1.0,
            }
            for left, right, output in indices
        ],
        "bias": False,
        "rescale": True,
    }


def _program(attrs=None):
    group = GroupSpec.o3()
    left = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse(HIDDEN_IRREPS, group.family),
        dtype="float64",
    )
    right = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse(EDGE_IRREPS, group.family),
        dtype="float64",
    )
    out = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("7x0e+12x1e+11x2e", group.family),
        dtype="float64",
    )
    return ArchitectureProgram(
        "2.13.0",
        "v1-internal-shared-depthwise-tp",
        (InputPort("left", left), InputPort("right", right)),
        (Node("tp", "core.tensor_product@5", {"left": ("input:left",), "right": ("input:right",)}, attrs or _attrs()),),
        (OutputPort("out", "tp", out),),
    )


def test_internal_shared_uvu_tp_matches_official_initialization_forward_gradients_and_o3():
    root = Path(__file__).resolve().parents[2] / "equiformer"
    official_module, _source = _load_official_module(root, torch)
    torch.manual_seed(20260751)
    official = official_module.DepthwiseTensorProduct(
        o3.Irreps(HIDDEN_IRREPS),
        o3.Irreps(EDGE_IRREPS),
        o3.Irreps("4x0e+2x1e+2x2e"),
        internal_weights=True,
        bias=False,
    ).double().eval()
    registry = core_registry()
    artifact = Compiler(registry).analyze(_program())
    torch.manual_seed(20260751)
    model = E3NNGraphBackend(registry).build(artifact.expanded_program, artifact.inference).double().eval()
    dsl_module = model.node_modules["tp"]

    assert dsl_module.tp.internal_weights
    assert dsl_module.tp.shared_weights
    assert dsl_module.tp.weight_numel == official.tp.weight_numel == 30
    assert torch.equal(dsl_module.tp.weight, official.tp.weight)
    assert [
        (item.i_in1, item.i_in2, item.i_out, item.connection_mode, item.has_weight)
        for item in dsl_module.tp.instructions
    ] == [
        (item.i_in1, item.i_in2, item.i_out, item.connection_mode, item.has_weight)
        for item in official.tp.instructions
    ]

    torch.manual_seed(20260752)
    left = torch.randn(8, 15, dtype=torch.float64, requires_grad=True)
    right = torch.randn(8, 9, dtype=torch.float64, requires_grad=True)
    official_left = left.detach().clone().requires_grad_(True)
    official_right = right.detach().clone().requires_grad_(True)
    actual = model({"left": left, "right": right}, {})["out"]
    expected = official(official_left, official_right)
    assert torch.equal(actual, expected)

    torch.manual_seed(20260753)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert torch.allclose(left.grad, official_left.grad, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(right.grad, official_right.grad, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(dsl_module.tp.weight.grad, official.tp.weight.grad, atol=1.0e-12, rtol=1.0e-12)

    with torch.no_grad():
        rotation = o3.rand_matrix(dtype=torch.float64)
        left_action = o3.Irreps(HIDDEN_IRREPS).D_from_matrix(rotation)
        right_action = o3.Irreps(EDGE_IRREPS).D_from_matrix(rotation)
        out_action = official.irreps_out.D_from_matrix(rotation)
        transformed = model(
            {
                "left": left.detach() @ left_action.transpose(0, 1),
                "right": right.detach() @ right_action.transpose(0, 1),
            },
            {},
        )["out"]
        expected_transformed = actual.detach() @ out_action.transpose(0, 1)
        relative_error = (
            (transformed - expected_transformed).norm()
            / expected_transformed.norm().clamp_min(1.0e-12)
        )
        assert float(relative_error) < 1.0e-7


def test_internal_shared_uvu_tp_rejects_bias_in_first_version():
    attrs = _attrs()
    attrs["bias"] = True
    with pytest.raises(DSLValidationError) as captured:
        Compiler(core_registry()).analyze(_program(attrs))
    assert any(item.code == "E_TP_V5_004" for item in captured.value.diagnostics)
