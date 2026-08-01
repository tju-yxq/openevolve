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
    RepresentationLayout,
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend


INPUT_IRREPS = "4x0e+2x1e+1x2e"
ATTR_IRREPS = "1x0e"


def _type(irreps, *, carrier=Carrier.NODE, layout=None):
    group = GroupSpec.o3()
    return EquivariantTensorType(
        group,
        carrier,
        Irreps.parse(irreps, group.family),
        dtype="float64",
        layout=layout or RepresentationLayout(),
    )


def _program(out_irreps, *, bias=True, rescale=True, left_layout=None):
    left = _type(INPUT_IRREPS, layout=left_layout)
    right = _type(ATTR_IRREPS)
    output = _type(out_irreps)
    return ArchitectureProgram(
        "2.10.0",
        "official-v1-fully-connected-tp",
        (InputPort("left", left), InputPort("right", right)),
        (
            Node(
                "fctp",
                "core.tensor_product@4",
                {"left": ("input:left",), "right": ("input:right",)},
                {"out_irreps": out_irreps, "bias": bias, "rescale": rescale},
            ),
        ),
        (OutputPort("out", "fctp", output),),
        program_id="official-v1-fully-connected-tp@1",
    )


def _official(out_irreps, *, bias=True, rescale=True):
    root = Path(__file__).resolve().parents[2] / "equiformer"
    module, _source = _load_official_module(root, torch)
    return module.FullyConnectedTensorProductRescale(
        o3.Irreps(INPUT_IRREPS),
        o3.Irreps(ATTR_IRREPS),
        o3.Irreps(out_irreps),
        bias=bias,
        rescale=rescale,
        internal_weights=True,
        shared_weights=True,
        normalization=None,
    ).double()


@pytest.mark.parametrize(
    ("out_irreps", "weight_numel", "bias_size"),
    (
        ("7x0e+2x1e+1x2e", 33, 7),
        ("4x0e+2x1e+1x2e", 21, 4),
    ),
)
def test_internal_shared_uvw_fctp_matches_official_initialization_forward_gradients_and_o3(
    out_irreps,
    weight_numel,
    bias_size,
):
    root = Path(__file__).resolve().parents[2] / "equiformer"
    _load_official_module(root, torch)
    torch.manual_seed(20260731)
    official = _official(out_irreps)
    torch.manual_seed(20260731)
    registry = core_registry()
    program = _program(out_irreps)
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference).double()
    dsl_parameters = dict(model.named_parameters())
    official_parameters = dict(official.named_parameters())
    mapping = {
        "node_modules.fctp.tp.weight": "tp.weight",
        "node_modules.fctp.bias.0": "bias.0",
    }
    assert tuple(dsl_parameters["node_modules.fctp.tp.weight"].shape) == (weight_numel,)
    assert tuple(dsl_parameters["node_modules.fctp.bias.0"].shape) == (bias_size,)
    assert set(dsl_parameters) == set(mapping)
    for dsl_name, official_name in mapping.items():
        assert torch.equal(dsl_parameters[dsl_name], official_parameters[official_name])

    left = torch.randn(8, o3.Irreps(INPUT_IRREPS).dim, dtype=torch.float64, requires_grad=True)
    right = torch.randn(8, o3.Irreps(ATTR_IRREPS).dim, dtype=torch.float64, requires_grad=True)
    official_left = left.detach().clone().requires_grad_(True)
    official_right = right.detach().clone().requires_grad_(True)
    actual = model({"left": left, "right": right}, {})["out"]
    expected = official(official_left, official_right)
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)

    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert torch.allclose(left.grad, official_left.grad, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(right.grad, official_right.grad, atol=1.0e-12, rtol=1.0e-12)
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
            left_action = o3.Irreps(INPUT_IRREPS).D_from_matrix(matrix)
            right_action = o3.Irreps(ATTR_IRREPS).D_from_matrix(matrix)
            output_action = o3.Irreps(out_irreps).D_from_matrix(matrix)
            transformed = model(
                {
                    "left": left.detach() @ left_action.transpose(0, 1),
                    "right": right.detach() @ right_action.transpose(0, 1),
                },
                {},
            )["out"]
            expected_transformed = reference @ output_action.transpose(0, 1)
            relative_error = (
                (transformed - expected_transformed).norm()
                / expected_transformed.norm().clamp_min(1.0e-12)
            )
            assert float(relative_error) < 1.0e-7


@pytest.mark.parametrize(
    ("program", "expected_code"),
    (
        (_program("1x3e"), "E_TP_V4_005"),
        (_program(INPUT_IRREPS, bias="yes"), "E_TP_V4_006"),
        (
            _program(
                INPUT_IRREPS,
                left_layout=RepresentationLayout(storage="m_primary", coefficient_order="canonical"),
            ),
            "E_TP_V4_002",
        ),
    ),
)
def test_internal_shared_uvw_fctp_rejects_invalid_contracts(program, expected_code):
    with pytest.raises(DSLValidationError) as error:
        Compiler(core_registry()).analyze(program)
    assert any(item.code == expected_code for item in error.value.diagnostics)
