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


IRREPS = "4x0e+2x1e+1x2e"


def _type(*, layout=None):
    group = GroupSpec.o3()
    return EquivariantTensorType(
        group,
        Carrier.NODE,
        Irreps.parse(IRREPS, group.family),
        dtype="float64",
        layout=layout or RepresentationLayout(),
    )


def _program(*, epsilon=1.0e-5, affine=True, normalization="component", layout=None):
    value_type = _type(layout=layout)
    return ArchitectureProgram(
        "2.10.0",
        "official-v1-irrep-layer-norm",
        (InputPort("x", value_type),),
        (
            Node(
                "norm",
                "core.irrep_layer_norm@1",
                {"x": ("input:x",)},
                {"epsilon": epsilon, "affine": affine, "normalization": normalization},
            ),
        ),
        (OutputPort("out", "norm", value_type),),
        program_id="official-v1-irrep-layer-norm@1",
    )


def _official(*, epsilon=1.0e-5, affine=True, normalization="component"):
    root = Path(__file__).resolve().parents[2] / "equiformer"
    module, _source = _load_official_module(root, torch)
    return module.EquivariantLayerNormV2(
        o3.Irreps(IRREPS),
        eps=epsilon,
        affine=affine,
        normalization=normalization,
    ).double()


@pytest.mark.parametrize("normalization", ("component", "norm"))
def test_irrep_layer_norm_matches_official_initialization_forward_gradients_and_o3(normalization):
    torch.manual_seed(20260731)
    registry = core_registry()
    program = _program(normalization=normalization)
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference).double()
    official = _official(normalization=normalization)
    dsl_parameters = dict(model.named_parameters())
    official_parameters = dict(official.named_parameters())
    mapping = {
        "node_modules.norm.affine_weight": "affine_weight",
        "node_modules.norm.affine_bias": "affine_bias",
    }
    assert set(dsl_parameters) == set(mapping)
    for dsl_name, official_name in mapping.items():
        assert tuple(dsl_parameters[dsl_name].shape) == tuple(official_parameters[official_name].shape)
        assert torch.equal(dsl_parameters[dsl_name], official_parameters[official_name])

    values = torch.randn(7, o3.Irreps(IRREPS).dim, dtype=torch.float64, requires_grad=True)
    official_values = values.detach().clone().requires_grad_(True)
    actual = model({"x": values}, {})["out"]
    expected = official(official_values)
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)

    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert torch.allclose(values.grad, official_values.grad, atol=1.0e-12, rtol=1.0e-12)
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
            action = o3.Irreps(IRREPS).D_from_matrix(matrix)
            transformed = model({"x": values.detach() @ action.transpose(0, 1)}, {})["out"]
            expected_transformed = reference @ action.transpose(0, 1)
            relative_error = (
                (transformed - expected_transformed).norm()
                / expected_transformed.norm().clamp_min(1.0e-12)
            )
            assert float(relative_error) < 1.0e-7


@pytest.mark.parametrize(
    ("kwargs", "expected_code"),
    (
        ({"epsilon": 0.0}, "E_IRREP_LN_003"),
        ({"affine": "yes"}, "E_IRREP_LN_004"),
        ({"affine": False}, "E_IRREP_LN_006"),
        ({"normalization": "batch"}, "E_IRREP_LN_005"),
        (
            {"layout": RepresentationLayout(storage="m_primary", coefficient_order="canonical")},
            "E_IRREP_LN_002",
        ),
    ),
)
def test_irrep_layer_norm_rejects_invalid_contracts(kwargs, expected_code):
    with pytest.raises(DSLValidationError) as error:
        Compiler(core_registry()).analyze(_program(**kwargs))
    assert any(item.code == expected_code for item in error.value.diagnostics)
