from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")
from e3nn import o3

from equivariant_nas.dsl import (
    ArchitectureProgram,
    AxisSpec,
    Carrier,
    Compiler,
    DSLValidationError,
    EquivariantTensorType,
    FeatureRole,
    GroupSpec,
    InputPort,
    InvariantTensorType,
    Irreps,
    Node,
    OutputPort,
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend


def _types(*, weight_measure="dimensionless", weight_role=FeatureRole.RADIAL_WEIGHT, path_role=FeatureRole.TP_PATH):
    group = GroupSpec.o3()
    left = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("1x1o", group.family),
        dtype="float64",
        measure="angstrom",
    )
    right = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("1x1o", group.family),
        dtype="float64",
        measure="dimensionless",
    )
    output = replace(
        left,
        irreps=Irreps.parse("1x0e+1x1e+1x2e", group.family),
        measure="angstrom",
    )
    weight = InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("3x0e", group.family),
        axes=("tp_path",),
        axis_specs=(AxisSpec("tp_path", 3, path_role, "independent", 0),),
        dtype="float64",
        measure=weight_measure,
        feature_role=weight_role,
    )
    return left, right, weight, output


def _program(*, weight=None, left=None, right=None, output=None):
    default_left, default_right, default_weight, default_output = _types()
    left = left or default_left
    right = right or default_right
    weight = weight or default_weight
    output = output or default_output
    return ArchitectureProgram(
        "2.4.0",
        "external-tensor-product-test",
        (InputPort("left", left), InputPort("right", right), InputPort("weight", weight)),
        (
            Node(
                "tp",
                "core.tensor_product@2",
                {"left": ("input:left",), "right": ("input:right",), "weight": ("input:weight",)},
                {"out_irreps": str(output.irreps)},
            ),
        ),
        (OutputPort("out", "tp", output),),
    )


def test_tensor_product_v2_infers_external_weight_contract_and_units():
    registry = core_registry()
    artifact = Compiler(registry).analyze(_program())
    contract = artifact.inference.parameter_contracts["tp"][0]
    output = artifact.inference.value_types["tp"]
    obligations = {item.kind.value: item for item in artifact.inference.obligations}

    assert contract.name == "weight"
    assert contract.storage == "external"
    assert contract.external_port == "weight"
    assert contract.shape == (3,)
    assert contract.trainable is False
    assert contract.backend_parameter_name == ""
    assert output.measure == "angstrom"
    assert obligations["IRREP_PATH_EXISTS"].details["weight_numel"] == 3
    assert obligations["IRREP_PATH_EXISTS"].details["connection_mode"] == "uvw"


def test_tensor_product_v2_lowering_is_rotation_equivariant_permutation_safe_and_differentiable():
    registry = core_registry()
    program = _program()
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference).double()
    module = model.node_modules["tp"]

    assert int(module.weight_numel) == 3
    assert dict(module.named_parameters()) == {}
    assert model.parameter_contract_manifest["tp"][0]["storage"] == "external"

    left = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    right = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    reference = model({"left": left, "right": right, "weight": weight}, {})["out"]
    direct = module(left, right, weight)
    assert torch.allclose(reference, direct, atol=1e-12, rtol=1e-12)

    rotation = o3.rand_matrix(dtype=torch.float64)
    input_action = o3.Irreps("1x1o").D_from_matrix(rotation)
    output_action = o3.Irreps("1x0e+1x1e+1x2e").D_from_matrix(rotation)
    rotated = model(
        {
            "left": left @ input_action.transpose(0, 1),
            "right": right @ input_action.transpose(0, 1),
            "weight": weight,
        },
        {},
    )["out"]
    expected = reference @ output_action.transpose(0, 1)
    relative_error = (rotated - expected).norm() / expected.norm().clamp_min(1e-12)
    assert float(relative_error.detach()) < 1e-7

    permutation = torch.tensor([7, 2, 0, 5, 3, 1, 6, 4], dtype=torch.long)
    permuted = model(
        {
            "left": left.index_select(0, permutation),
            "right": right.index_select(0, permutation),
            "weight": weight.index_select(0, permutation),
        },
        {},
    )["out"]
    assert torch.allclose(permuted, reference.index_select(0, permutation), atol=1e-12, rtol=1e-12)

    reference.square().sum().backward()
    assert left.grad is not None and torch.isfinite(left.grad).all()
    assert right.grad is not None and torch.isfinite(right.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


@pytest.mark.parametrize(
    "weight,code",
    (
        (
            replace(
                _types()[2],
                irreps=Irreps.parse("2x0e", "O3"),
                axis_specs=(AxisSpec("tp_path", 2, FeatureRole.TP_PATH, "independent", 0),),
            ),
            "E_TP_V2_010",
        ),
        (_types(weight_measure="angstrom")[2], "E_TP_V2_007"),
        (_types(weight_role=FeatureRole.CHANNEL)[2], "E_TP_V2_008"),
        (_types(path_role=FeatureRole.CHANNEL)[2], "E_TP_V2_009"),
    ),
)
def test_tensor_product_v2_rejects_invalid_external_weight_contracts(weight, code):
    with pytest.raises(DSLValidationError) as error:
        Compiler(core_registry()).analyze(_program(weight=weight))
    assert any(item.code == code for item in error.value.diagnostics)


def test_tensor_product_v2_rejects_axisful_inputs_and_illegal_output_paths():
    left, right, weight, output = _types()
    axisful_left = replace(
        left,
        axes=("channel",),
        axis_specs=(AxisSpec("channel", 1, FeatureRole.CHANNEL, "independent", 0),),
    )
    with pytest.raises(DSLValidationError) as axes:
        Compiler(core_registry()).analyze(_program(left=axisful_left))
    assert any(item.code == "E_TP_V2_002" for item in axes.value.diagnostics)

    illegal_output = replace(output, irreps=Irreps.parse("1x3e", output.group.family))
    with pytest.raises(DSLValidationError) as path:
        Compiler(core_registry()).analyze(_program(output=illegal_output))
    assert any(item.code == "E_TP_V2_004" for item in path.value.diagnostics)
