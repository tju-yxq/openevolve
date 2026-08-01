import ast
from dataclasses import replace
import math
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
pytest.importorskip("e3nn")

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
    reference_motif_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend


def _official_radial_profile_class():
    source_path = Path(__file__).resolve().parents[2] / "equiformer" / "nets" / "radial_func.py"
    if not source_path.exists():
        pytest.skip("local official Equiformer V1 radial_func.py is unavailable")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RadialProfile"
    ]
    if len(selected) != 1:
        raise RuntimeError("official radial_func.py no longer contains exactly one RadialProfile class")
    namespace = {"torch": torch, "nn": torch.nn, "init": torch.nn.init, "math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["RadialProfile"]


def _types():
    group = GroupSpec.o3()
    radial = InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("6x0e", group.family),
        axes=("radial_channel",),
        axis_specs=(AxisSpec("radial_channel", 6, FeatureRole.CHANNEL, "independent", 0),),
        dtype="float64",
        measure="dimensionless",
        feature_role=FeatureRole.CHANNEL,
    )
    weight = replace(
        radial,
        irreps=Irreps.parse("30x0e", group.family),
        axes=("tp_path",),
        axis_specs=(AxisSpec("tp_path", 30, FeatureRole.TP_PATH, "independent", 0),),
        feature_role=FeatureRole.RADIAL_WEIGHT,
    )
    left = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("4x0e+2x1e+1x2e", group.family),
        dtype="float64",
    )
    right = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("1x0e+1x1e+1x2e", group.family),
        dtype="float64",
    )
    tp_output = replace(left, irreps=Irreps.parse("7x0e+12x1e+11x2e", group.family))
    return radial, weight, left, right, tp_output


def _radial_program(scales=None, *, input_type=None):
    radial, weight, _left, _right, _tp_output = _types()
    radial = input_type or radial
    return ArchitectureProgram(
        "2.7.0",
        "official-v1-radial-profile",
        (InputPort("x", radial),),
        (
            Node(
                "radial",
                "motif.v1_radial_profile@1",
                {"x": ("input:x",)},
                {
                    "axis": "radial_channel",
                    "out_axis": "tp_path",
                    "hidden_features": 8,
                    "out_features": 30,
                    "output_scales": list(scales or [1.0] * 30),
                },
            ),
        ),
        (OutputPort("out", "radial", weight),),
    )


def _tp_attrs():
    path_blocks = [
        {"multiplicity": multiplicity, "irrep": irrep}
        for multiplicity, irrep in (
            (4, "0e"), (2, "0e"), (1, "0e"),
            (4, "1e"), (2, "1e"), (2, "1e"), (2, "1e"), (1, "1e"), (1, "1e"),
            (4, "2e"), (2, "2e"), (2, "2e"), (1, "2e"), (1, "2e"), (1, "2e"),
        )
    ]
    indices = (
        (0, 0, 0), (0, 1, 3), (0, 2, 9),
        (1, 0, 4), (1, 1, 1), (1, 1, 5), (1, 1, 10), (1, 2, 6), (1, 2, 11),
        (2, 0, 12), (2, 1, 7), (2, 1, 13), (2, 2, 2), (2, 2, 8), (2, 2, 14),
    )
    return {
        "path_blocks": path_blocks,
        "instructions": [
            {"left": left, "right": right, "out": output, "mode": "uvu", "has_weight": True, "path_weight": 1.0}
            for left, right, output in indices
        ],
    }


def _radial_tp_program():
    radial, _weight, left, right, tp_output = _types()
    return ArchitectureProgram(
        "2.7.0",
        "official-v1-radial-profile-to-depthwise-tp",
        (InputPort("radial", radial), InputPort("left", left), InputPort("right", right)),
        (
            Node(
                "profile",
                "motif.v1_radial_profile@1",
                {"x": ("input:radial",)},
                {
                    "axis": "radial_channel",
                    "out_axis": "tp_path",
                    "hidden_features": 8,
                    "out_features": 30,
                    "output_scales": [1.0] * 30,
                },
            ),
            Node(
                "tp",
                "core.tensor_product@3",
                {"left": ("input:left",), "right": ("input:right",), "weight": ("profile",)},
                _tp_attrs(),
            ),
        ),
        (OutputPort("out", "tp", tp_output),),
    )


def _parameter_mapping():
    return {
        "node_modules.radial__linear_in.linear.weight": "net.0.weight",
        "node_modules.radial__linear_in.linear.bias": "net.0.bias",
        "node_modules.radial__norm.layer_norm.weight": "net.1.weight",
        "node_modules.radial__norm.layer_norm.bias": "net.1.bias",
        "node_modules.radial__linear_out.linear.weight": "net.3.weight",
        "node_modules.radial__offset.offset": "offset",
    }


def test_v1_radial_profile_motif_expands_to_five_core_nodes_and_six_parameter_contracts():
    registry = core_registry()
    artifact = Compiler(registry, reference_motif_registry()).analyze(_radial_program())
    assert [node.op for node in artifact.expanded_program.nodes] == [
        "core.scalar_linear@2",
        "core.scalar_layer_norm@1",
        "core.scalar_activation@1",
        "core.scalar_linear@2",
        "core.scalar_offset@1",
    ]
    output = artifact.inference.value_types["radial__offset"]
    assert isinstance(output, InvariantTensorType)
    assert str(output.irreps) == "30x0e"
    assert output.feature_role == FeatureRole.RADIAL_WEIGHT
    assert [(axis.name, axis.size, axis.role) for axis in output.axis_specs] == [
        ("tp_path", 30, FeatureRole.TP_PATH)
    ]
    shapes = {
        node_id: [contract.shape for contract in contracts]
        for node_id, contracts in artifact.inference.parameter_contracts.items()
        if contracts
    }
    assert shapes == {
        "radial__linear_in": [(8, 6), (8,)],
        "radial__norm": [(8,), (8,)],
        "radial__linear_out": [(30, 8)],
        "radial__offset": [(30,)],
    }


@pytest.mark.parametrize("scaled", (False, True))
def test_v1_radial_profile_matches_official_initialization_forward_input_and_parameter_gradients(scaled):
    scales = [1.0] * 30
    if scaled:
        scales = [0.5] * 7 + [1.0] * 12 + [2.0] * 11
    RadialProfile = _official_radial_profile_class()
    seed = 20260731
    torch.manual_seed(seed)
    official = RadialProfile([6, 8, 30]).double()
    with torch.no_grad():
        scale_tensor = torch.tensor(scales, dtype=torch.float64)
        official.net[-1].weight.mul_(scale_tensor.reshape(-1, 1))
        official.offset.mul_(scale_tensor)

    registry = core_registry()
    program = _radial_program(scales)
    artifact = Compiler(registry, reference_motif_registry()).analyze(program)
    torch.manual_seed(seed)
    model = E3NNGraphBackend(registry).build(artifact.expanded_program, artifact.inference).double()

    official_parameters = dict(official.named_parameters())
    dsl_parameters = dict(model.named_parameters())
    mapping = _parameter_mapping()
    assert set(dsl_parameters) == set(mapping)
    for dsl_name, official_name in mapping.items():
        assert torch.equal(dsl_parameters[dsl_name], official_parameters[official_name])

    values = torch.randn(11, 6, dtype=torch.float64, requires_grad=True)
    official_values = values.detach().clone().requires_grad_(True)
    actual = model({"x": values}, {})["out"]
    expected = official(official_values)
    assert torch.equal(actual, expected)

    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert torch.equal(values.grad, official_values.grad)
    for dsl_name, official_name in mapping.items():
        assert torch.equal(dsl_parameters[dsl_name].grad, official_parameters[official_name].grad)


def test_v1_radial_profile_output_composes_directly_with_tensor_product_v3():
    registry = core_registry()
    program = _radial_tp_program()
    artifact = Compiler(registry, reference_motif_registry()).analyze(program)
    model = E3NNGraphBackend(registry).build(artifact.expanded_program, artifact.inference).double()
    output = model(
        {
            "radial": torch.randn(6, 6, dtype=torch.float64),
            "left": torch.randn(6, 15, dtype=torch.float64),
            "right": torch.randn(6, 9, dtype=torch.float64),
        },
        {},
    )["out"]
    assert output.shape == (6, 98)
    assert torch.isfinite(output).all()


def test_v1_radial_profile_rejects_invalid_initialization_scales_and_dimensional_norm_input():
    invalid_length = _radial_program([1.0] * 29)
    with pytest.raises(DSLValidationError) as length_error:
        Compiler(core_registry(), reference_motif_registry()).analyze(invalid_length)
    assert any(item.code == "E_SCALAR_LINEAR_V2_003" for item in length_error.value.diagnostics)

    invalid_value = [1.0] * 30
    invalid_value[3] = 0.0
    with pytest.raises(DSLValidationError) as value_error:
        Compiler(core_registry(), reference_motif_registry()).analyze(_radial_program(invalid_value))
    assert any(item.code == "E_SCALAR_LINEAR_V2_003" for item in value_error.value.diagnostics)

    radial = replace(_types()[0], measure="angstrom")
    with pytest.raises(DSLValidationError) as unit_error:
        Compiler(core_registry(), reference_motif_registry()).analyze(_radial_program(input_type=radial))
    assert any(item.code == "E_SCALAR_NORM_005" for item in unit_error.value.diagnostics)
