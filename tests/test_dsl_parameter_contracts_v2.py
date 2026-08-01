from dataclasses import replace

import pytest
import torch

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
    PARAMETER_CONTRACT_SCHEMA_VERSION,
    ParameterAxis,
    ParameterContract,
    architecture_id,
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.registry import PrimitiveRegistry


def _invariant_type(*, channel_size=3, dynamic_channel=False):
    group = GroupSpec.so3()
    return InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("6x0", group.family),
        axes=("head", "channel"),
        axis_specs=(
            AxisSpec("head", 2, FeatureRole.HEAD, order=0),
            AxisSpec(
                "channel",
                None if dynamic_channel else channel_size,
                FeatureRole.CHANNEL,
                sharing="per_head",
                order=1,
            ),
        ),
        feature_role=FeatureRole.CHANNEL,
    )


def _program(*, node_id="projection", out_features=4, bias=True, input_type=None):
    input_type = input_type or _invariant_type()
    output_axes = tuple(
        replace(axis, size=out_features) if axis.name == "channel" else axis
        for axis in input_type.axis_specs
    )
    output_type = replace(
        input_type,
        irreps=Irreps.parse("{}x0".format(2 * out_features), input_type.group.family),
        axis_specs=output_axes,
    )
    return ArchitectureProgram(
        "2.2.0",
        "parameter-contract-test",
        (InputPort("x", input_type),),
        (
            Node(
                node_id,
                "core.scalar_linear@1",
                {"x": ("input:x",)},
                {"axis": "channel", "out_features": out_features, "bias": bias},
            ),
        ),
        (OutputPort("out", node_id, output_type),),
    )


def test_parameter_contract_roundtrip_and_negative_schema_cases():
    contract = ParameterContract(
        "weight",
        (
            ParameterAxis("out_feature", 4, "output_feature", "channel"),
            ParameterAxis("in_feature", 3, "input_feature", "channel"),
        ),
        sharing_axes=("head",),
        initializer="kaiming_uniform",
        checkpoint_names=("projection.linear.weight",),
        backend_parameter_name="linear.weight",
    )
    assert ParameterContract.from_dict(contract.to_dict()) == contract
    assert contract.schema_version == PARAMETER_CONTRACT_SCHEMA_VERSION
    assert contract.shape == (4, 3)
    assert contract.concrete_shape == (4, 3)
    assert len(contract.content_hash()) == 64

    with pytest.raises(DSLValidationError) as external:
        ParameterContract(
            "weight",
            (ParameterAxis("path", "num_paths", "tensor_product_path"),),
            storage="external",
        )
    assert any(item.code == "E_PARAMETER_007" for item in external.value.diagnostics)

    with pytest.raises(DSLValidationError) as bias:
        ParameterContract(
            "bias",
            (ParameterAxis("feature", 4, "output_feature"),),
            is_bias=True,
            backend_parameter_name="bias",
        )
    assert any(item.code == "E_PARAMETER_011" for item in bias.value.diagnostics)


def test_scalar_linear_infers_axis_aware_types_and_parameter_contracts():
    registry = core_registry()
    program = _program()
    artifact = Compiler(registry).analyze(program)
    output = artifact.inference.value_types["projection"]
    contracts = artifact.inference.parameter_contracts["projection"]

    assert isinstance(output, InvariantTensorType)
    assert output.irreps.dimension == 8
    assert [(axis.name, axis.size) for axis in output.axis_specs] == [("head", 2), ("channel", 4)]
    assert [contract.name for contract in contracts] == ["weight", "bias"]
    assert contracts[0].shape == (4, 3)
    assert contracts[0].sharing_axes == ("head",)
    assert contracts[1].shape == (4,)
    assert contracts[1].bias_irreps == "4x0"
    assert all(contract.architecture_identity == "contract_only" for contract in contracts)


def test_scalar_linear_generic_lowering_matches_axiswise_torch_linear_and_gradients():
    registry = core_registry()
    program = _program()
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference)
    module = model.node_modules["projection"]

    assert model.parameter_contract_manifest["projection"][0]["backend_parameter_name"] == "linear.weight"
    assert tuple(module.linear.weight.shape) == (4, 3)
    assert tuple(module.linear.bias.shape) == (4,)

    with torch.no_grad():
        module.linear.weight.copy_(
            torch.tensor(
                [[1.0, 0.0, -1.0], [0.5, 0.25, 0.0], [-0.5, 1.0, 0.5], [0.0, -1.0, 2.0]],
                dtype=module.linear.weight.dtype,
            )
        )
        module.linear.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=module.linear.bias.dtype))

    value = torch.randn(5, 6, requires_grad=True)
    actual = model({"x": value}, {})["out"]
    expanded = value.reshape(5, 2, 3)
    expected = torch.nn.functional.linear(expanded, module.linear.weight, module.linear.bias).reshape(5, 8)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)

    actual.square().sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert module.linear.weight.grad is not None and torch.isfinite(module.linear.weight.grad).all()
    assert module.linear.bias.grad is not None and torch.isfinite(module.linear.bias.grad).all()


def test_scalar_linear_without_bias_has_one_contract_and_one_backend_parameter():
    registry = core_registry()
    program = _program(bias=False)
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference)

    assert [item.name for item in artifact.inference.parameter_contracts["projection"]] == ["weight"]
    assert set(dict(model.node_modules["projection"].named_parameters())) == {"linear.weight"}


def test_scalar_linear_contracts_are_canonical_and_affect_architecture_identity():
    registry = core_registry()
    renamed = _program(node_id="renamed")
    baseline = _program()
    wider = _program(out_features=5)
    without_bias = _program(bias=False)

    assert architecture_id(baseline, registry) == architecture_id(renamed, registry)
    assert architecture_id(baseline, registry) != architecture_id(wider, registry)
    assert architecture_id(baseline, registry) != architecture_id(without_bias, registry)
    assert len(registry.content_hash()) == 64

    drifted = PrimitiveRegistry()
    for name in registry.names():
        definition = registry.resolve(name)
        if name == "core.scalar_linear@1":
            definition = replace(definition, description=definition.description + " registry-drift")
        drifted.register(definition)
    assert drifted.content_hash() != registry.content_hash()
    assert architecture_id(baseline, drifted) != architecture_id(baseline, registry)


@pytest.mark.parametrize(
    "program,code",
    (
        (
            _program(
                input_type=EquivariantTensorType(
                    GroupSpec.so3(),
                    Carrier.EDGE,
                    Irreps.parse("2x0+1x1", "SO3"),
                )
            ),
            "E_SCALAR_LINEAR_001",
        ),
        (_program(input_type=_invariant_type(dynamic_channel=True)), "E_SCALAR_LINEAR_004"),
        (_program(input_type=_invariant_type(channel_size=2)), "E_SCALAR_LINEAR_005"),
    ),
)
def test_scalar_linear_rejects_noninvariant_dynamic_or_inconsistent_axis_contracts(program, code):
    with pytest.raises(DSLValidationError) as error:
        Compiler(core_registry()).analyze(program)
    assert any(item.code == code for item in error.value.diagnostics)


def test_scalar_linear_rejects_unknown_axis_and_nonintegral_output_width():
    baseline = _program()
    unknown_axis = replace(
        baseline,
        nodes=(replace(baseline.nodes[0], attrs={"axis": "missing", "out_features": 4, "bias": True}),),
    )
    with pytest.raises(DSLValidationError) as axis_error:
        Compiler(core_registry()).analyze(unknown_axis)
    assert any(item.code == "E_SCALAR_LINEAR_003" for item in axis_error.value.diagnostics)

    invalid_width = replace(
        baseline,
        nodes=(replace(baseline.nodes[0], attrs={"axis": "channel", "out_features": 4.5, "bias": True}),),
    )
    with pytest.raises(DSLValidationError) as width_error:
        Compiler(core_registry()).analyze(invalid_width)
    assert any(item.code == "E_SCALAR_LINEAR_006" for item in width_error.value.diagnostics)
