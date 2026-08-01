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
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend


def _flat_scalar_type(channels=12, *, sharing="independent"):
    group = GroupSpec.so3()
    return InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("{}x0".format(channels), group.family),
        axes=("channel",),
        axis_specs=(AxisSpec("channel", channels, FeatureRole.CHANNEL, sharing, 0),),
        feature_role=FeatureRole.CHANNEL,
    )


def _split_type(source, heads=3):
    return replace(
        source,
        axes=("head", "per_head_channel"),
        axis_specs=(
            AxisSpec("head", heads, FeatureRole.HEAD, "independent", 0),
            AxisSpec(
                "per_head_channel",
                source.irreps.dimension // heads,
                FeatureRole.CHANNEL,
                "per_head",
                1,
            ),
        ),
    )


def _alpha_type(source, heads=3):
    return replace(
        source,
        irreps=Irreps.parse("{}x0".format(heads), source.group.family),
        axes=("head",),
        axis_specs=(AxisSpec("head", heads, FeatureRole.HEAD, "independent", 0),),
        feature_role=FeatureRole.ALPHA,
    )


def _split_node():
    return Node(
        "split",
        "core.head_split@1",
        {"x": ("input:x",)},
        {
            "axis": "channel",
            "head_axis": "head",
            "channel_axis": "per_head_channel",
            "num_heads": 3,
        },
    )


def _split_contract_program(*, bias=True):
    source = _flat_scalar_type()
    alpha = _alpha_type(source)
    return ArchitectureProgram(
        "2.3.0",
        "invariant-multihead-test",
        (InputPort("x", source),),
        (
            _split_node(),
            Node(
                "alpha",
                "core.headwise_scalar_contraction@1",
                {"x": ("split",)},
                {"head_axis": "head", "channel_axis": "per_head_channel", "bias": bias},
            ),
        ),
        (OutputPort("alpha", "alpha", alpha),),
    )


def test_invariant_head_split_and_merge_are_typed_inverse_views():
    source = _flat_scalar_type()
    program = ArchitectureProgram(
        "2.3.0",
        "head-view-test",
        (InputPort("x", source),),
        (
            _split_node(),
            Node(
                "merge",
                "core.head_merge@1",
                {"x": ("split",)},
                {"head_axis": "head", "channel_axis": "per_head_channel", "out_axis": "channel"},
            ),
        ),
        (OutputPort("out", "merge", source),),
    )
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    split_type = artifact.inference.value_types["split"]

    assert isinstance(split_type, InvariantTensorType)
    assert [(axis.name, axis.size, axis.sharing) for axis in split_type.axis_specs] == [
        ("head", 3, "independent"),
        ("per_head_channel", 4, "per_head"),
    ]
    assert artifact.inference.value_types["merge"] == source
    assert artifact.inference.parameter_contracts["split"] == ()
    assert artifact.inference.parameter_contracts["merge"] == ()

    model = E3NNGraphBackend(registry).build(program, artifact.inference)
    value = torch.randn(7, 12, requires_grad=True)
    output = model({"x": value}, {})["out"]
    assert output.data_ptr() == value.data_ptr()
    output.square().sum().backward()
    assert torch.allclose(value.grad, 2.0 * value)


def test_headwise_scalar_contraction_has_per_head_contract_and_matches_reference():
    registry = core_registry()
    program = _split_contract_program()
    artifact = Compiler(registry).analyze(program)
    contracts = artifact.inference.parameter_contracts["alpha"]
    output_type = artifact.inference.value_types["alpha"]
    model = E3NNGraphBackend(registry).build(program, artifact.inference)
    module = model.node_modules["alpha"]

    assert output_type.feature_role == FeatureRole.ALPHA
    assert output_type.irreps.dimension == 3
    assert [contract.shape for contract in contracts] == [(3, 4), (3,)]
    assert [contract.backend_parameter_name for contract in contracts] == ["weight", "bias"]

    with torch.no_grad():
        module.weight.copy_(
            torch.tensor(
                [[1.0, 0.0, -1.0, 0.5], [0.5, 0.25, 0.0, -0.5], [-0.5, 1.0, 0.5, 2.0]],
                dtype=module.weight.dtype,
            )
        )
        module.bias.copy_(torch.tensor([0.1, -0.2, 0.3], dtype=module.bias.dtype))

    value = torch.randn(6, 12, requires_grad=True)
    actual = model({"x": value}, {})["alpha"]
    expected = (value.reshape(6, 3, 4) * module.weight).sum(dim=-1) + module.bias
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)

    actual.square().sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
    assert module.bias.grad is not None and torch.isfinite(module.bias.grad).all()


def test_headwise_scalar_contraction_without_bias_has_exact_parameter_set():
    registry = core_registry()
    program = _split_contract_program(bias=False)
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference)

    assert [item.name for item in artifact.inference.parameter_contracts["alpha"]] == ["weight"]
    assert set(dict(model.node_modules["alpha"].named_parameters())) == {"weight"}


def test_head_split_rejects_noninvariant_value_path_until_blockwise_layout_exists():
    group = GroupSpec.so3()
    value = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("4x0+4x1", group.family),
        axes=("channel",),
        axis_specs=(AxisSpec("channel", 8, FeatureRole.CHANNEL, "independent", 0),),
    )
    program = ArchitectureProgram(
        "2.3.0",
        "reject-equivariant-value-head-split",
        (InputPort("x", value),),
        (_split_node(),),
        (OutputPort("out", "split", value),),
    )
    with pytest.raises(DSLValidationError) as error:
        Compiler(core_registry()).analyze(program)
    assert any(item.code == "E_HEAD_001" for item in error.value.diagnostics)


@pytest.mark.parametrize(
    "source,attrs,code",
    (
        (
            _flat_scalar_type(channels=10),
            {"axis": "channel", "head_axis": "head", "channel_axis": "per_head_channel", "num_heads": 3},
            "E_HEAD_009",
        ),
        (
            _flat_scalar_type(sharing="per_head"),
            {"axis": "channel", "head_axis": "head", "channel_axis": "per_head_channel", "num_heads": 3},
            "E_HEAD_006",
        ),
    ),
)
def test_head_split_rejects_nondivisible_or_already_shared_source_axes(source, attrs, code):
    program = ArchitectureProgram(
        "2.3.0",
        "invalid-head-split",
        (InputPort("x", source),),
        (Node("split", "core.head_split@1", {"x": ("input:x",)}, attrs),),
        (OutputPort("out", "split", source),),
    )
    with pytest.raises(DSLValidationError) as error:
        Compiler(core_registry()).analyze(program)
    assert any(item.code == code for item in error.value.diagnostics)


def test_headwise_contraction_rejects_extra_feature_axes():
    source = _split_type(_flat_scalar_type())
    source = replace(
        source,
        irreps=Irreps.parse("24x0", source.group.family),
        axes=("batch_feature", "head", "per_head_channel"),
        axis_specs=(
            AxisSpec("batch_feature", 2, FeatureRole.BASIS, "independent", 0),
            replace(source.axis_specs[0], order=1),
            replace(source.axis_specs[1], order=2),
        ),
    )
    expected = _alpha_type(_flat_scalar_type())
    program = ArchitectureProgram(
        "2.3.0",
        "invalid-head-contraction",
        (InputPort("x", source),),
        (
            Node(
                "alpha",
                "core.headwise_scalar_contraction@1",
                {"x": ("input:x",)},
                {"head_axis": "head", "channel_axis": "per_head_channel", "bias": True},
            ),
        ),
        (OutputPort("alpha", "alpha", expected),),
    )
    with pytest.raises(DSLValidationError) as error:
        Compiler(core_registry()).analyze(program)
    assert any(item.code == "E_HEAD_013" for item in error.value.diagnostics)
