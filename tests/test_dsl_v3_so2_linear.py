import os
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from equivariant_nas.dsl import (  # noqa: E402
    ArchitectureProgram,
    AxisSpec,
    Carrier,
    DSLValidationError,
    EquivariantType,
    FeatureRole,
    Frame,
    GroupSpec,
    InputPort,
    InvariantTensorType,
    Irreps,
    Node,
    OutputPort,
    TypeChecker,
    core_registry,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend  # noqa: E402
from equivariant_nas.dsl.backends.v2_runtime import SO3RuntimeValue  # noqa: E402
from equivariant_nas.dsl.backends.v3_runtime import (  # noqa: E402
    build_v3_so2_linear_module,
    equiformer_v3_source_available,
    load_equiformer_v3_modules,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = Path(
    os.environ.get(
        "EQUIFORMER_V3_ROOT",
        str(REPOSITORY_ROOT.parent / "equiformer_v3_official"),
    )
)


def _require_v3_source():
    if not equiformer_v3_source_available(str(V3_ROOT)):
        pytest.skip("official Equiformer V3 operator source is unavailable")
    return str(V3_ROOT)


def _edge_type(channels=3, *, frame="edge", lmax=2):
    return EquivariantType(
        GroupSpec.so3(),
        Carrier.EDGE,
        Irreps.parse(
            "+".join("{}x{}".format(channels, degree) for degree in range(lmax + 1)),
            "SO3",
        ),
        frame=Frame(frame, "v3_edge" if frame == "edge" else ""),
    )


def _dual_program(**attrs):
    source = _edge_type()
    output = _edge_type(channels=2)
    extra = InvariantTensorType(
        group=GroupSpec.so3(),
        carrier=Carrier.EDGE,
        irreps=Irreps.parse("5x0", "SO3"),
        frame=Frame("invariant"),
        axes=("m0_channel",),
        axis_specs=(AxisSpec("m0_channel", 5, FeatureRole.CHANNEL, "independent", 0),),
    )
    node_attrs = {
        "out_irreps": str(output.irreps),
        "mmax": 1,
        "extra_m0_channels": 5,
        **attrs,
    }
    return ArchitectureProgram(
        "1.0.0",
        "v3_so2_linear_contract",
        (InputPort("x", source),),
        (
            Node(
                "blocks.0.ga.so2_linear_1",
                "core.so2_linear@2",
                {"x": ("input:x",)},
                node_attrs,
                outputs=("out", "extra_m0"),
            ),
        ),
        (
            OutputPort("out", "blocks.0.ga.so2_linear_1:out", output),
            OutputPort("extra", "blocks.0.ga.so2_linear_1:extra_m0", extra),
        ),
    )


def test_v3_so2_linear_dual_output_type_and_parameter_contract_are_explicit():
    inference = TypeChecker(core_registry()).check(_dual_program())
    contracts = inference.parameter_contracts["blocks.0.ga.so2_linear_1"]
    assert [item.backend_parameter_name for item in contracts] == [
        "fc_m0.weight",
        "fc_m0.bias",
        "so2_m_linear.0.fc.weight",
    ]
    assert [item.concrete_shape for item in contracts] == [(11, 9), (11,), (8, 6)]
    assert contracts[0].checkpoint_names == (
        "blocks.0.ga.so2_linear_1.fc_m0.weight",
    )
    assert inference.value_types["blocks.0.ga.so2_linear_1:out"].frame.kind == "edge"
    extra = inference.value_types["blocks.0.ga.so2_linear_1:extra_m0"]
    assert extra.frame.kind == "invariant"
    assert str(extra.irreps) == "5x0"


@pytest.mark.parametrize("extra_m0_channels", [0, 5])
def test_v3_so2_linear_matches_official_low_level_operator_and_gradients(extra_m0_channels):
    root = _require_v3_source()
    _so3, so2_ops, _activation, _layer_norm = load_equiformer_v3_modules(root)
    input_irreps = Irreps.parse("3x0+3x1+3x2", "SO3")
    output_irreps = Irreps.parse("2x0+2x1+2x2", "SO3")

    torch.manual_seed(7341)
    official = so2_ops.SO2Linear(
        3,
        2,
        2,
        1,
        extra_m0_out_channels=(extra_m0_channels or None),
    ).double()
    torch.manual_seed(7341)
    lowered = build_v3_so2_linear_module(
        input_irreps,
        output_irreps,
        mmax=1,
        extra_m0_channels=extra_m0_channels,
    ).double()

    assert official.state_dict().keys() == lowered.state_dict().keys()
    for name, expected in official.state_dict().items():
        torch.testing.assert_close(lowered.state_dict()[name], expected, rtol=0.0, atol=0.0)

    torch.manual_seed(7342)
    official_input = torch.randn(7, 7, 3, dtype=torch.float64, requires_grad=True)
    lowered_input = official_input.detach().clone().requires_grad_(True)
    official_output = official(official_input)
    runtime = SO3RuntimeValue(lowered_input, input_irreps, 2, 1, 3, "v3_edge", None)
    lowered_output = lowered(runtime)

    if extra_m0_channels:
        expected_value, expected_extra = official_output
        actual_value = lowered_output["out"].embedding
        actual_extra = lowered_output["extra_m0"]
        torch.testing.assert_close(actual_value, expected_value, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_extra, expected_extra, rtol=0.0, atol=0.0)
        expected_loss = expected_value.square().sum() + expected_extra.square().sum()
        actual_loss = actual_value.square().sum() + actual_extra.square().sum()
    else:
        actual_value = lowered_output.embedding
        torch.testing.assert_close(actual_value, official_output, rtol=0.0, atol=0.0)
        expected_loss = official_output.square().sum()
        actual_loss = actual_value.square().sum()

    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(lowered_input.grad, official_input.grad, rtol=1.0e-14, atol=1.0e-14)
    official_parameters = dict(official.named_parameters())
    lowered_parameters = dict(lowered.named_parameters())
    assert official_parameters.keys() == lowered_parameters.keys()
    for name in official_parameters:
        torch.testing.assert_close(
            lowered_parameters[name].grad,
            official_parameters[name].grad,
            rtol=1.0e-14,
            atol=1.0e-14,
        )


def test_v3_so2_linear_prefix_rescale_and_final_zero_bias_are_explicit():
    root = _require_v3_source()
    _so3, so2_ops, _activation, _layer_norm = load_equiformer_v3_modules(root)
    input_irreps = Irreps.parse("3x0+3x1+3x2", "SO3")
    output_irreps = Irreps.parse("2x0+2x1+2x2", "SO3")

    torch.manual_seed(9201)
    official = so2_ops.SO2Linear(3, 2, 2, 1, extra_m0_out_channels=5)
    official.fc_m0.weight.data[:4].mul_(1.0 / (2.0 ** 0.5))
    torch.nn.init.zeros_(official.fc_m0.bias)
    official = official.double()
    torch.manual_seed(9201)
    lowered = build_v3_so2_linear_module(
        input_irreps,
        output_irreps,
        mmax=1,
        extra_m0_channels=5,
        m0_prefix_rows=4,
        m0_prefix_scale=1.0 / (2.0 ** 0.5),
        zero_bias=True,
    ).double()
    for name, expected in official.state_dict().items():
        torch.testing.assert_close(lowered.state_dict()[name], expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("program", "code"),
    [
        (
            ArchitectureProgram(
                "1.0.0",
                "bad_frame",
                (InputPort("x", _edge_type(frame="global")),),
                (
                    Node(
                        "bad",
                        "core.so2_linear@1",
                        {"x": ("input:x",)},
                        {"out_irreps": "2x0+2x1+2x2", "mmax": 1},
                    ),
                ),
                (OutputPort("out", "bad", _edge_type(channels=2)),),
            ),
            "E_SO2_LINEAR_002",
        ),
        (_dual_program(extra_m0_channels=0), "E_SO2_LINEAR_007"),
        (_dual_program(m0_prefix_rows=12), "E_SO2_LINEAR_009"),
    ],
)
def test_v3_so2_linear_contract_rejects_invalid_layouts(program, code):
    with pytest.raises(DSLValidationError) as caught:
        TypeChecker(core_registry()).check(program)
    assert any(item.code == code for item in caught.value.diagnostics)


def test_v3_so2_linear_rules_are_present_in_generic_lowering_registry():
    registry = core_registry()
    backend = E3NNGraphBackend(registry)
    assert len(registry.names()) == 102
    assert backend.lowering_rules.resolve("core.so2_linear@1").exactness == "constructive_exact"
    assert backend.lowering_rules.resolve("core.so2_linear@2").exactness == "constructive_exact"
    assert backend.lowering_rules.audit()["rule_count"] == 102
