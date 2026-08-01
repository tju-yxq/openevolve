import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
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
    IndexMapType,
    InputPort,
    InvariantTensorType,
    Irreps,
    Node,
    OutputPort,
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend


HIDDEN_IRREPS = "4x0e+2x1e+1x2e"


def _official_drop_module():
    source = Path(__file__).resolve().parents[2] / "equiformer" / "nets" / "drop.py"
    spec = importlib.util.spec_from_file_location("equiformer_v1_drop_oracle", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load official Equiformer V1 drop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hidden_type(*, axisful=False):
    group = GroupSpec.o3()
    return EquivariantTensorType(
        group,
        Carrier.NODE,
        Irreps.parse(HIDDEN_IRREPS, group.family),
        axes=("channel",) if axisful else (),
        axis_specs=(AxisSpec("channel", 1, FeatureRole.CHANNEL, "independent", 0),) if axisful else (),
        dtype="float64",
    )


def _alpha_type():
    group = GroupSpec.o3()
    return InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("2x0e", group.family),
        axes=("head",),
        axis_specs=(AxisSpec("head", 2, FeatureRole.HEAD, "independent", 0),),
        dtype="float64",
        feature_role=FeatureRole.ALPHA,
    )


def _graph_drop_program(*, p=0.25, batch_type=None):
    hidden = _hidden_type()
    batch = batch_type or IndexMapType(
        hidden.group,
        Carrier.NODE,
        Carrier.GRAPH,
        "batch",
        target_size=2,
        allows_empty_targets=False,
    )
    return ArchitectureProgram(
        "2.12.0",
        "official-v1-graph-drop-path",
        (InputPort("x", hidden), InputPort("batch", batch)),
        (
            Node(
                "drop",
                "core.graph_stochastic_depth@1",
                {"x": ("input:x",), "batch": ("input:batch",)},
                {"p": p},
            ),
        ),
        (OutputPort("out", "drop", hidden),),
        program_id="official-v1-graph-drop-path@1",
    )


def _unary_program(op, value_type, *, p=0.25):
    return ArchitectureProgram(
        "2.12.0",
        op,
        (InputPort("x", value_type),),
        (Node("drop", op, {"x": ("input:x",)}, {"p": p}),),
        (OutputPort("out", "drop", value_type),),
    )


def _build(program):
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    return E3NNGraphBackend(registry).build(artifact.expanded_program, artifact.inference).double()


def test_graph_stochastic_depth_matches_official_graph_shared_mask_forward_gradient_eval_and_o3():
    official_drop = _official_drop_module()
    p = 0.25
    official = official_drop.GraphDropPath(p).double().train()
    model = _build(_graph_drop_program(p=p)).train()
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    values = torch.randn(5, o3.Irreps(HIDDEN_IRREPS).dim, dtype=torch.float64, requires_grad=True)
    official_values = values.detach().clone().requires_grad_(True)
    payload = {"indices": batch, "target_size": 2}

    torch.manual_seed(17)
    actual = model({"x": values, "batch": payload}, {})["out"]
    torch.manual_seed(17)
    expected = official(official_values, batch)
    assert torch.equal(actual, expected)

    for graph_id in (0, 1):
        rows = actual[batch == graph_id]
        inputs = values.detach()[batch == graph_id]
        nonzero = inputs.abs() > 1.0e-12
        ratios = rows[nonzero] / inputs[nonzero]
        assert torch.allclose(ratios, ratios[0].expand_as(ratios), atol=1.0e-15, rtol=1.0e-15)

    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert torch.equal(values.grad, official_values.grad)

    rotation = o3.rand_matrix(dtype=torch.float64)
    action = o3.Irreps(HIDDEN_IRREPS).D_from_matrix(rotation)
    torch.manual_seed(29)
    reference = model({"x": values.detach(), "batch": payload}, {})["out"]
    torch.manual_seed(29)
    transformed = model(
        {"x": values.detach() @ action.transpose(0, 1), "batch": payload},
        {},
    )["out"]
    assert torch.allclose(transformed, reference @ action.transpose(0, 1), atol=1.0e-12, rtol=1.0e-12)

    official.eval()
    model.eval()
    assert torch.equal(model({"x": values.detach(), "batch": payload}, {})["out"], values.detach())
    assert torch.equal(official(values.detach(), batch), values.detach())


def test_scalar_dropout_matches_official_attention_dropout_forward_and_gradient():
    p = 0.2
    model = _build(_unary_program("core.scalar_dropout@1", _alpha_type(), p=p)).train()
    official = torch.nn.Dropout(p).double().train()
    values = torch.randn(8, 2, dtype=torch.float64, requires_grad=True)
    official_values = values.detach().clone().requires_grad_(True)
    torch.manual_seed(41)
    actual = model({"x": values}, {})["out"]
    torch.manual_seed(41)
    expected = official(official_values.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(actual, expected)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert torch.equal(values.grad, official_values.grad)
    model.eval()
    assert torch.equal(model({"x": values.detach()}, {})["out"], values.detach())


def test_equivariant_dropout_matches_official_irrep_instance_masks_gradient_and_o3():
    official_drop = _official_drop_module()
    p = 0.3
    hidden = _hidden_type()
    model = _build(_unary_program("core.equivariant_dropout@1", hidden, p=p)).train()
    official = official_drop.EquivariantDropout(o3.Irreps(HIDDEN_IRREPS), p).double().train()
    values = torch.randn(7, o3.Irreps(HIDDEN_IRREPS).dim, dtype=torch.float64, requires_grad=True)
    official_values = values.detach().clone().requires_grad_(True)
    torch.manual_seed(53)
    actual = model({"x": values}, {})["out"]
    torch.manual_seed(53)
    expected = official(official_values)
    assert torch.equal(actual, expected)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert torch.equal(values.grad, official_values.grad)

    rotation = o3.rand_matrix(dtype=torch.float64)
    action = o3.Irreps(HIDDEN_IRREPS).D_from_matrix(rotation)
    torch.manual_seed(67)
    reference = model({"x": values.detach()}, {})["out"]
    torch.manual_seed(67)
    transformed = model({"x": values.detach() @ action.transpose(0, 1)}, {})["out"]
    assert torch.allclose(transformed, reference @ action.transpose(0, 1), atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.parametrize(
    ("program", "expected_code"),
    (
        (
            _graph_drop_program(
                batch_type=IndexMapType(
                    GroupSpec.o3(),
                    Carrier.NODE,
                    Carrier.GRAPH,
                    "batch",
                    target_size=2,
                    allows_empty_targets=True,
                )
            ),
            "E_GRAPH_DROP_003",
        ),
        (
            _graph_drop_program(
                batch_type=IndexMapType(
                    GroupSpec.o3(),
                    Carrier.NODE,
                    Carrier.GRAPH,
                    "segment",
                    target_size=2,
                    allows_empty_targets=False,
                )
            ),
            "E_GRAPH_DROP_003",
        ),
        (_unary_program("core.scalar_dropout@1", _hidden_type()), "E_SCALAR_DROP_001"),
        (_unary_program("core.equivariant_dropout@1", _hidden_type(axisful=True)), "E_EQ_DROPOUT_001"),
        (_graph_drop_program(p=1.0), "E_ATTR_003"),
    ),
)
def test_v1_stochastic_regularization_rejects_invalid_static_contracts(program, expected_code):
    with pytest.raises(DSLValidationError) as error:
        Compiler(core_registry()).analyze(program)
    assert any(item.code == expected_code for item in error.value.diagnostics)


def test_graph_stochastic_depth_rejects_runtime_empty_graph_target():
    model = _build(_graph_drop_program()).train()
    values = torch.randn(5, o3.Irreps(HIDDEN_IRREPS).dim, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="contiguous graph ids|empty graph target"):
        model(
            {
                "x": values,
                "batch": {
                    "indices": torch.tensor([0, 0, 2, 2, 2], dtype=torch.long),
                    "target_size": 3,
                },
            },
            {},
        )
