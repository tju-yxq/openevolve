import ast
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
    InputPort,
    Irreps,
    Node,
    OutputPort,
    RepresentationLayout,
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend


CASES = (
    ("merge_src", "4x0e+2x1e+1x2e", "4x0e+2x1e+1x2e", True, 21, (4,)),
    ("merge_dst", "4x0e+2x1e+1x2e", "4x0e+2x1e+1x2e", False, 21, ()),
    ("post_tp", "7x0e+12x1e+11x2e", "8x0e+2x1e+2x2e", True, 102, (8,)),
    ("projection", "4x0e+2x1e+2x2e", "4x0e+2x1e+1x2e", True, 22, (4,)),
)


def _official_linear_rs_class():
    source_path = Path(__file__).resolve().parents[2] / "equiformer" / "nets" / "tensor_product_rescale.py"
    if not source_path.exists():
        pytest.skip("local official Equiformer V1 tensor_product_rescale.py is unavailable")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    names = {"TensorProductRescale", "FullyConnectedTensorProductRescale", "LinearRS"}
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise RuntimeError("official source no longer contains the LinearRS inheritance chain")
    namespace = {"torch": torch, "o3": o3}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["LinearRS"]


def _type(irreps, *, layout=None, axisful=False):
    group = GroupSpec.o3()
    axes = ("channel",) if axisful else ()
    axis_specs = (AxisSpec("channel", 1, FeatureRole.CHANNEL, "independent", 0),) if axisful else ()
    return EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse(irreps, group.family),
        axes=axes,
        axis_specs=axis_specs,
        layout=layout or RepresentationLayout(),
        dtype="float64",
    )


def _program(input_irreps, output_irreps, *, bias=True, input_type=None):
    source = input_type or _type(input_irreps)
    output = replace(source, irreps=Irreps.parse(output_irreps, source.group.family), axes=(), axis_specs=())
    return ArchitectureProgram(
        "2.8.0",
        "official-v1-linear-rs",
        (InputPort("x", source),),
        (
            Node(
                "linear",
                "core.irrep_linear@2",
                {"x": ("input:x",)},
                {"out_irreps": output_irreps, "bias": bias, "rescale": True},
            ),
        ),
        (OutputPort("out", "linear", output),),
    )


def _mapping(bias_shapes):
    mapping = {"node_modules.linear.tp.weight": "tp.weight"}
    for index in range(len(bias_shapes)):
        mapping["node_modules.linear.bias.{}".format(index)] = "bias.{}".format(index)
    return mapping


@pytest.mark.parametrize("_name,input_irreps,output_irreps,bias,weight_numel,bias_shapes", CASES)
def test_linear_rs_parameter_contracts_and_lowering_match_official_initialization_forward_and_gradients(
    _name, input_irreps, output_irreps, bias, weight_numel, bias_shapes
):
    registry = core_registry()
    program = _program(input_irreps, output_irreps, bias=bias)
    artifact = Compiler(registry).analyze(program)
    contracts = artifact.inference.parameter_contracts["linear"]
    assert contracts[0].name == "weight"
    assert contracts[0].shape == (weight_numel,)
    assert [contract.shape for contract in contracts[1:]] == [(size,) for size in bias_shapes]

    LinearRS = _official_linear_rs_class()
    seed = 20260731
    torch.manual_seed(seed)
    official = LinearRS(o3.Irreps(input_irreps), o3.Irreps(output_irreps), bias=bias, rescale=True).double()
    torch.manual_seed(seed)
    model = E3NNGraphBackend(registry).build(program, artifact.inference).double()
    mapping = _mapping(bias_shapes)
    official_parameters = dict(official.named_parameters())
    dsl_parameters = dict(model.named_parameters())
    assert set(dsl_parameters) == set(mapping)
    for dsl_name, official_name in mapping.items():
        assert torch.equal(dsl_parameters[dsl_name], official_parameters[official_name])

    values = torch.randn(9, o3.Irreps(input_irreps).dim, dtype=torch.float64, requires_grad=True)
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


@pytest.mark.parametrize("matrix_kind", ("rotation", "inversion"))
def test_linear_rs_is_o3_equivariant(matrix_kind):
    input_irreps = "4x0e+2x1e+2x2e"
    output_irreps = "4x0e+2x1e+1x2e"
    registry = core_registry()
    program = _program(input_irreps, output_irreps)
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference).double()
    values = torch.randn(7, o3.Irreps(input_irreps).dim, dtype=torch.float64)
    output = model({"x": values}, {})["out"]
    matrix = o3.rand_matrix(dtype=torch.float64) if matrix_kind == "rotation" else -torch.eye(3, dtype=torch.float64)
    input_action = o3.Irreps(input_irreps).D_from_matrix(matrix)
    output_action = o3.Irreps(output_irreps).D_from_matrix(matrix)
    transformed = model({"x": values @ input_action.transpose(0, 1)}, {})["out"]
    expected = output @ output_action.transpose(0, 1)
    relative_error = (transformed - expected).norm() / expected.norm().clamp_min(1.0e-12)
    assert float(relative_error) < 1.0e-7


def test_linear_rs_rejects_absent_irrep_axisful_and_noncanonical_layout():
    with pytest.raises(DSLValidationError) as missing:
        Compiler(core_registry()).analyze(_program("2x0e", "2x1e"))
    assert any(item.code == "E_IRREP_011" for item in missing.value.diagnostics)

    axisful = _type("4x0e+2x1e+1x2e", axisful=True)
    with pytest.raises(DSLValidationError) as axes:
        Compiler(core_registry()).analyze(
            _program("4x0e+2x1e+1x2e", "4x0e+2x1e+1x2e", input_type=axisful)
        )
    assert any(item.code == "E_LINEAR_RS_001" for item in axes.value.diagnostics)

    grid = _type(
        "4x0e+2x1e+1x2e",
        layout=RepresentationLayout(storage="grid", truncation_state="grid_projected"),
    )
    with pytest.raises(DSLValidationError) as layout:
        Compiler(core_registry()).analyze(
            _program("4x0e+2x1e+1x2e", "4x0e+2x1e+1x2e", input_type=grid)
        )
    assert any(item.code == "E_LINEAR_RS_002" for item in layout.value.diagnostics)
