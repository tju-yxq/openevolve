import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from equivariant_nas.dsl import (  # noqa: E402
    ArchitectureProgram,
    AxisSpec,
    Carrier,
    CategoricalTensorType,
    EquivarianceLevel,
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V3_PACKAGE = (
    REPOSITORY_ROOT.parent
    / "equiformer_v3_official"
    / "experimental"
    / "models"
    / "equiformer_v3"
)


def _load_plain_module(name):
    path = V3_PACKAGE / "{}.py".format(name)
    if not path.is_file():
        pytest.skip("official Equiformer V3 {} source is unavailable".format(name))
    spec = importlib.util.spec_from_file_location("v3_test_{}".format(name), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _species_type(vocabulary=128):
    return CategoricalTensorType(
        group=GroupSpec.so3(),
        carrier=Carrier.NODE,
        vocabulary_size=vocabulary,
        dtype="int64",
    )


def _embedding_program(**attrs):
    source = _species_type()
    output = InvariantTensorType(
        group=GroupSpec.so3(),
        carrier=Carrier.NODE,
        irreps=Irreps.parse("4x0", "SO3"),
        frame=Frame("invariant"),
        axes=("channel",),
        axis_specs=(AxisSpec("channel", 4, FeatureRole.CHANNEL, "independent", 0),),
        level=EquivarianceLevel.CORE_CERTIFIED,
        feature_role=FeatureRole.CHANNEL,
    )
    node_attrs = {"embedding_dim": 4, **attrs}
    return ArchitectureProgram(
        "1.0.0",
        "v3_categorical_embedding",
        (InputPort("atomic_numbers", source),),
        (
            Node(
                "sphere_embedding",
                "core.categorical_embedding",
                {"x": ("input:atomic_numbers",)},
                node_attrs,
            ),
        ),
        (OutputPort("out", "sphere_embedding", output),),
    )


@pytest.mark.parametrize(
    "attrs",
    [
        {},
        {"initializer": "normal_then_uniform", "init_min": -0.001, "init_max": 0.001},
    ],
)
def test_v3_categorical_embedding_matches_pytorch_initialization_forward_and_gradients(attrs):
    registry = core_registry()
    program = _embedding_program(**attrs)
    inference = TypeChecker(registry).check(program)
    torch.manual_seed(5101)
    expected = torch.nn.Embedding(128, 4)
    if attrs:
        torch.nn.init.uniform_(expected.weight.data, attrs["init_min"], attrs["init_max"])
    torch.manual_seed(5101)
    actual = E3NNGraphBackend(registry).build(program, inference)
    actual_embedding = actual.node_modules["sphere_embedding"]
    torch.testing.assert_close(actual_embedding.weight, expected.weight, rtol=0.0, atol=0.0)

    categories = torch.tensor([1, 6, 6, 8, 79], dtype=torch.long)
    expected_output = expected(categories)
    actual_output = actual({"atomic_numbers": categories}, {})["out"]
    torch.testing.assert_close(actual_output, expected_output, rtol=0.0, atol=0.0)
    expected_output.square().sum().backward()
    actual_output.square().sum().backward()
    torch.testing.assert_close(actual_embedding.weight.grad, expected.weight.grad, rtol=0.0, atol=0.0)

    contract = inference.parameter_contracts["sphere_embedding"][0]
    assert contract.concrete_shape == (128, 4)
    assert contract.backend_parameter_name == "weight"
    assert contract.checkpoint_names == ("sphere_embedding.weight",)


def _distance_type():
    return InvariantTensorType(
        group=GroupSpec.so3(),
        carrier=Carrier.EDGE,
        irreps=Irreps.parse("1x0", "SO3"),
        frame=Frame("invariant"),
        measure="length",
        feature_role=FeatureRole.CHANNEL,
    )


def _fixed_rbf_program():
    source = _distance_type()
    output = InvariantTensorType(
        group=GroupSpec.so3(),
        carrier=Carrier.EDGE,
        irreps=Irreps.parse("8x0", "SO3"),
        frame=Frame("invariant"),
        axes=("radial_basis",),
        axis_specs=(AxisSpec("radial_basis", 8, FeatureRole.BASIS, "independent", 0),),
        feature_role=FeatureRole.BASIS,
    )
    return ArchitectureProgram(
        "1.0.0",
        "v3_fixed_gaussian",
        (InputPort("distance", source),),
        (
            Node(
                "distance_expansion",
                "core.fixed_gaussian_radial_basis",
                {"distance": ("input:distance",)},
                {
                    "start": 0.0,
                    "stop": 12.0,
                    "num_basis": 8,
                    "basis_width_scalar": 2.0,
                    "axis": "radial_basis",
                    "construction_dtype": "float32",
                },
            ),
        ),
        (OutputPort("out", "distance_expansion", output),),
    )


def test_v3_fixed_gaussian_smearing_matches_official_forward_and_input_gradient():
    radial = _load_plain_module("radial_function")
    expected = radial.GaussianSmearing(0.0, 12.0, 8, 2.0).double()
    registry = core_registry()
    program = _fixed_rbf_program()
    inference = TypeChecker(registry).check(program)
    actual = E3NNGraphBackend(registry).build(program, inference).double()

    distances_expected = torch.tensor([[0.0], [1.25], [5.5], [11.9], [12.5]], dtype=torch.float64, requires_grad=True)
    distances_actual = distances_expected.detach().clone().requires_grad_(True)
    expected_output = expected(distances_expected)
    actual_output = actual({"distance": distances_actual}, {})["out"]
    torch.testing.assert_close(actual_output, expected_output, rtol=0.0, atol=0.0)
    expected_output.square().sum().backward()
    actual_output.square().sum().backward()
    torch.testing.assert_close(distances_actual.grad, distances_expected.grad, rtol=1.0e-14, atol=1.0e-14)
    assert not list(actual.parameters())


def test_v3_polynomial_envelope_is_already_exactly_expressible_by_cutoff_envelope():
    envelope = _load_plain_module("envelope")
    expected = envelope.PolynomialEnvelope(cutoff=12.0, exponent=5).double()
    source = _distance_type()
    program = ArchitectureProgram(
        "1.0.0",
        "v3_polynomial_envelope",
        (InputPort("distance", source),),
        (
            Node(
                "envelope_func",
                "core.cutoff_envelope",
                {"x": ("input:distance",)},
                {"cutoff": 12.0, "order": 5},
            ),
        ),
        (OutputPort("out", "envelope_func", source),),
    )
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    actual = E3NNGraphBackend(registry).build(program, inference).double()
    distance = torch.tensor([[0.0], [3.0], [11.9], [12.0], [14.0]], dtype=torch.float64)
    torch.testing.assert_close(
        actual({"distance": distance}, {})["out"],
        expected(distance),
        rtol=0.0,
        atol=0.0,
    )


def test_v3_input_primitives_remain_registered_in_expanded_generic_registry():
    registry = core_registry()
    backend = E3NNGraphBackend(registry)
    assert len(registry.names()) == 102
    assert "core.categorical_embedding@1" in registry.names()
    assert "core.fixed_gaussian_radial_basis@1" in registry.names()
    assert backend.lowering_rules.audit()["rule_count"] == 102
