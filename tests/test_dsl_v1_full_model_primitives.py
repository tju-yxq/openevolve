from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest
import torch
from e3nn import o3

from equivariant_nas.dsl import (
    ArchitectureProgram,
    AxisSpec,
    Carrier,
    CategoricalTensorType,
    Compiler,
    DSLValidationError,
    EquivarianceLevel,
    EquivariantTensorType,
    FeatureRole,
    Frame,
    GroupSpec,
    InputPort,
    InvariantTensorType,
    Irreps,
    Node,
    OutputPort,
    core_registry,
    value_type_from_dict,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.backends.lowering import RuntimeValueKind, runtime_kind_for_value_type


ATOM_MAPPING = (-1, 0, -1, -1, -1, -1, 1, 2, 3, 4)


def _categorical_program():
    group = GroupSpec.o3()
    raw = CategoricalTensorType(group, Carrier.NODE, 10, labels=tuple(str(index) for index in range(10)))
    remapped = CategoricalTensorType(group, Carrier.NODE, 5)
    encoded = InvariantTensorType(
        group,
        Carrier.NODE,
        Irreps.parse("5x0e", group.family),
        frame=Frame("invariant"),
        axes=("species",),
        dtype="float32",
        level=EquivarianceLevel.CORE_CERTIFIED,
        axis_specs=(AxisSpec("species", 5, FeatureRole.SPECIES, "independent", 0),),
        feature_role=FeatureRole.SPECIES,
    )
    return ArchitectureProgram(
        "2.13.0",
        "v1-atom-categorical-flow",
        (InputPort("node_atom", raw),),
        (
            Node(
                "remap",
                "core.categorical_remap@1",
                {"x": ("input:node_atom",)},
                {"mapping": list(ATOM_MAPPING), "output_vocabulary_size": 5},
            ),
            Node(
                "one_hot",
                "core.categorical_one_hot@1",
                {"x": ("remap",)},
                {"axis": "species", "dtype": "float32"},
            ),
        ),
        (
            OutputPort("remapped", "remap", remapped),
            OutputPort("one_hot", "one_hot", encoded),
        ),
    )


def _gaussian_program(num_basis=6, cutoff=2.5):
    group = GroupSpec.o3()
    vector = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("1x1o", group.family),
        dtype="float32",
        measure="length",
    )
    distance = InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("1x0e", group.family),
        frame=Frame("invariant"),
        dtype="float32",
        measure="length",
    )
    radial = InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("{}x0e".format(num_basis), group.family),
        frame=Frame("invariant"),
        axes=("radial_channel",),
        dtype="float32",
        axis_specs=(AxisSpec("radial_channel", num_basis, FeatureRole.BASIS, "independent", 0),),
        feature_role=FeatureRole.BASIS,
    )
    return ArchitectureProgram(
        "2.13.0",
        "v1-learnable-gaussian-rbf",
        (InputPort("edge_vector", vector),),
        (
            Node("distance", "core.distance@2", {"vector": ("input:edge_vector",)}),
            Node(
                "rbf",
                "core.gaussian_radial_basis@1",
                {"distance": ("distance",)},
                {"num_basis": num_basis, "cutoff": cutoff, "axis": "radial_channel"},
            ),
        ),
        (
            OutputPort("distance", "distance", distance),
            OutputPort("out", "rbf", radial),
        ),
    )


def _atom_embedding_program():
    categorical = _categorical_program()
    raw = categorical.inputs[0].value_type
    one_hot = categorical.outputs[1].expected_type
    flattened = replace(one_hot, axes=(), axis_specs=())
    scalar_embedding = replace(
        flattened,
        irreps=Irreps.parse("4x0e", flattened.group.family),
    )
    mixed_embedding = EquivariantTensorType(
        raw.group,
        Carrier.NODE,
        Irreps.parse("4x0e+2x1e+1x2e", raw.group.family),
        dtype="float32",
        level=EquivarianceLevel.CORE_CERTIFIED,
    )
    return ArchitectureProgram(
        "2.13.0",
        "official-v1-atom-embedding",
        (InputPort("node_atom", raw),),
        (
            categorical.nodes[0],
            categorical.nodes[1],
            Node("flatten_species", "core.flatten_invariant_axes@1", {"x": ("one_hot",)}),
            Node(
                "embedding",
                "core.irrep_linear@2",
                {"x": ("flatten_species",)},
                {
                    "out_irreps": "4x0e",
                    "bias": True,
                    "rescale": True,
                    "initializer_scale": 5.0 ** 0.5,
                },
            ),
            Node(
                "pad",
                "core.irrep_pad@1",
                {"x": ("embedding",)},
                {"out_irreps": "4x0e+2x1e+1x2e"},
            ),
        ),
        (
            OutputPort("one_hot", "one_hot", one_hot),
            OutputPort("scalar_embedding", "embedding", scalar_embedding),
            OutputPort("node_embedding", "pad", mixed_embedding),
        ),
    )


def _load_official_gaussian_class():
    source = Path(__file__).resolve().parents[2] / "equiformer" / "nets" / "gaussian_rbf.py"
    spec = importlib.util.spec_from_file_location("official_equiformer_v1_gaussian_rbf", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GaussianRadialBasisLayer


def _load_official_v1_module():
    from scripts.audit_equiformer_v1_graph_attention_contract import _load_official_module

    root = Path(__file__).resolve().parents[2] / "equiformer"
    module, _source = _load_official_module(root, torch)
    return module


def test_categorical_type_roundtrip_and_runtime_kind_are_explicit():
    value_type = CategoricalTensorType(
        GroupSpec.o3(),
        Carrier.NODE,
        5,
        labels=("H", "C", "N", "O", "F"),
    )
    assert value_type_from_dict(value_type.to_dict()) == value_type
    assert runtime_kind_for_value_type(value_type) == RuntimeValueKind.CATEGORICAL_TENSOR


def test_v1_atom_mapping_and_one_hot_lower_without_an_embedding_constructor():
    program = _categorical_program()
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    backend = E3NNGraphBackend(registry)
    support = backend.support_report(artifact.expanded_program)
    model = backend.build(artifact.expanded_program, artifact.inference)

    assert support.supported
    assert dict(support.runtime_kinds)["input:node_atom"] == RuntimeValueKind.CATEGORICAL_TENSOR
    assert len(model.node_modules) == 0

    node_atom = torch.tensor([1, 6, 8, 7, 9], dtype=torch.long)
    outputs = model({"node_atom": node_atom}, {})
    expected_remapped = torch.tensor([0, 1, 3, 2, 4], dtype=torch.long)
    assert torch.equal(outputs["remapped"], expected_remapped)
    assert torch.equal(outputs["one_hot"], torch.nn.functional.one_hot(expected_remapped, 5).float())

    with pytest.raises(RuntimeError, match="rejected unsupported"):
        model({"node_atom": torch.tensor([2], dtype=torch.long)}, {})


def test_categorical_contract_rejects_wrong_mapping_shape_and_output_values():
    program = _categorical_program()
    wrong_length = replace(
        program,
        nodes=(replace(program.nodes[0], attrs={"mapping": [0, 1], "output_vocabulary_size": 5}), program.nodes[1]),
    )
    with pytest.raises(DSLValidationError) as length_error:
        Compiler(core_registry()).analyze(wrong_length)
    assert any(item.code == "E_CATEGORICAL_REMAP_003" for item in length_error.value.diagnostics)

    invalid_value = replace(
        program,
        nodes=(
            replace(
                program.nodes[0],
                attrs={"mapping": list(ATOM_MAPPING[:-1]) + [5], "output_vocabulary_size": 5},
            ),
            program.nodes[1],
        ),
    )
    with pytest.raises(DSLValidationError) as value_error:
        Compiler(core_registry()).analyze(invalid_value)
    assert any(item.code == "E_CATEGORICAL_REMAP_004" for item in value_error.value.diagnostics)


def test_v1_atom_embedding_matches_official_initialization_forward_and_parameter_gradients():
    registry = core_registry()
    program = _atom_embedding_program()
    artifact = Compiler(registry).analyze(program)
    official_module = _load_official_v1_module()
    seed = 20260731

    torch.manual_seed(seed)
    official = official_module.NodeEmbeddingNetwork(o3.Irreps("4x0e+2x1e+1x2e"), 5)
    torch.manual_seed(seed)
    dsl_model = E3NNGraphBackend(registry).build(program, artifact.inference)

    mapping = {
        "node_modules.embedding.tp.weight": "atom_type_lin.tp.weight",
        "node_modules.embedding.bias.0": "atom_type_lin.bias.0",
    }
    official_parameters = dict(official.named_parameters())
    dsl_parameters = dict(dsl_model.named_parameters())
    assert set(dsl_parameters) == set(mapping)
    for dsl_name, official_name in mapping.items():
        assert torch.equal(dsl_parameters[dsl_name], official_parameters[official_name])

    raw_atom = torch.tensor([1, 6, 8, 7, 9], dtype=torch.long)
    remapped_atom = torch.tensor([0, 1, 3, 2, 4], dtype=torch.long)
    official_embedding, official_attr, official_one_hot = official(remapped_atom)
    outputs = dsl_model({"node_atom": raw_atom}, {})
    assert torch.equal(outputs["one_hot"], official_one_hot)
    assert torch.equal(outputs["node_embedding"], official_embedding)
    assert torch.equal(official_attr, official_one_hot)
    assert torch.count_nonzero(outputs["node_embedding"][:, 4:]) == 0

    probe = torch.randn_like(official_embedding)
    (official_embedding * probe).sum().backward()
    (outputs["node_embedding"] * probe).sum().backward()
    for dsl_name, official_name in mapping.items():
        assert torch.equal(dsl_parameters[dsl_name].grad, official_parameters[official_name].grad)


def test_learnable_gaussian_rbf_matches_official_initialization_forward_and_gradients():
    registry = core_registry()
    program = _gaussian_program()
    artifact = Compiler(registry).analyze(program)
    seed = 20260731

    official_class = _load_official_gaussian_class()
    torch.manual_seed(seed)
    official = official_class(6, cutoff=2.5)
    torch.manual_seed(seed)
    dsl_model = E3NNGraphBackend(registry).build(program, artifact.inference)
    dsl_rbf = dsl_model.node_modules["rbf"]

    assert type(dsl_rbf).__module__.startswith("equivariant_nas.dsl.backends")
    assert not type(dsl_rbf).__module__.startswith("nets")
    assert set(dict(official.named_parameters())) == {"mean", "std", "weight", "bias"}
    assert set(dict(dsl_rbf.named_parameters())) == {"mean", "std", "weight", "bias"}
    for name, official_parameter in official.named_parameters():
        assert torch.equal(official_parameter, dict(dsl_rbf.named_parameters())[name])

    contracts = artifact.inference.parameter_contracts["rbf"]
    assert [item.name for item in contracts] == ["mean", "std", "weight", "bias"]
    assert [item.concrete_shape for item in contracts] == [(1, 6), (1, 6), (1, 1), (1, 1)]

    base_vector = torch.tensor(
        [[0.3, -0.2, 0.7], [1.0, 0.2, -0.4], [-0.5, 0.6, 0.9], [0.1, 0.2, 0.3]],
        dtype=torch.float32,
    )
    official_vector = base_vector.clone().requires_grad_(True)
    dsl_vector = base_vector.clone().requires_grad_(True)
    official_output = official(torch.linalg.vector_norm(official_vector, dim=-1))
    dsl_outputs = dsl_model({"edge_vector": dsl_vector}, {})
    dsl_output = dsl_outputs["out"]
    assert torch.equal(dsl_outputs["distance"], torch.linalg.vector_norm(dsl_vector, dim=-1, keepdim=True))
    assert torch.allclose(dsl_output, official_output, atol=1.0e-7, rtol=1.0e-7)

    official_targets = (official_vector,) + tuple(official.parameters())
    dsl_targets = (dsl_vector,) + tuple(dsl_rbf.parameters())
    official_gradients = torch.autograd.grad(
        official_output.square().sum() + official_output.sum(),
        official_targets,
    )
    dsl_gradients = torch.autograd.grad(
        dsl_output.square().sum() + dsl_output.sum(),
        dsl_targets,
    )
    for official_gradient, dsl_gradient in zip(official_gradients, dsl_gradients):
        assert torch.allclose(dsl_gradient, official_gradient, atol=1.0e-6, rtol=1.0e-6)


def test_constant_scale_preserves_type_and_applies_v1_edge_direction_sign():
    group = GroupSpec.o3()
    vector = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("1x1o", group.family),
        measure="length",
    )
    program = ArchitectureProgram(
        "2.13.0",
        "v1-source-minus-target-sign",
        (InputPort("x", vector),),
        (Node("negate", "core.constant_scale@1", {"x": ("input:x",)}, {"factor": -1.0}),),
        (OutputPort("out", "negate", vector),),
    )
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference)
    value = torch.randn(4, 3)
    assert torch.equal(model({"x": value}, {})["out"], -value)
