import pytest
import torch

from equivariant_nas.dsl import (
    AffinePointType,
    ArchitectureProgram,
    Carrier,
    Compiler,
    DSLValidationError,
    EquivariantTensorType,
    GraphTopologyType,
    GroupSpec,
    IndexMapType,
    InputPort,
    Irreps,
    LatticeShiftType,
    LatticeType,
    Node,
    OutputPort,
    core_registry,
    value_type_from_dict,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.backends.lowering import RuntimeValueKind


def _endpoint_types(group, edge_count=None):
    source = IndexMapType(group, Carrier.NODE, Carrier.EDGE, "source", target_size=edge_count)
    target = IndexMapType(group, Carrier.NODE, Carrier.EDGE, "target", target_size=edge_count)
    return source, target


def _nonperiodic_program(group=None, dtype="float64"):
    group = group or GroupSpec.so3()
    positions = AffinePointType(group, dtype=dtype, measure="angstrom")
    source, target = _endpoint_types(group, 3)
    vector = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("1x1o" if group.family == "O3" else "1x1", group.family),
        dtype=dtype,
        measure="angstrom",
    )
    program = ArchitectureProgram(
        "2.1.0",
        "relative-displacement",
        (InputPort("positions", positions), InputPort("source_index", source), InputPort("target_index", target)),
        (
            Node(
                "displacement",
                "core.relative_displacement@2",
                {
                    "positions": ("input:positions",),
                    "source_index": ("input:source_index",),
                    "target_index": ("input:target_index",),
                },
            ),
        ),
        (OutputPort("vectors", "displacement", vector),),
    )
    return program, positions, source, target, vector


def _periodic_program(dtype="float64"):
    group = GroupSpec("SO3", 3, periodicity="lattice")
    positions = AffinePointType(group, dtype=dtype, measure="angstrom")
    source, target = _endpoint_types(group, 2)
    lattice = LatticeType(group, "cell-0", dtype=dtype, measure="angstrom")
    shift = LatticeShiftType(group, "cell-0")
    vector = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("1x1", group.family),
        dtype=dtype,
        measure="angstrom",
    )
    program = ArchitectureProgram(
        "2.1.0",
        "periodic-displacement",
        (
            InputPort("positions", positions),
            InputPort("source_index", source),
            InputPort("target_index", target),
            InputPort("lattice", lattice),
            InputPort("lattice_shift", shift),
        ),
        (
            Node(
                "displacement",
                "core.periodic_displacement",
                {
                    "positions": ("input:positions",),
                    "source_index": ("input:source_index",),
                    "target_index": ("input:target_index",),
                    "lattice": ("input:lattice",),
                    "lattice_shift": ("input:lattice_shift",),
                },
            ),
        ),
        (OutputPort("vectors", "displacement", vector),),
    )
    return program, positions, lattice, shift, vector


def _index_payload(indices, target_size):
    return {"indices": torch.as_tensor(indices, dtype=torch.long), "target_size": target_size}


def test_affine_lattice_and_shift_types_roundtrip_and_enforce_contracts():
    periodic = GroupSpec("SO3", 3, periodicity="lattice")
    affine = AffinePointType(periodic, dtype="float64", measure="angstrom")
    lattice = LatticeType(periodic, "cell", dtype="float64", measure="angstrom")
    shift = LatticeShiftType(periodic, "cell")
    source, target = _endpoint_types(periodic, 5)
    topology = GraphTopologyType(periodic, source, target, periodic_mapping="lattice_shift")

    assert value_type_from_dict(affine.to_dict()) == affine
    assert value_type_from_dict(lattice.to_dict()) == lattice
    assert value_type_from_dict(shift.to_dict()) == shift
    assert value_type_from_dict(topology.to_dict()) == topology

    with pytest.raises(DSLValidationError) as dimension:
        AffinePointType(GroupSpec.so3(), coordinate_dimension=2)
    assert any(item.code == "E_AFFINE_002" for item in dimension.value.diagnostics)

    with pytest.raises(DSLValidationError) as nonperiodic:
        LatticeType(GroupSpec.so3(), "cell")
    assert any(item.code == "E_LATTICE_001" for item in nonperiodic.value.diagnostics)


@pytest.mark.parametrize(
    "action",
    (
        torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
        torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)),
    ),
)
def test_relative_displacement_is_translation_invariant_and_o3_covariant(action):
    program, _positions_type, _source_type, _target_type, vector_type = _nonperiodic_program(GroupSpec.o3())
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    backend = E3NNGraphBackend(registry)
    report = backend.support_report(program)
    model = backend.build(program, artifact.inference)

    assert report.supported
    kinds = dict(report.runtime_kinds)
    assert kinds["input:positions"] == RuntimeValueKind.AFFINE_POINT
    assert kinds["input:source_index"] == RuntimeValueKind.INDEX_MAP
    assert kinds["displacement"] == RuntimeValueKind.DENSE_TENSOR
    assert str(vector_type.irreps) == "1x1o"

    positions = torch.tensor(
        [[0.2, -0.1, 0.4], [1.1, 0.3, -0.2], [-0.4, 0.8, 0.5], [0.6, -0.7, 1.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    source = _index_payload([0, 1, 3], 3)
    target = _index_payload([1, 2, 0], 3)
    inputs = {"positions": positions, "source_index": source, "target_index": target}
    reference = model(inputs, {})["vectors"]
    translated = model({**inputs, "positions": positions + torch.tensor([3.0, -2.0, 5.0])}, {})["vectors"]
    transformed = model({**inputs, "positions": positions @ action.T}, {})["vectors"]

    expected = positions.index_select(0, target["indices"]) - positions.index_select(0, source["indices"])
    assert torch.allclose(reference, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(translated, reference, atol=1e-12, rtol=1e-12)
    assert torch.allclose(transformed, reference @ action.T, atol=1e-12, rtol=1e-12)

    reference.square().sum().backward()
    assert positions.grad is not None
    assert torch.allclose(positions.grad.sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-12, rtol=0.0)


def test_periodic_displacement_uses_explicit_target_image_and_is_differentiable():
    program, _positions_type, _lattice_type, _shift_type, _vector_type = _periodic_program()
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    backend = E3NNGraphBackend(registry)
    report = backend.support_report(program)
    model = backend.build(program, artifact.inference)

    assert report.supported
    kinds = dict(report.runtime_kinds)
    assert kinds["input:lattice"] == RuntimeValueKind.LATTICE
    assert kinds["input:lattice_shift"] == RuntimeValueKind.LATTICE_SHIFT

    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.2, -0.4, 0.8], [-0.2, 1.1, 0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    lattice = torch.tensor(
        [[2.0, 0.0, 0.0], [0.3, 1.8, 0.0], [0.1, 0.2, 2.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    source = _index_payload([0, 1], 2)
    target = _index_payload([1, 2], 2)
    shifts = torch.tensor([[1, 0, 0], [0, -1, 1]], dtype=torch.long)
    inputs = {
        "positions": positions,
        "source_index": source,
        "target_index": target,
        "lattice": lattice,
        "lattice_shift": shifts,
    }
    output = model(inputs, {})["vectors"]
    expected = (
        positions.index_select(0, target["indices"])
        + shifts.to(torch.float64) @ lattice
        - positions.index_select(0, source["indices"])
    )
    translated = model({**inputs, "positions": positions + torch.tensor([7.0, -3.0, 2.0])}, {})["vectors"]
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    rotated = model({**inputs, "positions": positions @ rotation.T, "lattice": lattice @ rotation.T}, {})["vectors"]

    assert torch.allclose(output, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(translated, output, atol=1e-12, rtol=1e-12)
    assert torch.allclose(rotated, output @ rotation.T, atol=1e-12, rtol=1e-12)

    output.square().sum().backward()
    assert positions.grad is not None
    assert lattice.grad is not None
    assert torch.isfinite(positions.grad).all()
    assert torch.isfinite(lattice.grad).all()


def test_periodic_image_relabeling_leaves_displacement_unchanged():
    program, *_ = _periodic_program()
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(program, artifact.inference)
    positions = torch.tensor([[0.1, 0.2, 0.3], [1.2, -0.4, 0.8]], dtype=torch.float64)
    lattice = torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    source = _index_payload([0], 1)
    target = _index_payload([1], 1)
    shift = torch.tensor([[1, -1, 0]], dtype=torch.long)
    base = model({
        "positions": positions,
        "source_index": source,
        "target_index": target,
        "lattice": lattice,
        "lattice_shift": shift,
    }, {})["vectors"]

    relabel = torch.tensor([1, 0, -1], dtype=torch.long)
    relabeled_positions = positions.clone()
    relabeled_positions[1] = relabeled_positions[1] + relabel.to(torch.float64) @ lattice
    relabeled = model({
        "positions": relabeled_positions,
        "source_index": source,
        "target_index": target,
        "lattice": lattice,
        "lattice_shift": shift - relabel,
    }, {})["vectors"]
    assert torch.allclose(relabeled, base, atol=1e-12, rtol=1e-12)


def test_periodic_displacement_rejects_mismatched_lattice_identity():
    program, positions, lattice, _shift, vector = _periodic_program()
    wrong_shift = LatticeShiftType(lattice.group, "other-cell")
    bad = ArchitectureProgram(
        program.language_version,
        program.task_contract,
        tuple(
            InputPort(item.name, wrong_shift if item.name == "lattice_shift" else item.value_type)
            for item in program.inputs
        ),
        program.nodes,
        (OutputPort("vectors", "displacement", vector),),
    )
    with pytest.raises(DSLValidationError) as error:
        Compiler(core_registry()).analyze(bad)
    assert any(item.code == "E_LATTICE_015" for item in error.value.diagnostics)
