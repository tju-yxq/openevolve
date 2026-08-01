from pathlib import Path

from equivariant_nas.dsl import (
    ArchitectureProgram,
    Carrier,
    EquivariantType,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    TypeChecker,
    core_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.backends.lowering import RuntimeValueKind


def _v2_root():
    return str(Path(__file__).resolve().parents[2] / "equiformer_v3")


def _edge_type(channels=2):
    group = GroupSpec.so3()
    return EquivariantType(
        group,
        Carrier.EDGE,
        Irreps.parse("{0}x0+{0}x1+{0}x2".format(channels), "SO3"),
    )


def test_support_report_tracks_edge_frame_runtime_values_through_a_legal_residual():
    edge_type = _edge_type()
    program = ArchitectureProgram(
        "1.0.0",
        "runtime_kind_residual",
        (InputPort("x", edge_type),),
        (
            Node("to", "core.to_edge_frame", {"x": ("input:x",)}, {"frame_id": "bond"}),
            Node("copy", "core.identity", {"x": ("to",)}),
            Node("sum", "core.residual_add", {"left": ("to",), "right": ("copy",)}),
            Node("back", "core.from_edge_frame", {"x": ("sum",)}, {"frame_id": "bond"}),
        ),
        (OutputPort("out", "back", edge_type),),
    )
    registry = core_registry()
    TypeChecker(registry).check(program)
    report = E3NNGraphBackend(registry, equiformer_v2_root=_v2_root()).support_report(program)
    kinds = dict(report.runtime_kinds)

    assert not report.composition_errors
    assert kinds["to"] == RuntimeValueKind.SO3_EDGE_FRAME
    assert kinds["sum"] == RuntimeValueKind.SO3_EDGE_FRAME
    assert kinds["back"] == RuntimeValueKind.DENSE_TENSOR


def test_support_report_rejects_dense_only_concat_on_edge_frame_values():
    edge_type = _edge_type()
    concatenated_type = _edge_type(channels=4)
    program = ArchitectureProgram(
        "1.0.0",
        "runtime_kind_concat_negative",
        (InputPort("x", edge_type),),
        (
            Node("to", "core.to_edge_frame", {"x": ("input:x",)}, {"frame_id": "bond"}),
            Node("concat", "core.irrep_concat", {"xs": ("to", "to")}),
            Node("back", "core.from_edge_frame", {"x": ("concat",)}, {"frame_id": "bond"}),
        ),
        (OutputPort("out", "back", concatenated_type),),
    )
    registry = core_registry()
    TypeChecker(registry).check(program)
    report = E3NNGraphBackend(registry, equiformer_v2_root=_v2_root()).support_report(program)

    assert not report.supported
    assert any(node_id == "concat" for node_id, _message in report.composition_errors)
