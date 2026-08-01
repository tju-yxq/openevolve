from dataclasses import replace
from pathlib import Path

import pytest

from equivariant_nas.dsl import (
    ArchitectureProgram,
    Carrier,
    Compiler,
    EquivariantType,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    TypeChecker,
    core_registry,
    expand_motifs,
    import_equiformer_v1,
    reference_motif_registry,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend, EquiformerV2GraphBackend, find_v2_fusion_patterns
from equivariant_nas.dsl.backends import baseline_spec


def test_v1_representation_flow_is_covered_by_node_backend_operations():
    program = expand_motifs(import_equiformer_v1(baseline_spec()), reference_motif_registry())
    report = E3NNGraphBackend(core_registry()).support_report(program)
    assert not report.unsupported_nodes
    # Local orchestration environments may intentionally omit e3nn.
    assert set(report.missing_dependencies).issubset({"torch", "e3nn"})
    assert dict(report.node_exactness)
    assert set(dict(report.node_exactness)) == {node.id for node in program.nodes}


def test_v2_so2_path_reports_its_reference_dependency_when_root_is_missing():
    changed = _expanded_v2_program()
    report = E3NNGraphBackend(core_registry()).support_report(changed)
    assert not report.unsupported_nodes
    assert "equiformer_v2_reference" in report.missing_dependencies


def _expanded_v2_program():
    group = GroupSpec.so3()
    hidden = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0+2x1+2x2", "SO3"))
    source = ArchitectureProgram(
        "1.0.0",
        "v2_backend",
        (InputPort("x", hidden),),
        (
            Node(
                "v2",
                "motif.v2_so2_residual_message",
                {"x": ("input:x",)},
                {"hidden_irreps": str(hidden.irreps), "frame_id": "v2_edge"},
            ),
        ),
        (OutputPort("out", "v2", hidden),),
    )
    return expand_motifs(source, reference_motif_registry())


def test_v2_graph_backend_recognizes_only_the_closed_edge_frame_pattern(tmp_path):
    program = _expanded_v2_program()
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    patterns = find_v2_fusion_patterns(program, inference)
    assert len(patterns) == 1
    assert tuple(program.nodes[index].id for index in range(1, 6)) == patterns[0].node_ids

    package = tmp_path / "nets" / "equiformer_v2"
    package.mkdir(parents=True)
    for name in ("so3.py", "so2_ops.py", "edge_rot_mat.py", "activation.py"):
        (package / name).write_text("", encoding="utf-8")
    backend = EquiformerV2GraphBackend(registry, str(tmp_path))
    report = backend.support_report(program, inference)
    assert not report.unsupported_nodes
    exactness = dict(report.node_exactness)
    assert all(exactness[node_id] == "library_exact_fusion" for node_id in patterns[0].node_ids)
    plan = Compiler(registry).plan_lowering(program, graph_backend=backend)
    assert plan.backend_family == "equiformer_v2_graph"
    assert plan.details["backend_support"]["supported"] is True
    assert set(patterns[0].node_ids) <= set(plan.details["certified_nodes"])


def test_v2_backend_falls_back_to_independent_rules_for_an_external_consumer():
    program = _expanded_v2_program()
    registry = core_registry()
    base_inference = TypeChecker(registry).check(program)
    so2 = next(node for node in program.nodes if node.op == "core.so2_convolution")
    tap = Node("external_tap", "core.identity", {"x": (so2.id,)})
    branched = replace(
        program,
        nodes=program.nodes + (tap,),
        outputs=program.outputs + (OutputPort("tap", "external_tap", base_inference.value_types[so2.id]),),
    )
    inference = TypeChecker(registry).check(branched)
    assert not find_v2_fusion_patterns(branched, inference)
    root = str(Path(__file__).resolve().parents[2] / "equiformer_v3")
    if not Path(root).is_dir():
        pytest.skip("Equiformer V2 reference source is unavailable")
    report = EquiformerV2GraphBackend(registry, root).support_report(branched, inference)
    assert not report.unsupported_nodes
    assert report.supported
