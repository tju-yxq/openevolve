from dataclasses import replace

import pytest

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
    expand_motifs,
    import_equiformer_v1,
    reference_motif_registry,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend, EquiformerV2GraphBackend, find_v2_fusion_patterns
from equivariant_nas.spec import baseline_spec


def test_v1_representation_flow_is_covered_by_node_backend_operations():
    program = expand_motifs(import_equiformer_v1(baseline_spec()), reference_motif_registry())
    report = E3NNGraphBackend(core_registry()).support_report(program)
    assert not report.unsupported_nodes
    # Local orchestration environments may intentionally omit e3nn.
    assert set(report.missing_dependencies).issubset({"torch", "e3nn"})


def test_v2_so2_path_is_rejected_until_its_numerical_backend_is_available():
    source = import_equiformer_v1(baseline_spec())
    # Presence of a V2 primitive is enough to prove support reporting is explicit.
    from dataclasses import replace
    changed_node = replace(source.nodes[1], op="core.so2_convolution")
    changed = replace(source, nodes=source.nodes[:1] + (changed_node,) + source.nodes[2:])
    report = E3NNGraphBackend(core_registry()).support_report(changed)
    assert (changed_node.id, "core.so2_convolution@1") in report.unsupported_nodes


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
    report = EquiformerV2GraphBackend(registry, str(tmp_path)).support_report(program, inference)
    assert not report.unsupported_nodes


def test_v2_fusion_rejects_an_intermediate_value_with_an_external_consumer(tmp_path):
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
    report = EquiformerV2GraphBackend(registry, str(tmp_path)).support_report(branched, inference)
    assert any(node_id == so2.id for node_id, _ in report.unsupported_nodes)
