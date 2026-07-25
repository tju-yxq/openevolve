from dataclasses import replace

from equivariant_nas.dsl import ArchitectureProgram, Carrier, EquivariantType, GroupSpec, InputPort, Irreps, Node, OutputPort, architecture_id, core_registry


def make_program(first_name, second_name, reverse=False):
    group = GroupSpec.o3()
    value_type = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0e", "O3"))
    first = Node(first_name, "core.identity", {"x": ("input:x",)})
    second = Node(second_name, "core.identity", {"x": (first_name,)})
    nodes = (second, first) if reverse else (first, second)
    return ArchitectureProgram(
        "1.0.0",
        "canonical",
        (InputPort("x", value_type),),
        nodes,
        (OutputPort("out", second_name, value_type),),
    )


def test_architecture_id_ignores_node_names_and_source_order():
    registry = core_registry()
    left = make_program("human_readable_a", "human_readable_b")
    right = make_program("renamed_1", "renamed_2", reverse=True)
    assert architecture_id(left, registry) == architecture_id(right, registry)
