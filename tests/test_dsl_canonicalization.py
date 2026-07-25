from dataclasses import replace

from equivariant_nas.dsl import ArchitectureProgram, Carrier, Compiler, EquivariantType, GroupSpec, InputPort, Irreps, Node, OutputPort, apply_strict_rewrites, architecture_id, core_registry


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


def test_identity_elimination_has_a_proof_trace_and_reaches_a_fixed_point():
    registry = core_registry()
    with_identities = make_program("identity_a", "identity_b")
    direct = replace(with_identities, nodes=(), outputs=(replace(with_identities.outputs[0], source="input:x"),))
    result = apply_strict_rewrites(with_identities)
    assert result.program == direct
    assert [step.rule_id for step in result.trace] == ["core.eliminate_identity", "core.eliminate_identity"]
    assert all(step.equivalence == "exact_function" for step in result.trace)
    assert apply_strict_rewrites(result.program).trace == ()
    assert architecture_id(with_identities, registry) == architecture_id(direct, registry)


def test_residual_operands_are_canonicalized_but_concat_order_is_not_assumed_commutative():
    group = GroupSpec.o3()
    scalar = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0e", "O3"))
    vector = EquivariantType(group, Carrier.NODE, Irreps.parse("1x1o", "O3"))
    residual_left = ArchitectureProgram(
        "1.0.0",
        "canonical",
        (InputPort("a", scalar), InputPort("b", scalar)),
        (Node("sum", "core.residual_add", {"left": ("input:a",), "right": ("input:b",)}),),
        (OutputPort("out", "sum", scalar),),
    )
    residual_right = replace(
        residual_left,
        nodes=(Node("sum", "core.residual_add", {"left": ("input:b",), "right": ("input:a",)}),),
    )
    assert architecture_id(residual_left, core_registry()) == architecture_id(residual_right, core_registry())
    trace_count = len(apply_strict_rewrites(residual_left).trace) + len(apply_strict_rewrites(residual_right).trace)
    assert trace_count == 1

    concat_left = ArchitectureProgram(
        "1.0.0",
        "canonical",
        (InputPort("a", scalar), InputPort("b", vector)),
        (Node("concat", "core.irrep_concat", {"xs": ("input:a", "input:b")}),),
        (OutputPort("out", "concat", scalar.with_irreps(Irreps.parse("1x0e+1x1o", "O3"))),),
    )
    concat_right = replace(
        concat_left,
        nodes=(Node("concat", "core.irrep_concat", {"xs": ("input:b", "input:a")}),),
    )
    assert architecture_id(concat_left, core_registry()) != architecture_id(concat_right, core_registry())


def test_compiler_exposes_rewrite_trace_and_registry_version():
    artifact = Compiler(core_registry()).analyze(make_program("first", "second"))
    assert len(artifact.rewrite_trace) == 2
    assert artifact.rewrite_registry_hash
    assert artifact.expanded_program.nodes == ()
