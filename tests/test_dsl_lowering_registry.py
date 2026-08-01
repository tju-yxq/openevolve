import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from equivariant_nas.dsl import (  # noqa: E402
    ArchitectureProgram,
    Carrier,
    Compiler,
    DSLValidationError,
    EquivariantType,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    TypeChecker,
    core_registry,
)
from equivariant_nas.dsl.backends import (  # noqa: E402
    BackendSupportReport,
    E3NNGraphBackend,
    LoweringRuleRegistry,
)
from equivariant_nas.dsl.backends.e3nn_backend import (  # noqa: E402
    _SUPPORTED,
    build_e3nn_lowering_registry,
)


def _single_node_program(op, value_type, *, attrs=None, input_name="x", node_port="x"):
    return ArchitectureProgram(
        "1.0.0",
        "lowering_rule_probe",
        (InputPort(input_name, value_type),),
        (Node("node", op, {node_port: ("input:{}".format(input_name),)}, attrs or {}),),
        (OutputPort("out", "node", value_type),),
    )


def test_lowering_registry_is_the_single_supported_primitive_manifest():
    rules = build_e3nn_lowering_registry()
    manifest = rules.audit()

    assert isinstance(rules, LoweringRuleRegistry)
    assert set(rules.names()) == set(_SUPPORTED)
    assert manifest["rule_count"] == 102
    assert "core.endpoint_gather@1" in rules.names()
    assert "core.irrep_linear@2" in rules.names()
    assert "core.irrep_layer_norm@1" in rules.names()
    assert "core.segment_reduce@1" in rules.names()
    assert "core.relative_displacement@2" in rules.names()
    assert "core.periodic_displacement@1" in rules.names()
    assert "core.scalar_linear@1" in rules.names()
    assert "core.scalar_linear@2" in rules.names()
    assert "core.scalar_layer_norm@1" in rules.names()
    assert "core.scalar_offset@1" in rules.names()
    assert "core.head_split@1" in rules.names()
    assert "core.head_merge@1" in rules.names()
    assert "core.headwise_scalar_contraction@1" in rules.names()
    assert "core.head_split@2" in rules.names()
    assert "core.head_merge@2" in rules.names()
    assert "core.headwise_scalar_contraction@2" in rules.names()
    assert "core.irrep_select@2" in rules.names()
    assert "core.invariant_scale@1" in rules.names()
    assert "core.segment_softmax@2" in rules.names()
    assert "core.tensor_product@2" in rules.names()
    assert "core.tensor_product@3" in rules.names()
    assert "core.tensor_product@4" in rules.names()
    assert sum(manifest["exactness_counts"].values()) == 102
    assert all(item["primitive"] in _SUPPORTED for item in manifest["rules"])
    assert all(item["exactness"] for item in manifest["rules"])
    assert all(item["has_executor"] for item in manifest["rules"])


def test_generic_lowering_plan_consumes_support_report_and_records_certification():
    group = GroupSpec.so3()
    scalar = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0", "SO3"))
    program = _single_node_program(
        "core.scalar_activation",
        scalar,
        attrs={"activation": "silu"},
    )
    plan = Compiler(core_registry()).plan_lowering(program)

    assert plan.mode == "experimental_node_graph"
    assert plan.backend_family == "e3nn_graph"
    assert plan.unsupported_nodes == ()
    assert plan.details["backend_support"]["supported"] is True
    assert plan.details["rule_exactness"] == {"node": "numerically_tested"}
    assert plan.details["certified_nodes"] == []
    assert plan.details["uncertified_nodes"] == ["node"]


def test_generic_lowering_plan_records_unsupported_nodes_and_missing_dependencies():
    group = GroupSpec.so3()
    scalar = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0", "SO3"))
    program = _single_node_program("core.identity", scalar)

    class UnsupportedBackend:
        semantic_version = "unsupported-probe@1"

        def support_report(self, expanded_program):
            del expanded_program
            return BackendSupportReport(
                "unsupported_probe",
                False,
                (("node", "core.identity@1"),),
                ("missing_probe",),
            )

    plan = Compiler(core_registry()).plan_lowering(
        program,
        graph_backend=UnsupportedBackend(),
    )

    assert plan.backend_family == "unsupported_probe"
    assert plan.backend_semantics_version == "unsupported-probe@1"
    assert plan.unsupported_nodes == ("node",)
    assert plan.details["backend_support"]["supported"] is False
    assert plan.details["backend_support"]["missing_dependencies"] == ["missing_probe"]


def test_segment_softmax_has_no_hidden_torch_geometric_dependency():
    group = GroupSpec.so3()
    scalar = EquivariantType(group, Carrier.EDGE, Irreps.parse("1x0", "SO3"))
    program = _single_node_program("core.segment_softmax", scalar, input_name="logits", node_port="logits")
    backend = E3NNGraphBackend(core_registry())
    report = backend.support_report(program)

    assert "torch_geometric" not in report.missing_dependencies
    model = backend.build(program, TypeChecker(core_registry()).check(program)).double()
    logits = torch.tensor([[0.2], [1.1], [-0.4], [0.3]], dtype=torch.float64, requires_grad=True)
    edge_dst = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    output = model({"logits": logits}, {"edge_dst": edge_dst})["out"]
    sums = torch.zeros(2, dtype=torch.float64)
    sums.index_add_(0, edge_dst, output[:, 0])
    output.square().sum().backward()

    assert torch.allclose(sums, torch.ones_like(sums), atol=1.0e-12, rtol=1.0e-12)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_cutoff_envelope_has_real_smooth_compact_support_and_gradient():
    group = GroupSpec.so3()
    scalar = EquivariantType(group, Carrier.EDGE, Irreps.parse("1x0", "SO3"))
    program = _single_node_program(
        "core.cutoff_envelope",
        scalar,
        attrs={"cutoff": 2.0, "order": 5},
    )
    registry = core_registry()
    model = E3NNGraphBackend(registry).build(program, TypeChecker(registry).check(program)).double()
    values = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float64, requires_grad=True)
    output = model({"x": values}, {})["out"]

    assert torch.allclose(output[0], torch.ones_like(output[0]))
    assert 0.0 < float(output[1].detach()) < 1.0
    assert torch.equal(output[2:], torch.zeros_like(output[2:]))
    output.sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()

    wider_program = _single_node_program(
        "core.cutoff_envelope",
        scalar,
        attrs={"cutoff": 4.0, "order": 5},
    )
    wider_model = E3NNGraphBackend(registry).build(
        wider_program,
        TypeChecker(registry).check(wider_program),
    ).double()
    wider_output = wider_model({"x": values.detach()}, {})["out"]
    assert not torch.allclose(wider_output[1], output.detach()[1])
    assert 0.0 < float(wider_output[2]) < 1.0


@pytest.mark.parametrize(
    ("attrs", "code"),
    (
        ({"cutoff": 0.0, "order": 5}, "E_BACKEND_010"),
        ({"cutoff": 2.0, "order": 0}, "E_BACKEND_011"),
    ),
)
def test_cutoff_envelope_rejects_invalid_static_parameters(attrs, code):
    group = GroupSpec.so3()
    scalar = EquivariantType(group, Carrier.EDGE, Irreps.parse("1x0", "SO3"))
    program = _single_node_program("core.cutoff_envelope", scalar, attrs=attrs)
    registry = core_registry()

    with pytest.raises(DSLValidationError) as captured:
        E3NNGraphBackend(registry).build(program, TypeChecker(registry).check(program))

    assert captured.value.diagnostics[0].code == code


def test_compiled_model_records_the_exact_lowering_rule_manifest():
    group = GroupSpec.so3()
    scalar = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0", "SO3"))
    output_type = EquivariantType(group, Carrier.NODE, Irreps.parse("3x0", "SO3"))
    program = ArchitectureProgram(
        "1.0.0",
        "manifest_probe",
        (InputPort("x", scalar),),
        (Node("linear", "core.irrep_linear", {"x": ("input:x",)}, {"out_irreps": "3x0"}),),
        (OutputPort("out", "linear", output_type),),
    )
    registry = core_registry()
    model = E3NNGraphBackend(registry).build(program, TypeChecker(registry).check(program))

    assert model.backend_semantics_version == "e3nn-graph-lowering-registry-v23"
    assert model.lowering_rule_manifest["rule_count"] == 102
    assert any(
        item["primitive"] == "core.irrep_linear@1" and item["has_module_builder"]
        for item in model.lowering_rule_manifest["rules"]
    )
