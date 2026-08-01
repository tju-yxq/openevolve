import sys
import types

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from equivariant_nas.dsl import (  # noqa: E402
    ArchitectureProgram,
    Carrier,
    Compiler,
    EquivariantType,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    core_registry,
)
from equivariant_nas.dsl.backends import BackendSupportReport, E3NNGraphBackend  # noqa: E402
from equivariant_nas.dsl.backends.qm9_model import build_qm9_dsl_model  # noqa: E402
from equivariant_nas.dsl.diagnostics import DSLValidationError  # noqa: E402


def _generic_scalar_program():
    group = GroupSpec.so3()
    node_input = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0", "SO3"))
    node_scalar = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0", "SO3"))
    graph_scalar = EquivariantType(group, Carrier.GRAPH, Irreps.parse("1x0", "SO3"))
    return ArchitectureProgram(
        "1.0.0",
        "qm9_alpha",
        (InputPort("node_features", node_input),),
        (
            Node(
                "project",
                "core.irrep_linear",
                {"x": ("input:node_features",)},
                {"out_irreps": "1x0"},
                declared_types={"out": node_scalar},
            ),
            Node(
                "pool",
                "core.global_pool",
                {"x": ("project",)},
                {"reduce": "sum"},
                declared_types={"out": graph_scalar},
            ),
        ),
        (OutputPort("prediction", "pool", graph_scalar),),
        annotations={},
    )


def _install_empty_radius_graph(monkeypatch):
    module = types.ModuleType("torch_cluster")

    def radius_graph(pos, r, batch, max_num_neighbors):
        del r, batch, max_num_neighbors
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)

    module.radius_graph = radius_graph
    monkeypatch.setitem(sys.modules, "torch_cluster", module)


def test_generic_qm9_lowering_is_rejected_by_default():
    program = _generic_scalar_program()
    compiler = Compiler(core_registry())

    with pytest.raises(DSLValidationError) as captured:
        build_qm9_dsl_model(program, compiler)

    assert captured.value.diagnostics[0].code == "E_QM9_BACKEND_007"


def test_explicit_generic_qm9_lowering_builds_runs_and_backpropagates(monkeypatch):
    _install_empty_radius_graph(monkeypatch)
    program = _generic_scalar_program()
    compiler = Compiler(core_registry())
    delegate = E3NNGraphBackend(compiler.primitives)

    class TrackingBackend:
        def __init__(self):
            self.support_report_called = False

        def support_report(self, expanded_program):
            self.support_report_called = True
            return delegate.support_report(expanded_program)

        def build(self, expanded_program, inference):
            assert self.support_report_called
            return delegate.build(expanded_program, inference)

    backend = TrackingBackend()
    model = build_qm9_dsl_model(
        program,
        compiler,
        graph_backend=backend,
        allow_experimental_generic_lowering=True,
    ).double()
    features = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    positions = torch.randn(4, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    prediction = model(features, positions, batch)
    prediction.square().sum().backward()

    assert backend.support_report_called
    assert prediction.shape == (2, 1)
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert model.lowering_mode == "experimental_node_graph"
    assert model.lowering_plan["details"]["backend_support"]["supported"] is True
    assert model.lowering_plan["details"]["rule_exactness"] == {
        "pool": "constructive_exact",
        "project": "library_exact",
    }
    assert model.lowering_plan["details"]["uncertified_nodes"] == []
    assert model.generic_lowering_admission["explicitly_enabled"] is True
    assert model.generic_lowering_admission["formal_ranking_admitted"] is False
    assert model.lowering_rule_manifest["rule_count"] == 102

    model.eval()
    reference = model(features.detach(), positions, batch)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    rotated = model(features.detach(), positions @ rotation.transpose(0, 1), batch)
    translated = model(
        features.detach(),
        positions + torch.tensor([[1.5, -0.5, 2.0]], dtype=torch.float64),
        batch,
    )
    permutation = torch.tensor([1, 0, 3, 2], dtype=torch.long)
    permuted = model(
        features.detach().index_select(0, permutation),
        positions.index_select(0, permutation),
        batch.index_select(0, permutation),
    )
    assert torch.allclose(rotated, reference, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(translated, reference, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(permuted, reference, atol=1.0e-12, rtol=1.0e-12)


def test_generic_qm9_preflight_rejects_backend_support_failures_before_build():
    program = _generic_scalar_program()
    compiler = Compiler(core_registry())

    class UnsupportedBackend:
        def support_report(self, expanded_program):
            del expanded_program
            return BackendSupportReport(
                "unsupported_probe",
                False,
                (("project", "core.irrep_linear@1"),),
                ("missing_probe_dependency",),
            )

        def build(self, expanded_program, inference):
            del expanded_program, inference
            raise AssertionError("build must not run after a failed support report")

    with pytest.raises(DSLValidationError) as captured:
        build_qm9_dsl_model(
            program,
            compiler,
            graph_backend=UnsupportedBackend(),
            allow_experimental_generic_lowering=True,
        )

    diagnostic = captured.value.diagnostics[0]
    assert diagnostic.code == "E_QM9_BACKEND_009"
    assert diagnostic.details["backend"] == "unsupported_probe"
    assert diagnostic.details["unsupported_nodes"] == [["project", "core.irrep_linear@1"]]
    assert diagnostic.details["missing_dependencies"] == ["missing_probe_dependency"]
