from dataclasses import replace
from pathlib import Path

import pytest

from equivariant_nas.dsl import (
    DSLValidationError,
    EquivarianceLevel,
    TypeChecker,
    canonicalize,
    core_registry,
    equiformer_v3_backbone_program,
    equiformer_v3_direct_model_program,
    equiformer_v3_energy_head_program,
    equiformer_v3_energy_model_program,
    equiformer_v3_force_head_program,
    equiformer_v3_transblock_program,
)
from equivariant_nas.dsl.backends.equiformer_v3_spec import EquiformerV3Spec
from equivariant_nas.dsl.backends import E3NNGraphBackend
from equivariant_nas.dsl.backends.lowering import RuntimeValueKind
from equivariant_nas.dsl.backends.v3_runtime import equiformer_v3_source_available


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = REPOSITORY_ROOT.parent / "equiformer_v3_official"


def _spec():
    return EquiformerV3Spec(
        use_pbc=False,
        num_radial_basis=8,
        num_channels=3,
        attn_hidden_channels=2,
        num_heads=2,
        attn_alpha_channels=2,
        attn_value_channels=1,
        ffn_hidden_channels=4,
        edge_channels=4,
        lmax=2,
        mmax=1,
        attn_grid_resolution=(8, 4),
        ffn_grid_resolution=(8, 8),
        drop_path_rate=0.1,
        proj_drop=0.1,
        ffn_activation="sep-merge_gates2_swiglu",
        attn_activation="sep-merge_gates2_swiglu",
        norm_type="merge_layer_norm",
    )


def test_v3_transblock_is_expanded_from_typed_subprograms():
    registry = core_registry()
    program = equiformer_v3_transblock_program(_spec())
    inference = TypeChecker(registry).check(program)
    canonical = canonicalize(program, registry)
    TypeChecker(registry).check(canonical)

    operations = {node.op for node in program.nodes}
    assert "core.equivariant_merge_norm@1" in operations
    assert "core.graph_stochastic_depth@1" in operations
    assert "core.equivariant_dropout@1" in operations
    assert "core.grid_project@1" in operations
    assert "core.grid_unproject@1" in operations
    assert inference.value_types["ffn_residual"].level == EquivarianceLevel.EMPIRICAL
    assert program.annotations["constructor_bypass"] is False


def test_v3_transblock_has_complete_generic_lowering_support():
    registry = core_registry()
    program = equiformer_v3_transblock_program(_spec())
    report = E3NNGraphBackend(registry).support_report(program)

    assert report.unsupported_nodes == ()
    runtime = dict(report.runtime_kinds)
    assert runtime["attn_drop_path"] == RuntimeValueKind.DENSE_TENSOR
    assert runtime["ffn_so3_linear2"] == RuntimeValueKind.DENSE_TENSOR
    assert runtime["ffn_residual"] == RuntimeValueKind.DENSE_TENSOR


def test_v3_transblock_requires_explicit_batch_for_graph_drop_path():
    program = equiformer_v3_transblock_program(_spec())
    assert any(port.name == "batch" for port in program.inputs)
    for node in program.nodes:
        if node.op == "core.graph_stochastic_depth@1":
            assert node.inputs["batch"] == ("input:batch",)


def test_v3_backbone_expands_input_and_every_transformer_block():
    registry = core_registry()
    spec = replace(_spec(), num_layers=2)
    program = equiformer_v3_backbone_program(spec)
    inference = TypeChecker(registry).check(program)
    canonical = canonicalize(program, registry)
    TypeChecker(registry).check(canonical)
    report = E3NNGraphBackend(registry).support_report(program)

    assert len(program.nodes) == 24 + 2 * 62
    assert program.outputs[0].source == "block1_ffn_residual"
    assert inference.value_types["block1_ffn_residual"].level == EquivarianceLevel.EMPIRICAL
    assert report.unsupported_nodes == ()
    assert report.composition_errors == ()
    assert program.annotations["block_count"] == 2
    assert program.annotations["task_heads_complete"] is False


def test_v3_energy_head_uses_explicit_scalar_mlp_and_batch_reduction():
    registry = core_registry()
    program = equiformer_v3_energy_head_program(_spec())
    inference = TypeChecker(registry).check(program)
    report = E3NNGraphBackend(registry).support_report(program)

    assert len(program.nodes) == 8
    assert program.outputs[0].source == "energy_rescale"
    assert inference.value_types["energy_rescale"].carrier == "graph"
    assert report.unsupported_nodes == ()
    assert [node.id for node in program.nodes][-2:] == ["energy_reduce", "energy_rescale"]


def test_v3_energy_model_composes_backbone_and_head_without_constructor_bypass():
    registry = core_registry()
    spec = replace(_spec(), num_layers=2, regress_forces=False)
    program = equiformer_v3_energy_model_program(spec)
    inference = TypeChecker(registry).check(program)
    canonical = canonicalize(program, registry)
    TypeChecker(registry).check(canonical)
    report = E3NNGraphBackend(registry).support_report(program)

    assert len(program.nodes) == 24 + 2 * 62 + 8
    assert program.outputs[0].source == "head_energy_rescale"
    assert inference.value_types["head_energy_rescale"].carrier == "graph"
    assert report.unsupported_nodes == ()
    assert report.composition_errors == ()
    assert program.annotations["constructor_bypass"] is False
    assert program.annotations["task_outputs_complete"] is True


def test_v3_force_head_expands_gate_activation_and_vector_readout():
    registry = core_registry()
    program = equiformer_v3_force_head_program(_spec())
    inference = TypeChecker(registry).check(program)
    canonical = canonicalize(program, registry)
    TypeChecker(registry).check(canonical)
    report = E3NNGraphBackend(registry).support_report(program)

    assert len(program.nodes) == 31
    assert "core.edge_frame_gate_activation@1" in {node.op for node in program.nodes}
    assert program.outputs[0].source == "force_vector"
    assert str(program.outputs[0].expected_type.irreps) == "1x1"
    assert inference.value_types["force_vector"].carrier == "node"
    assert report.unsupported_nodes == ()
    assert report.composition_errors == ()
    assert dict(report.runtime_kinds)["gated_activation"] == RuntimeValueKind.SO3_EDGE_FRAME
    assert program.annotations["constructor_bypass"] is False


def test_v3_force_head_rejects_a_gate_count_that_breaks_the_degree_contract():
    registry = core_registry()
    program = equiformer_v3_force_head_program(_spec())
    broken_nodes = []
    for node in program.nodes:
        if node.id == "activation_gates":
            broken_nodes.append(
                replace(node, attrs={**dict(node.attrs), "length": int(node.attrs["length"]) - 1})
            )
        else:
            broken_nodes.append(node)
    broken = replace(program, nodes=tuple(broken_nodes))

    with pytest.raises(DSLValidationError) as error:
        TypeChecker(registry).check(broken)
    assert any(item.code == "E_V3_EDGE_GATE_004" for item in error.value.diagnostics)


def test_v3_direct_model_shares_final_norm_between_energy_and_force_heads():
    registry = core_registry()
    spec = replace(
        _spec(),
        num_layers=2,
        direct_prediction=True,
        regress_forces=True,
        regress_stress=False,
    )
    program = equiformer_v3_direct_model_program(spec)
    inference = TypeChecker(registry).check(program)
    canonical = canonicalize(program, registry)
    TypeChecker(registry).check(canonical)
    report = E3NNGraphBackend(registry).support_report(program)

    assert len(program.nodes) == 24 + 2 * 62 + 1 + 7 + 31
    assert [(item.name, item.source) for item in program.outputs] == [
        ("energy", "energy_energy_rescale"),
        ("forces", "force_force_vector"),
    ]
    assert [node.id for node in program.nodes if node.id.endswith("final_norm")] == [
        "final_norm"
    ]
    final_norm_consumers = {
        node.id
        for node in program.nodes
        if any("final_norm" in reference for refs in node.inputs.values() for reference in refs)
    }
    assert final_norm_consumers == {
        "energy_scalar_input",
        "force_source_features",
        "force_target_features",
    }
    assert inference.value_types["energy_energy_rescale"].carrier == "graph"
    assert str(inference.value_types["force_force_vector"].irreps) == "1x1"
    assert report.unsupported_nodes == ()
    assert report.composition_errors == ()
    assert program.parameters["lowering_contract"]["shared_final_norm"]["node"] == "final_norm"
    canonical_contract = canonical.parameters["lowering_contract"]["shared_final_norm"]
    canonical_node_ids = {node.id for node in canonical.nodes}
    assert canonical_contract["node"] in canonical_node_ids
    assert set(canonical_contract["consumers"]) <= canonical_node_ids
    assert program.annotations["constructor_bypass"] is False
    assert program.annotations["task_outputs_complete"] is True


def test_v3_direct_model_generic_lowering_runs_energy_force_forward_and_backward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    if not equiformer_v3_source_available(str(V3_ROOT)):
        pytest.skip("official Equiformer V3 operator source is unavailable")

    registry = core_registry()
    spec = replace(
        _spec(),
        num_layers=1,
        direct_prediction=True,
        regress_forces=True,
        regress_stress=False,
        drop_path_rate=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        attn_weights_drop=0.0,
    )
    program = equiformer_v3_direct_model_program(spec)
    inference = TypeChecker(registry).check(program)
    model = E3NNGraphBackend(
        registry,
        equiformer_v3_root=str(V3_ROOT),
    ).build(program, inference).eval()

    atomic_numbers = torch.tensor([1, 6, 8, 14], dtype=torch.long)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1], [0.3, 1.2, 0.4], [-0.4, 0.6, 1.3]],
        dtype=torch.float32,
        requires_grad=True,
    )
    source = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    target = torch.tensor([1, 2, 3, 0, 2, 1], dtype=torch.long)
    batch = torch.zeros(atomic_numbers.numel(), dtype=torch.long)

    def index_map(indices, target_size):
        return {"indices": indices, "target_size": target_size}

    outputs = model(
        {
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "source_index": index_map(source, source.numel()),
            "target_index": index_map(target, target.numel()),
            "target_segment": index_map(target, atomic_numbers.numel()),
            "batch": index_map(batch, 1),
        },
        {},
    )
    assert tuple(outputs["energy"].shape) == (1, 1)
    assert tuple(outputs["forces"].shape) == (4, 3)
    assert torch.isfinite(outputs["energy"]).all()
    assert torch.isfinite(outputs["forces"]).all()

    (outputs["energy"].square().sum() + outputs["forces"].square().sum()).backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())
