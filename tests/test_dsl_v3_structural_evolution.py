import json
from types import SimpleNamespace

import pytest

from equivariant_nas.dsl import (
    TypeChecker,
    architecture_id,
    core_registry,
    v3_program_from_spec,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend, EquiformerV3Spec
from equivariant_nas.dsl.pipeline import _observable_symmetry_report
from equivariant_nas.dsl.v3_structural_evolution import (
    apply_v3_structural_patch,
    build_v3_structural_patch,
    v3_structural_mutation_catalog,
    v3_structural_novelty_report,
)
from equivariant_nas.training.v3_qm9_runtime import build_lowered_v3_qm9_model
from scripts.run_dsl_v3_evolution import _router_prompt, _validate_candidate


def _small_spec(num_layers=1):
    return EquiformerV3Spec(
        num_layers=num_layers,
        num_channels=4,
        attn_hidden_channels=4,
        num_heads=1,
        attn_alpha_channels=2,
        attn_value_channels=2,
        ffn_hidden_channels=8,
        lmax=1,
        mmax=1,
        attn_grid_resolution=(4, 4),
        ffn_grid_resolution=(4, 4),
        edge_channels=4,
        num_radial_basis=4,
        max_num_elements=10,
        use_pbc=False,
        otf_graph=False,
        regress_forces=False,
        regress_stress=False,
        alpha_drop=0.02,
        attn_weights_drop=0.02,
        value_drop=0.02,
        drop_path_rate=0.02,
        proj_drop=0.02,
        ffn_drop=0.02,
    )


def _matching_inverse(actions, original):
    matches = [
        action
        for action in actions
        if action.field == original.field and action.block_index == original.block_index
    ]
    if "first_id" in original.parameters:
        matches = [
            action
            for action in matches
            if action.parameters.get("first_id") == original.parameters.get("first_id")
        ]
    if "norm_id" in original.parameters:
        matches = [
            action
            for action in matches
            if action.parameters.get("norm_id") == original.parameters.get("norm_id")
        ]
    assert len(matches) == 1
    return matches[0]


def test_production_v3_parent_exposes_block_operator_and_operator_internal_rewrites():
    payload = json.loads(open("configs/dsl_v3_qm9_alpha_seed.json", encoding="utf-8").read())
    spec, _ = EquiformerV3Spec.from_official_config(payload)
    program = v3_program_from_spec(spec)
    actions = v3_structural_mutation_catalog(program)

    assert len(actions) == 56
    assert len({action.action_id for action in actions}) == 56
    assert {
        field: sum(action.field == field for action in actions)
        for field in {action.field for action in actions}
    } == {
        "block_branch_topology": 8,
        "radial_norm_presence": 16,
        "radial_norm_activation_order": 16,
        "alpha_norm_activation_order": 8,
        "ffn_scalar_gate_orientation": 8,
    }
    assert {action.level for action in actions} == {"block", "operator", "operator_internal"}


def test_structural_rewrites_are_unique_direct_children_and_have_inverse_edits():
    registry = core_registry()
    parent = v3_program_from_spec(_small_spec(num_layers=2))
    parent_id = architecture_id(parent, registry)
    actions = v3_structural_mutation_catalog(parent)
    child_ids = set()

    for action in actions:
        patch = build_v3_structural_patch(parent, action.action_id, registry)
        assert patch.parent_architecture_id == parent_id
        child, child_spec = apply_v3_structural_patch(parent, patch, registry)
        TypeChecker(registry).check(child)
        assert child_spec.to_dict() == _small_spec(num_layers=2).to_dict()
        child_id = architecture_id(child, registry)
        assert child_id != parent_id
        assert child_id not in child_ids
        child_ids.add(child_id)
        novelty = v3_structural_novelty_report(parent, child, registry)
        assert novelty["is_structurally_novel"] is True
        assert novelty["only_attribute_changed"] is False
        assert novelty["graph_edit_count"] > 0

    assert len(child_ids) == len(actions) == 14

    for field in sorted({action.field for action in actions}):
        action = next(item for item in actions if item.field == field)
        child, _ = apply_v3_structural_patch(
            parent,
            build_v3_structural_patch(parent, action.action_id, registry),
            registry,
        )
        inverse = _matching_inverse(v3_structural_mutation_catalog(child), action)
        restored, _ = apply_v3_structural_patch(
            child,
            build_v3_structural_patch(child, inverse.action_id, registry),
            registry,
        )
        assert architecture_id(restored, registry) == parent_id


def test_eight_structural_siblings_are_unique_without_updating_the_parent():
    registry = core_registry()
    parent = v3_program_from_spec(_small_spec(num_layers=2))
    parent_id = architecture_id(parent, registry)
    actions = v3_structural_mutation_catalog(parent)
    child_ids = []
    for action in actions[:8]:
        child, _ = apply_v3_structural_patch(
            parent,
            build_v3_structural_patch(parent, action.action_id, registry),
            registry,
        )
        child_ids.append(architecture_id(child, registry))
    assert architecture_id(parent, registry) == parent_id
    assert len(set(child_ids)) == 8
    assert parent_id not in child_ids


def test_llm_router_receives_structural_regions_and_not_dropout_only_state():
    parent = v3_program_from_spec(_small_spec(num_layers=2))
    actions = v3_structural_mutation_catalog(parent)
    prompt = _router_prompt(parent, actions, [], 1, mutation_mode="structural")
    payload = json.loads(prompt["user"])
    assert payload["mutation_mode"] == "structural"
    assert len(payload["regions"]) == 5
    assert {item["mutation_level"] for item in payload["regions"]} == {
        "block",
        "operator",
        "operator_internal",
    }
    assert "current_stochastic_spec" not in payload


def test_ffn_scalar_gate_reimplementation_rewires_an_internal_operator_subgraph():
    registry = core_registry()
    parent = v3_program_from_spec(_small_spec())
    action = next(
        item
        for item in v3_structural_mutation_catalog(parent)
        if item.field == "ffn_scalar_gate_orientation"
    )
    patch = build_v3_structural_patch(parent, action.action_id, registry)
    assert [edit.kind for edit in patch.edits] == ["rewire_port", "rewire_port", "rewire_port"]
    child, _ = apply_v3_structural_patch(parent, patch, registry)
    nodes = {node.id: node for node in child.nodes}
    assert nodes["block0_ffn_scalar_gate_act"].inputs["x"] == ("block0_ffn_scalar_up",)
    assert nodes["block0_ffn_scalar_product"].inputs["left"] == ("block0_ffn_scalar_gate",)
    assert nodes["block0_ffn_scalar_product"].inputs["right"] == ("block0_ffn_scalar_gate_act",)
    novelty = v3_structural_novelty_report(parent, child, registry)
    assert novelty["is_structurally_novel"] is True
    assert novelty["input_rewires"]


def test_each_structural_family_generic_lowers_and_preserves_so3_translation_permutation():
    torch = pytest.importorskip("torch")
    registry = core_registry()
    parent = v3_program_from_spec(_small_spec())
    actions = v3_structural_mutation_catalog(parent)
    batch = SimpleNamespace(
        x=torch.zeros(5, 5),
        z=torch.tensor([1, 6, 8, 1, 1]),
        atomic_numbers=torch.tensor([1, 6, 8, 1, 1]),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 1.1, 0.3], [0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]
        ),
        edge_index=torch.tensor([[0, 1, 2, 0, 3, 4], [1, 2, 0, 2, 4, 3]]),
        edge_d_index=torch.tensor([[0, 1, 2, 0, 3, 4], [1, 2, 0, 2, 4, 3]]),
        edge_d_attr=torch.zeros(6, 3),
        batch=torch.tensor([0, 0, 0, 1, 1]),
        num_graphs=2,
    )

    for field in sorted({action.field for action in actions}):
        action = next(item for item in actions if item.field == field)
        child, _ = apply_v3_structural_patch(
            parent,
            build_v3_structural_patch(parent, action.action_id, registry),
            registry,
        )
        model = build_lowered_v3_qm9_model(
            child,
            equiformer_v3_root="../equiformer_v3_official",
        )
        outputs = model(batch)
        assert outputs["energy"].shape == (2,)
        outputs["energy"].sum().backward()
        assert all(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        report = _observable_symmetry_report(model, batch)
        assert report["maximum"] < 1.0e-5
        assert report["maximum_absolute"] < 1.0e-6


def test_formal_full_generation_gate_audits_energy_only_rotation_translation_permutation_and_gradients():
    registry = core_registry()
    parent = v3_program_from_spec(_small_spec())
    action = v3_structural_mutation_catalog(parent)[0]
    child, _ = apply_v3_structural_patch(
        parent,
        build_v3_structural_patch(parent, action.action_id, registry),
        registry,
    )
    backend = E3NNGraphBackend(
        registry,
        equiformer_v3_root="../equiformer_v3_official",
    )

    validation = _validate_candidate(
        child,
        registry,
        backend,
        validation_level="full",
        seed=201,
    )

    runtime = validation["runtime"]
    assert runtime["force_shape"] is None
    assert runtime["audited_transformations"] == {
        "rotations": 2,
        "translations": 2,
        "permutations": 2,
    }
    assert runtime["maximum_relative_error"] <= runtime["relative_threshold"] or (
        runtime["maximum_absolute_error"] <= runtime["absolute_threshold"]
    )
    assert runtime["position_gradient_finite"] is True
    assert runtime["trainable_parameter_gradients_complete"] is True
