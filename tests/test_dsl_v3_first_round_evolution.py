import json
from pathlib import Path

import pytest

from equivariant_nas.dsl import (
    DSLValidationError,
    PatchEdit,
    TypedPatch,
    apply_typed_patch,
    apply_v3_first_round_patch,
    architecture_id,
    build_v3_first_round_patch,
    choose_deterministic_v3_action,
    core_registry,
    equiformer_v3_direct_model_program,
    v3_evolution_state,
    v3_first_round_mutation_catalog,
)
from equivariant_nas.dsl.backends import EquiformerV3Spec
from scripts.export_dsl_v3_seed import export_seed
from scripts.run_dsl_v3_evolution import get_parser


def _spec():
    return EquiformerV3Spec(
        num_layers=2,
        num_channels=8,
        attn_hidden_channels=8,
        num_heads=2,
        attn_alpha_channels=4,
        attn_value_channels=2,
        ffn_hidden_channels=16,
        lmax=2,
        mmax=1,
        attn_grid_resolution=(6, 4),
        ffn_grid_resolution=(6, 6),
        edge_channels=8,
        num_radial_basis=8,
        alpha_drop=0.02,
        attn_weights_drop=0.10,
        value_drop=0.02,
        drop_path_rate=0.05,
        proj_drop=0.02,
        ffn_drop=0.02,
        direct_prediction=True,
        regress_forces=True,
        regress_stress=False,
    )


def test_v3_first_round_catalog_covers_all_six_shape_preserving_stochastic_families():
    program = equiformer_v3_direct_model_program(_spec())
    actions = v3_first_round_mutation_catalog(program)
    assert {item.field for item in actions} == {
        "alpha_drop",
        "attn_weights_drop",
        "value_drop",
        "drop_path_rate",
        "proj_drop",
        "ffn_drop",
    }
    assert all(item.value != item.current_value for item in actions)
    assert all(item.targets for item in actions)


@pytest.mark.parametrize(
    "action_id",
    (
        "alpha_drop=0.05",
        "attn_weights_drop=0.05",
        "value_drop=0.05",
        "drop_path_rate=0.10",
        "proj_drop=0.05",
        "ffn_drop=0.05",
    ),
)
def test_v3_first_round_patch_updates_every_runtime_site_and_equals_spec_regeneration(action_id):
    registry = core_registry()
    parent = equiformer_v3_direct_model_program(_spec())
    patch = build_v3_first_round_patch(
        parent,
        action_id,
        registry,
        hypothesis={"rationale": "unit test"},
    )
    child, spec = apply_v3_first_round_patch(parent, patch, registry)
    field, raw_value = action_id.split("=", 1)
    assert getattr(spec, field) == float(raw_value)
    assert architecture_id(child, registry) != architecture_id(parent, registry)
    parameter_edits = [item for item in patch.edits if item.kind == "change_parameters"]
    assert [item.target for item in parameter_edits] == [
        "program.parameters.equiformer_v3_spec.{}".format(field)
    ]


def test_ten_deterministic_v3_rounds_form_one_unique_typed_lineage():
    registry = core_registry()
    parent = equiformer_v3_direct_model_program(_spec())
    seen_states = [v3_evolution_state(parent)]
    seen_ids = {architecture_id(parent, registry)}
    for round_index in range(1, 11):
        action = choose_deterministic_v3_action(
            parent,
            round_index=round_index,
            seen_states=seen_states,
        )
        patch = build_v3_first_round_patch(parent, action.action_id, registry)
        child, _specification = apply_v3_first_round_patch(parent, patch, registry)
        child_id = architecture_id(child, registry)
        child_state = v3_evolution_state(child)
        assert patch.parent_architecture_id == architecture_id(parent, registry)
        assert child_id not in seen_ids
        assert child_state not in seen_states
        seen_ids.add(child_id)
        seen_states.append(child_state)
        parent = child
    assert len(seen_ids) == 11


def test_program_parameter_patch_is_existing_path_only():
    registry = core_registry()
    parent = equiformer_v3_direct_model_program(_spec())
    parent_id = architecture_id(parent, registry)
    patch = TypedPatch(
        "1.0",
        parent_id,
        parent.language_version,
        {"claim": "invalid metadata insertion"},
        ("program.parameters.equiformer_v3_spec.unknown",),
        (
            PatchEdit(
                "change_parameters",
                "program.parameters.equiformer_v3_spec.unknown",
                {"value": 0.1},
            ),
        ),
    )
    with pytest.raises(DSLValidationError) as error:
        apply_typed_patch(parent, patch, registry)
    assert error.value.diagnostics[0].code == "E_PATCH_020"


def test_v3_seed_export_and_evolution_cli_default_to_ten_rounds(tmp_path):
    config = tmp_path / "v3.json"
    config.write_text(
        json.dumps({"model": {"name": "equiformer_v3", **_spec().official_constructor_kwargs()}}),
        encoding="utf-8",
    )
    identity = export_seed(config, tmp_path / "seed")
    assert identity["node_count"] == 190
    assert identity["outputs"] == ["energy", "forces"]
    assert identity["constructor_bypass"] is False
    args = get_parser().parse_args([
        "--model-config",
        str(config),
        "--output",
        str(tmp_path / "run"),
    ])
    assert args.rounds == 10
    assert args.selection_mode == "glm"
