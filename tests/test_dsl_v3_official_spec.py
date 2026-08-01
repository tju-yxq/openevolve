import ast
from pathlib import Path

import pytest


from equivariant_nas.dsl import import_equiformer_v3
from equivariant_nas.dsl.backends import (
    EquiformerV3Spec,
    V3_REFERENCE_COMMIT,
    official_v3_oc_spec,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = REPOSITORY_ROOT.parent / "equiformer_v3_official"
V3_MODEL_SOURCE = (
    V3_ROOT
    / "experimental"
    / "models"
    / "equiformer_v3"
    / "equiformer_v3.py"
)
V3_TRAIN_CONFIG = (
    V3_ROOT
    / "experimental"
    / "configs"
    / "oc20"
    / "2M"
    / "equiformer_v3"
    / "experiments"
    / "base_N@8-L@6-C@128-attn-hidden@64-ffn@512-envelope-num-rbf@128_merge-layer-norm_gates2-gridmlp_use-gate-force-head_wd@1e-3-grad-clip@100_lin-ref-e@4.yml"
)


def _official_init_argument_names():
    if not V3_MODEL_SOURCE.is_file():
        pytest.skip("official Equiformer V3 source is unavailable")
    tree = ast.parse(V3_MODEL_SOURCE.read_text(encoding="utf-8"))
    model = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EquiformerV3_OC"
    )
    init = next(
        node for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    return tuple(argument.arg for argument in init.args.args if argument.arg != "self")


def test_v3_spec_covers_every_official_constructor_argument():
    official = set(_official_init_argument_names())
    schema = set(EquiformerV3Spec.field_names())
    aliases = {"attn_grid_resolution_list", "ffn_grid_resolution_list"}
    canonical = {"attn_grid_resolution", "ffn_grid_resolution"}
    assert (official - aliases) | canonical == schema
    assert len(official) == len(schema)


def test_official_default_spec_round_trips_all_constructor_kwargs():
    spec = official_v3_oc_spec()
    kwargs = spec.official_constructor_kwargs()
    assert set(kwargs) == set(_official_init_argument_names())
    restored = EquiformerV3Spec.from_mapping(kwargs, strict=True)
    assert restored == spec
    assert kwargs["attn_grid_resolution_list"] == [20, 8]
    assert kwargs["ffn_grid_resolution_list"] == [20, 20]
    assert kwargs["gradient_checkpointing_block_list"] is None


def test_actual_official_training_yaml_is_strictly_consumed_without_unknown_model_fields():
    yaml = pytest.importorskip("yaml")
    if not V3_TRAIN_CONFIG.is_file():
        pytest.skip("official Equiformer V3 training config is unavailable")
    config = yaml.safe_load(V3_TRAIN_CONFIG.read_text(encoding="utf-8"))
    spec, manifest = EquiformerV3Spec.from_official_config(config)
    assert manifest.source_commit == V3_REFERENCE_COMMIT
    assert manifest.model_name == "equiformer_v3"
    assert manifest.architecture_id == spec.architecture_id()
    assert "name" in manifest.dispatch_fields
    assert "num_layers" in manifest.consumed_fields
    assert "attn_eps" in manifest.defaulted_fields
    assert spec.num_layers == 8
    assert spec.lmax == 6
    assert spec.num_radial_basis == 128
    assert spec.gradient_checkpointing_block_list == (0,) * 8


def test_v3_official_config_import_rejects_silent_field_loss_and_dens_dispatch():
    with pytest.raises(ValueError, match="unknown Equiformer V3 constructor fields"):
        EquiformerV3Spec.from_official_config(
            {"model": {"name": "equiformer_v3", "num_layers": 1, "misspelled_lmax": 2}}
        )
    with pytest.raises(ValueError, match="dedicated DeNS importer"):
        EquiformerV3Spec.from_official_config({"model": {"name": "equiformer_v3_dens"}})


def test_legacy_compositional_importer_refuses_full_config_variants_it_cannot_represent():
    spec = EquiformerV3Spec(num_layers=1, norm_type="merge_rms_norm")
    spec.validate()
    with pytest.raises(ValueError, match="legacy compositional V3 importer"):
        import_equiformer_v3(spec)


@pytest.mark.parametrize(
    "changes",
    [
        {"softcap": 0.0},
        {"attn_mask_rate": 1.0},
        {"use_add_merge": True, "use_rad_l_parametrization": False},
        {"direct_prediction": False, "regress_stress": True, "regress_forces": False},
        {"num_layers": 2, "gradient_checkpointing_block_list": (0,)},
    ],
)
def test_v3_full_spec_rejects_officially_invalid_or_nonfunctional_combinations(changes):
    values = dict(EquiformerV3Spec().to_dict())
    values.update(changes)
    with pytest.raises(ValueError):
        EquiformerV3Spec.from_mapping(values, strict=True)
