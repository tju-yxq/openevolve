from collections import OrderedDict
from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

import tests.test_dsl_v3_official_direct_model as oracle  # noqa: E402
from equivariant_nas.dsl import (  # noqa: E402
    CheckpointMappingError,
    TypeChecker,
    core_registry,
    equiformer_v3_direct_model_program,
    export_source_state_dict,
    translate_checkpoint_state_dict,
)
from equivariant_nas.dsl.backends import (  # noqa: E402
    E3NNGraphBackend,
    equiformer_v3_direct_checkpoint_manifest,
    equiformer_v3_direct_parameter_mapping,
    export_equiformer_v3_direct_checkpoint,
    load_equiformer_v3_direct_checkpoint,
)


def _build_lowered(spec, program, seed):
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    torch.manual_seed(seed)
    model = E3NNGraphBackend(
        registry,
        equiformer_v3_root=str(oracle.V3_ROOT),
    ).build(program, inference)
    return model


def _official_checkpoint(spec, state_dict, *, prefixed=False):
    if prefixed:
        state_dict = OrderedDict(
            ("module._orig_mod." + name, value.detach().clone())
            for name, value in state_dict.items()
        )
    else:
        state_dict = OrderedDict(
            (name, value.detach().clone()) for name, value in state_dict.items()
        )
    return {
        "config": {
            "model": {
                "name": "equiformer_v3",
                **spec.official_constructor_kwargs(),
            }
        },
        "state_dict": state_dict,
    }


def _assert_state_dict_equal(actual, expected):
    assert set(actual) == set(expected)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0.0, atol=0.0)


def test_v3_direct_checkpoint_manifest_covers_every_official_and_lowered_state_tensor():
    modules = oracle._official_modules()
    spec = oracle._stochastic_spec()
    program = equiformer_v3_direct_model_program(spec)
    torch.manual_seed(31001)
    official = oracle._official_model(spec, modules)
    lowered = _build_lowered(spec, program, 31002)
    manifest = equiformer_v3_direct_checkpoint_manifest(spec, program)

    assert len(official.state_dict()) == len(manifest.source_keys) == 158
    assert len(lowered.state_dict()) == len(manifest.target_keys) == 157
    assert set(manifest.source_keys) == set(official.state_dict())
    assert set(manifest.target_keys) == set(lowered.state_dict())
    assert len(manifest.reconstructed_sources) == 2 * spec.num_layers + 1
    assert manifest.architecture_id == spec.architecture_id()
    assert dict(equiformer_v3_direct_parameter_mapping(spec, program)) == (
        oracle._direct_model_parameter_mapping(spec, program)
    )


def test_v3_official_checkpoint_load_export_round_trip_and_train_eval_behavior(tmp_path):
    modules = oracle._official_modules()
    spec = oracle._stochastic_spec()
    program = equiformer_v3_direct_model_program(spec)
    torch.manual_seed(31101)
    expected = oracle._official_model(spec, modules)
    official_state = expected.state_dict()
    checkpoint = _official_checkpoint(spec, official_state, prefixed=True)
    checkpoint_path = tmp_path / "official_v3_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    actual = _build_lowered(spec, program, 31102)
    manifest, translated = load_equiformer_v3_direct_checkpoint(
        actual,
        checkpoint_path,
        spec,
        program,
    )
    _assert_state_dict_equal(actual.state_dict(), translated)

    exported_state = export_source_state_dict(actual.state_dict(), manifest)
    _assert_state_dict_equal(exported_state, official_state)
    exported_checkpoint = export_equiformer_v3_direct_checkpoint(
        actual,
        spec,
        program,
    )
    _assert_state_dict_equal(exported_checkpoint["state_dict"], official_state)

    reloaded = _build_lowered(spec, program, 31103)
    load_equiformer_v3_direct_checkpoint(
        reloaded,
        exported_checkpoint,
        spec,
        program,
    )
    _assert_state_dict_equal(reloaded.state_dict(), actual.state_dict())

    atomic_numbers = torch.tensor([1, 6, 8, 14, 16], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.2, -0.1],
            [0.3, 1.2, 0.4],
            [-0.4, 0.6, 1.3],
            [0.8, -0.7, 0.5],
        ],
        dtype=torch.float32,
    )
    source = torch.tensor([0, 1, 2, 3, 4, 3, 4, 2], dtype=torch.long)
    target = torch.tensor([1, 0, 3, 4, 2, 2, 3, 4], dtype=torch.long)
    edge_index = torch.stack((source, target), dim=0)
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    parameter_mapping = equiformer_v3_direct_parameter_mapping(spec, program)

    for training in (True, False):
        expected.train(training)
        actual.train(training)
        expected.zero_grad(set_to_none=True)
        actual.zero_grad(set_to_none=True)
        positions_expected = positions.clone().requires_grad_(True)
        positions_actual = positions.clone().requires_grad_(True)

        torch.manual_seed(31110 + int(training))
        expected_outputs = oracle._official_forward(
            expected,
            atomic_numbers,
            positions_expected,
            edge_index,
            batch,
        )
        expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(31110 + int(training))
        actual_outputs = actual(
            {
                "atomic_numbers": atomic_numbers,
                "positions": positions_actual,
                "source_index": oracle._index(source, source.numel()),
                "target_index": oracle._index(target, target.numel()),
                "target_segment": oracle._index(target, atomic_numbers.numel()),
                "batch": oracle._index(batch, 2),
            },
            {},
        )
        actual_rng = torch.random.get_rng_state().clone()
        assert torch.equal(actual_rng, expected_rng)
        torch.testing.assert_close(
            actual_outputs["energy"],
            expected_outputs["energy"],
            rtol=3.0e-5,
            atol=3.0e-6,
        )
        torch.testing.assert_close(
            actual_outputs["forces"],
            expected_outputs["forces"],
            rtol=4.0e-5,
            atol=4.0e-7,
        )

        expected_loss = sum(value.square().sum() for value in expected_outputs.values())
        actual_loss = sum(value.square().sum() for value in actual_outputs.values())
        expected_loss.backward()
        actual_loss.backward()
        torch.testing.assert_close(
            positions_actual.grad,
            positions_expected.grad,
            rtol=1.0e-4,
            atol=1.0e-6,
        )
        expected_parameters = dict(expected.named_parameters())
        actual_parameters = dict(actual.named_parameters())
        for actual_name, expected_name in parameter_mapping.items():
            torch.testing.assert_close(
                actual_parameters[actual_name].grad,
                expected_parameters[expected_name].grad,
                rtol=1.5e-4,
                atol=1.5e-6,
            )


def test_v3_checkpoint_mapping_rejects_incomplete_corrupt_or_wrong_architecture_state():
    modules = oracle._official_modules()
    spec = oracle._spec()
    program = equiformer_v3_direct_model_program(spec)
    torch.manual_seed(31201)
    official = oracle._official_model(spec, modules)
    actual = _build_lowered(spec, program, 31202)
    manifest = equiformer_v3_direct_checkpoint_manifest(spec, program)
    state = official.state_dict()

    missing = OrderedDict((name, value.clone()) for name, value in state.items())
    missing.pop("sphere_embedding.weight")
    with pytest.raises(CheckpointMappingError, match="missing"):
        translate_checkpoint_state_dict(missing, actual.state_dict(), manifest)

    unexpected = OrderedDict((name, value.clone()) for name, value in state.items())
    unexpected["unexpected.weight"] = torch.zeros(1)
    with pytest.raises(CheckpointMappingError, match="unexpected"):
        translate_checkpoint_state_dict(unexpected, actual.state_dict(), manifest)

    wrong_shape = OrderedDict((name, value.clone()) for name, value in state.items())
    wrong_shape["sphere_embedding.weight"] = wrong_shape["sphere_embedding.weight"][:-1]
    with pytest.raises(CheckpointMappingError, match="shapes differ"):
        translate_checkpoint_state_dict(wrong_shape, actual.state_dict(), manifest)

    broken_alias = OrderedDict((name, value.clone()) for name, value in state.items())
    alias_key = "blocks.0.ga.so3_rotation.wigner_index_to_m_array"
    broken_alias[alias_key] = broken_alias[alias_key] + 1.0
    with pytest.raises(CheckpointMappingError, match="source aliases"):
        translate_checkpoint_state_dict(broken_alias, actual.state_dict(), manifest)

    broken_reconstruction = OrderedDict(
        (name, value.clone()) for name, value in state.items()
    )
    reconstructed_key = "blocks.0.ga.rad_func.expand_index"
    broken_reconstruction[reconstructed_key][0] = 1
    with pytest.raises(CheckpointMappingError, match="constant contract"):
        translate_checkpoint_state_dict(
            broken_reconstruction,
            actual.state_dict(),
            manifest,
        )

    wrong_spec = replace(spec, num_channels=spec.num_channels + 1)
    wrong_config = _official_checkpoint(wrong_spec, state)
    with pytest.raises(CheckpointMappingError, match="does not match target"):
        load_equiformer_v3_direct_checkpoint(
            actual,
            wrong_config,
            spec,
            program,
        )
