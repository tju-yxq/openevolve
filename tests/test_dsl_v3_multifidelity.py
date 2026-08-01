import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from equivariant_nas.dsl import TypeChecker, V3MultiFidelityProtocol, core_registry, v3_program_from_spec, rank_v3_stage_records
from equivariant_nas.dsl.backends import EquiformerV3Spec
from equivariant_nas.dsl.pipeline import (
    _count_parameters,
    _observable_symmetry_report,
    _representation_statistics,
    _v3_reference_parameter_count,
)
from equivariant_nas.training.v3_qm9_runtime import build_lowered_v3_qm9_model
from scripts.run_dsl_v3_cohort import load_frozen_protocol
from scripts import run_dsl_v3_cycles, run_dsl_v3_multifidelity


def _record(protocol, stage, index, metric):
    return {
        "architecture_id": "candidate_{:02d}".format(index),
        "protocol_hash": protocol.content_hash(),
        "dataset_manifest_sha256": protocol.dataset_manifest_sha256,
        "quarter_subset_sha256": protocol.quarter_subset_sha256,
        "equivariance_contract_sha256": protocol.equivariance_contract_sha256,
        "endpoint_step": stage.endpoint_steps,
        "validation_alpha_mae": metric,
        "test_evaluated": False,
    }


def test_frozen_v3_protocol_is_exactly_batch8_and_8_to_4_to_2():
    protocol = load_frozen_protocol("configs/dsl_v3_qm9_alpha_8_4_2_protocol.json")
    assert isinstance(protocol, V3MultiFidelityProtocol)
    assert protocol.cycle_count == 10
    assert protocol.batch_size == 8
    assert [item.endpoint_steps for item in protocol.stages] == [8000, 80000, 250000]
    assert [item.candidate_count for item in protocol.stages] == [8, 4, 2]
    assert [item.training_data for item in protocol.stages] == [
        "fixed_quarter",
        "fixed_quarter",
        "full_train",
    ]


def test_v3_stage_ranking_promotes_8_to_4_to_2_to_one_winner():
    protocol = load_frozen_protocol("configs/dsl_v3_qm9_alpha_8_4_2_protocol.json")
    stage8, stage80, stage250 = protocol.stages
    records8 = [_record(protocol, stage8, index, 0.8 - index * 0.01) for index in range(8)]
    top4 = rank_v3_stage_records(records8, stage=stage8, protocol=protocol)
    assert len(top4) == 4
    assert [item["architecture_id"] for item in top4] == [
        "candidate_07",
        "candidate_06",
        "candidate_05",
        "candidate_04",
    ]

    records80 = [_record(protocol, stage80, index, 0.4 - index * 0.01) for index in range(4)]
    top2 = rank_v3_stage_records(records80, stage=stage80, protocol=protocol)
    assert len(top2) == 2

    records250 = [_record(protocol, stage250, index, 0.2 - index * 0.01) for index in range(2)]
    winner = rank_v3_stage_records(records250, stage=stage250, protocol=protocol)
    assert len(winner) == 1
    assert winner[0]["architecture_id"] == "candidate_01"


def test_v3_stage_ranking_rejects_dataset_equivariance_or_test_changes():
    protocol = load_frozen_protocol("configs/dsl_v3_qm9_alpha_8_4_2_protocol.json")
    stage = protocol.stages[0]
    records = [_record(protocol, stage, index, 1.0 + index) for index in range(8)]
    records[0] = {**records[0], "dataset_manifest_sha256": "changed"}
    with pytest.raises(ValueError, match="dataset"):
        rank_v3_stage_records(records, stage=stage, protocol=protocol)
    records[0] = {**_record(protocol, stage, 0, 1.0), "quarter_subset_sha256": "changed"}
    with pytest.raises(ValueError, match="quarter subset"):
        rank_v3_stage_records(records, stage=stage, protocol=protocol)
    records[0] = {**_record(protocol, stage, 0, 1.0), "test_evaluated": True}
    with pytest.raises(ValueError, match="test leakage"):
        rank_v3_stage_records(records, stage=stage, protocol=protocol)


def test_v3_protocol_rejects_the_old_batch32_setting():
    protocol = load_frozen_protocol("configs/dsl_v3_qm9_alpha_8_4_2_protocol.json")
    payload = protocol.to_dict()
    payload["batch_size"] = 32
    with pytest.raises(ValueError, match="batch_size=8"):
        V3MultiFidelityProtocol.from_mapping(json.loads(json.dumps(payload)))


def test_energy_only_v3_generic_lowering_accepts_a_pyg_qm9_batch():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    spec = EquiformerV3Spec(
        num_layers=1,
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
    program = v3_program_from_spec(spec)
    model = build_lowered_v3_qm9_model(
        program,
        equiformer_v3_root="../equiformer_v3_official",
    )
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
    outputs = model(batch)
    assert outputs["energy"].shape == (2,)
    outputs["energy"].sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    symmetry = _observable_symmetry_report(model, batch)
    assert symmetry["maximum"] < 1.0e-5
    assert symmetry["maximum_absolute"] < 1.0e-6


def test_v3_pipeline_representation_statistics_ignore_categorical_and_topology_values():
    inference = TypeChecker(core_registry()).check(v3_program_from_spec(EquiformerV3Spec(
        num_layers=1,
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
    )))
    assert any(not hasattr(value_type, "irreps") for value_type in inference.value_types.values())
    statistics = _representation_statistics(inference.value_types.values())
    assert statistics["lmax"] == 1
    assert 0.0 < statistics["higher_order_fraction"] < 1.0


def test_v3_parameter_budget_uses_the_frozen_v3_parent_not_the_v1_baseline():
    program = v3_program_from_spec(EquiformerV3Spec(
        num_layers=1,
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
    ))
    model = build_lowered_v3_qm9_model(program, equiformer_v3_root="../equiformer_v3_official")
    assert _v3_reference_parameter_count(program, "../equiformer_v3_official") == _count_parameters(model)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_one_cycle_controller_trains_exactly_8_then_4_then_2(monkeypatch, tmp_path):
    protocol_path = Path("configs/dsl_v3_qm9_alpha_8_4_2_protocol.json").resolve()
    protocol = load_frozen_protocol(protocol_path)
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    parent_id = "parent_000"
    _write_json(
        cohort / "cohort_manifest.json",
        {
            "protocol_hash": protocol.content_hash(),
            "cycle_index": 1,
            "fixed_parent_architecture_id": parent_id,
        },
    )
    generation = []
    for index in range(8):
        program = cohort / "candidate_{:02d}.dsl.json".format(index)
        program.write_text("{}", encoding="utf-8")
        generation.append(
            {
                "candidate_index": index,
                "architecture_id": "candidate_{:02d}".format(index),
                "parent_architecture_id": parent_id,
                "candidate_path": str(program),
            }
        )
    (cohort / "cohort.jsonl").write_text(
        "\n".join(json.dumps(item) for item in generation) + "\n", encoding="utf-8"
    )

    calls = []

    def fake_pipeline(args, candidate, *, stage, checkpoint, protocol, log_path):
        del args, log_path
        calls.append((stage.name, candidate["architecture_id"], checkpoint))
        index = int(candidate["architecture_id"].split("_")[-1])
        if stage.name == "quarter_8k":
            metric = float(index)
        elif stage.name == "quarter_80k":
            metric = float(3 - index)
        else:
            metric = float(3 - index)
        checkpoint_path = tmp_path / "checkpoints" / stage.name / (candidate["architecture_id"] + ".pth")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("checkpoint", encoding="utf-8")
        return {
            **candidate,
            "protocol_hash": protocol.content_hash(),
            "dataset_manifest_sha256": protocol.dataset_manifest_sha256,
            "quarter_subset_sha256": protocol.quarter_subset_sha256,
            "equivariance_contract_sha256": protocol.equivariance_contract_sha256,
            "endpoint_step": stage.endpoint_steps,
            "validation_alpha_mae": metric,
            "test_evaluated": False,
            "checkpoint": str(checkpoint_path),
            "runtime_manifest_sha256": "runtime-{}".format(stage.name),
            "executable_id": "executable-{}".format(stage.name),
            "lowering_plan_hash": "lowering-{}".format(stage.name),
        }

    monkeypatch.setattr(run_dsl_v3_multifidelity, "_run_pipeline", fake_pipeline)
    args = SimpleNamespace(
        root=str(tmp_path / "training"),
        cohort_dir=str(cohort),
        protocol_config=str(protocol_path),
        equiformer_root="unused-v1",
        equiformer_v3_root="unused-v3",
        data_path="unused-data",
        quarter_subset_file=str(Path("data_splits/qm9_train_quarter_seed201.npz").resolve()),
        task_contract="",
        python="python",
        gpu_budget_hours=1.0,
        eval_interval_epochs=10,
    )
    state = run_dsl_v3_multifidelity.run(args)

    assert [sum(stage == name for stage, _, _ in calls) for name in (
        "quarter_8k", "quarter_80k", "full_250k"
    )] == [8, 4, 2]
    assert all(not checkpoint for stage, _, checkpoint in calls if stage == "quarter_8k")
    assert all(checkpoint for stage, _, checkpoint in calls if stage != "quarter_8k")
    assert state["stage"] == "cycle_complete"
    assert state["next_cycle_parent"]["architecture_id"] == "candidate_03"
    assert state["next_cycle_parent"]["endpoint_step"] == 250000

    calls.clear()
    resumed = run_dsl_v3_multifidelity.run(args)
    assert resumed["stage"] == "cycle_complete"
    assert calls == []


def test_pipeline_commands_encode_exact_resume_and_data_transition():
    protocol = load_frozen_protocol("configs/dsl_v3_qm9_alpha_8_4_2_protocol.json")
    candidate = {"program": "candidate.dsl.json"}
    args = SimpleNamespace(
        python="python",
        equiformer_root="v1",
        equiformer_v3_root="v3",
        data_path="data",
        eval_interval_epochs=10,
        task_contract="",
        quarter_subset_file="quarter.npz",
    )
    stage8, stage80, stage250 = protocol.stages
    command80 = run_dsl_v3_multifidelity._pipeline_command(
        args, candidate, stage=stage80, checkpoint="8k.pth", protocol=protocol
    )
    assert command80[command80.index("--max-steps") + 1] == "80000"
    assert command80[command80.index("--resume-checkpoint") + 1] == "8k.pth"
    assert "--train-subset-file" in command80
    assert "--resume-model-only" not in command80
    assert "--allow-data-transition" not in command80

    command250 = run_dsl_v3_multifidelity._pipeline_command(
        args, candidate, stage=stage250, checkpoint="80k.pth", protocol=protocol
    )
    assert command250[command250.index("--max-steps") + 1] == "250000"
    assert "--train-subset-file" not in command250
    assert "--resume-model-only" in command250
    assert "--allow-data-transition" in command250
    assert command250[command250.index("--data-epoch-origin-step") + 1] == "80000"
    assert command250[command250.index("--lr-schedule-origin-step") + 1] == "80000"


def test_ten_cycle_controller_chains_each_250k_winner_as_next_parent(monkeypatch, tmp_path):
    protocol_path = Path("configs/dsl_v3_qm9_alpha_8_4_2_protocol.json").resolve()
    protocol = load_frozen_protocol(protocol_path)
    root = tmp_path / "cycles"
    observed_seed_programs = []

    def option(command, name):
        return command[command.index(name) + 1]

    def fake_run_logged(command, log_path, environment):
        del log_path, environment
        if command[1].endswith("run_dsl_v3_cohort.py"):
            cycle = int(option(command, "--cycle-index"))
            cohort_dir = Path(option(command, "--output"))
            if "--seed-program" in command:
                observed_seed_programs.append(option(command, "--seed-program"))
                parent_id = "winner_{:03d}".format(cycle - 1)
            else:
                parent_id = "initial_parent"
            _write_json(
                cohort_dir / "cohort_manifest.json",
                {
                    "protocol_hash": protocol.content_hash(),
                    "fixed_parent_architecture_id": parent_id,
                },
            )
            candidate_ids = ["winner_{:03d}".format(cycle)] + [
                "cycle_{:03d}_candidate_{:02d}".format(cycle, index)
                for index in range(2, 9)
            ]
            (cohort_dir / "cohort.jsonl").write_text(
                "\n".join(
                    json.dumps({"architecture_id": architecture_id})
                    for architecture_id in candidate_ids
                )
                + "\n",
                encoding="utf-8",
            )
            return
        cycle_root = Path(option(command, "--root"))
        cycle = int(cycle_root.parent.name.split("_")[-1])
        parent_id = "initial_parent" if cycle == 1 else "winner_{:03d}".format(cycle - 1)
        winner_program = tmp_path / "winner_{:03d}.dsl.json".format(cycle)
        winner_program.write_text("{}", encoding="utf-8")
        _write_json(
            cycle_root / "state.json",
            {
                "stage": "cycle_complete",
                "parent_architecture_id": parent_id,
                "next_cycle_parent": {
                    "architecture_id": "winner_{:03d}".format(cycle),
                    "program": str(winner_program),
                    "checkpoint": "winner_{:03d}.pth".format(cycle),
                    "endpoint_step": 250000,
                    "protocol_hash": protocol.content_hash(),
                },
            },
        )

    monkeypatch.setattr(run_dsl_v3_cycles, "_run_logged", fake_run_logged)
    args = SimpleNamespace(
        root=str(root),
        model_config=str(Path("configs/dsl_v3_qm9_alpha_seed.json").resolve()),
        protocol_config=str(protocol_path),
        equiformer_root="v1",
        equiformer_v3_root="v3",
        data_path="data",
        quarter_subset_file=str(Path("data_splits/qm9_train_quarter_seed201.npz").resolve()),
        task_contract="",
        training_python="python",
        cycles=10,
        selection_mode="deterministic",
        mutation_mode="structural",
        model="",
        generation_validation_level="static",
        gpu_budget_hours=1.0,
        eval_interval_epochs=10,
    )
    state = run_dsl_v3_cycles.run(args)

    assert state["stage"] == "all_cycles_complete"
    assert len(state["completed_cycles"]) == 10
    assert state["final_parent"]["architecture_id"] == "winner_010"
    assert len(state["architecture_archive"]) == 81
    assert observed_seed_programs == [
        str(tmp_path / "winner_{:03d}.dsl.json".format(index)) for index in range(1, 10)
    ]
