import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "finalize_dsl_formal_v1.py"
    spec = importlib.util.spec_from_file_location("finalize_dsl_formal_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_freeze_requires_validation_only_winner_and_binds_files(tmp_path):
    program = tmp_path / "winner.dsl.json"
    checkpoint = tmp_path / "checkpoint_last.pth"
    parent_checkpoint = tmp_path / "parent_checkpoint_last.pth"
    state_path = tmp_path / "state.json"
    program.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    parent_checkpoint.write_bytes(b"parent checkpoint")
    state = {
        "stage": "completed_validation_selection",
        "winner_by_validation": {
            "architecture_id": "arch",
            "program_id": "program",
            "executable_id": "exe",
            "program": str(program),
            "checkpoint_250000": str(checkpoint),
            "metrics_250000": {"validation_alpha_mae": 0.1, "test_evaluated": False},
        },
        "parent_baseline": {
            "checkpoint_250000": str(parent_checkpoint),
            "metrics_250000": {
                "validation_alpha_mae": 0.12,
                "endpoint_step": 250000,
                "valid": True,
                "test_evaluated": False,
            },
        },
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    freeze = _module().selection_payload(state, state_path)
    assert freeze["selection_split"] == "validation"
    assert freeze["test_evaluated_before_freeze"] is False
    assert len(freeze["program_sha256"]) == 64
    assert len(freeze["checkpoint_sha256"]) == 64
    assert freeze["candidate_beats_parent"] is True


def test_selection_freeze_rejects_prior_test_exposure(tmp_path):
    state = {
        "stage": "completed_validation_selection",
        "winner_by_validation": {
            "program": str(tmp_path / "missing"),
            "checkpoint_250000": str(tmp_path / "missing2"),
            "metrics_250000": {"validation_alpha_mae": 0.1, "test_evaluated": True},
        },
    }
    with pytest.raises(RuntimeError, match="exposed to Test"):
        _module().selection_payload(state, tmp_path / "state.json")


def test_materialize_candidate_evidence_archives_complete_training_bundle(tmp_path):
    module = _module()
    root = tmp_path / "formal_v1"
    search_dir = tmp_path / "search"
    run_dir = tmp_path / "candidate_run"
    training_dir = run_dir / "training"
    (search_dir / "candidates").mkdir(parents=True)
    training_dir.mkdir(parents=True)

    (tmp_path / "initial_program.dsl.json").write_text('{"parent": true}', encoding="utf-8")
    (search_dir / "candidates" / "iteration_0001_arch.dsl.json").write_text(
        '{"candidate": true}', encoding="utf-8"
    )
    (run_dir / "runtime_manifest.json").write_text('{"protocol_id": "p"}', encoding="utf-8")
    (run_dir / "posttrain_symmetry_report.json").write_text('{"passed": true}', encoding="utf-8")
    (run_dir / "result.json").write_text('{"validation_alpha_mae": 0.1}', encoding="utf-8")
    (training_dir / "progress.json").write_text('{"global_step": 8000}', encoding="utf-8")
    (training_dir / "metrics.jsonl").write_text('{"step": 8000, "mae": 0.1}\n', encoding="utf-8")
    (training_dir / "checkpoint_last.pth").write_bytes(b"checkpoint")

    record = {
        "iteration": 1,
        "architecture_id": "arch",
        "patch": {"factor_id": "F2.2"},
        "router_response": {"factor_id": "F2.2"},
        "critic_response": {"hypothesis": "radial"},
        "planner_response": {"edits": []},
        "region_audit": {"valid": True},
        "metrics": {
            "run_dir": str(run_dir),
            "valid": True,
            "test_evaluated": False,
            "compiler_obligations": [],
            "lowering_plan": {"mode": "exact_constructor"},
        },
    }
    (search_dir / "evolution.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    module.materialize_candidate_evidence(root, search_dir)

    material = root / "candidate_materials" / "iteration_0001_arch"
    expected = {
        "parent.dsl.json",
        "candidate.dsl.json",
        "typed_patch.json",
        "factor_router.json",
        "factor_critic.json",
        "synthesizer.json",
        "generation_record.json",
        "compiler_report.json",
        "lowering_plan.json",
        "runtime_manifest.json",
        "symmetry_audit.json",
        "training/progress.json",
        "training/metrics.jsonl",
        "training/checkpoint_last.pth",
        "result.json",
    }
    actual = {path.relative_to(material).as_posix() for path in material.rglob("*") if path.is_file()}
    assert expected <= actual
    assert json.loads((material / "factor_router.json").read_text(encoding="utf-8"))["factor_id"] == "F2.2"
    assert (material / "training" / "checkpoint_last.pth").read_bytes() == b"checkpoint"
