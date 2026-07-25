import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_dsl_multifidelity_cycles.py"
    spec = importlib.util.spec_from_file_location("dsl_multifidelity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_v1_multifidelity_source_locks_validation_only_protocol():
    source = (ROOT / "scripts/run_dsl_multifidelity_cycles.py").read_text(encoding="utf-8")
    assert "formal-v1-8-4-2@2" in source
    assert '"stage_steps": [8000, 80000, 250000]' in source
    assert '"candidate_counts": [args.cohort, args.promote_80k, args.promote_250k]' in source
    assert 'metrics.get("test_evaluated") is not False' in source
    assert 'selection_split="validation"' in source
    assert '"parent_baseline": "same_seed_same_data_schedule_to_250000"' in source
    assert 'state["stage"] = "train_parent_baseline"' in source


def test_collect_8k_candidates_requires_dsl_identity_and_checkpoint(tmp_path):
    module = _module()
    search = tmp_path / "search"
    candidates = search / "candidates"
    candidates.mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint_last.pth"
    checkpoint.write_bytes(b"checkpoint")
    architecture_id = "abc123"
    (candidates / "iteration_0001_abc123.dsl.json").write_text("{}", encoding="utf-8")
    record = {
        "iteration": 1,
        "region_audit": {"factor_id": "F2.2"},
        "metrics": {
            "valid": True,
            "test_evaluated": False,
            "endpoint_step": 8000,
            "architecture_id": architecture_id,
            "program_id": architecture_id,
            "executable_id": "exec",
            "protocol_id": "protocol",
            "checkpoint_last": str(checkpoint),
            "validation_alpha_mae": 0.1,
        },
    }
    (search / "evolution.jsonl").write_text(__import__("json").dumps(record) + "\n", encoding="utf-8")
    result = module.collect_8k_candidates(search, 1)
    assert result[0]["factor_id"] == "F2.2"
    assert result[0]["executable_id"] == "exec"


def test_parent_baseline_requires_full_fidelity_checkpoint_and_test_isolation(tmp_path):
    module = _module()
    checkpoint = tmp_path / "checkpoint_last.pth"
    checkpoint.write_bytes(b"checkpoint")
    parent = {
        "checkpoint_250000": str(checkpoint),
        "metrics_250000": {
            "valid": True,
            "test_evaluated": False,
            "endpoint_step": 250000,
        },
    }
    assert module.parent_stage_complete(parent, 250000)
    parent["metrics_250000"]["test_evaluated"] = True
    assert not module.parent_stage_complete(parent, 250000)
