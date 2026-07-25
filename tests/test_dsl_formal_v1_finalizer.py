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
    state_path = tmp_path / "state.json"
    program.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
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
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    freeze = _module().selection_payload(state, state_path)
    assert freeze["selection_split"] == "validation"
    assert freeze["test_evaluated_before_freeze"] is False
    assert len(freeze["program_sha256"]) == 64
    assert len(freeze["checkpoint_sha256"]) == 64


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
