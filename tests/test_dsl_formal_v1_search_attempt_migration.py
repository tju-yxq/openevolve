import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "migrate_dsl_formal_v1_search_attempts.py"
    spec = importlib.util.spec_from_file_location("migrate_dsl_formal_v1_search_attempts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize(tmp_path):
    search = tmp_path / "search"
    search.mkdir()
    config = {"gpu_budget_hours": 40.0, "batch_size": 32}
    config_bytes = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    (tmp_path / "formal_v1_config.json").write_bytes(config_bytes)
    protocol = {"config_sha256": __import__("hashlib").sha256(config_bytes).hexdigest(), "test_during_search": False}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (tmp_path / "READY_TO_LAUNCH.json").write_text(json.dumps({"ready": True, "protocol": protocol}), encoding="utf-8")
    (search / "summary.json").write_text(json.dumps({
        "status": "generation_attempts_exhausted",
        "last_completed_iteration": 32,
        "valid_candidate_count": 6,
        "valid_candidate_target": 8,
    }), encoding="utf-8")
    rows = [
        {"iteration": 1, "metrics": {"valid": True, "test_evaluated": False}},
        {"iteration": 32, "error": "E_LLM_014: schema"},
    ]
    (search / "evolution.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return search


def test_search_attempt_migration_preserves_training_and_selection_protocol(tmp_path):
    search = materialize(tmp_path)
    result = load_module().migrate(tmp_path, search, old_attempts=32, new_attempts=48, reason="repair observed critic failures")
    config = json.loads((tmp_path / "formal_v1_config.json").read_text(encoding="utf-8"))
    protocol = json.loads((tmp_path / "protocol.json").read_text(encoding="utf-8"))
    assert config["max_generation_attempts"] == 48
    assert config["batch_size"] == 32
    assert protocol["max_generation_attempts"] == 48
    assert result["training_protocol_changed"] is False
    assert result["selection_protocol_changed"] is False
    assert result["test_evaluated"] is False
    assert result["failure_code_counts"] == {"E_LLM_014": 1}


def test_search_attempt_migration_requires_exhaustion(tmp_path):
    search = materialize(tmp_path)
    summary = json.loads((search / "summary.json").read_text(encoding="utf-8"))
    summary["status"] = "running"
    (search / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not in generation_attempts_exhausted"):
        load_module().migrate(tmp_path, search, old_attempts=32, new_attempts=48, reason="invalid")
