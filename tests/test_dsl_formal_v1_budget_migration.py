import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "migrate_dsl_formal_v1_budget.py"
    spec = importlib.util.spec_from_file_location("migrate_dsl_formal_v1_budget", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize_run(tmp_path):
    ledger = tmp_path / "budget_ledger.jsonl"
    ledger.write_text(json.dumps({"gpu_seconds": 3600.0}) + "\n", encoding="utf-8")
    config = {"gpu_budget_hours": 40.0, "batch_size": 32, "stage_steps": [8000, 80000, 250000]}
    config_bytes = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    (tmp_path / "formal_v1_config.json").write_bytes(config_bytes)
    protocol = {
        "gpu_budget_hours": 40.0,
        "config_sha256": __import__("hashlib").sha256(config_bytes).hexdigest(),
        "budget_ledger": str(ledger),
        "test_during_search": False,
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (tmp_path / "READY_TO_LAUNCH.json").write_text(json.dumps({"ready": True, "protocol": protocol}), encoding="utf-8")


def test_budget_migration_changes_only_the_ceiling_and_records_evidence(tmp_path):
    materialize_run(tmp_path)
    module = load_module()
    migration = module.migrate(tmp_path, old_hours=40.0, new_hours=80.0, reason="measured full-cycle cost")
    config = json.loads((tmp_path / "formal_v1_config.json").read_text(encoding="utf-8"))
    protocol = json.loads((tmp_path / "protocol.json").read_text(encoding="utf-8"))
    assert config["gpu_budget_hours"] == 80.0
    assert config["batch_size"] == 32
    assert config["stage_steps"] == [8000, 80000, 250000]
    assert protocol["gpu_budget_hours"] == 80.0
    assert protocol["test_during_search"] is False
    assert migration["training_protocol_changed"] is False
    assert migration["selection_protocol_changed"] is False
    assert migration["test_evaluated"] is False
    assert migration["used_gpu_hours_at_migration"] == 1.0
    assert (tmp_path / "protocol_migrations" / "migrations.jsonl").is_file()


def test_budget_migration_rejects_decrease_or_source_mismatch(tmp_path):
    materialize_run(tmp_path)
    module = load_module()
    with pytest.raises(ValueError):
        module.migrate(tmp_path, old_hours=40.0, new_hours=20.0, reason="invalid")
    with pytest.raises(RuntimeError):
        module.migrate(tmp_path, old_hours=30.0, new_hours=80.0, reason="mismatch")
