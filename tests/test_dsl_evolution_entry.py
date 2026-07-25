import json
from types import SimpleNamespace

import pytest

from openevolve_adapter import evaluator
from scripts.run_dsl_evolution import (
    _ensure_compiler_manifest,
    _is_exact_constructor_capability_label_fix,
    _is_formal_root_parent_policy_fix,
    _lineage_root,
    _next_forced_factor,
    _unique_initial_parent,
    _valid_factor_counts,
    get_parser,
)


def _manifest_with_capabilities(constructor_capability):
    return {
        "compiler_semantics_version": "evoequilang-3",
        "rewrite_registry_hash": "rules-a",
        "region_registry": [
            {"factor_id": "F2.2", "region_id": "radial", "backend_capability": constructor_capability},
            {"factor_id": "F4.4", "region_id": "heads", "backend_capability": constructor_capability},
            {"factor_id": "F5.3", "region_id": "norm", "backend_capability": constructor_capability},
            {"factor_id": "F6.3", "region_id": "readout", "backend_capability": "exact_hybrid"},
        ],
    }


def test_dsl_evolution_entry_requires_program_task_evaluator_and_config():
    args = get_parser().parse_args([
        "--initial-program", "initial.json",
        "--task-contract", "task.json",
        "--evaluator-file", "evaluator.py",
        "--config", "openevolve.yaml",
        "--output", "run",
        "--iterations", "3",
    ])
    assert args.initial_program == "initial.json"
    assert args.task_contract == "task.json"
    assert args.iterations == 3
    assert args.max_steps == 0
    assert args.eval_interval_epochs == 10


def test_compiler_manifest_prevents_silent_resume_under_new_semantics(tmp_path):
    manifest = {
        "compiler_semantics_version": "evoequilang-2",
        "rewrite_registry_hash": "rules-a",
    }
    path = _ensure_compiler_manifest(tmp_path, manifest, database_has_programs=False)
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
    assert _ensure_compiler_manifest(tmp_path, manifest, database_has_programs=True) == path
    with pytest.raises(RuntimeError):
        _ensure_compiler_manifest(
            tmp_path,
            {**manifest, "rewrite_registry_hash": "rules-b"},
            database_has_programs=True,
        )


def test_legacy_database_without_manifest_is_not_adopted(tmp_path):
    with pytest.raises(RuntimeError):
        _ensure_compiler_manifest(tmp_path, {"compiler_semantics_version": "evoequilang-2"}, database_has_programs=True)


def test_exact_constructor_capability_label_fix_is_narrow_and_audited(tmp_path):
    old = _manifest_with_capabilities("exact_v1_constructor")
    new = _manifest_with_capabilities("exact_constructor")
    assert _is_exact_constructor_capability_label_fix(old, new)
    _ensure_compiler_manifest(tmp_path, old, database_has_programs=False)
    (tmp_path / "evolution.jsonl").write_text(
        json.dumps({
            "iteration": 8,
            "region_audit": {"factor_id": "F6.3"},
            "metrics": {
                "architecture_id": "readout-child",
                "valid": True,
                "test_evaluated": False,
                "lowering_mode": "exact_hybrid",
            },
        }) + "\n",
        encoding="utf-8",
    )
    path = _ensure_compiler_manifest(tmp_path, new, database_has_programs=True)
    assert json.loads(path.read_text(encoding="utf-8")) == new
    migrations = (tmp_path / "compiler_manifest_migrations" / "migrations.jsonl").read_text(encoding="utf-8")
    assert "exact_constructor_capability_label_fix" in migrations
    assert "readout-child" in migrations


def test_exact_constructor_capability_label_fix_refuses_affected_valid_candidate(tmp_path):
    old = _manifest_with_capabilities("exact_v1_constructor")
    new = _manifest_with_capabilities("exact_constructor")
    _ensure_compiler_manifest(tmp_path, old, database_has_programs=False)
    (tmp_path / "evolution.jsonl").write_text(
        json.dumps({
            "iteration": 9,
            "region_audit": {"factor_id": "F4.4"},
            "metrics": {
                "architecture_id": "constructor-child",
                "valid": True,
                "test_evaluated": False,
                "lowering_mode": "exact_constructor",
            },
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="admission semantics could have changed"):
        _ensure_compiler_manifest(tmp_path, new, database_has_programs=True)


def test_language_lineage_prefers_the_openevolve_island_identity():
    root = SimpleNamespace(id="root", parent_id=None, metadata={"island": 0})
    child = SimpleNamespace(id="child", parent_id="root", metadata={"island": 2})
    assert _lineage_root(child, {"root": root, "child": child}) == "island:2"


def test_formal_coverage_selects_the_unique_iteration_zero_parent():
    root = SimpleNamespace(id="root", iteration_found=0)
    evolved = SimpleNamespace(id="evolved", iteration_found=9)
    assert _unique_initial_parent({"root": root, "evolved": evolved}) is root
    with pytest.raises(RuntimeError, match="exactly one iteration-0 parent"):
        _unique_initial_parent({"a": root, "b": SimpleNamespace(id="b", iteration_found=0)})


def test_root_parent_policy_manifest_migration_is_narrow():
    old = _manifest_with_capabilities("exact_constructor")
    old["formal_v1_search_protocol"] = {"forced_factor_sequence": ["F2.2"], "valid_per_factor_target": 2}
    new = json.loads(json.dumps(old))
    new["formal_v1_search_protocol"]["forced_factor_parent_policy"] = "root_isolated"
    assert _is_formal_root_parent_policy_fix(old, new)
    changed = json.loads(json.dumps(new))
    changed["formal_v1_search_protocol"]["valid_per_factor_target"] = 3
    assert not _is_formal_root_parent_policy_fix(old, changed)


def test_evaluator_forwards_the_preregistered_seed(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout='{"valid": true}', stderr="")

    monkeypatch.setenv("NAS_SEED", "201")
    monkeypatch.setenv("NAS_TRAIN_SUBSET_FILE", "/tmp/quarter.npz")
    monkeypatch.setenv("NAS_EVAL_INTERVAL_EPOCHS", "10")
    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
    assert evaluator.evaluate("candidate.dsl.json")["valid"] is True
    seed_index = captured["command"].index("--seed")
    assert captured["command"][seed_index + 1] == "201"
    subset_index = captured["command"].index("--train-subset-file")
    assert captured["command"][subset_index + 1] == "/tmp/quarter.npz"
    interval_index = captured["command"].index("--eval-interval-epochs")
    assert captured["command"][interval_index + 1] == "10"


def test_formal_factor_coverage_routes_to_a_factor_with_remaining_quota(tmp_path):
    evolution = tmp_path / "evolution.jsonl"
    records = [
        {
            "iteration": 1,
            "region_audit": {"factor_id": "F2.2"},
            "metrics": {"valid": True, "test_evaluated": False, "architecture_id": "a"},
        },
        {
            "iteration": 2,
            "region_audit": {"factor_id": "F2.2"},
            "metrics": {"valid": True, "test_evaluated": False, "architecture_id": "b"},
        },
    ]
    evolution.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    counts = _valid_factor_counts(evolution)
    assert counts == {"F2.2": 2}
    assert _next_forced_factor(("F2.2", "F4.4", "F5.3", "F6.3"), counts, 2, 5) == "F4.4"
