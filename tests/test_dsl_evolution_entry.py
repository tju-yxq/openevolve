import json
from types import SimpleNamespace

import pytest

from openevolve_adapter import evaluator
from scripts.run_dsl_evolution import (
    _ensure_compiler_manifest,
    _lineage_root,
    _next_forced_factor,
    _valid_factor_counts,
    get_parser,
)


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


def test_language_lineage_prefers_the_openevolve_island_identity():
    root = SimpleNamespace(id="root", parent_id=None, metadata={"island": 0})
    child = SimpleNamespace(id="child", parent_id="root", metadata={"island": 2})
    assert _lineage_root(child, {"root": root, "child": child}) == "island:2"


def test_evaluator_forwards_the_preregistered_seed(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout='{"valid": true}', stderr="")

    monkeypatch.setenv("NAS_SEED", "201")
    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
    assert evaluator.evaluate("candidate.dsl.json")["valid"] is True
    seed_index = captured["command"].index("--seed")
    assert captured["command"][seed_index + 1] == "201"


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
