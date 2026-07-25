import json
from types import SimpleNamespace

import pytest

from scripts.run_dsl_evolution import _ensure_compiler_manifest, _lineage_root, get_parser


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
