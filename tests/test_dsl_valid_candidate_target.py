import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_dsl_evolution.py"
    spec = importlib.util.spec_from_file_location("run_dsl_evolution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_valid_candidate_count_excludes_parent_rejections_test_leaks_and_duplicates(tmp_path):
    records = [
        {"iteration": 0, "metrics": {"valid": True, "test_evaluated": False, "architecture_id": "parent"}},
        {"iteration": 1, "metrics": {"valid": True, "test_evaluated": False, "architecture_id": "a"}},
        {"iteration": 2, "metrics": {"valid": False, "test_evaluated": False, "architecture_id": "b"}},
        {"iteration": 3, "metrics": {"valid": True, "test_evaluated": True, "architecture_id": "c"}},
        {"iteration": 4, "metrics": {"valid": True, "test_evaluated": False, "architecture_id": "a"}},
        {"iteration": 5, "metrics": {"valid": True, "test_evaluated": False, "architecture_id": "d"}},
    ]
    path = tmp_path / "evolution.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    assert _module()._valid_candidate_ids(path) == {"a", "d"}
