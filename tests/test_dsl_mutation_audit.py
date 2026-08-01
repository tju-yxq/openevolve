import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_dsl_mutation_audit.py"
    spec = importlib.util.spec_from_file_location("run_dsl_mutation_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_controlled_mutation_audit_separates_validity_novelty_and_equivariance(tmp_path):
    summary = _module().run_audit(tmp_path, seed=31, rotations=2, permutations=2)
    by_id = {item["mutation_id"]: item for item in summary["mutations"]}

    assert summary["counts"]["generated"] == 8
    assert summary["counts"]["valid"] == 6
    assert summary["all_expectations_met"]
    assert not by_id["M1_identity_insertion"]["canonical_unique_vs_parent"]
    assert by_id["M4_parallel_branch_concat"]["mechanism_novel_by_preregistered_class"]
    assert by_id["M4_parallel_branch_concat"]["behavior_distance_vs_parent"] < 1.0e-12
    assert by_id["M5_tensor_product_path"]["mechanism_novel_by_preregistered_class"]
    assert by_id["M6_pool_select_swap"]["behavior_distance_vs_parent"] < 1.0e-12
    assert not by_id["M7_illegal_scalar_activation"]["typecheck_passed"]
    assert not by_id["M8_illegal_irrep_creation"]["typecheck_passed"]
    assert all(
        item["equivariance"]["passed"]
        for item in summary["mutations"]
        if item["typecheck_passed"]
    )
    assert (tmp_path / "mutation_audit.json").is_file()
    assert (tmp_path / "mutation_audit.csv").is_file()
    assert (tmp_path / "DSL变异创新性与等变性测试报告.md").is_file()
