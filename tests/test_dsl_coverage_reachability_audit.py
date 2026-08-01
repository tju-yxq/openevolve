import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_dsl_coverage_reachability_audit.py"
    spec = importlib.util.spec_from_file_location("run_dsl_coverage_reachability_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_expressivity_audit_distinguishes_exact_coverage_from_mechanism_witnesses():
    result = _module().audit_expressivity(seed=43)
    by_family = {item["family"]: item for item in result["families"]}

    assert result["witnesses"]["tfn_message"]["passed"]
    assert result["witnesses"]["egnn_coordinate"]["passed"]
    assert by_family["Equiformer V1"]["status"] == "exact_reference_covered"
    assert by_family["Tensor Field Network"]["status"] == "executable_mechanism_witness"
    assert "affine_coordinate_type" in by_family["EGNN"]["missing_capabilities"]
    assert "angle_triplet_geometry" in by_family["DimeNet"]["missing_capabilities"]
    assert "symmetric_contraction" in by_family["MACE"]["missing_capabilities"]
    assert not result["backend_dependency_audit"]["false_positive_support_report"]
    assert result["backend_dependency_audit"]["runtime_passed"]


def test_formal_v1_space_is_reference_reachable_but_not_strongly_connected():
    result = _module().audit_reachability()

    assert result["space"]["state_count"] == 72
    assert result["reachable_from_reference"] == 72
    assert result["maximum_shortest_path_from_reference"] <= 4
    assert not result["strongly_connected"]
    assert result["strong_component_count"] > 1
    assert result["sink_state_count"] > 0
    assert result["all_state_architecture_ids_unique"]
