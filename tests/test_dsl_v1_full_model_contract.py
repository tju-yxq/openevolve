from pathlib import Path


def test_official_v1_full_model_contract_freezes_construction_and_forward(tmp_path):
    from scripts.audit_equiformer_v1_full_model_contract import build_audit

    official_root = Path(__file__).resolve().parents[2] / "equiformer"
    evidence = build_audit(tmp_path, official_root)
    summary = evidence["summary"]

    assert summary["status"] == "pass"
    assert summary["constructor_config"]["num_layers"] == 2
    assert summary["edge_count"] > 0
    assert summary["parameter_tensor_count"] > 0
    assert summary["forward_max_abs_error"] <= 1.0e-6
    assert summary["intermediate_max_abs_error"] <= 1.0e-6
    assert summary["position_gradient_max_abs_error"] <= 1.0e-5
    assert summary["parameter_gradient_max_abs_error"] <= 1.0e-5
    assert summary["missing_parameter_gradient_mismatches"] == []
    assert summary["official_constructor_used_for_dsl_execution"] is False
    assert summary["counts_as_full_model_lowering"] is False

    assert "atom_embed" in evidence["official_trace"]
    assert "rbf" in evidence["official_trace"]
    assert "edge_degree_embedding" in evidence["official_trace"]
    assert "block_0" in evidence["official_trace"]
    assert "block_1" in evidence["official_trace"]
    assert "norm" in evidence["official_trace"]
    assert "head" in evidence["official_trace"]
    assert "scale_scatter" in evidence["official_trace"]
