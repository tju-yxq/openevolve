from equivariant_nas.dsl.online_promotion import fidelity_trust_report


def test_fidelity_report_pairs_by_identity_not_raw_scale():
    low = [{"architecture_id": key, "validation_alpha_mae": value} for key, value in zip("abc", [3, 2, 1])]
    high = [{"architecture_id": key, "validation_alpha_mae": value} for key, value in zip("abc", [300, 200, 100])]
    report = fidelity_trust_report(low, high)
    assert report == {"paired_count": 3, "spearman_rank_correlation": 1.0}

