import pytest

from equivariant_nas.fidelity_trust import assess_fidelity_trust


def test_rank_reversal_disables_proxy():
    report = assess_fidelity_trust(
        {"baseline": 3.4780, "factorized": 2.8402, "random": 3.1377},
        {"baseline": 0.753540, "factorized": 1.170596, "random": 0.705551},
        top_k=1,
        minimum_cohort=3,
    )
    assert report.spearman == pytest.approx(-0.5)
    assert report.kendall_tau == pytest.approx(-1.0 / 3.0)
    assert report.top_k_recall == 0.0
    assert not report.trustworthy


def test_consistent_proxy_passes_when_cohort_is_large_enough():
    proxy = {str(index): float(index) for index in range(8)}
    reference = {str(index): float(index) * 0.5 + 1.0 for index in range(8)}
    report = assess_fidelity_trust(proxy, reference, top_k=2)
    assert report.spearman == pytest.approx(1.0)
    assert report.kendall_tau == pytest.approx(1.0)
    assert report.top_k_recall == 1.0
    assert report.trustworthy
