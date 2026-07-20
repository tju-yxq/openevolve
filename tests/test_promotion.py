import pytest

from equivariant_nas.promotion import UntrustedFidelityError, authorize_promotion


def test_calibration_promotion_cannot_influence_selection():
    assert authorize_promotion("calibration") is False


def test_selection_promotion_requires_trusted_report():
    with pytest.raises(UntrustedFidelityError):
        authorize_promotion("selection")
    with pytest.raises(UntrustedFidelityError):
        authorize_promotion("selection", {"trustworthy": False, "reason": "rank reversal"})
    assert authorize_promotion("selection", {"trustworthy": True}) is True
