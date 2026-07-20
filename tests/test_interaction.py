import pytest

from equivariant_nas.interaction import (
    interaction_contrast,
    rescue_requires_counterfactual,
)


def test_rescue_chain_triggers_counterfactual():
    assert rescue_requires_counterfactual(0.7535, 1.2996, 0.6399)
    assert not rescue_requires_counterfactual(0.7535, 0.7000, 0.6500)


def test_interaction_contrast_has_difference_in_differences_semantics():
    result = interaction_contrast(1.0, 1.2, 0.9, 0.7)
    assert result.factor_b_gain_without_a == pytest.approx(0.1)
    assert result.factor_b_gain_with_a == pytest.approx(0.5)
    assert result.epistasis_mae == pytest.approx(-0.4)
