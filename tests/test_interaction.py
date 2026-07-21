import pytest
import json

from equivariant_nas.interaction import (
    interaction_contrast,
    rescue_requires_counterfactual,
    resolved_counterfactual_credits,
)


def test_rescue_chain_triggers_counterfactual():
    assert rescue_requires_counterfactual(0.7535, 1.2996, 0.6399)
    assert not rescue_requires_counterfactual(0.7535, 0.7000, 0.6500)


def test_interaction_contrast_has_difference_in_differences_semantics():
    result = interaction_contrast(1.0, 1.2, 0.9, 0.7)
    assert result.factor_b_gain_without_a == pytest.approx(0.1)
    assert result.factor_b_gain_with_a == pytest.approx(0.5)
    assert result.epistasis_mae == pytest.approx(-0.4)


def test_resolved_credit_uses_counterfactual_main_effect_once(tmp_path):
    source = tmp_path / "resolved.json"
    source.write_text(
        json.dumps(
            [
                {
                    "status": "resolved",
                    "selected_factor": "OPERATOR",
                    "child_architecture_id": "child",
                    "counterfactual_architecture_id": "sibling",
                    "interaction_contrast": {
                        "factor_b_gain_without_a": 0.177,
                        "factor_b_gain_with_a": 0.660,
                        "epistasis_mae": -0.483,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    credits = resolved_counterfactual_credits(str(source), set())
    assert credits[0]["mae_gain"] == pytest.approx(0.177)
    assert resolved_counterfactual_credits(
        str(source), {credits[0]["key"]}
    ) == []
