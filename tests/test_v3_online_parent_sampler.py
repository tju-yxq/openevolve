import pytest

from equivariant_nas.dsl.fidelity_aware_parent_sampler import FidelityAwareParentSampler


PROBABILITIES = {"elite_archive": .30, "island_uniform": .25, "island_rank_weighted": .20, "novelty": .15, "global_cross_island": .10}


def test_probabilities_are_validated():
    with pytest.raises(ValueError):
        FidelityAwareParentSampler({**PROBABILITIES, "elite_archive": .31})


def test_negative_mae_does_not_collapse_rank_weights():
    candidates = [
        {"architecture_id": "a", "highest_completed_fidelity": 20000, "validation_alpha_mae": -3.0},
        {"architecture_id": "b", "highest_completed_fidelity": 20000, "validation_alpha_mae": -2.0},
        {"architecture_id": "c", "highest_completed_fidelity": 20000, "validation_alpha_mae": -1.0},
    ]
    weights = FidelityAwareParentSampler.rank_weights(candidates)
    assert weights[0] > weights[1] > weights[2] > 0


def test_rank_weighting_rejects_mixed_fidelity():
    with pytest.raises(ValueError, match="different fidelities"):
        FidelityAwareParentSampler.rank_weights([
            {"highest_completed_fidelity": 20000, "validation_alpha_mae": .1},
            {"highest_completed_fidelity": 80000, "validation_alpha_mae": .2},
        ])

