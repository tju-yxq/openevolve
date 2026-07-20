import pytest

from equivariant_nas.search_statistics import (
    best_so_far,
    bootstrap_mean_difference_ci,
    exact_paired_sign_flip_pvalue,
    summarize_trajectory,
)


def test_trajectory_summary_rewards_early_good_architectures():
    early = summarize_trajectory([0.8, 0.7, 0.6], 1.0, 0.65)
    late = summarize_trajectory([0.95, 0.9, 0.6], 1.0, 0.65)
    assert early.final_best_mae == late.final_best_mae == 0.6
    assert early.normalized_best_so_far_auc < late.normalized_best_so_far_auc
    assert early.time_to_threshold == late.time_to_threshold == 3
    assert best_so_far([0.9, 1.1, 0.8], 1.0) == (1.0, 0.9, 0.9, 0.8)


def test_exact_sign_flip_and_bootstrap_have_lower_is_better_semantics():
    full = [0.5, 0.6, 0.55]
    random = [0.8, 0.9, 0.85]
    assert exact_paired_sign_flip_pvalue(full, random) == pytest.approx(0.125)
    lower, upper = bootstrap_mean_difference_ci(full, random, samples=1000)
    assert upper < 0.0
