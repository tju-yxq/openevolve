import pytest

from equivariant_nas.budget import BudgetLedger


def test_repeated_stage_estimate_uses_robust_observed_cost(tmp_path):
    ledger = BudgetLedger(str(tmp_path / "budget.jsonl"), 5.0)
    for seconds in (1000.0, 1100.0, 5000.0):
        ledger.append({"stage": "steps5000", "gpu_seconds": seconds})

    estimate = ledger.estimate_stage_gpu_hours(
        "steps5000", fallback_gpu_hours=0.75, safety_multiplier=1.2
    )
    assert estimate == pytest.approx(1100.0 * 1.2 / 3600.0)


def test_unseen_stage_uses_fallback(tmp_path):
    ledger = BudgetLedger(str(tmp_path / "budget.jsonl"), 5.0)
    assert ledger.estimate_stage_gpu_hours("steps20000", 2.5) == 2.5
