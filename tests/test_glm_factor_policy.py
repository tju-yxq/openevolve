from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "dsl_glm_mutation_novelty_20260730"
    / "scripts"
    / "generate_glm_local_mutants.py"
)
SPEC = importlib.util.spec_from_file_location("glm_factor_policy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_minimum_factor_targets_rejects_impossible_budget():
    with pytest.raises(ValueError):
        MODULE.minimum_factor_targets(("E6.1", "E6.2"), total=1, minimum_per_factor=1)


def test_minimum_coverage_precedes_epsilon_exploitation():
    selected, snapshot = MODULE.select_annealed_epsilon_factor(
        sequence=("E6.1", "E6.2"),
        targets={"E6.1": 1, "E6.2": 1},
        accepted=({"factor_id": "E6.1"},),
        experience=(),
        attempt=1,
        seed=201,
        epsilon_start=0.75,
        epsilon_end=0.12,
        epsilon_decay_attempts=24.0,
        stagnation_patience=6,
        stagnation_boost=0.20,
    )
    assert selected == "E6.2"
    assert snapshot["selection_mode"] == "minimum_coverage"


def test_epsilon_anneals_but_keeps_nonzero_floor():
    common = dict(
        sequence=("E6.1", "E6.2"),
        targets={"E6.1": 0, "E6.2": 0},
        accepted=(),
        experience=(),
        seed=201,
        epsilon_start=0.75,
        epsilon_end=0.12,
        epsilon_decay_attempts=24.0,
        stagnation_patience=0,
        stagnation_boost=0.20,
    )
    _, early = MODULE.select_annealed_epsilon_factor(attempt=1, **common)
    _, late = MODULE.select_annealed_epsilon_factor(attempt=200, **common)
    assert early["epsilon"] > late["epsilon"]
    assert late["epsilon"] >= 0.12


def test_multiseed_acceptance_has_more_exploitation_weight():
    experience = (
        {
            "factor_id": "E6.1",
            "outcome": "accepted_multiseed",
            "operator_signature": ["core.irrep_concat@1"],
        },
        {
            "factor_id": "E6.2",
            "outcome": "rejected_multiseed",
            "operator_signature": ["core.tensor_product@1"],
        },
    )
    stats = MODULE.factor_policy_statistics(("E6.1", "E6.2"), experience, ())
    assert stats["E6.1"]["exploitation_score"] > stats["E6.2"]["exploitation_score"]

