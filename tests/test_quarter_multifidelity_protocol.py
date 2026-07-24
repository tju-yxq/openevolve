import hashlib
import json
from pathlib import Path

import numpy as np

from equivariant_nas.training.data_protocol import (
    is_epoch_validation_step,
    next_epoch_validation_step,
    steps_per_data_epoch,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_quarter_subset_is_complete_and_reproducible():
    artifact = ROOT / "data_splits/qm9_train_quarter_seed201.npz"
    manifest_path = artifact.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(artifact) as payload:
        local = payload["train_local_indices"]
        global_indices = payload["train_global_indices"]
    assert manifest["source_train_size"] == 110000
    assert manifest["subset_size"] == 27500
    assert len(local) == len(np.unique(local)) == 27500
    assert len(global_indices) == len(np.unique(global_indices)) == 27500
    assert local.min() >= 0 and local.max() < 110000
    assert manifest["validation_overlap"] == 0
    assert manifest["test_overlap"] == 0
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == manifest["npz_sha256"]


def test_validation_uses_real_epochs_for_each_data_phase():
    quarter_steps = steps_per_data_epoch(27500, 32)
    full_steps = steps_per_data_epoch(110000, 32)
    assert quarter_steps == 859
    assert full_steps == 3437
    assert next_epoch_validation_step(0, 0, quarter_steps, 10) == 8590
    assert next_epoch_validation_step(8000, 0, quarter_steps, 10) == 8590
    assert is_epoch_validation_step(8590, 0, quarter_steps, 10)
    assert next_epoch_validation_step(80000, 80000, full_steps, 10) == 114370
    assert is_epoch_validation_step(114370, 80000, full_steps, 10)


def test_protocol_is_8_to_4_to_2_and_keeps_test_locked():
    protocol = json.loads(
        (ROOT / "configs/quarter_multifidelity_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    stages = protocol["stages"]
    assert [stage["candidate_count"] for stage in stages] == [8, 4, 2]
    assert [stage["max_optimizer_steps"] for stage in stages] == [8000, 80000, 250000]
    assert [stage["training_data"] for stage in stages] == [
        "fixed_quarter",
        "fixed_quarter",
        "full_train",
    ]
    assert protocol["validation_interval_data_epochs"] == 10
    assert protocol["test_during_search"] is False
    assert protocol["final_selection_split"] == "validation"


def test_resumed_total_step_budget():
    total = 8 * 8000 + 4 * (80000 - 8000) + 2 * (250000 - 80000)
    assert total == 692000