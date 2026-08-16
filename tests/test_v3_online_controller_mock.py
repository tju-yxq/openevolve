from pathlib import Path

from equivariant_nas.dsl.online_controller import OnlineV3Controller, _validated_checkpoint
from equivariant_nas.dsl.online_protocol import OnlineV3Protocol


def test_full_60_15_10_mock_and_resume(tmp_path):
    protocol = OnlineV3Protocol.load(Path(__file__).parents[1] / "configs" / "dsl_v3_online_20k_60_protocol.json")
    calls = []

    def generate(attempt, population, parent):
        return {"architecture_id": f"arch-{attempt:03d}", "parent_architecture_id": parent["architecture_id"] if parent else "seed", "program": f"/mock/arch-{attempt:03d}.json", "novelty_score": float(attempt)}

    def train(candidate, endpoint, checkpoint):
        calls.append((candidate["architecture_id"], endpoint))
        return {"valid": True, "endpoint_step": endpoint, "checkpoint_global_step": endpoint, "checkpoint": f"/mock/{candidate['architecture_id']}-{endpoint}.pt", "validation_alpha_mae": 1000.0 - float(candidate["architecture_id"].split("-")[-1]) + endpoint / 1e7, "test_evaluated": False, "start_global_step": {20000: 0, 80000: 20000, 250000: 80000}[endpoint], "training_data": "full_train" if endpoint == 250000 else "fixed_quarter"}

    def evaluate_test(winner):
        return {"architecture_id": winner["architecture_id"], "split": "test", "evaluation_only": True, "test_alpha_mae": .1}

    controller = OnlineV3Controller(tmp_path, protocol, generator=generate, trainer=train, test_evaluator=evaluate_test)
    result = controller.run()
    assert result["stage"] == "ready_for_tos_archive"
    assert [len(result[key]) for key in ("completed_20k", "completed_80k", "completed_250k")] == [60, 15, 10]
    assert result["test_evaluated"] is True
    assert {item["island"] for item in result["completed_20k"]} == set(range(5))
    assert len(calls) == 85
    resumed = OnlineV3Controller(tmp_path, protocol, generator=generate, trainer=train, test_evaluator=evaluate_test).run()
    assert resumed["stage"] == "ready_for_tos_archive"
    assert len(calls) == 85


def test_checkpoint_last_is_normalized_and_bulky_evidence_is_not_in_state(tmp_path):
    protocol = OnlineV3Protocol.load(Path(__file__).parents[1] / "configs" / "dsl_v3_online_20k_60_protocol.json")
    controller = OnlineV3Controller(tmp_path, protocol, generator=lambda *_: {}, trainer=lambda *_: {}, test_evaluator=lambda *_: {})
    record = controller._fidelity_record({
        "architecture_id": "arch",
        "valid": True,
        "endpoint_step": 20000,
        "checkpoint_last": "/checkpoints/arch-20000.pth",
        "validation_alpha_mae": 0.5,
        "test_evaluated": False,
        "training_data": "fixed_quarter",
        "lowering_plan": {"large": "x" * 10000},
        "compiler_obligations": [{"large": "x" * 10000}],
    }, 20000)
    assert record["checkpoint"] == "/checkpoints/arch-20000.pth"
    assert record["checkpoint_global_step"] == 20000
    assert record["metrics_by_fidelity"]["20000"]["checkpoint"] == "/checkpoints/arch-20000.pth"
    assert _validated_checkpoint(record, expected_step=20000) == "/checkpoints/arch-20000.pth"
    assert "lowering_plan" not in record
    assert "compiler_obligations" not in record
