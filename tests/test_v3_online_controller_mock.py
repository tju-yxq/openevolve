from pathlib import Path

from equivariant_nas.dsl.online_controller import OnlineV3Controller
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
