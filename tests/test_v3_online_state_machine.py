import json

from equivariant_nas.dsl.online_state import OnlineEvolutionState, atomic_write_json


def test_invalid_candidate_does_not_take_valid_slot(tmp_path):
    state = OnlineEvolutionState(tmp_path / "state.json", protocol_hash="p", valid_target=2, maximum_attempts=3)
    assert not state.add_generation_attempt({"architecture_id": "bad", "valid": False, "endpoint_step": 20000, "test_evaluated": False})
    assert state.add_generation_attempt({"architecture_id": "ok", "valid": True, "endpoint_step": 20000, "test_evaluated": False, "protocol_hash": "p"})
    assert len(state.data["completed_20k"]) == 1


def test_atomic_state_file_is_complete_json(tmp_path):
    path = tmp_path / "nested" / "state.json"
    atomic_write_json(path, {"large": list(range(1000))})
    assert json.loads(path.read_text())["large"][-1] == 999
    assert not list(path.parent.glob("*.tmp"))


def test_test_is_exactly_once_and_only_for_frozen_winner(tmp_path):
    state = OnlineEvolutionState(tmp_path / "state.json", protocol_hash="p")
    state.data.update(stage="evaluate_test", winner={"architecture_id": "winner"}); state.save()
    state.record_test({"architecture_id": "winner", "split": "test"})
    try:
        state.record_test({"architecture_id": "winner", "split": "test"})
        assert False
    except RuntimeError:
        pass
