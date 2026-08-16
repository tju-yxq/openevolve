import pytest

from equivariant_nas.dsl.online_promotion import select_20k_to_80k, select_80k_to_250k


HASH = "protocol"


def records(count, endpoint):
    return [{
        "architecture_id": f"a{index:02d}", "endpoint_step": endpoint,
        "validation_alpha_mae": float(index), "novelty_score": float(count - index),
        "protocol_hash": HASH, "test_evaluated": False,
        "checkpoint": f"/{index}.pt", "checkpoint_global_step": endpoint,
    } for index in range(count)]


def test_20k_selection_is_unique_and_reproducible():
    first = select_20k_to_80k(records(60, 20000), protocol_hash=HASH, seed=201)
    second = select_20k_to_80k(records(60, 20000), protocol_hash=HASH, seed=201)
    ids = [item["architecture_id"] for item in first["selected"]]
    assert len(ids) == len(set(ids)) == 15
    assert [item["promotion_reason"] for item in first["selected"]].count("validation_top10") == 10
    assert [item["promotion_reason"] for item in first["selected"]].count("novelty_top3") == 3
    assert [item["promotion_reason"] for item in first["selected"]].count("random_lower_half") == 2
    assert first == second


def test_promotion_rejects_test_leakage():
    values = records(60, 20000); values[0]["test_evaluated"] = True
    with pytest.raises(ValueError, match="test leakage"):
        select_20k_to_80k(values, protocol_hash=HASH)


def test_80k_top_ten():
    snapshot = select_80k_to_250k(records(15, 80000), protocol_hash=HASH)
    assert [item["architecture_id"] for item in snapshot["selected"]] == [f"a{i:02d}" for i in range(10)]

