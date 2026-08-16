"""Frozen, reproducible promotion and cross-fidelity trust reporting."""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence


def _validate(records: Sequence[Mapping[str, Any]], endpoint: int, protocol_hash: str) -> list[Mapping[str, Any]]:
    unique = {}
    for record in records:
        architecture_id = str(record.get("architecture_id", ""))
        if not architecture_id or architecture_id in unique:
            raise ValueError("promotion records require unique architecture_id values")
        if int(record.get("endpoint_step", 0)) != endpoint:
            raise ValueError("candidate did not reach the required endpoint")
        if str(record.get("protocol_hash", "")) != protocol_hash:
            raise ValueError("candidate protocol identity changed")
        if record.get("test_evaluated") is not False:
            raise ValueError("test leakage detected")
        if record.get("validation_alpha_mae") is None:
            raise ValueError("candidate omitted validation_alpha_mae")
        unique[architecture_id] = record
    return list(unique.values())


def select_20k_to_80k(records: Sequence[Mapping[str, Any]], *, protocol_hash: str, seed: int = 201) -> Mapping[str, Any]:
    valid = _validate(records, 20000, protocol_hash)
    if len(valid) != 60:
        raise ValueError(f"20k selection requires 60 candidates, got {len(valid)}")
    ranked = sorted(valid, key=lambda item: (float(item["validation_alpha_mae"]), str(item["architecture_id"])))
    top = ranked[:10]
    selected_ids = {str(item["architecture_id"]) for item in top}
    novelty_pool = [item for item in valid if str(item["architecture_id"]) not in selected_ids]
    novelty = sorted(novelty_pool, key=lambda item: (-float(item.get("novelty_score", 0.0)), str(item["architecture_id"])))[:3]
    selected_ids.update(str(item["architecture_id"]) for item in novelty)
    lower_half = [item for item in ranked[len(ranked) // 2:] if str(item["architecture_id"]) not in selected_ids]
    random_low = random.Random(seed).sample(lower_half, 2)
    selected = top + novelty + random_low
    snapshot = {
        "source_endpoint_step": 20000,
        "ranking": [dict(item, rank_20k=index + 1) for index, item in enumerate(ranked)],
        "selected": [
            dict(item, promotion_reason=("validation_top10" if index < 10 else "novelty_top3" if index < 13 else "random_lower_half"))
            for index, item in enumerate(selected)
        ],
        "seed": seed,
        "protocol_hash": protocol_hash,
    }
    snapshot["snapshot_sha256"] = _hash_without_self(snapshot)
    return snapshot


def select_80k_to_250k(records: Sequence[Mapping[str, Any]], *, protocol_hash: str) -> Mapping[str, Any]:
    valid = _validate(records, 80000, protocol_hash)
    if len(valid) != 15:
        raise ValueError(f"80k selection requires 15 candidates, got {len(valid)}")
    ranked = sorted(valid, key=lambda item: (float(item["validation_alpha_mae"]), str(item["architecture_id"])))
    snapshot = {
        "source_endpoint_step": 80000,
        "ranking": [dict(item, rank_80k=index + 1) for index, item in enumerate(ranked)],
        "selected": [dict(item, promotion_reason="validation_top10") for item in ranked[:10]],
        "protocol_hash": protocol_hash,
    }
    snapshot["snapshot_sha256"] = _hash_without_self(snapshot)
    return snapshot


def fidelity_trust_report(low: Sequence[Mapping[str, Any]], high: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    low_map = {str(item["architecture_id"]): float(item["validation_alpha_mae"]) for item in low}
    high_map = {str(item["architecture_id"]): float(item["validation_alpha_mae"]) for item in high}
    ids = sorted(set(low_map) & set(high_map))
    if len(ids) < 2:
        return {"paired_count": len(ids), "spearman_rank_correlation": None}
    low_ranks = _ranks([low_map[key] for key in ids])
    high_ranks = _ranks([high_map[key] for key in ids])
    mean_low, mean_high = sum(low_ranks) / len(ids), sum(high_ranks) / len(ids)
    numerator = sum((a - mean_low) * (b - mean_high) for a, b in zip(low_ranks, high_ranks))
    denominator = math.sqrt(sum((a - mean_low) ** 2 for a in low_ranks) * sum((b - mean_high) ** 2 for b in high_ranks))
    return {"paired_count": len(ids), "spearman_rank_correlation": numerator / denominator if denominator else 0.0}


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    for rank, index in enumerate(order, 1):
        result[index] = float(rank)
    return result


def _hash_without_self(payload: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "snapshot_sha256"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

