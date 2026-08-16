"""Frozen protocol model for online V3 multi-fidelity evolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class OnlineV3Protocol:
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "OnlineV3Protocol":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OnlineV3Protocol":
        protocol = cls(dict(value))
        protocol.validate()
        return protocol

    def validate(self) -> None:
        required = {
            "protocol_version", "seed", "batch_size", "dataset", "search",
            "parent_sampling", "promotion", "storage", "test_policy",
        }
        missing = required - set(self.raw)
        if missing:
            raise ValueError(f"online protocol omitted keys: {sorted(missing)}")
        search = self.raw["search"]
        if int(search["valid_child_target"]) <= 0:
            raise ValueError("valid_child_target must be positive")
        if int(search["maximum_generation_attempts"]) < int(search["valid_child_target"]):
            raise ValueError("maximum_generation_attempts cannot be below valid_child_target")
        if int(search["num_islands"]) <= 0:
            raise ValueError("num_islands must be positive")
        if int(self.raw["batch_size"]) != 8:
            raise ValueError("formal V3 online protocol requires batch_size=8")
        weights = self.raw["parent_sampling"]
        expected = {"elite_archive", "island_uniform", "island_rank_weighted", "novelty", "global_cross_island"}
        if set(weights) != expected:
            raise ValueError(f"parent_sampling must contain exactly {sorted(expected)}")
        if any(float(value) < 0 for value in weights.values()) or abs(sum(map(float, weights.values())) - 1.0) > 1e-9:
            raise ValueError("parent sampling probabilities must be non-negative and sum to one")
        stages = self.raw["promotion"]["stages"]
        endpoints = [int(item["endpoint_steps"]) for item in stages]
        counts = [int(item["candidate_count"]) for item in stages]
        if endpoints != [20000, 80000, 250000] or counts != [60, 15, 10]:
            raise ValueError("formal online stages must be 60@20k -> 15@80k -> 10@250k")
        rule = self.raw["promotion"]["from_20k_to_80k"]
        if [int(rule[key]) for key in ("top_validation", "top_novelty", "random_lower_half")] != [10, 3, 2]:
            raise ValueError("20k promotion must be top10 + novelty3 + lower-half random2")
        if self.raw["test_policy"].get("during_search") is not False:
            raise ValueError("test split must remain inaccessible during search")
        dataset = self.raw["dataset"]
        for key in ("dataset_id", "dataset_manifest_sha256", "quarter_subset_sha256", "equivariance_contract_sha256"):
            if not str(dataset.get(key, "")):
                raise ValueError(f"dataset identity omitted {key}")

    @property
    def content_hash(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

