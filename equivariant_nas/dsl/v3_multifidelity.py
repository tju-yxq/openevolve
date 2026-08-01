"""Frozen 8 -> 4 -> 2 multi-fidelity protocol for iterative V3 evolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class V3FidelityStage:
    name: str
    endpoint_steps: int
    candidate_count: int
    training_data: str
    resume_from: str = ""
    optimizer_transition: str = "resume_full_state"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "V3FidelityStage":
        required = {
            "name",
            "endpoint_steps",
            "candidate_count",
            "training_data",
            "resume_from",
            "optimizer_transition",
        }
        if set(value) != required:
            raise ValueError("V3 fidelity stage must contain exactly {}".format(sorted(required)))
        return cls(
            name=str(value["name"]),
            endpoint_steps=int(value["endpoint_steps"]),
            candidate_count=int(value["candidate_count"]),
            training_data=str(value["training_data"]),
            resume_from=str(value["resume_from"]),
            optimizer_transition=str(value["optimizer_transition"]),
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "endpoint_steps": self.endpoint_steps,
            "candidate_count": self.candidate_count,
            "training_data": self.training_data,
            "resume_from": self.resume_from,
            "optimizer_transition": self.optimizer_transition,
        }


@dataclass(frozen=True)
class V3MultiFidelityProtocol:
    protocol_version: str
    cycle_count: int
    batch_size: int
    seed: int
    dataset_id: str
    dataset_manifest_sha256: str
    quarter_subset_sha256: str
    equivariance_contract_sha256: str
    selection_metric: str
    selection_direction: str
    test_during_search: bool
    stages: Tuple[V3FidelityStage, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "V3MultiFidelityProtocol":
        required = {
            "protocol_version",
            "cycle_count",
            "batch_size",
            "seed",
            "dataset_id",
            "dataset_manifest_sha256",
            "quarter_subset_sha256",
            "equivariance_contract_sha256",
            "selection_metric",
            "selection_direction",
            "test_during_search",
            "stages",
        }
        if set(value) != required:
            raise ValueError("V3 multi-fidelity protocol must contain exactly {}".format(sorted(required)))
        stages = tuple(V3FidelityStage.from_mapping(item) for item in value["stages"])
        protocol = cls(
            protocol_version=str(value["protocol_version"]),
            cycle_count=int(value["cycle_count"]),
            batch_size=int(value["batch_size"]),
            seed=int(value["seed"]),
            dataset_id=str(value["dataset_id"]),
            dataset_manifest_sha256=str(value["dataset_manifest_sha256"]),
            quarter_subset_sha256=str(value["quarter_subset_sha256"]),
            equivariance_contract_sha256=str(value["equivariance_contract_sha256"]),
            selection_metric=str(value["selection_metric"]),
            selection_direction=str(value["selection_direction"]),
            test_during_search=bool(value["test_during_search"]),
            stages=stages,
        )
        protocol.validate()
        return protocol

    def validate(self) -> None:
        if self.cycle_count <= 0:
            raise ValueError("cycle_count must be positive")
        if self.batch_size != 8:
            raise ValueError("formal V3 evolution requires batch_size=8")
        if self.test_during_search:
            raise ValueError("test split must remain unavailable during search")
        if self.selection_direction not in {"min", "max"}:
            raise ValueError("selection_direction must be min or max")
        frozen = (
            self.dataset_id,
            self.dataset_manifest_sha256,
            self.quarter_subset_sha256,
            self.equivariance_contract_sha256,
            self.selection_metric,
        )
        if any(not item for item in frozen):
            raise ValueError("dataset, subset, equivariance, and selection identities must be frozen")
        expected = (
            V3FidelityStage("quarter_8k", 8000, 8, "fixed_quarter", "", "new_optimizer"),
            V3FidelityStage(
                "quarter_80k",
                80000,
                4,
                "fixed_quarter",
                "quarter_8k",
                "resume_full_state",
            ),
            V3FidelityStage(
                "full_250k",
                250000,
                2,
                "full_train",
                "quarter_80k",
                "model_only_optimizer_restart",
            ),
        )
        if self.stages != expected:
            raise ValueError("formal V3 stages must be exactly 8k/80k/250k with 8->4->2 admission")

    @property
    def cohort_size(self) -> int:
        return self.stages[0].candidate_count

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "cycle_count": self.cycle_count,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "quarter_subset_sha256": self.quarter_subset_sha256,
            "equivariance_contract_sha256": self.equivariance_contract_sha256,
            "selection_metric": self.selection_metric,
            "selection_direction": self.selection_direction,
            "test_during_search": self.test_during_search,
            "stages": [item.to_dict() for item in self.stages],
        }

    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rank_v3_stage_records(
    records: Sequence[Mapping[str, Any]],
    *,
    stage: V3FidelityStage,
    protocol: V3MultiFidelityProtocol,
) -> Tuple[Mapping[str, Any], ...]:
    """Validate one completed cohort stage and return its admitted candidates."""

    unique = {}
    for record in records:
        architecture_id = str(record.get("architecture_id", ""))
        if not architecture_id:
            raise ValueError("stage result omitted architecture_id")
        if architecture_id in unique:
            raise ValueError("stage result contains duplicate architecture_id")
        if str(record.get("protocol_hash", "")) != protocol.content_hash():
            raise ValueError("stage result changed the frozen multi-fidelity protocol")
        if str(record.get("dataset_manifest_sha256", "")) != protocol.dataset_manifest_sha256:
            raise ValueError("stage result changed the frozen dataset")
        if str(record.get("quarter_subset_sha256", "")) != protocol.quarter_subset_sha256:
            raise ValueError("stage result changed the frozen quarter subset")
        if str(record.get("equivariance_contract_sha256", "")) != protocol.equivariance_contract_sha256:
            raise ValueError("stage result changed the frozen equivariance rules")
        if bool(record.get("test_evaluated", False)):
            raise ValueError("test leakage detected during V3 evolution")
        if int(record.get("endpoint_step", 0)) != stage.endpoint_steps:
            raise ValueError("stage result did not reach its frozen endpoint")
        metric = record.get(protocol.selection_metric)
        if metric is None:
            raise ValueError("stage result omitted selection metric {}".format(protocol.selection_metric))
        unique[architecture_id] = record
    if len(unique) != stage.candidate_count:
        raise ValueError(
            "stage {} requires exactly {} unique completed candidates, got {}".format(
                stage.name,
                stage.candidate_count,
                len(unique),
            )
        )
    reverse = protocol.selection_direction == "max"
    ordered = sorted(unique.values(), key=lambda item: float(item[protocol.selection_metric]), reverse=reverse)
    next_index = protocol.stages.index(stage) + 1
    admitted = protocol.stages[next_index].candidate_count if next_index < len(protocol.stages) else 1
    return tuple(ordered[:admitted])
