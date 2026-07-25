"""Immutable task and resource contracts kept outside candidate programs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from .diagnostics import DSLValidationError, Diagnostic
from .groups import GroupSpec
from .types import EquivariantType


def task_reasoning_context(task: "TaskContract") -> Mapping[str, Any]:
    """Return trusted scientific facts exposed to candidate-generation agents."""

    context = dict(task.metadata)
    if task.task_id == "qm9_alpha":
        context.update({
            "target_semantics": "QM9 alpha is the graph-level isotropic scalar polarizability target, not a rank-2 or tensor-valued output target.",
            "representation_semantics": "Hidden l>0 irreps can still improve the scalar prediction by legal equivariant coupling into l=0 before readout.",
            "output_constraint": "The final prediction must remain one invariant graph scalar.",
        })
    return context


def validate_task_reasoning(task: "TaskContract", payload: Mapping[str, Any]) -> None:
    """Apply deterministic task-specific scientific guards before synthesis."""

    if task.task_id != "qm9_alpha":
        return
    from ..semantics import ScientificSemanticsError, validate_qm9_alpha_reasoning

    try:
        validate_qm9_alpha_reasoning(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True))
    except ScientificSemanticsError as exc:
        raise DSLValidationError([
            Diagnostic(
                "E_SCIENCE_001",
                "planner claim contradicts the trusted QM9 alpha target semantics",
                actual=str(exc),
                repairs=(
                    "treat alpha as an invariant graph scalar",
                    "describe l>0 features as hidden representations that may couple into l=0",
                ),
            )
        ])


@dataclass(frozen=True)
class ResourceContract:
    max_parameters: int
    max_peak_memory_bytes: int
    max_step_time_ratio: float
    max_static_flops: int = 0

    def __post_init__(self) -> None:
        if self.max_parameters <= 0 or self.max_peak_memory_bytes <= 0 or self.max_step_time_ratio <= 0:
            raise DSLValidationError([Diagnostic("E_RESOURCE_001", "resource limits must be positive")])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_parameters": self.max_parameters,
            "max_peak_memory_bytes": self.max_peak_memory_bytes,
            "max_step_time_ratio": self.max_step_time_ratio,
            "max_static_flops": self.max_static_flops,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResourceContract":
        allowed = {"max_parameters", "max_peak_memory_bytes", "max_step_time_ratio", "max_static_flops"}
        unknown = set(data) - allowed
        if unknown:
            raise DSLValidationError([Diagnostic("E_TASK_007", "unknown resource contract fields", details={"fields": sorted(unknown)})])
        return cls(
            int(data["max_parameters"]),
            int(data["max_peak_memory_bytes"]),
            float(data["max_step_time_ratio"]),
            int(data.get("max_static_flops", 0)),
        )


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    group: GroupSpec
    output_type: EquivariantType
    training_protocol_hash: str
    resource_contract: ResourceContract
    allowed_evidence_splits: Tuple[str, ...] = ("train", "validation")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output_type.group != self.group:
            raise DSLValidationError([Diagnostic("E_TASK_001", "task output type uses a different group")])
        if "test" in self.allowed_evidence_splits:
            raise DSLValidationError([Diagnostic("E_TASK_002", "candidate-generation contracts must not expose the test split")])
        if not self.training_protocol_hash:
            raise DSLValidationError([Diagnostic("E_TASK_003", "training protocol hash is required")])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "group": self.group.to_dict(),
            "output_type": self.output_type.to_dict(),
            "training_protocol_hash": self.training_protocol_hash,
            "resource_contract": self.resource_contract.to_dict(),
            "allowed_evidence_splits": list(self.allowed_evidence_splits),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskContract":
        allowed = {
            "task_id",
            "group",
            "output_type",
            "training_protocol_hash",
            "resource_contract",
            "allowed_evidence_splits",
            "metadata",
        }
        unknown = set(data) - allowed
        if unknown:
            raise DSLValidationError([Diagnostic("E_TASK_008", "unknown task contract fields", details={"fields": sorted(unknown)})])
        return cls(
            str(data["task_id"]),
            GroupSpec.from_dict(data["group"]),
            EquivariantType.from_dict(data["output_type"]),
            str(data["training_protocol_hash"]),
            ResourceContract.from_dict(data["resource_contract"]),
            tuple(str(item) for item in data.get("allowed_evidence_splits", ("train", "validation"))),
            dict(data.get("metadata", {})),
        )

    def content_hash(self) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def assert_evidence_visible(self, split: str) -> None:
        if split not in self.allowed_evidence_splits:
            raise DSLValidationError([Diagnostic("E_TASK_004", "evidence split is not visible to candidate generation", actual=split)])
