"""Proof obligations created and discharged during static checking."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple


class ObligationKind(str, Enum):
    IRREP_PATH_EXISTS = "IRREP_PATH_EXISTS"
    PARITY_MATCH = "PARITY_MATCH"
    FRAME_BALANCE = "FRAME_BALANCE"
    INVARIANT_ATTENTION_WEIGHT = "INVARIANT_ATTENTION_WEIGHT"
    PERMUTATION_SAFE_AGGREGATION = "PERMUTATION_SAFE_AGGREGATION"
    OUTPUT_CONTRACT_MATCH = "OUTPUT_CONTRACT_MATCH"
    RESOURCE_BOUND = "RESOURCE_BOUND"
    BACKEND_AVAILABLE = "BACKEND_AVAILABLE"
    NUMERICAL_EQUIVARIANCE_CHECK = "NUMERICAL_EQUIVARIANCE_CHECK"


@dataclass(frozen=True)
class ProofObligation:
    obligation_id: str
    kind: ObligationKind
    source_node: str
    status: str = "open"
    discharged_by: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def discharge(self, source: str) -> "ProofObligation":
        return ProofObligation(self.obligation_id, self.kind, self.source_node, "discharged", source, dict(self.details))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "kind": self.kind.value,
            "source_node": self.source_node,
            "status": self.status,
            "discharged_by": self.discharged_by,
            "details": dict(self.details),
        }
