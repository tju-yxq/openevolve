"""Symmetry-semantic parent-to-child state transfer for search acceleration.

Exact tensor names and shapes are necessary but not sufficient: an OPERATOR
edit can leave shapes unchanged while changing the meaning of radial features
or attention heads.  The policy below combines exact state compatibility with
the typed ArchitectureSpec diff and blocks semantically unsafe module regions.
Inherited runs are calibration proxies until a separate trust test grants them
selection authority; final models must always be trained from scratch.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Tuple

from .spec import ArchitectureSpec, EvolutionFactor


@dataclass(frozen=True)
class TransferReport:
    parent_architecture_id: str
    child_architecture_id: str
    changed_factor: str
    child_state_tensors: int
    child_state_elements: int
    transferred_tensors: int
    transferred_elements: int
    tensor_coverage: float
    element_coverage: float
    blocked_reason_counts: Dict[str, int]
    blocked_keys: Dict[str, str]
    transferable_keys: Tuple[str, ...]
    selection_eligible: bool = False
    final_training_allowed: bool = False
    test_split_allowed: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _numel(value) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    result = 1
    for dimension in getattr(value, "shape", ()):
        result *= int(dimension)
    return result


def _shape(value) -> Tuple[int, ...]:
    return tuple(int(item) for item in getattr(value, "shape", ()))


def _semantic_block_reason(
    key: str,
    parent: ArchitectureSpec,
    child: ArchitectureSpec,
    factor: EvolutionFactor,
) -> str:
    lower = key.lower()
    if factor == EvolutionFactor.REPRESENTATION:
        return "representation_irrep_layout_changed"
    if factor == EvolutionFactor.ACTION:
        if parent.action.norm_layer != child.action.norm_layer and "norm" in lower:
            return "normalization_semantics_changed"
        return ""
    if factor == EvolutionFactor.MACRO:
        # Common block indices are homogeneous Equiformer blocks. Radius has no
        # learned coordinate frame, while new/removed layers are handled by the
        # exact-key test.
        return ""

    operator = parent.operator
    new_operator = child.operator
    head_or_message_changed = (
        operator.num_heads != new_operator.num_heads
        or operator.nonlinear_message != new_operator.nonlinear_message
    )
    radial_semantics_changed = (
        operator.basis_type != new_operator.basis_type
        or operator.num_basis != new_operator.num_basis
        or operator.radial_hidden != new_operator.radial_hidden
    )
    if head_or_message_changed and (
        ".ga." in key or key.startswith("edge_deg_embed") or key.startswith("rbf.")
    ):
        return "attention_operator_semantics_changed"
    if radial_semantics_changed and key.startswith("rbf."):
        return "radial_basis_semantics_changed"
    if radial_semantics_changed and (
        key.startswith("edge_deg_embed.rad.") or ".ga.sep_act.dtp_rad." in key
    ):
        return "radial_network_semantics_changed"
    return ""


def analyze_transfer(
    parent_spec: ArchitectureSpec,
    child_spec: ArchitectureSpec,
    parent_state: Mapping[str, object],
    child_state: Mapping[str, object],
) -> TransferReport:
    changed = parent_spec.changed_factors(child_spec)
    if len(changed) != 1:
        raise ValueError("state transfer requires exactly one changed factor")
    factor = changed[0]
    transferable = []
    blocked = {}
    transferred_elements = 0
    total_elements = sum(_numel(value) for value in child_state.values())
    for key, child_value in child_state.items():
        if key not in parent_state:
            blocked[key] = "missing_in_parent"
            continue
        if _shape(parent_state[key]) != _shape(child_value):
            blocked[key] = "shape_changed"
            continue
        reason = _semantic_block_reason(key, parent_spec, child_spec, factor)
        if reason:
            blocked[key] = reason
            continue
        transferable.append(key)
        transferred_elements += _numel(child_value)
    reason_counts = dict(sorted(Counter(blocked.values()).items()))
    return TransferReport(
        parent_architecture_id=parent_spec.architecture_id(),
        child_architecture_id=child_spec.architecture_id(),
        changed_factor=factor.value,
        child_state_tensors=len(child_state),
        child_state_elements=total_elements,
        transferred_tensors=len(transferable),
        transferred_elements=transferred_elements,
        tensor_coverage=len(transferable) / max(1, len(child_state)),
        element_coverage=transferred_elements / max(1, total_elements),
        blocked_reason_counts=reason_counts,
        blocked_keys=blocked,
        transferable_keys=tuple(transferable),
    )


def apply_transfer(
    parent_spec: ArchitectureSpec,
    child_spec: ArchitectureSpec,
    parent_state: Mapping[str, object],
    child_state: Mapping[str, object],
):
    """Return a child state with only policy-approved tensors inherited."""

    report = analyze_transfer(parent_spec, child_spec, parent_state, child_state)
    merged = dict(child_state)
    for key in report.transferable_keys:
        value = parent_state[key]
        merged[key] = value.detach().clone() if hasattr(value, "detach") else value
    return merged, report
