"""Static, uncertainty-aware architecture cost estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from .ast import ArchitectureProgram
from .diagnostics import DSLValidationError, Diagnostic
from .inference import InferenceResult
from .task import ResourceContract


@dataclass(frozen=True)
class CostEstimate:
    parameter_estimate: int
    activation_elements_per_entity: int
    flops_per_entity_estimate: int
    unknown_nodes: Tuple[str, ...]
    model_version: str = "static-cost-v1"
    certified_upper_bound: bool = False

    @property
    def is_complete(self) -> bool:
        return not self.unknown_nodes


def estimate_static_cost(program: ArchitectureProgram, inference: InferenceResult) -> CostEstimate:
    parameters = 0
    activations = 0
    flops = 0
    unknown = []
    for node in program.nodes:
        output_key = node.id if node.id in inference.value_types else "{}:out".format(node.id)
        output = inference.value_types.get(output_key)
        if output is None:
            unknown.append(node.id)
            continue
        out_dim = output.irreps.dimension
        activations += out_dim
        if node.op.startswith("core.irrep_linear") or node.op.startswith("core.change_multiplicity"):
            source_ref = node.inputs["x"][0]
            source = inference.value_types.get(source_ref) or inference.value_types.get("{}:out".format(source_ref))
            if source is None:
                unknown.append(node.id)
                continue
            in_mul = sum(mul for mul, _ in source.irreps)
            out_mul = sum(mul for mul, _ in output.irreps)
            parameters += in_mul * out_mul
            flops += source.irreps.dimension * out_dim
        elif node.op.startswith("core.tensor_product"):
            left_ref = node.inputs["left"][0]
            right_ref = node.inputs["right"][0]
            left = inference.value_types.get(left_ref) or inference.value_types.get("{}:out".format(left_ref))
            right = inference.value_types.get(right_ref) or inference.value_types.get("{}:out".format(right_ref))
            if left is None or right is None:
                unknown.append(node.id)
                continue
            flops += left.irreps.dimension * right.irreps.dimension * out_dim
            parameters += sum(mul for mul, _ in left.irreps) * sum(mul for mul, _ in output.irreps)
        elif node.op.startswith("core.so2_convolution"):
            parameters += out_dim * out_dim
            flops += out_dim * out_dim
        elif node.op.startswith("core."):
            flops += out_dim
        else:
            unknown.append(node.id)
    return CostEstimate(parameters, activations, flops, tuple(sorted(set(unknown))))


def enforce_static_resource_contract(cost: CostEstimate, contract: ResourceContract, *, require_certified: bool = True) -> None:
    diagnostics = []
    if require_certified and not cost.certified_upper_bound:
        diagnostics.append(Diagnostic("E_RESOURCE_005", "heuristic cost estimates cannot discharge a certified resource-bound obligation"))
    if cost.parameter_estimate > contract.max_parameters:
        diagnostics.append(Diagnostic("E_RESOURCE_002", "static parameter estimate exceeds contract", expected=str(contract.max_parameters), actual=str(cost.parameter_estimate)))
    if contract.max_static_flops and cost.flops_per_entity_estimate > contract.max_static_flops:
        diagnostics.append(Diagnostic("E_RESOURCE_003", "static FLOP estimate exceeds contract", expected=str(contract.max_static_flops), actual=str(cost.flops_per_entity_estimate)))
    if cost.unknown_nodes:
        diagnostics.append(Diagnostic("E_RESOURCE_004", "static cost is incomplete for one or more nodes", details={"nodes": list(cost.unknown_nodes)}))
    if diagnostics:
        raise DSLValidationError(diagnostics)
