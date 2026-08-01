"""Optional fusion for Equiformer V2-style primitive paths.

The primitive implementations live in :mod:`v2_runtime`.  This module only
recognizes closed subgraphs and composes those same implementations into an
optimization.  Graphs that cannot be fused remain executable by the generic
``E3NNGraphBackend`` when a V2 source root is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..ast import ArchitectureProgram, Node
from ..diagnostics import DSLValidationError, Diagnostic
from ..inference import InferenceResult, TypeChecker
from ..registry import PrimitiveRegistry
from .e3nn_backend import BackendSupportReport, E3NNGraphBackend, FusedSubgraph
from .v2_runtime import (
    V2_FUSION_SEMANTICS,
    V2_REFERENCE_COMMIT,
    build_s2_activation_module,
    build_so2_convolution_module,
    equiformer_v2_source_available,
    from_edge_frame_value,
    load_equiformer_v2_modules,
    select_runtime_scalars,
    to_edge_frame_value,
    uniform_so3_layout,
)


def _qualified(node: Node) -> str:
    return node.op if "@" in node.op else "{}@1".format(node.op)


def _single_reference(node: Node, port: str) -> Optional[str]:
    values = node.inputs.get(port, ())
    return values[0] if len(values) == 1 else None


@dataclass(frozen=True)
class V2FusionPattern:
    end_node: str
    node_ids: Tuple[str, ...]
    input_reference: str
    input_irreps: object
    output_irreps: object
    mmax: Optional[int]
    grid_resolution: int


def find_v2_fusion_patterns(program: ArchitectureProgram, inference: InferenceResult) -> Tuple[V2FusionPattern, ...]:
    """Find closed V2 edge-frame paths that can be lowered as one trusted unit."""

    nodes = {node.id: node for node in program.nodes}
    consumers: Dict[str, set] = {node.id: set() for node in program.nodes}
    for node in program.nodes:
        for references in node.inputs.values():
            for reference in references:
                root = reference.split(":", 1)[0]
                if root in consumers:
                    consumers[root].add(node.id)
    for output in program.outputs:
        root = output.source.split(":", 1)[0]
        if root in consumers:
            consumers[root].add("output:{}".format(output.name))

    def referenced_node(node: Node, port: str) -> Optional[Node]:
        reference = _single_reference(node, port)
        if reference is None:
            return None
        return nodes.get(reference.split(":", 1)[0])

    patterns = []
    claimed = set()
    for from_edge in program.nodes:
        if _qualified(from_edge) != "core.from_edge_frame@1":
            continue
        activation = referenced_node(from_edge, "x")
        if activation is None or _qualified(activation) != "core.separable_s2_activation@1":
            continue
        so2 = referenced_node(activation, "x")
        scalars = referenced_node(activation, "scalars")
        if so2 is None or scalars is None:
            continue
        if _qualified(so2) != "core.so2_convolution@1" or _qualified(scalars) != "core.select_scalars@1":
            continue
        if referenced_node(scalars, "x") is not so2:
            continue
        to_edge = referenced_node(so2, "x")
        if to_edge is None or _qualified(to_edge) != "core.to_edge_frame@1":
            continue
        input_reference = _single_reference(to_edge, "x")
        if input_reference is None:
            continue
        expected_consumers = {
            to_edge.id: {so2.id},
            so2.id: {scalars.id, activation.id},
            scalars.id: {activation.id},
            activation.id: {from_edge.id},
        }
        if any(consumers[node_id] != expected for node_id, expected in expected_consumers.items()):
            continue
        node_ids = (to_edge.id, so2.id, scalars.id, activation.id, from_edge.id)
        if claimed.intersection(node_ids):
            raise DSLValidationError([
                Diagnostic("E_V2_BACKEND_008", "V2 fusion patterns overlap", details={"nodes": sorted(claimed.intersection(node_ids))})
            ])
        input_type = inference.value_types.get(input_reference) or inference.value_types.get("{}:out".format(input_reference))
        output_type = inference.value_types.get(so2.id) or inference.value_types.get("{}:out".format(so2.id))
        if input_type is None or output_type is None:
            raise DSLValidationError([Diagnostic("E_V2_BACKEND_009", "V2 fusion types are unavailable", node_id=from_edge.id)])
        raw_mmax = so2.attrs.get("mmax")
        pattern = V2FusionPattern(
            end_node=from_edge.id,
            node_ids=node_ids,
            input_reference=input_reference,
            input_irreps=input_type.irreps,
            output_irreps=output_type.irreps,
            mmax=None if raw_mmax is None else int(raw_mmax),
            grid_resolution=int(activation.attrs.get("grid_resolution", 18)),
        )
        patterns.append(pattern)
        claimed.update(node_ids)
    return tuple(patterns)


def build_v2_so2_path(input_irreps, output_irreps, root: str, *, mmax: int = None, activation: str = "none", grid_resolution: int = 18):
    """Compose the same independent V2 primitive Lowerings as one module."""

    in_lmax, _in_channels = uniform_so3_layout(input_irreps)
    out_lmax, out_channels = uniform_so3_layout(output_irreps)
    if in_lmax != out_lmax:
        raise DSLValidationError([Diagnostic("E_V2_BACKEND_005", "current V2 adapter requires equal input and output lmax")])
    lmax = in_lmax
    mmax = lmax if mmax is None else int(mmax)
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([Diagnostic("E_V2_BACKEND_006", "mmax must satisfy 0 <= mmax <= lmax")])
    if activation not in ("none", "separable_s2"):
        raise DSLValidationError([Diagnostic("E_V2_BACKEND_007", "unsupported V2 path activation", actual=activation)])
    modules = load_equiformer_v2_modules(root)

    import torch

    class EquiformerV2SO2Path(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.convolution = build_so2_convolution_module(
                input_irreps,
                output_irreps,
                mmax,
                modules,
            )
            self.activation_name = activation
            if activation == "separable_s2":
                self.s2_activation = build_s2_activation_module(
                    output_irreps,
                    mmax=mmax,
                    resolution=grid_resolution,
                    normalization="component",
                    separable=True,
                    modules=modules,
                )
            self.lmax = lmax
            self.mmax = mmax
            self.out_channels = out_channels
            self.reference_commit = V2_REFERENCE_COMMIT

        def forward(self, flat_features, edge_vectors):
            frame_id = "__fused_v2_edge_frame__"
            value = to_edge_frame_value(
                flat_features,
                input_irreps,
                edge_vectors,
                frame_id,
                modules,
            )
            value = self.convolution(value)
            if self.activation_name == "separable_s2":
                scalars = select_runtime_scalars(value, self.out_channels)
                value = self.s2_activation(value, scalars)
            return from_edge_frame_value(value, output_irreps, frame_id, modules)

    return EquiformerV2SO2Path()


class EquiformerV2GraphBackend:
    """Compile a typed graph by fusing verified V2 paths into the e3nn executor."""

    semantic_version = V2_FUSION_SEMANTICS

    def __init__(self, registry: PrimitiveRegistry, root: str):
        self.registry = registry
        self.root = str(root)
        self.e3nn_backend = E3NNGraphBackend(
            registry,
            equiformer_v2_root=self.root,
        )

    def _analyze(self, program: ArchitectureProgram, inference: InferenceResult = None):
        inferred = inference or TypeChecker(self.registry).check(program)
        patterns = find_v2_fusion_patterns(program, inferred)
        fused = {node_id for pattern in patterns for node_id in pattern.node_ids}
        return inferred, patterns, fused

    def support_report(self, program: ArchitectureProgram, inference: InferenceResult = None) -> BackendSupportReport:
        _, patterns, fused = self._analyze(program, inference)
        base = self.e3nn_backend._support_report(program, fused)
        missing = list(base.missing_dependencies)
        if patterns and not equiformer_v2_source_available(self.root):
            missing.append("equiformer_v2_reference@{}".format(V2_REFERENCE_COMMIT))
        fused_exactness = tuple(
            (node_id, "library_exact_fusion")
            for pattern in patterns
            for node_id in pattern.node_ids
        )
        return BackendSupportReport(
            "equiformer_v2_graph",
            not base.unsupported_nodes and not missing and not base.composition_errors,
            base.unsupported_nodes,
            tuple(sorted(set(missing))),
            tuple(sorted(base.node_exactness + fused_exactness)),
            base.runtime_kinds,
            base.composition_errors,
        )

    def build(self, program: ArchitectureProgram, inference: InferenceResult = None):
        inferred, patterns, _ = self._analyze(program, inference)
        report = self.support_report(program, inferred)
        if report.missing_dependencies:
            raise DSLValidationError([
                Diagnostic("E_V2_BACKEND_010", "V2 graph backend dependencies are unavailable", details={"missing": list(report.missing_dependencies)})
            ])
        if report.unsupported_nodes:
            raise DSLValidationError([
                Diagnostic("E_V2_BACKEND_011", "program contains operations unsupported by both V2 fusion and generic Lowering", details={"nodes": list(report.unsupported_nodes)})
            ])
        if report.composition_errors:
            raise DSLValidationError([
                Diagnostic("E_V2_BACKEND_012", "V2 graph contains incompatible runtime value compositions", details={"errors": [list(item) for item in report.composition_errors]})
            ])
        fusions = []
        for pattern in patterns:
            module = build_v2_so2_path(
                pattern.input_irreps,
                pattern.output_irreps,
                self.root,
                mmax=pattern.mmax,
                activation="separable_s2",
                grid_resolution=pattern.grid_resolution,
            )
            fusions.append(
                FusedSubgraph(
                    pattern.end_node,
                    pattern.node_ids,
                    pattern.input_reference,
                    module,
                    "edge_vectors",
                    self.semantic_version,
                )
            )
        return self.e3nn_backend.build(program, inferred, fused_subgraphs=tuple(fusions))
