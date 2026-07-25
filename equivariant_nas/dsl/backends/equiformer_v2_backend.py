"""Trusted adapter for Equiformer V2 SO(3)-embedding and SO(2)-convolution paths."""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from ..ast import ArchitectureProgram, Node
from ..diagnostics import DSLValidationError, Diagnostic
from ..inference import InferenceResult, TypeChecker
from ..registry import PrimitiveRegistry
from .e3nn_backend import BackendSupportReport, E3NNGraphBackend, FusedSubgraph


V2_REFERENCE_COMMIT = "d5ad4be729b56f74012ebb7f097f77c5b00a1004"
V2_FUSION_SEMANTICS = "equiformer-v2-so2-fusion@{}".format(V2_REFERENCE_COMMIT)


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


def load_equiformer_v2_modules(root: str):
    """Load the operator package without executing its OC20 top-level model."""

    package_path = Path(root) / "nets" / "equiformer_v2"
    required = ("so3.py", "so2_ops.py", "edge_rot_mat.py", "activation.py")
    missing = [name for name in required if not (package_path / name).is_file()]
    if missing:
        raise DSLValidationError([Diagnostic("E_V2_BACKEND_001", "Equiformer V2 reference source is incomplete", details={"missing": missing})])
    package_name = "evoequilang_equiformer_v2_ref"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return (
        importlib.import_module(package_name + ".so3"),
        importlib.import_module(package_name + ".so2_ops"),
        importlib.import_module(package_name + ".edge_rot_mat"),
        importlib.import_module(package_name + ".activation"),
    )


def _uniform_so3_layout(irreps) -> Tuple[int, int]:
    if irreps.family != "SO3" or not irreps.terms:
        raise DSLValidationError([Diagnostic("E_V2_BACKEND_002", "V2 adapter requires nonempty SO(3) irreps")])
    degrees = [ir.degree for _, ir in irreps]
    lmax = max(degrees)
    if degrees != list(range(lmax + 1)):
        raise DSLValidationError([Diagnostic("E_V2_BACKEND_003", "V2 layout requires every degree from zero through lmax")])
    multiplicities = {mul for mul, _ in irreps}
    if len(multiplicities) != 1:
        raise DSLValidationError([Diagnostic("E_V2_BACKEND_004", "V2 layout requires uniform multiplicity across degrees")])
    return lmax, next(iter(multiplicities))


def build_v2_so2_path(input_irreps, output_irreps, root: str, *, mmax: int = None, activation: str = "none", grid_resolution: int = 18):
    """Build edge-frame rotate -> official SO(2) convolution -> rotate-back."""

    in_lmax, in_channels = _uniform_so3_layout(input_irreps)
    out_lmax, out_channels = _uniform_so3_layout(output_irreps)
    if in_lmax != out_lmax:
        raise DSLValidationError([Diagnostic("E_V2_BACKEND_005", "current V2 adapter requires equal input and output lmax")])
    lmax = in_lmax
    mmax = lmax if mmax is None else int(mmax)
    if mmax < 0 or mmax > lmax:
        raise DSLValidationError([Diagnostic("E_V2_BACKEND_006", "mmax must satisfy 0 <= mmax <= lmax")])
    if activation not in ("none", "separable_s2"):
        raise DSLValidationError([Diagnostic("E_V2_BACKEND_007", "unsupported V2 path activation", actual=activation)])
    so3, so2_ops, edge_rot_mat, activation_module = load_equiformer_v2_modules(root)

    import torch

    class EquiformerV2SO2Path(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mapping = so3.CoefficientMappingModule([lmax], [mmax])
            self.rotation = so3.SO3_Rotation(lmax)
            self.convolution = so2_ops.SO2_Convolution(
                in_channels,
                out_channels,
                [lmax],
                [mmax],
                self.mapping,
                internal_weights=True,
                edge_channels_list=None,
            )
            self.activation_name = activation
            if activation == "separable_s2":
                self.s2_activation = activation_module.SeparableS2Activation(lmax, mmax)
                grids = []
                for l_value in range(lmax + 1):
                    grids.append(torch.nn.ModuleList([
                        so3.SO3_Grid(l_value, m_value, resolution=grid_resolution, normalization="component")
                        for m_value in range(lmax + 1)
                    ]))
                self.so3_grid = torch.nn.ModuleList(grids)
            self.lmax = lmax
            self.mmax = mmax
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.reference_commit = V2_REFERENCE_COMMIT

        @staticmethod
        def _to_embedding(flat, irreps, channels):
            pieces = []
            offset = 0
            for multiplicity, irrep in irreps:
                width = multiplicity * irrep.dimension
                block = flat[..., offset : offset + width]
                block = block.reshape(flat.shape[0], multiplicity, irrep.dimension).transpose(1, 2)
                pieces.append(block)
                offset += width
            embedding = so3.SO3_Embedding(0, [lmax], channels, flat.device, flat.dtype)
            embedding.set_embedding(torch.cat(pieces, dim=1))
            return embedding

        @staticmethod
        def _from_embedding(embedding):
            pieces = []
            offset = 0
            for degree in range(lmax + 1):
                width = 2 * degree + 1
                block = embedding.embedding[:, offset : offset + width]
                pieces.append(block.transpose(1, 2).reshape(block.shape[0], -1))
                offset += width
            return torch.cat(pieces, dim=-1)

        def forward(self, flat_features, edge_vectors):
            embedding = self._to_embedding(flat_features, input_irreps, self.in_channels)
            rotation_matrix = edge_rot_mat.init_edge_rot_mat(edge_vectors)
            self.rotation.set_wigner(rotation_matrix)
            embedding._rotate([self.rotation], [self.lmax], [self.mmax])
            dummy_edge_scalars = edge_vectors.new_zeros((edge_vectors.shape[0], 1))
            embedding = self.convolution(embedding, dummy_edge_scalars)
            if self.activation_name == "separable_s2":
                scalars = embedding.embedding[:, 0, :]
                embedding.set_embedding(self.s2_activation(scalars, embedding.embedding, self.so3_grid))
            embedding._rotate_inv([self.rotation], self.mapping)
            return self._from_embedding(embedding)

    return EquiformerV2SO2Path()


class EquiformerV2GraphBackend:
    """Compile a typed graph by fusing verified V2 paths into the e3nn executor."""

    semantic_version = V2_FUSION_SEMANTICS

    def __init__(self, registry: PrimitiveRegistry, root: str):
        self.registry = registry
        self.root = str(root)
        self.e3nn_backend = E3NNGraphBackend(registry)

    def _analyze(self, program: ArchitectureProgram, inference: InferenceResult = None):
        inferred = inference or TypeChecker(self.registry).check(program)
        patterns = find_v2_fusion_patterns(program, inferred)
        fused = {node_id for pattern in patterns for node_id in pattern.node_ids}
        return inferred, patterns, fused

    def support_report(self, program: ArchitectureProgram, inference: InferenceResult = None) -> BackendSupportReport:
        _, _, fused = self._analyze(program, inference)
        base = self.e3nn_backend._support_report(program, fused)
        missing = list(base.missing_dependencies)
        package_path = Path(self.root) / "nets" / "equiformer_v2"
        required = ("so3.py", "so2_ops.py", "edge_rot_mat.py", "activation.py")
        if any(not (package_path / name).is_file() for name in required):
            missing.append("equiformer_v2_reference@{}".format(V2_REFERENCE_COMMIT))
        return BackendSupportReport(
            "equiformer_v2_graph",
            not base.unsupported_nodes and not missing,
            base.unsupported_nodes,
            tuple(sorted(set(missing))),
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
                Diagnostic("E_V2_BACKEND_011", "program contains V2 operations outside a supported closed fusion", details={"nodes": list(report.unsupported_nodes)})
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
