"""Node-level e3nn/PyTorch execution for the backend-neutral core graph.

The module imports torch and e3nn only when `build` is called, keeping the DSL
front end usable in orchestration environments without GPU dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ..ast import ArchitectureProgram, Node
from ..diagnostics import DSLValidationError, Diagnostic
from ..inference import InferenceResult, TypeChecker
from ..registry import PrimitiveRegistry


_SUPPORTED = {
    "core.identity@1",
    "core.irrep_linear@1",
    "core.change_multiplicity@1",
    "core.irrep_concat@1",
    "core.irrep_slice@1",
    "core.residual_add@1",
    "core.tensor_product@1",
    "core.scalar_activation@1",
    "core.norm_activation@1",
    "core.gate@1",
    "core.equivariant_norm@1",
    "core.invariant_weight@1",
    "core.invariant_compatibility@1",
    "core.segment_softmax@1",
    "core.edge_lift@1",
    "core.segment_sum@1",
    "core.segment_mean@1",
    "core.global_pool@1",
    "core.select_scalars@1",
    "core.relative_position@1",
    "core.distance@1",
    "core.radial_basis@1",
    "core.cutoff_envelope@1",
    "core.spherical_harmonics@1",
    "core.stochastic_depth@1",
    "core.invariant_dropout@1",
}


def to_e3nn_irreps(irreps) -> str:
    """Materialize SO(3) types in e3nn's O(3)-labelled storage format."""

    if irreps.family == "SO3":
        return "+".join("{}x{}e".format(mul, ir.degree) for mul, ir in irreps)
    return str(irreps)


@dataclass(frozen=True)
class BackendSupportReport:
    backend: str
    supported: bool
    unsupported_nodes: Tuple[Tuple[str, str], ...]
    missing_dependencies: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FusedSubgraph:
    """A verified backend fusion over nodes that have no external consumers."""

    end_node: str
    node_ids: Tuple[str, ...]
    input_reference: str
    module: Any
    context_key: str
    backend_semantics: str


class E3NNGraphBackend:
    semantic_version = "e3nn-graph-v1"

    def __init__(self, registry: PrimitiveRegistry):
        self.registry = registry

    def _support_report(self, program: ArchitectureProgram, ignored_nodes: Sequence[str] = ()) -> BackendSupportReport:
        ignored = set(ignored_nodes)
        unsupported = []
        for node in program.nodes:
            if node.id in ignored:
                continue
            qualified = node.op if "@" in node.op else "{}@1".format(node.op)
            if qualified not in _SUPPORTED:
                unsupported.append((node.id, qualified))
        missing = []
        try:
            import torch  # noqa: F401
        except ImportError:
            missing.append("torch")
        try:
            import e3nn  # noqa: F401
        except ImportError:
            missing.append("e3nn")
        return BackendSupportReport(
            "e3nn_graph",
            not unsupported and not missing,
            tuple(unsupported),
            tuple(missing),
        )

    def support_report(self, program: ArchitectureProgram) -> BackendSupportReport:
        return self._support_report(program)

    def build(
        self,
        program: ArchitectureProgram,
        inference: InferenceResult = None,
        *,
        fused_subgraphs: Sequence[FusedSubgraph] = (),
    ):
        if inference is None:
            inference = TypeChecker(self.registry).check(program)
        fusion_by_end = {}
        fused_node_ids = set()
        for fusion in fused_subgraphs:
            if fusion.end_node not in fusion.node_ids:
                raise DSLValidationError([
                    Diagnostic("E_BACKEND_006", "fused subgraph end node is outside its node set", node_id=fusion.end_node)
                ])
            overlap = fused_node_ids.intersection(fusion.node_ids)
            if fusion.end_node in fusion_by_end or overlap:
                raise DSLValidationError([
                    Diagnostic("E_BACKEND_007", "fused subgraphs overlap", details={"nodes": sorted(overlap)})
                ])
            fusion_by_end[fusion.end_node] = fusion
            fused_node_ids.update(fusion.node_ids)
        report = self._support_report(program, fused_node_ids)
        if report.missing_dependencies:
            raise DSLValidationError([
                Diagnostic("E_BACKEND_003", "e3nn backend dependencies are unavailable", details={"missing": list(report.missing_dependencies)})
            ])
        if report.unsupported_nodes:
            raise DSLValidationError([
                Diagnostic("E_BACKEND_004", "program contains operations unsupported by the e3nn graph backend", details={"nodes": list(report.unsupported_nodes)})
            ])

        import torch
        from e3nn import o3
        from e3nn.nn import BatchNorm, NormActivation

        node_by_id = {node.id: node for node in program.nodes}
        ordered_nodes = tuple(node_by_id[item] for item in inference.node_order)

        def value_type(reference: str):
            return inference.value_types.get(reference) or inference.value_types.get("{}:out".format(reference))

        modules = {}
        for node in ordered_nodes:
            if node.id in fused_node_ids:
                if node.id in fusion_by_end:
                    modules[node.id] = fusion_by_end[node.id].module
                continue
            op = node.op if "@" in node.op else "{}@1".format(node.op)
            output = value_type(node.id)
            if op in ("core.irrep_linear@1", "core.change_multiplicity@1"):
                source = value_type(node.inputs["x"][0])
                modules[node.id] = o3.Linear(to_e3nn_irreps(source.irreps), to_e3nn_irreps(output.irreps))
            elif op == "core.tensor_product@1":
                left = value_type(node.inputs["left"][0])
                right = value_type(node.inputs["right"][0])
                modules[node.id] = o3.FullyConnectedTensorProduct(
                    to_e3nn_irreps(left.irreps),
                    to_e3nn_irreps(right.irreps),
                    to_e3nn_irreps(output.irreps),
                    internal_weights=True,
                    shared_weights=True,
                )
            elif op == "core.norm_activation@1":
                source = value_type(node.inputs["x"][0])
                activation = str(node.attrs.get("function", "silu"))
                functions = {"silu": torch.nn.functional.silu, "relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}
                if activation not in functions:
                    raise DSLValidationError([Diagnostic("E_BACKEND_005", "unsupported norm activation", node_id=node.id, actual=activation)])
                modules[node.id] = NormActivation(
                    to_e3nn_irreps(source.irreps),
                    functions[activation],
                    normalize=bool(node.attrs.get("normalize", True)),
                    epsilon=float(node.attrs.get("epsilon", 1e-8)),
                    bias=bool(node.attrs.get("bias", False)),
                )
            elif op == "core.equivariant_norm@1":
                source = value_type(node.inputs["x"][0])
                modules[node.id] = BatchNorm(
                    to_e3nn_irreps(source.irreps),
                    eps=float(node.attrs.get("epsilon", 1e-5)),
                    affine=bool(node.attrs.get("affine", True)),
                    instance=bool(node.attrs.get("instance", False)),
                    normalization=str(node.attrs.get("normalization", "component")),
                )

        class CompiledE3NNGraph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.node_modules = torch.nn.ModuleDict(modules)
                self.architecture_program = program
                semantics = [E3NNGraphBackend.semantic_version]
                semantics.extend(sorted(item.backend_semantics for item in fused_subgraphs))
                self.backend_semantics_version = "+".join(semantics)
                self.fused_subgraphs = tuple(fused_subgraphs)

            @staticmethod
            def _resolve(reference, values):
                if reference in values:
                    return values[reference]
                key = "{}:out".format(reference)
                if key in values:
                    return values[key]
                raise KeyError(reference)

            @staticmethod
            def _index_select_irreps(tensor, source_irreps, target_irreps):
                offsets = {}
                start = 0
                for multiplicity, irrep in source_irreps:
                    width = multiplicity * irrep.dimension
                    offsets.setdefault(irrep, []).append((start, multiplicity, irrep.dimension))
                    start += width
                selected = []
                consumed = {}
                for multiplicity, irrep in target_irreps:
                    available = offsets.get(irrep, [])
                    remaining = multiplicity
                    for start, count, dimension in available:
                        already = consumed.get((irrep, start), 0)
                        take = min(remaining, count - already)
                        if take > 0:
                            begin = start + already * dimension
                            selected.append(tensor[..., begin : begin + take * dimension])
                            consumed[(irrep, start)] = already + take
                            remaining -= take
                        if remaining == 0:
                            break
                    if remaining:
                        raise RuntimeError("static irrep_slice invariant was violated")
                return torch.cat(selected, dim=-1) if selected else tensor[..., :0]

            @staticmethod
            def _segment_reduce(values, indices, count, mean=False):
                output = values.new_zeros((count,) + values.shape[1:])
                output.index_add_(0, indices, values)
                if mean:
                    denominator = values.new_zeros(count)
                    denominator.index_add_(0, indices, values.new_ones(indices.shape[0]))
                    shape = (count,) + (1,) * (values.dim() - 1)
                    output = output / denominator.clamp_min_(1).reshape(shape)
                return output

            @staticmethod
            def _gate(gates, values, irreps):
                if gates.shape[-1] == 1:
                    return values * gates
                pieces = []
                value_offset = 0
                gate_offset = 0
                for multiplicity, irrep in irreps:
                    width = multiplicity * irrep.dimension
                    block = values[..., value_offset : value_offset + width]
                    if irrep.degree == 0:
                        pieces.append(block)
                    else:
                        block = block.reshape(block.shape[:-1] + (multiplicity, irrep.dimension))
                        block_gates = gates[..., gate_offset : gate_offset + multiplicity].unsqueeze(-1)
                        pieces.append((block * block_gates).reshape(block.shape[:-2] + (width,)))
                        gate_offset += multiplicity
                    value_offset += width
                return torch.cat(pieces, dim=-1)

            @staticmethod
            def _irrep_dropout(values, irreps, probability, training, whole_value=False):
                if not training or probability <= 0.0:
                    return values
                if whole_value:
                    shape = values.shape[:-1] + (1,)
                    mask = (torch.rand(shape, device=values.device) >= probability).to(values.dtype) / (1.0 - probability)
                    return values * mask
                pieces = []
                offset = 0
                for multiplicity, irrep in irreps:
                    width = multiplicity * irrep.dimension
                    block = values[..., offset : offset + width].reshape(values.shape[:-1] + (multiplicity, irrep.dimension))
                    mask_shape = values.shape[:-1] + (multiplicity, 1)
                    mask = (torch.rand(mask_shape, device=values.device) >= probability).to(values.dtype) / (1.0 - probability)
                    pieces.append((block * mask).reshape(values.shape[:-1] + (width,)))
                    offset += width
                return torch.cat(pieces, dim=-1)

            def forward(self, inputs: Mapping[str, Any], context: Mapping[str, Any]):
                values = {"input:{}".format(name): value for name, value in inputs.items()}
                for node in ordered_nodes:
                    if node.id in fused_node_ids:
                        fusion = fusion_by_end.get(node.id)
                        if fusion is None:
                            continue
                        if fusion.context_key not in context:
                            raise RuntimeError(
                                "fused backend {} requires context value {}".format(
                                    fusion.backend_semantics,
                                    fusion.context_key,
                                )
                            )
                        output = self.node_modules[node.id](
                            self._resolve(fusion.input_reference, values),
                            context[fusion.context_key],
                        )
                        values[node.id] = output
                        values["{}:out".format(node.id)] = output
                        continue
                    op = node.op if "@" in node.op else "{}@1".format(node.op)
                    resolved = {
                        port: tuple(self._resolve(ref, values) for ref in refs)
                        for port, refs in node.inputs.items()
                    }
                    output_type = value_type(node.id)
                    if op == "core.identity@1":
                        output = resolved["x"][0]
                    elif op in ("core.irrep_linear@1", "core.change_multiplicity@1"):
                        output = self.node_modules[node.id](resolved["x"][0])
                    elif op == "core.tensor_product@1":
                        output = self.node_modules[node.id](resolved["left"][0], resolved["right"][0])
                    elif op == "core.irrep_concat@1":
                        output = torch.cat(resolved["xs"], dim=-1)
                    elif op in ("core.irrep_slice@1", "core.select_scalars@1"):
                        source_ref = node.inputs["x"][0]
                        source_type = value_type(source_ref)
                        output = self._index_select_irreps(resolved["x"][0], source_type.irreps, output_type.irreps)
                    elif op == "core.residual_add@1":
                        output = resolved["left"][0] + resolved["right"][0]
                    elif op == "core.scalar_activation@1":
                        function = str(node.attrs.get("function", "silu"))
                        functions = {"silu": torch.nn.functional.silu, "relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}
                        if function not in functions:
                            raise RuntimeError("unsupported scalar activation {}".format(function))
                        output = functions[function](resolved["x"][0])
                    elif op in ("core.norm_activation@1", "core.equivariant_norm@1"):
                        output = self.node_modules[node.id](resolved["x"][0])
                    elif op == "core.gate@1":
                        output = self._gate(resolved["gates"][0], resolved["value"][0], output_type.irreps)
                    elif op == "core.invariant_weight@1":
                        weight = resolved["weight"][0]
                        value = resolved["value"][0]
                        output = weight * value
                    elif op == "core.invariant_compatibility@1":
                        output = (resolved["query"][0] * resolved["key"][0]).sum(dim=-1, keepdim=True)
                    elif op == "core.segment_softmax@1":
                        from torch_geometric.utils import softmax

                        output = softmax(resolved["logits"][0], context["edge_dst"])
                    elif op == "core.edge_lift@1":
                        endpoint = str(node.attrs.get("endpoint", "source"))
                        index = context["edge_src" if endpoint == "source" else "edge_dst"]
                        output = resolved["x"][0].index_select(0, index)
                    elif op in ("core.segment_sum@1", "core.segment_mean@1"):
                        count = int(context.get("num_nodes", inputs[next(iter(inputs))].shape[0]))
                        output = self._segment_reduce(resolved["x"][0], context["edge_dst"], count, mean=op == "core.segment_mean@1")
                    elif op == "core.global_pool@1":
                        batch = context["batch"]
                        count = int(context.get("num_graphs", int(batch.max().item()) + 1))
                        output = self._segment_reduce(resolved["x"][0], batch, count, mean=str(node.attrs.get("reduce", "sum")) == "mean")
                    elif op == "core.relative_position@1":
                        source = resolved["source"][0].index_select(0, context["edge_src"])
                        target = resolved["target"][0].index_select(0, context["edge_dst"])
                        output = target - source
                    elif op == "core.distance@1":
                        output = torch.linalg.vector_norm(resolved["vector"][0], dim=-1, keepdim=True)
                    elif op == "core.radial_basis@1":
                        distance = resolved["distance"][0]
                        count = int(node.attrs["num_basis"])
                        cutoff = float(node.attrs.get("cutoff", 5.0))
                        centers = torch.linspace(0.0, cutoff, count, dtype=distance.dtype, device=distance.device)
                        width = float(node.attrs.get("width", cutoff / max(count - 1, 1)))
                        output = torch.exp(-((distance - centers) / max(width, 1e-8)) ** 2)
                    elif op == "core.cutoff_envelope@1":
                        output = resolved["x"][0]
                    elif op == "core.spherical_harmonics@1":
                        output = o3.spherical_harmonics(
                            to_e3nn_irreps(output_type.irreps),
                            resolved["direction"][0],
                            normalize=True,
                            normalization=str(node.attrs.get("normalization", "component")),
                        )
                    elif op in ("core.stochastic_depth@1", "core.invariant_dropout@1"):
                        probability = float(node.attrs.get("p", 0.0))
                        output = self._irrep_dropout(
                            resolved["x"][0],
                            output_type.irreps,
                            probability,
                            self.training,
                            whole_value=op == "core.stochastic_depth@1",
                        )
                    else:
                        raise RuntimeError("support report and executor diverged for {}".format(op))
                    values[node.id] = output
                    values["{}:out".format(node.id)] = output
                return {
                    output.name: self._resolve(output.source, values)
                    for output in program.outputs
                }

        return CompiledE3NNGraph()
