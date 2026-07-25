"""Exact Equiformer V1 reference and certified readout-region hybrid models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...spec import ArchitectureSpec
from ..ast import ArchitectureProgram
from ..diagnostics import DSLValidationError, Diagnostic


@dataclass(frozen=True)
class V1ReadoutHybridSpec:
    auxiliary_block: int

    def to_dict(self):
        return {"kind": "v1_multilevel_readout", "auxiliary_block": self.auxiliary_block}


def parse_v1_readout_hybrid(program: ArchitectureProgram) -> Optional[V1ReadoutHybridSpec]:
    """Recognize the only modified V1 topology certified by the showcase backend."""

    if program.annotations.get("legacy_backend") != "equiformer_v1":
        return None
    nodes = {node.id: node for node in program.nodes}
    pool = nodes.get("graph_pool")
    if pool is None or pool.op not in ("motif.v1_multilevel_readout", "motif.v1_multilevel_readout@1"):
        return None
    if set(pool.inputs) != {"terminal", "aux"}:
        return None
    terminal = tuple(pool.inputs["terminal"])
    auxiliary = tuple(pool.inputs["aux"])
    if terminal != ("scalar_readout",) or len(auxiliary) != 1:
        return None
    source = auxiliary[0]
    if not source.startswith("block") or not source[5:].isdigit():
        return None
    block_index = int(source[5:])
    spec = ArchitectureSpec.from_dict(program.annotations["legacy_architecture_spec"])
    if block_index < 0 or block_index >= spec.macro.num_layers - 1:
        return None
    if len(program.outputs) != 1 or program.outputs[0].source not in ("graph_pool", "graph_pool:out"):
        return None
    return V1ReadoutHybridSpec(block_index)


def build_v1_readout_hybrid(base_model, hybrid: V1ReadoutHybridSpec, architecture_id: str, language_version: str):
    """Add an invariant intermediate-block readout while retaining the official V1 body."""

    import torch

    block_index = int(hybrid.auxiliary_block)
    if block_index >= len(base_model.blocks) - 1:
        raise DSLValidationError([
            Diagnostic("E_HYBRID_001", "auxiliary readout must tap a nonterminal V1 block", actual=str(block_index))
        ])
    irreps = base_model.irreps_node_embedding
    scalar_width = sum(multiplicity for multiplicity, irrep in irreps if irrep.l == 0)
    if scalar_width <= 0:
        raise DSLValidationError([Diagnostic("E_HYBRID_002", "selected V1 block exposes no scalar irreps")])

    class ExactV1ReadoutHybrid(torch.nn.Module):
        backend_family = "equiformer_v1"
        backend_semantics_version = "equiformer-v1-readout-hybrid-v1"
        lowering_mode = "exact_hybrid"

        def __init__(self):
            super().__init__()
            self.base_model = base_model
            self.auxiliary_block = block_index
            hidden = max(16, min(128, scalar_width))
            self.auxiliary_head = torch.nn.Sequential(
                torch.nn.Linear(scalar_width, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, 1),
            )
            self.combine = torch.nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                self.combine.weight.zero_()
                self.combine.weight[0, 0] = 1.0
            self.dsl_architecture_id = architecture_id
            self.dsl_language_version = language_version
            self.reference_model_identity = "official_equiformer_v1_graph_attention_transformer"
            self._captured = None
            self._hook = self.base_model.blocks[block_index].register_forward_hook(self._capture)

        def _capture(self, _module, _inputs, output):
            self._captured = output

        def no_weight_decay(self):
            if not hasattr(self.base_model, "no_weight_decay"):
                return set()
            return {"base_model." + item for item in self.base_model.no_weight_decay()}

        def forward(self, f_in, pos, batch, node_atom=None, edge_d_index=None, edge_d_attr=None):
            self._captured = None
            terminal = self.base_model(
                f_in=f_in,
                pos=pos,
                batch=batch,
                node_atom=node_atom,
                edge_d_index=edge_d_index,
                edge_d_attr=edge_d_attr,
            )
            if self._captured is None:
                raise RuntimeError("official V1 block hook did not capture the auxiliary representation")
            scalars = self._captured.narrow(-1, 0, scalar_width)
            auxiliary_node = self.auxiliary_head(scalars)
            auxiliary_graph = self.base_model.scale_scatter(auxiliary_node, batch, dim=0)
            self._captured = None
            return self.combine(torch.cat((terminal, auxiliary_graph), dim=-1))

    return ExactV1ReadoutHybrid()
