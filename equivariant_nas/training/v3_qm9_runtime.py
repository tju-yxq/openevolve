"""PyG/QM9 adapter for a V3 model built by generic Typed DSL Lowering."""

from __future__ import annotations

from typing import Any, Mapping

from equivariant_nas.dsl import TypeChecker, architecture_id, core_registry
from equivariant_nas.dsl.backends import E3NNGraphBackend


def _index(indices, target_size: int) -> Mapping[str, Any]:
    return {"indices": indices, "target_size": int(target_size)}


def build_lowered_v3_qm9_model(program, *, equiformer_v3_root: str):
    """Build a trainable PyG-compatible wrapper without an official V3 constructor bypass."""

    import torch

    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    backend = E3NNGraphBackend(registry, equiformer_v3_root=equiformer_v3_root)
    graph = backend.build(program, inference)
    forbidden = {
        "EquiformerV3_OC",
        "TransBlockV3",
        "EquivariantGraphAttention",
        "ScalarFeedForwardNetwork",
    }
    bypass = sorted({type(module).__name__ for module in graph.modules()} & forbidden)
    if bypass:
        raise RuntimeError("official V3 constructor bypass detected: {}".format(bypass))

    class LoweredV3QM9Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.graph = graph
            self.architecture_id = architecture_id(program, registry)
            self.dsl_architecture_id = self.architecture_id
            self.dsl_language_version = program.language_version
            self.backend_semantics_version = graph.backend_semantics_version
            self.constructor_bypass = False
            self.lowering_mode = "v3_generic_typed_graph"
            self.lowering_plan = {
                "mode": self.lowering_mode,
                "backend_semantics_version": self.backend_semantics_version,
                "constructor_bypass": False,
            }
            self.generic_lowering_admission = {
                "admitted": True,
                "reason": "all V3 nodes lowered through the certified generic e3nn graph backend",
            }

        def no_weight_decay(self):
            # The generic graph parameter tree does not expose official module
            # class names.  Keep this frozen empty set across every candidate.
            return set()

        def forward(
            self,
            batch=None,
            *,
            f_in=None,
            pos=None,
            node_atom=None,
            edge_d_index=None,
            edge_d_attr=None,
        ):
            del f_in, edge_d_attr
            legacy_tensor_call = edge_d_index is not None
            if legacy_tensor_call:
                edge_index = edge_d_index
                graph_index = batch
                atomic_numbers = node_atom
                positions = pos
                num_graphs = int(graph_index.max()) + 1 if graph_index.numel() else 0
            else:
                if not hasattr(batch, "edge_index"):
                    raise ValueError("QM9 batch lacks the frozen precomputed edge_index")
                edge_index = batch.edge_index
                atomic_numbers = getattr(batch, "atomic_numbers", None)
                if atomic_numbers is None:
                    atomic_numbers = batch.z
                graph_index = batch.batch
                positions = batch.pos
                num_graphs = int(getattr(batch, "num_graphs", 0) or (int(graph_index.max()) + 1 if graph_index.numel() else 0))
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("QM9 edge_index must have shape [2, E]")
            num_nodes = int(atomic_numbers.numel())
            num_edges = int(edge_index.shape[1])
            inputs = {
                "atomic_numbers": atomic_numbers.long(),
                "positions": positions,
                "source_index": _index(edge_index[0].long(), num_edges),
                "target_index": _index(edge_index[1].long(), num_edges),
                "target_segment": _index(edge_index[1].long(), num_nodes),
                "batch": _index(graph_index.long(), num_graphs),
            }
            input_names = {item.name for item in program.inputs}
            if "lattice" in input_names or "lattice_shift" in input_names:
                raise ValueError("QM9 protocol requires the non-periodic V3 input contract")
            outputs = self.graph(inputs, {})
            return outputs["energy"] if legacy_tensor_call else outputs

    return LoweredV3QM9Model()
