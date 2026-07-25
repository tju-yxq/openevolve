"""QM9 model adapter for executable core DSL programs."""

from __future__ import annotations

from ..compiler import Compiler
from ..diagnostics import DSLValidationError, Diagnostic
from ..types import Carrier
from .e3nn_backend import E3NNGraphBackend, to_e3nn_irreps
from .equiformer_v2_backend import EquiformerV2GraphBackend


def build_qm9_dsl_model(
    program,
    compiler: Compiler,
    *,
    radius: float = 5.0,
    max_num_neighbors: int = 1000,
    graph_backend=None,
    equiformer_v2_root: str = None,
    equiformer_root: str = None,
    task_mean=None,
    task_std=None,
    atomref=None,
    task=None,
):
    """Build a trainable model with the Equiformer QM9 forward signature."""

    artifact = compiler.analyze(program, task)
    lowering = compiler.plan_lowering(program, task)
    if lowering.mode == "representation_only":
        raise DSLValidationError([
            Diagnostic(
                "E_QM9_BACKEND_005",
                "DSL program has representation-flow semantics only and cannot enter training",
                details=lowering.to_dict(),
            )
        ])
    if lowering.mode in ("exact_reference", "exact_hybrid") and not equiformer_root:
        raise DSLValidationError([
            Diagnostic("E_QM9_BACKEND_006", "exact V1 lowering requires equiformer_root")
        ])
    if lowering.mode == "exact_reference":
        model = compiler.lower_legacy_equiformer_v1(
            program,
            equiformer_root,
            task_mean=task_mean,
            task_std=task_std,
            atomref=atomref,
        )
        model.dsl_architecture_id = artifact.architecture_id
        model.dsl_language_version = program.language_version
        model.backend_family = lowering.backend_family
        model.backend_semantics_version = lowering.backend_semantics_version
        model.lowering_mode = lowering.mode
        model.reference_model_identity = lowering.reference_model_identity
        model.lowering_plan = lowering.to_dict()
        return model
    if lowering.mode == "exact_hybrid":
        from ...builder import build_equiformer
        from ...spec import ArchitectureSpec
        from .hybrid_v1 import build_v1_readout_hybrid, parse_v1_readout_hybrid

        spec = ArchitectureSpec.from_dict(program.annotations["legacy_architecture_spec"])
        base = build_equiformer(
            spec,
            equiformer_root,
            task_mean=task_mean,
            task_std=task_std,
            atomref=atomref,
        )
        hybrid = parse_v1_readout_hybrid(program)
        model = build_v1_readout_hybrid(
            base,
            hybrid,
            artifact.architecture_id,
            program.language_version,
        )
        model.lowering_plan = lowering.to_dict()
        return model
    if len(program.outputs) != 1 or program.outputs[0].expected_type.carrier != Carrier.GRAPH:
        raise DSLValidationError([Diagnostic("E_QM9_BACKEND_001", "QM9 adapter requires one graph-carried output")])
    supported_inputs = {"node_features", "positions", "edge_sh"}
    unknown_inputs = {item.name for item in artifact.expanded_program.inputs} - supported_inputs
    if unknown_inputs:
        raise DSLValidationError([Diagnostic("E_QM9_BACKEND_002", "QM9 adapter cannot materialize program inputs", details={"inputs": sorted(unknown_inputs)})])

    import torch
    from e3nn import o3

    try:
        from torch_cluster import radius_graph
    except ImportError:
        try:
            from torch_geometric.nn import radius_graph
        except ImportError:
            raise DSLValidationError([Diagnostic("E_QM9_BACKEND_003", "torch_cluster or torch_geometric radius_graph is required")])

    if graph_backend is not None and equiformer_v2_root is not None:
        raise DSLValidationError([
            Diagnostic("E_QM9_BACKEND_004", "choose either an explicit graph backend or an Equiformer V2 root")
        ])
    if graph_backend is None:
        graph_backend = (
            EquiformerV2GraphBackend(compiler.primitives, equiformer_v2_root)
            if equiformer_v2_root is not None
            else E3NNGraphBackend(compiler.primitives)
        )
    graph_model = graph_backend.build(artifact.expanded_program, artifact.inference)
    input_types = {item.name: item.value_type for item in artifact.expanded_program.inputs}
    output_name = artifact.expanded_program.outputs[0].name

    class QM9DSLModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.graph_model = graph_model
            self.max_radius = float(radius)
            self.max_num_neighbors = int(max_num_neighbors)
            self.dsl_architecture_id = artifact.architecture_id
            self.dsl_language_version = program.language_version
            self.backend_family = lowering.backend_family
            self.backend_semantics_version = lowering.backend_semantics_version
            self.lowering_mode = lowering.mode
            self.reference_model_identity = lowering.reference_model_identity
            self.lowering_plan = lowering.to_dict()

        def forward(self, f_in, pos, batch, node_atom=None, edge_d_index=None, edge_d_attr=None):
            edge_index = radius_graph(pos, r=self.max_radius, batch=batch, max_num_neighbors=self.max_num_neighbors)
            edge_src, edge_dst = edge_index[0], edge_index[1]
            edge_vector = pos.index_select(0, edge_dst) - pos.index_select(0, edge_src)
            values = {}
            if "node_features" in input_types:
                expected = input_types["node_features"].irreps.dimension
                if f_in.shape[-1] != expected:
                    raise RuntimeError("node feature width {} does not match DSL input {}".format(f_in.shape[-1], expected))
                values["node_features"] = f_in
            if "positions" in input_types:
                values["positions"] = pos
            if "edge_sh" in input_types:
                values["edge_sh"] = o3.spherical_harmonics(to_e3nn_irreps(input_types["edge_sh"].irreps), edge_vector, normalize=True, normalization="component")
            outputs = self.graph_model(
                values,
                {
                    "edge_src": edge_src,
                    "edge_dst": edge_dst,
                    "edge_vectors": edge_vector,
                    "num_nodes": pos.shape[0],
                    "batch": batch,
                    "num_graphs": int(batch.max().item()) + 1 if batch.numel() else 0,
                },
            )
            return outputs[output_name]

    return QM9DSLModel()
