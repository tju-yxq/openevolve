"""Symmetry diagnostics used before expensive candidate training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class ErrorSummary:
    maximum: float
    mean: float
    values: List[float]


@dataclass
class SymmetryReport:
    rotation_invariance: ErrorSummary
    translation_invariance: ErrorSummary
    permutation_invariance: ErrorSummary
    layerwise_rotation_equivariance: Dict[str, ErrorSummary]

    def to_dict(self):
        return asdict(self)


def _relative_error(reference, observed, eps: float = 1.0e-12) -> float:
    import torch

    numerator = torch.linalg.vector_norm((reference - observed).reshape(-1))
    denominator = torch.linalg.vector_norm(reference.reshape(-1)).clamp_min(eps)
    return float((numerator / denominator).detach().cpu())


def _summary(values: Sequence[float]) -> ErrorSummary:
    if not values:
        return ErrorSummary(maximum=0.0, mean=0.0, values=[])
    return ErrorSummary(
        maximum=max(values), mean=sum(values) / len(values), values=list(values)
    )


def _single_graph(data):
    """Extract the first molecule so arbitrary node permutations remain valid."""

    import copy
    import torch

    graph = copy.copy(data)
    if hasattr(data, "batch"):
        node_mask = data.batch == int(data.batch.min())
    else:
        node_mask = torch.ones(data.pos.shape[0], dtype=torch.bool, device=data.pos.device)
    graph.pos = data.pos[node_mask]
    graph.x = data.x[node_mask]
    graph.z = data.z[node_mask]
    graph.batch = torch.zeros(graph.pos.shape[0], dtype=torch.long, device=graph.pos.device)
    return graph


def _model_call(model, graph, positions=None, permutation=None):
    import torch

    pos = graph.pos if positions is None else positions
    x = graph.x
    z = graph.z
    batch = graph.batch
    if permutation is not None:
        pos = pos[permutation]
        x = x[permutation]
        z = z[permutation]
        batch = batch[permutation]
    return model(f_in=x, pos=pos, batch=batch, node_atom=z)


def symmetry_report(
    model,
    data,
    rotations: int = 3,
    translations: int = 2,
    freeze_neighbors: bool = False,
) -> SymmetryReport:
    """Measure scalar E(3) invariance and a provisional block-wise profile.

    Final scalar rotation/translation/permutation errors have direct observable
    semantics. The block-wise transform depends on internal hook coordinate
    conventions and is therefore diagnostic rather than a standalone rejection
    criterion until independently calibrated for the selected backbone.
    """

    import torch
    from e3nn import o3

    graph = _single_graph(data)
    model.eval()
    rotation_errors: List[float] = []
    translation_errors: List[float] = []
    permutation_errors: List[float] = []
    layer_errors: Dict[str, List[float]] = {
        "block_{}".format(index): [] for index in range(len(model.blocks))
    }

    captures: List[torch.Tensor] = []
    hooks = []
    for block in model.blocks:
        hooks.append(block.register_forward_hook(lambda _m, _i, out: captures.append(out.detach())))

    graph_module = None
    original_radius_graph = None
    if freeze_neighbors:
        import importlib

        graph_module = importlib.import_module(model.__class__.__module__)
        original_radius_graph = graph_module.radius_graph
        fixed_edges = original_radius_graph(
            graph.pos,
            r=model.max_radius,
            batch=graph.batch,
            max_num_neighbors=1000,
        )

        def fixed_radius_graph(_pos, r, batch, max_num_neighbors=1000):
            return fixed_edges

        graph_module.radius_graph = fixed_radius_graph

    try:
        with torch.no_grad():
            captures.clear()
            reference = _model_call(model, graph)
            reference_layers = [item.clone() for item in captures]

            for _ in range(rotations):
                # e3nn 0.4.4 checks det(R)==1 with a tolerance that is brittle
                # for float32 random matrices. Construct SO(3) in float64, then
                # cast only the position transform and representation matrix.
                rotation = o3.rand_matrix(dtype=torch.float64, device=graph.pos.device)
                rotated_pos = graph.pos @ rotation.to(graph.pos.dtype).transpose(0, 1)
                captures.clear()
                rotated = _model_call(model, graph, positions=rotated_pos)
                rotated_layers = [item.clone() for item in captures]
                rotation_errors.append(_relative_error(reference, rotated))

                for index, (base_features, rotated_features) in enumerate(
                    zip(reference_layers, rotated_layers)
                ):
                    irreps = (
                        model.irreps_node_embedding
                        if index < len(model.blocks) - 1
                        else model.irreps_feature
                    )
                    representation = irreps.D_from_matrix(rotation).to(base_features.dtype)
                    expected = base_features @ representation.transpose(0, 1)
                    layer_errors["block_{}".format(index)].append(
                        _relative_error(expected, rotated_features)
                    )

            for _ in range(translations):
                shift = torch.randn(1, 3, dtype=graph.pos.dtype, device=graph.pos.device)
                translated = _model_call(model, graph, positions=graph.pos + shift)
                translation_errors.append(_relative_error(reference, translated))

            generator = torch.Generator(device=graph.pos.device)
            generator.manual_seed(20260721)
            permutation = torch.randperm(
                graph.pos.shape[0], generator=generator, device=graph.pos.device
            )
            permuted = _model_call(model, graph, permutation=permutation)
            permutation_errors.append(_relative_error(reference, permuted))
    finally:
        for hook in hooks:
            hook.remove()
        if graph_module is not None:
            graph_module.radius_graph = original_radius_graph

    return SymmetryReport(
        rotation_invariance=_summary(rotation_errors),
        translation_invariance=_summary(translation_errors),
        permutation_invariance=_summary(permutation_errors),
        layerwise_rotation_equivariance={
            name: _summary(values) for name, values in layer_errors.items()
        },
    )
