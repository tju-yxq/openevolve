"""Narrow runtime compatibility for the pinned QM9 training environment."""

from __future__ import annotations

from contextlib import contextmanager
import sys
import types


_TORCHVISION_SCHEMA_LIBRARY = None


def install_torchvision_schema_stubs() -> str:
    """Declare missing detection schemas needed only while importing old timm.

    Some CPU-incompatible torchvision builds omit their compiled operator
    schemas, while torchvision's meta-registration module still expects them.
    The QM9 trainer never executes these operators; declaring schemas is enough
    to import timm's scheduler and EMA utilities without inventing kernels.
    """

    import torch

    schemas = {
        "nms": "nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor",
        "qnms": "qnms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor",
    }
    missing = []
    namespace = torch.ops.torchvision
    for name in schemas:
        try:
            getattr(namespace, name)
        except AttributeError:
            missing.append(name)
    if not missing:
        return "native"
    global _TORCHVISION_SCHEMA_LIBRARY
    if _TORCHVISION_SCHEMA_LIBRARY is None:
        _TORCHVISION_SCHEMA_LIBRARY = torch.library.Library("torchvision", "FRAGMENT")
    for name in missing:
        _TORCHVISION_SCHEMA_LIBRARY.define(schemas[name])
    return "schema_stub"


@contextmanager
def trusted_legacy_torch_load():
    """Load trusted legacy PyG artifacts under the PyTorch 2.6+ default.

    The pinned Equiformer QM9 loader calls ``torch.load(path)`` on processed
    PyG ``Data`` objects.  PyTorch 2.6 changed that call to
    ``weights_only=True`` by default, which cannot deserialize the historical
    ``GlobalStorage`` payload.  Keep the compatibility override scoped to the
    dataset constructors; explicit caller choices are left unchanged.
    """

    import torch

    original_load = torch.load

    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compatible_load
    try:
        yield
    finally:
        torch.load = original_load


def load_trusted_training_checkpoint(path, *, map_location="cpu"):
    """Load a checkpoint produced by this repository's fixed-step trainer.

    These resumable checkpoints intentionally contain optimizer, scheduler,
    and NumPy/PyTorch RNG state in addition to tensor weights.  PyTorch 2.6's
    ``weights_only=True`` default rejects that trusted local payload, so every
    internal checkpoint consumer must opt in explicitly instead of relying on
    the version-dependent default.
    """

    import torch

    return torch.load(path, map_location=map_location, weights_only=False)


def scatter_fallback(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    """Differentiable PyTorch implementation of the torch_scatter.scatter API subset."""

    import torch

    if index.dtype != torch.long:
        index = index.long()
    dim = int(dim)
    if dim < 0:
        dim += src.ndim
    if dim < 0 or dim >= src.ndim:
        raise IndexError("scatter dim is outside the source rank")
    if index.ndim == 1:
        shape = [1] * src.ndim
        shape[dim] = index.numel()
        index = index.reshape(shape)
    try:
        expanded_index = index.expand_as(src)
    except RuntimeError as error:
        raise ValueError("scatter index is not broadcastable to the source tensor") from error
    resolved_size = int(dim_size) if dim_size is not None else (
        int(index.max().item()) + 1 if index.numel() else 0
    )
    output_shape = list(src.shape)
    output_shape[dim] = resolved_size
    normalized_reduce = str(reduce).lower()
    if normalized_reduce in ("add", "sum", "mean"):
        result = out if out is not None else src.new_zeros(output_shape)
        result.scatter_add_(dim, expanded_index, src)
        if normalized_reduce == "mean":
            counts = src.new_zeros(output_shape)
            counts.scatter_add_(dim, expanded_index, torch.ones_like(src))
            result = result / counts.clamp_min_(1)
        return result
    reduce_map = {
        "max": "amax",
        "min": "amin",
        "mul": "prod",
        "prod": "prod",
    }
    if normalized_reduce not in reduce_map:
        raise ValueError("unsupported scatter reduction: {}".format(reduce))
    if out is not None:
        result = out
    else:
        fill = 1 if reduce_map[normalized_reduce] == "prod" else 0
        result = src.new_full(output_shape, fill)
    return result.scatter_reduce(
        dim,
        expanded_index,
        src,
        reduce=reduce_map[normalized_reduce],
        include_self=False,
    )


def radius_graph_fallback(
    x,
    r,
    batch=None,
    loop=False,
    max_num_neighbors=32,
    flow="source_to_target",
    num_workers=1,
    batch_size=None,
):
    """Small differentiable-topology-compatible replacement for torch_cluster.radius_graph."""

    del num_workers, batch_size
    import torch

    if x.ndim != 2:
        raise ValueError("radius_graph positions must have shape [node, coordinate]")
    node_count = int(x.shape[0])
    if batch is None:
        batch = torch.zeros(node_count, dtype=torch.long, device=x.device)
    else:
        batch = batch.to(device=x.device, dtype=torch.long)
    if batch.ndim != 1 or int(batch.numel()) != node_count:
        raise ValueError("radius_graph batch must contain one graph id per node")
    squared_distance = (x[:, None, :] - x[None, :, :]).square().sum(dim=-1)
    same_graph = batch[:, None].eq(batch[None, :])
    mask = same_graph & squared_distance.le(float(r) * float(r))
    if not loop:
        mask.fill_diagonal_(False)
    target_source_pairs = []
    neighbor_limit = int(max_num_neighbors)
    for target in range(node_count):
        sources = torch.nonzero(mask[target], as_tuple=False).flatten()
        if neighbor_limit > 0 and int(sources.numel()) > neighbor_limit:
            order = torch.argsort(squared_distance[target].index_select(0, sources))
            sources = sources.index_select(0, order[:neighbor_limit])
        if sources.numel():
            targets = torch.full_like(sources, target)
            target_source_pairs.append((targets, sources))
    if not target_source_pairs:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    targets = torch.cat([item[0] for item in target_source_pairs])
    sources = torch.cat([item[1] for item in target_source_pairs])
    if flow == "source_to_target":
        return torch.stack((sources, targets), dim=0)
    if flow == "target_to_source":
        return torch.stack((targets, sources), dim=0)
    raise ValueError("unsupported radius_graph flow: {}".format(flow))


def install_torch_scatter_fallback() -> str:
    """Return ``native`` or install a fallback when the binary extension cannot load."""

    try:
        from torch_scatter import scatter as native_scatter  # noqa: F401

        return "native"
    except (ImportError, OSError):
        module = types.ModuleType("torch_scatter")
        module.scatter = scatter_fallback
        module.scatter_add = lambda src, index, dim=-1, out=None, dim_size=None: scatter_fallback(
            src, index, dim=dim, out=out, dim_size=dim_size, reduce="sum"
        )
        module.scatter_mean = lambda src, index, dim=-1, out=None, dim_size=None: scatter_fallback(
            src, index, dim=dim, out=out, dim_size=dim_size, reduce="mean"
        )
        sys.modules["torch_scatter"] = module
        return "pytorch_fallback"


def install_torch_cluster_fallback() -> str:
    """Return ``native`` or install a radius-graph fallback for an unloadable extension."""

    try:
        from torch_cluster import radius_graph as native_radius_graph  # noqa: F401

        return "native"
    except (ImportError, OSError):
        module = types.ModuleType("torch_cluster")
        module.radius_graph = radius_graph_fallback
        sys.modules["torch_cluster"] = module
        return "pytorch_fallback"
