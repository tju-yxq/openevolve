"""Narrow torch_scatter compatibility for legacy QM9 loader imports."""

from __future__ import annotations

import sys
import types


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
