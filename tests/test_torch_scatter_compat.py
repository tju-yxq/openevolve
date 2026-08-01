import sys

import pytest

from equivariant_nas.training.torch_scatter_compat import (
    install_torch_cluster_fallback,
    install_torch_scatter_fallback,
    install_torchvision_schema_stubs,
    scatter_fallback,
    radius_graph_fallback,
    trusted_legacy_torch_load,
)


def test_scatter_fallback_sum_mean_and_gradient():
    torch = pytest.importorskip("torch")
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    index = torch.tensor([0, 1, 0])

    summed = scatter_fallback(source, index, dim=0, dim_size=2, reduce="sum")
    mean = scatter_fallback(source, index, dim=0, dim_size=2, reduce="mean")

    assert torch.equal(summed, torch.tensor([[6.0, 8.0], [3.0, 4.0]]))
    assert torch.equal(mean, torch.tensor([[3.0, 4.0], [3.0, 4.0]]))
    summed.sum().backward()
    assert torch.equal(source.grad, torch.ones_like(source))


def test_radius_graph_fallback_filters_graphs_self_loops_and_neighbor_count():
    torch = pytest.importorskip("torch")
    positions = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.9, 0.0], [0.0, 0.0]])
    batch = torch.tensor([0, 0, 0, 1])

    edge_index = radius_graph_fallback(
        positions,
        r=1.0,
        batch=batch,
        loop=False,
        max_num_neighbors=1,
    )

    assert edge_index.shape == (2, 3)
    assert not torch.any(edge_index[0] == edge_index[1])
    assert torch.equal(torch.bincount(edge_index[1], minlength=4), torch.tensor([1, 1, 1, 0]))
    assert torch.all(batch.index_select(0, edge_index[0]) == batch.index_select(0, edge_index[1]))


def test_install_fallback_replaces_an_unloadable_extension(monkeypatch):
    original = sys.modules.pop("torch_scatter", None)
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch_scatter":
            raise OSError("incompatible GLIBC")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    try:
        assert install_torch_scatter_fallback() == "pytorch_fallback"
        assert sys.modules["torch_scatter"].scatter is scatter_fallback
    finally:
        sys.modules.pop("torch_scatter", None)
        if original is not None:
            sys.modules["torch_scatter"] = original


def test_install_cluster_fallback_replaces_an_unloadable_extension(monkeypatch):
    original = sys.modules.pop("torch_cluster", None)
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch_cluster":
            raise OSError("incompatible GLIBC")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    try:
        assert install_torch_cluster_fallback() == "pytorch_fallback"
        assert sys.modules["torch_cluster"].radius_graph is radius_graph_fallback
    finally:
        sys.modules.pop("torch_cluster", None)
        if original is not None:
            sys.modules["torch_cluster"] = original


def test_trusted_legacy_torch_load_only_changes_the_implicit_default(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    def fake_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return "loaded"

    monkeypatch.setattr(torch, "load", fake_load)
    original = torch.load
    with trusted_legacy_torch_load():
        assert torch.load("legacy.pt") == "loaded"
        assert torch.load("explicit.pt", weights_only=True) == "loaded"

    assert torch.load is original
    assert calls == [{"weights_only": False}, {"weights_only": True}]


def test_torchvision_schema_compat_is_idempotent():
    torch = pytest.importorskip("torch")
    first = install_torchvision_schema_stubs()
    second = install_torchvision_schema_stubs()

    assert first in {"native", "schema_stub"}
    assert second == "native"
    assert torch.ops.torchvision.nms is not None
    assert torch.ops.torchvision.qnms is not None
