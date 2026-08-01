import sys

import pytest

from equivariant_nas.training.torch_scatter_compat import (
    install_torch_scatter_fallback,
    scatter_fallback,
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
