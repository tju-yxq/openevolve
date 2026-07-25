import pytest
import os
from pathlib import Path


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")
pytest.importorskip("torch_cluster")

from equivariant_nas.dsl import Compiler, core_registry, import_equiformer_v1, reference_motif_registry
from equivariant_nas.dsl.backends import build_qm9_dsl_model
from equivariant_nas.spec import baseline_spec


def test_imported_v1_representation_flow_runs_with_qm9_signature_and_gradients():
    equiformer_root = Path(os.environ.get("EQUIFORMER_ROOT", "/home/20262202788/equiformer"))
    if not equiformer_root.is_dir():
        pytest.skip("official Equiformer V1 source is unavailable")
    program = import_equiformer_v1(baseline_spec())
    compiler = Compiler(core_registry(), reference_motif_registry())
    model = build_qm9_dsl_model(program, compiler, radius=10.0, equiformer_root=str(equiformer_root))
    features = torch.randn(7, 5, dtype=torch.float32)
    positions = torch.randn(7, 3, dtype=torch.float32)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    node_atom = torch.tensor([1, 6, 7, 8, 1, 6, 8], dtype=torch.long)
    prediction = model(features, positions, batch, node_atom)
    assert prediction.shape == (2, 1)
    loss = prediction.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)

    model.eval()
    from e3nn import o3

    reference = model(features, positions, batch, node_atom)
    rotation = o3.rand_matrix(dtype=torch.float32)
    rotated = model(features, positions @ rotation.transpose(0, 1), batch, node_atom)
    translated = model(
        features,
        positions + torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float32),
        batch,
        node_atom,
    )
    permutation = torch.tensor([3, 2, 1, 0, 6, 5, 4], dtype=torch.long)
    permuted = model(
        features[permutation],
        positions[permutation],
        batch[permutation],
        node_atom[permutation],
    )
    assert torch.allclose(rotated, reference, atol=1e-5, rtol=1e-5)
    assert torch.allclose(translated, reference, atol=1e-5, rtol=1e-5)
    assert torch.allclose(permuted, reference, atol=1e-5, rtol=1e-5)
