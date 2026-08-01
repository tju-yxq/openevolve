import importlib
import math
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from equivariant_nas.dsl import (  # noqa: E402
    TypeChecker,
    canonicalize,
    core_registry,
    equiformer_v3_input_program,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend  # noqa: E402
from equivariant_nas.dsl.backends.equiformer_v3_spec import EquiformerV3Spec  # noqa: E402
from equivariant_nas.dsl.backends.v2_runtime import flat_to_embedding_tensor  # noqa: E402
from equivariant_nas.dsl.backends.v3_runtime import (  # noqa: E402
    equiformer_v3_source_available,
    load_equiformer_v3_modules,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = REPOSITORY_ROOT.parent / "equiformer_v3_official"


def _spec(*, use_pbc=False):
    return EquiformerV3Spec(
        use_pbc=use_pbc,
        use_pbc_single=False,
        otf_graph=False,
        regress_forces=True,
        regress_stress=False,
        direct_prediction=True,
        max_neighbors=8,
        max_radius=4.5,
        num_radial_basis=6,
        max_num_elements=32,
        num_layers=1,
        num_channels=3,
        attn_hidden_channels=2,
        num_heads=1,
        attn_alpha_channels=2,
        attn_value_channels=2,
        ffn_hidden_channels=4,
        lmax=2,
        mmax=1,
        edge_channels=5,
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        avg_degree=3.25,
    )


def _require_modules():
    if not equiformer_v3_source_available(str(V3_ROOT)):
        pytest.skip("official Equiformer V3 operator source is unavailable")
    modules = load_equiformer_v3_modules(str(V3_ROOT))
    package = modules[0].__package__
    return {
        "so3": modules[0],
        "radial": importlib.import_module(package + ".radial_function"),
        "envelope": importlib.import_module(package + ".envelope"),
        "input_block": importlib.import_module(package + ".input_block"),
        "edge_rot_mat": importlib.import_module(package + ".edge_rot_mat"),
    }


def _index(indices, target_size):
    return {"indices": indices, "target_size": target_size}


def _official_input_oracle(spec, modules):
    class OfficialInput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sphere_embedding = torch.nn.Embedding(spec.max_num_elements, spec.num_channels)
            self.distance_expansion = modules["radial"].GaussianSmearing(
                0.0,
                spec.max_radius,
                spec.num_radial_basis,
                2.0,
            )
            self.envelope_func = modules["envelope"].PolynomialEnvelope(spec.max_radius, 5)
            self.so3_rotation = modules["so3"].SO3Rotation(spec.lmax, spec.mmax, use_rotation_mask=False)
            self.edge_degree_embedding = modules["input_block"].EdgeDegreeEmbedding(
                num_channels=spec.num_channels,
                lmax=spec.lmax,
                mmax=spec.mmax,
                so3_rotation=self.so3_rotation,
                max_num_elements=spec.max_num_elements,
                edge_channels_list=[spec.num_radial_basis, spec.edge_channels, spec.edge_channels],
                use_atom_edge_embedding=spec.use_atom_edge_embedding,
                rescale_factor=spec.avg_degree,
            )
            self.apply(self._init_weights)

        @staticmethod
        def _uniform_init_linear_weights(module):
            if isinstance(module, torch.nn.Linear):
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
                std = 1.0 / math.sqrt(module.in_features)
                torch.nn.init.uniform_(module.weight, -std, std)

        def _init_weights(self, module):
            if isinstance(module, torch.nn.Linear):
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.constant_(module.bias, 0)
                torch.nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, modules["radial"].RadialFunction):
                module.apply(self._uniform_init_linear_weights)

        def forward(self, atomic_numbers, positions, edge_index):
            source = edge_index[0]
            target = edge_index[1]
            edge_vector = positions.index_select(0, source) - positions.index_select(0, target)
            edge_rotation = modules["edge_rot_mat"].init_edge_rot_mat(edge_vector, use_rotation_mask=False)
            self.so3_rotation.set_wigner(edge_rotation)
            edge_distance_scalar = torch.linalg.vector_norm(edge_vector, dim=-1, keepdim=True)
            edge_envelope = self.envelope_func(edge_distance_scalar)
            edge_radial = self.distance_expansion(edge_distance_scalar)
            atom_embedding = self.sphere_embedding(atomic_numbers)
            node_embedding = atom_embedding.new_zeros(
                atomic_numbers.shape[0],
                (spec.lmax + 1) ** 2,
                spec.num_channels,
            )
            node_embedding[:, 0, :] = atom_embedding
            node_embedding = node_embedding + self.edge_degree_embedding(
                atomic_numbers,
                edge_radial,
                edge_index,
                edge_envelope,
            )
            return node_embedding, edge_radial, edge_envelope, edge_vector

    return OfficialInput()


def test_v3_official_input_program_typechecks_and_canonicalization_preserves_rng_contract():
    modules = _require_modules()
    spec = _spec(use_pbc=False)
    registry = core_registry()
    program = equiformer_v3_input_program(spec)
    canonical = canonicalize(program, registry)
    inference = TypeChecker(registry).check(canonical)
    assert len(registry.names()) == 102
    assert len(program.nodes) == 24
    assert set(item.op for item in program.nodes).issubset(registry.names())
    assert all("equiformer_v3" not in item.op for item in program.nodes)
    canonical_order = canonical.parameters["lowering_contract"]["module_construction_order"]
    assert len(canonical_order) == len(canonical.nodes)
    assert set(canonical_order) == {node.id for node in canonical.nodes}

    torch.manual_seed(7311)
    original_model = E3NNGraphBackend(
        registry,
        equiformer_v3_root=str(V3_ROOT),
    ).build(program, TypeChecker(registry).check(program))
    torch.manual_seed(7311)
    canonical_model = E3NNGraphBackend(
        registry,
        equiformer_v3_root=str(V3_ROOT),
    ).build(canonical, inference)
    original_parameters = list(original_model.parameters())
    canonical_parameters = list(canonical_model.parameters())
    assert len(original_parameters) == len(canonical_parameters)
    for original, normalized in zip(original_parameters, canonical_parameters):
        torch.testing.assert_close(original, normalized, rtol=0.0, atol=0.0)
    del modules


def test_v3_official_input_program_matches_official_edge_degree_initialization_forward_and_gradients():
    modules = _require_modules()
    spec = _spec(use_pbc=False)
    registry = core_registry()
    program = equiformer_v3_input_program(spec)
    inference = TypeChecker(registry).check(program)

    torch.manual_seed(8123)
    expected = _official_input_oracle(spec, modules)
    torch.manual_seed(8123)
    actual = E3NNGraphBackend(
        registry,
        equiformer_v3_root=str(V3_ROOT),
    ).build(program, inference)

    expected_parameters = list(expected.parameters())
    actual_parameters = list(actual.parameters())
    assert [tuple(item.shape) for item in actual_parameters] == [
        tuple(item.shape) for item in expected_parameters
    ]
    for actual_parameter, expected_parameter in zip(actual_parameters, expected_parameters):
        torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0.0, atol=0.0)

    atomic_numbers = torch.tensor([1, 6, 8, 14], dtype=torch.long)
    positions_expected = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1], [0.3, 1.2, 0.4], [-0.4, 0.6, 1.3]],
        dtype=torch.float32,
        requires_grad=True,
    )
    positions_actual = positions_expected.detach().clone().requires_grad_(True)
    source = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    target = torch.tensor([1, 2, 3, 0, 2, 1], dtype=torch.long)
    edge_index = torch.stack((source, target), dim=0)

    torch.manual_seed(9901)
    expected_node, expected_radial, expected_envelope, expected_vector = expected(
        atomic_numbers,
        positions_expected,
        edge_index,
    )
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(9901)
    outputs = actual(
        {
            "atomic_numbers": atomic_numbers,
            "positions": positions_actual,
            "source_index": _index(source, source.numel()),
            "target_index": _index(target, target.numel()),
            "target_segment": _index(target, atomic_numbers.numel()),
        },
        {},
    )
    actual_rng = torch.random.get_rng_state().clone()
    actual_node = flat_to_embedding_tensor(outputs["node_embedding"], inference.value_types["node_embedding"].irreps)
    torch.testing.assert_close(actual_node, expected_node, rtol=2.0e-6, atol=2.0e-6)
    torch.testing.assert_close(outputs["edge_radial"], expected_radial, rtol=0.0, atol=0.0)
    torch.testing.assert_close(outputs["edge_envelope"], expected_envelope, rtol=0.0, atol=0.0)
    torch.testing.assert_close(outputs["edge_vector"], expected_vector, rtol=0.0, atol=0.0)
    assert torch.equal(actual_rng, expected_rng)

    expected_loss = expected_node.square().sum() + expected_radial.square().sum()
    actual_loss = actual_node.square().sum() + outputs["edge_radial"].square().sum()
    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(positions_actual.grad, positions_expected.grad, rtol=3.0e-5, atol=3.0e-5)
    for actual_parameter, expected_parameter in zip(actual_parameters, expected_parameters):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=3.0e-5, atol=3.0e-5)


def test_v3_periodic_input_adapter_reconstructs_official_fairchem_edge_vectors():
    _require_modules()
    spec = _spec(use_pbc=True)
    registry = core_registry()
    program = equiformer_v3_input_program(spec)
    inference = TypeChecker(registry).check(program)
    model = E3NNGraphBackend(
        registry,
        equiformer_v3_root=str(V3_ROOT),
    ).build(program, inference)

    atomic_numbers = torch.tensor([6, 8, 14], dtype=torch.long)
    positions = torch.tensor([[0.1, 0.2, 0.3], [1.0, 0.4, 0.2], [0.3, 1.1, 0.8]])
    source = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    target = torch.tensor([1, 2, 0, 2], dtype=torch.long)
    cell = torch.tensor([[2.0, 0.0, 0.0], [0.1, 2.1, 0.0], [0.0, 0.2, 2.2]])
    official_cell_offsets = torch.tensor([[0, 0, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.long)
    outputs = model(
        {
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "source_index": _index(source, source.numel()),
            "target_index": _index(target, target.numel()),
            "target_segment": _index(target, atomic_numbers.numel()),
            "lattice": cell,
            "lattice_shift": official_cell_offsets,
        },
        {},
    )
    expected_vector = (
        positions.index_select(0, source)
        - positions.index_select(0, target)
        + official_cell_offsets.to(dtype=cell.dtype) @ cell
    )
    torch.testing.assert_close(outputs["edge_vector"], expected_vector, rtol=0.0, atol=0.0)


def test_v3_official_input_program_rejects_unsupported_full_model_float64_path():
    with pytest.raises(ValueError, match="frozen to float32"):
        equiformer_v3_input_program(_spec(), dtype="float64")
