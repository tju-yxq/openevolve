import importlib
import math
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from equivariant_nas.dsl import (  # noqa: E402
    DSLValidationError,
    TypeChecker,
    core_registry,
    equiformer_v3_backbone_program,
    equiformer_v3_feed_forward_program,
    equiformer_v3_transblock_program,
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


def _spec():
    return EquiformerV3Spec(
        use_pbc=False,
        use_pbc_single=False,
        otf_graph=False,
        regress_forces=True,
        regress_stress=False,
        direct_prediction=True,
        max_neighbors=8,
        max_radius=4.5,
        num_radial_basis=6,
        max_num_elements=32,
        num_layers=2,
        num_channels=3,
        attn_hidden_channels=2,
        num_heads=1,
        attn_alpha_channels=2,
        attn_value_channels=2,
        ffn_hidden_channels=4,
        lmax=2,
        mmax=1,
        attn_grid_resolution=(8, 4),
        ffn_grid_resolution=(8, 8),
        edge_channels=5,
        attn_activation="sep-merge_gates2_swiglu",
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        avg_degree=3.25,
    )


def _install_torch_geometric_oracle_stub():
    try:
        package = importlib.import_module("torch_geometric")
        utils = importlib.import_module("torch_geometric.utils")
        num_nodes = importlib.import_module("torch_geometric.utils.num_nodes")
        if (
            hasattr(utils, "scatter")
            and hasattr(utils, "segment")
            and hasattr(num_nodes, "maybe_num_nodes")
        ):
            return
    except ModuleNotFoundError:
        package = types.ModuleType("torch_geometric")
        utils = types.ModuleType("torch_geometric.utils")
        num_nodes = types.ModuleType("torch_geometric.utils.num_nodes")

    def maybe_num_nodes(index, requested=None):
        if requested is not None:
            return int(requested)
        return int(index.max().item()) + 1 if index.numel() else 0

    def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
        if dim != 0:
            raise NotImplementedError("the V3 backbone oracle uses dim=0")
        count = maybe_num_nodes(index, dim_size)
        if reduce == "max":
            output = src.new_full((count,) + src.shape[1:], float("-inf"))
            expanded = index.reshape(
                (index.shape[0],) + (1,) * (src.dim() - 1)
            ).expand_as(src)
            output.scatter_reduce_(
                0,
                expanded,
                src,
                reduce="amax",
                include_self=True,
            )
            return output
        if reduce == "sum":
            output = src.new_zeros((count,) + src.shape[1:])
            output.index_add_(0, index, src)
            return output
        raise NotImplementedError(reduce)

    def segment(*_args, **_kwargs):
        raise NotImplementedError("the oracle exercises the explicit index path")

    utils.scatter = scatter
    utils.segment = segment
    num_nodes.maybe_num_nodes = maybe_num_nodes
    package.utils = utils
    sys.modules["torch_geometric"] = package
    sys.modules["torch_geometric.utils"] = utils
    sys.modules["torch_geometric.utils.num_nodes"] = num_nodes


def _official_modules():
    if not equiformer_v3_source_available(str(V3_ROOT)):
        pytest.skip("official Equiformer V3 operator source is unavailable")
    _install_torch_geometric_oracle_stub()
    loaded = load_equiformer_v3_modules(str(V3_ROOT))
    package = loaded[0].__package__
    return {
        "so3": loaded[0],
        "radial": importlib.import_module(package + ".radial_function"),
        "envelope": importlib.import_module(package + ".envelope"),
        "input_block": importlib.import_module(package + ".input_block"),
        "edge_rot_mat": importlib.import_module(package + ".edge_rot_mat"),
        "transformer": importlib.import_module(package + ".transformer_block"),
    }


class _OfficialBackbone(torch.nn.Module):
    def __init__(self, spec, modules):
        super().__init__()
        self.spec = spec
        self.modules_by_name = modules
        self.sphere_embedding = torch.nn.Embedding(
            spec.max_num_elements,
            spec.num_channels,
        )
        self.distance_expansion = modules["radial"].GaussianSmearing(
            0.0,
            spec.max_radius,
            spec.num_radial_basis,
            2.0,
        )
        self.envelope_func = modules["envelope"].PolynomialEnvelope(
            spec.max_radius,
            5,
        )
        self.so3_rotation = modules["so3"].SO3Rotation(
            spec.lmax,
            spec.mmax,
            use_rotation_mask=not spec.direct_prediction,
        )
        edge_channels = [
            spec.num_radial_basis,
            spec.edge_channels,
            spec.edge_channels,
        ]
        self.edge_degree_embedding = modules["input_block"].EdgeDegreeEmbedding(
            num_channels=spec.num_channels,
            lmax=spec.lmax,
            mmax=spec.mmax,
            so3_rotation=self.so3_rotation,
            max_num_elements=spec.max_num_elements,
            edge_channels_list=edge_channels,
            use_atom_edge_embedding=spec.use_atom_edge_embedding,
            rescale_factor=spec.avg_degree,
        )
        self.blocks = torch.nn.ModuleList(
            [
                modules["transformer"].TransBlockV3(
                    num_in_channels=spec.num_channels,
                    attn_hidden_channels=spec.attn_hidden_channels,
                    num_heads=spec.num_heads,
                    attn_alpha_channels=spec.attn_alpha_channels,
                    attn_value_channels=spec.attn_value_channels,
                    ffn_hidden_channels=spec.ffn_hidden_channels,
                    num_out_channels=spec.num_channels,
                    lmax=spec.lmax,
                    mmax=spec.mmax,
                    so3_rotation=self.so3_rotation,
                    attn_grid_resolution_list=list(spec.attn_grid_resolution),
                    ffn_grid_resolution_list=list(spec.ffn_grid_resolution),
                    max_num_elements=spec.max_num_elements,
                    edge_channels_list=edge_channels,
                    use_atom_edge_embedding=spec.use_atom_edge_embedding,
                    attn_activation=spec.attn_activation,
                    use_attn_renorm=spec.use_attn_renorm,
                    use_add_merge=spec.use_add_merge,
                    use_rad_l_parametrization=spec.use_rad_l_parametrization,
                    softcap=spec.softcap,
                    attn_eps=spec.attn_eps,
                    ffn_activation=spec.ffn_activation,
                    use_grid_mlp=spec.use_grid_mlp,
                    norm_type=spec.norm_type,
                    alpha_drop=spec.alpha_drop,
                    attn_mask_rate=spec.attn_mask_rate,
                    attn_weights_drop=spec.attn_weights_drop,
                    value_drop=spec.value_drop,
                    drop_path_rate=spec.drop_path_rate,
                    proj_drop=spec.proj_drop,
                    ffn_drop=spec.ffn_drop,
                )
                for _ in range(spec.num_layers)
            ]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _uniform_init_linear_weights(module):
        if isinstance(module, torch.nn.Linear):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
            bound = 1.0 / math.sqrt(module.in_features)
            torch.nn.init.uniform_(module.weight, -bound, bound)

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, self.modules_by_name["so3"].SO3Linear)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, self.modules_by_name["radial"].RadialFunction):
            module.apply(self._uniform_init_linear_weights)

    def forward(self, atomic_numbers, positions, edge_index, batch):
        source = edge_index[0]
        target = edge_index[1]
        edge_vector = positions.index_select(0, source) - positions.index_select(0, target)
        rotation_matrix = self.modules_by_name["edge_rot_mat"].init_edge_rot_mat(
            edge_vector,
            use_rotation_mask=not self.spec.direct_prediction,
        )
        self.so3_rotation.set_wigner(rotation_matrix)
        edge_distance = torch.linalg.vector_norm(edge_vector, dim=-1, keepdim=True)
        edge_envelope = self.envelope_func(edge_distance)
        edge_radial = self.distance_expansion(edge_distance)
        atom_embedding = self.sphere_embedding(atomic_numbers)
        node_embedding = atom_embedding.new_zeros(
            atomic_numbers.shape[0],
            (self.spec.lmax + 1) ** 2,
            self.spec.num_channels,
        )
        node_embedding[:, 0, :] = atom_embedding
        node_embedding = node_embedding + self.edge_degree_embedding(
            atomic_numbers,
            edge_radial,
            edge_index,
            edge_envelope,
        )
        source_atomic_numbers = atomic_numbers.index_select(0, source)
        target_atomic_numbers = atomic_numbers.index_select(0, target)
        for block in self.blocks:
            node_embedding = block(
                node_embedding,
                source_atomic_numbers,
                target_atomic_numbers,
                edge_radial,
                edge_index,
                edge_envelope,
                batch,
            )
        return node_embedding


def _attention_parameter_mapping(spec):
    mapping = {
        "node_modules.source_embedding.weight": "source_embedding.weight",
        "node_modules.target_embedding.weight": "target_embedding.weight",
        "node_modules.radial_linear0.linear.weight": "rad_func.net.0.weight",
        "node_modules.radial_linear0.linear.bias": "rad_func.net.0.bias",
        "node_modules.radial_norm0.layer_norm.weight": "rad_func.net.1.weight",
        "node_modules.radial_norm0.layer_norm.bias": "rad_func.net.1.bias",
        "node_modules.radial_linear1.linear.weight": "rad_func.net.3.weight",
        "node_modules.radial_linear1.linear.bias": "rad_func.net.3.bias",
        "node_modules.radial_norm1.layer_norm.weight": "rad_func.net.4.weight",
        "node_modules.radial_norm1.layer_norm.bias": "rad_func.net.4.bias",
        "node_modules.radial_linear2.linear.weight": "rad_func.net.6.weight",
        "node_modules.radial_linear2.linear.bias": "rad_func.net.6.bias",
        "node_modules.so2_linear1.fc_m0.weight": "so2_linear_1.fc_m0.weight",
        "node_modules.so2_linear1.fc_m0.bias": "so2_linear_1.fc_m0.bias",
        "node_modules.alpha_norm.layer_norm.weight": "alpha_norm.weight",
        "node_modules.alpha_norm.layer_norm.bias": "alpha_norm.bias",
        "node_modules.alpha_dot.weight": "alpha_dot",
        "node_modules.so2_linear2.fc_m0.weight": "so2_linear_2.fc_m0.weight",
        "node_modules.so2_linear2.fc_m0.bias": "so2_linear_2.fc_m0.bias",
        "node_modules.projection.weight": "proj.weight",
        "node_modules.projection.bias": "proj.bias",
    }
    for order in range(1, spec.mmax + 1):
        index = order - 1
        mapping[
            "node_modules.so2_linear1.so2_m_linear.{}.fc.weight".format(index)
        ] = "so2_linear_1.so2_m_linear.{}.fc.weight".format(index)
        mapping[
            "node_modules.so2_linear2.so2_m_linear.{}.fc.weight".format(index)
        ] = "so2_linear_2.so2_m_linear.{}.fc.weight".format(index)
    return mapping


def _transblock_parameter_mapping(spec):
    mapping = {
        "node_modules.norm1.norm.affine_weight": "norm_1.affine_weight",
        "node_modules.norm1.norm.affine_bias": "norm_1.affine_bias",
        "node_modules.norm2.norm.affine_weight": "norm_2.affine_weight",
        "node_modules.norm2.norm.affine_bias": "norm_2.affine_bias",
    }
    for local_name, official_name in _attention_parameter_mapping(spec).items():
        mapping[local_name.replace("node_modules.", "node_modules.attn_")] = (
            "ga." + official_name
        )
    ffn_mapping = equiformer_v3_feed_forward_program(spec).parameters[
        "parameter_mapping"
    ]
    for local_name, official_name in ffn_mapping.items():
        mapping[local_name.replace("node_modules.", "node_modules.ffn_")] = (
            "ffn." + official_name
        )
    return mapping


def _backbone_parameter_mapping(spec, program):
    mapping = {}
    input_mapping = program.parameters["parameter_mapping"]["input"]
    for local_name, official_name in input_mapping.items():
        mapping[local_name.replace("node_modules.", "node_modules.input_")] = official_name
    block_mapping = _transblock_parameter_mapping(spec)
    for index in range(spec.num_layers):
        for local_name, official_name in block_mapping.items():
            mapping[
                local_name.replace("node_modules.", "node_modules.block{}_".format(index))
            ] = "blocks.{}.{}".format(index, official_name)
    return mapping


def _index(indices, target_size):
    return {"indices": indices, "target_size": target_size}


def test_v3_backbone_rejects_an_empty_shared_frame_cache_id():
    registry = core_registry()
    program = equiformer_v3_backbone_program(_spec())
    broken_nodes = []
    for node in program.nodes:
        if node.id == "block0_attn_message_rotate":
            broken_nodes.append(
                replace(node, attrs={**dict(node.attrs), "frame_cache_id": ""})
            )
        else:
            broken_nodes.append(node)
    with pytest.raises(DSLValidationError) as error:
        TypeChecker(registry).check(replace(program, nodes=tuple(broken_nodes)))
    assert any(item.code == "E_FRAME_V2_004" for item in error.value.diagnostics)


def test_v3_two_layer_backbone_matches_official_parameters_forward_and_gradients():
    modules = _official_modules()
    spec = _spec()
    registry = core_registry()
    program = equiformer_v3_backbone_program(spec)
    inference = TypeChecker(registry).check(program)

    torch.manual_seed(9501)
    expected = _OfficialBackbone(spec, modules)
    expected_init_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(9501)
    actual = E3NNGraphBackend(
        registry,
        equiformer_v3_root=str(V3_ROOT),
    ).build(program, inference)
    actual_init_rng = torch.random.get_rng_state().clone()
    assert torch.equal(actual_init_rng, expected_init_rng)

    expected_parameters = dict(expected.named_parameters())
    actual_parameters = dict(actual.named_parameters())
    mapping = _backbone_parameter_mapping(spec, program)
    assert len(program.nodes) == 24 + spec.num_layers * len(
        equiformer_v3_transblock_program(spec).nodes
    )
    assert set(mapping) == set(actual_parameters)
    assert set(mapping.values()) == set(expected_parameters)
    for actual_name, expected_name in mapping.items():
        torch.testing.assert_close(
            actual_parameters[actual_name],
            expected_parameters[expected_name],
            rtol=0.0,
            atol=0.0,
        )

    atomic_numbers = torch.tensor([1, 6, 8, 14], dtype=torch.long)
    positions_expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.2, -0.1],
            [0.3, 1.2, 0.4],
            [-0.4, 0.6, 1.3],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    positions_actual = positions_expected.detach().clone().requires_grad_(True)
    source = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    target = torch.tensor([1, 2, 3, 0, 2, 1], dtype=torch.long)
    edge_index = torch.stack((source, target), dim=0)
    batch = torch.zeros(atomic_numbers.numel(), dtype=torch.long)

    torch.manual_seed(9502)
    expected_output = expected(
        atomic_numbers,
        positions_expected,
        edge_index,
        batch,
    )
    expected_forward_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(9502)
    actual_outputs = actual(
        {
            "atomic_numbers": atomic_numbers,
            "positions": positions_actual,
            "source_index": _index(source, source.numel()),
            "target_index": _index(target, target.numel()),
            "target_segment": _index(target, atomic_numbers.numel()),
            "batch": batch,
        },
        {},
    )
    actual_forward_rng = torch.random.get_rng_state().clone()
    actual_output = flat_to_embedding_tensor(
        actual_outputs["node_features"],
        inference.value_types["block1_ffn_residual"].irreps,
    )
    torch.testing.assert_close(actual_output, expected_output, rtol=3.0e-5, atol=3.0e-5)
    assert torch.equal(actual_forward_rng, expected_forward_rng)

    expected_loss = expected_output.square().sum()
    actual_loss = actual_output.square().sum()
    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(
        positions_actual.grad,
        positions_expected.grad,
        rtol=5.0e-5,
        atol=5.0e-5,
    )
    for actual_name, expected_name in mapping.items():
        torch.testing.assert_close(
            actual_parameters[actual_name].grad,
            expected_parameters[expected_name].grad,
            rtol=8.0e-5,
            atol=8.0e-5,
        )
