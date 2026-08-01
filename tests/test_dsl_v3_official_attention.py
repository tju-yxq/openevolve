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
    TypeChecker,
    canonicalize,
    core_registry,
    equiformer_v3_attention_program,
    equiformer_v3_feed_forward_program,
    equiformer_v3_force_head_program,
    equiformer_v3_transblock_program,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend  # noqa: E402
from equivariant_nas.dsl.backends.equiformer_v3_spec import EquiformerV3Spec  # noqa: E402
from equivariant_nas.dsl.backends.v2_runtime import (  # noqa: E402
    embedding_tensor_to_flat,
    flat_to_embedding_tensor,
)
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
        num_layers=1,
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
            raise NotImplementedError("the V3 attention oracle uses dim=0")
        count = maybe_num_nodes(index, dim_size)
        if reduce == "max":
            output = src.new_full((count,) + src.shape[1:], float("-inf"))
            expanded = index.reshape((index.shape[0],) + (1,) * (src.dim() - 1)).expand_as(src)
            output.scatter_reduce_(0, expanded, src, reduce="amax", include_self=True)
            return output
        if reduce == "sum":
            output = src.new_zeros((count,) + src.shape[1:])
            output.index_add_(0, index, src)
            return output
        raise NotImplementedError(reduce)

    def segment(*_args, **_kwargs):
        raise NotImplementedError("the oracle test exercises the explicit index path")

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
    modules = load_equiformer_v3_modules(str(V3_ROOT))
    package = modules[0].__package__
    return {
        "so3": modules[0],
        "radial": importlib.import_module(package + ".radial_function"),
        "edge_rot_mat": importlib.import_module(package + ".edge_rot_mat"),
        "transformer": importlib.import_module(package + ".transformer_block"),
    }


def _official_attention(spec, modules):
    rotation = modules["so3"].SO3Rotation(
        spec.lmax,
        spec.mmax,
        use_rotation_mask=not spec.direct_prediction,
    )
    attention = modules["transformer"].EquivariantGraphAttention(
        num_in_channels=spec.num_channels,
        num_hidden_channels=spec.attn_hidden_channels,
        num_heads=spec.num_heads,
        attn_alpha_channels=spec.attn_alpha_channels,
        attn_value_channels=spec.attn_value_channels,
        num_out_channels=spec.num_channels,
        lmax=spec.lmax,
        mmax=spec.mmax,
        so3_rotation=rotation,
        grid_resolution_list=list(spec.attn_grid_resolution),
        max_num_elements=spec.max_num_elements,
        edge_channels_list=[spec.num_radial_basis, spec.edge_channels, spec.edge_channels],
        use_atom_edge_embedding=spec.use_atom_edge_embedding,
        activation=spec.attn_activation,
        use_attn_renorm=spec.use_attn_renorm,
        use_add_merge=spec.use_add_merge,
        use_rad_l_parametrization=spec.use_rad_l_parametrization,
        softcap=spec.softcap,
        eps=spec.attn_eps,
        alpha_drop=spec.alpha_drop,
        attn_mask_rate=spec.attn_mask_rate,
        attn_weights_drop=spec.attn_weights_drop,
        value_drop=spec.value_drop,
    )

    def uniform_init_linear(module):
        if isinstance(module, torch.nn.Linear):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
            bound = 1.0 / math.sqrt(module.in_features)
            torch.nn.init.uniform_(module.weight, -bound, bound)

    def init_weights(module):
        if isinstance(module, (torch.nn.Linear, modules["so3"].SO3Linear)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, modules["radial"].RadialFunction):
            module.apply(uniform_init_linear)

    attention.apply(init_weights)
    return attention, rotation


def _official_transblock(spec, modules):
    rotation = modules["so3"].SO3Rotation(
        spec.lmax,
        spec.mmax,
        use_rotation_mask=not spec.direct_prediction,
    )
    block = modules["transformer"].TransBlockV3(
        num_in_channels=spec.num_channels,
        attn_hidden_channels=spec.attn_hidden_channels,
        num_heads=spec.num_heads,
        attn_alpha_channels=spec.attn_alpha_channels,
        attn_value_channels=spec.attn_value_channels,
        ffn_hidden_channels=spec.ffn_hidden_channels,
        num_out_channels=spec.num_channels,
        lmax=spec.lmax,
        mmax=spec.mmax,
        so3_rotation=rotation,
        attn_grid_resolution_list=list(spec.attn_grid_resolution),
        ffn_grid_resolution_list=list(spec.ffn_grid_resolution),
        max_num_elements=spec.max_num_elements,
        edge_channels_list=[spec.num_radial_basis, spec.edge_channels, spec.edge_channels],
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

    def uniform_init_linear(module):
        if isinstance(module, torch.nn.Linear):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
            bound = 1.0 / math.sqrt(module.in_features)
            torch.nn.init.uniform_(module.weight, -bound, bound)

    def init_weights(module):
        if isinstance(module, (torch.nn.Linear, modules["so3"].SO3Linear)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, modules["radial"].RadialFunction):
            module.apply(uniform_init_linear)

    block.apply(init_weights)
    return block, rotation


def _official_force_head(spec, modules):
    rotation = modules["so3"].SO3Rotation(
        spec.lmax,
        spec.mmax,
        use_rotation_mask=not spec.direct_prediction,
    )
    force_head = modules["transformer"].EquivariantGraphAttention(
        num_in_channels=spec.num_channels,
        num_hidden_channels=spec.attn_hidden_channels,
        num_heads=spec.num_heads,
        attn_alpha_channels=spec.attn_alpha_channels,
        attn_value_channels=spec.attn_value_channels,
        num_out_channels=1,
        lmax=spec.lmax,
        mmax=spec.mmax,
        so3_rotation=rotation,
        grid_resolution_list=list(spec.attn_grid_resolution),
        max_num_elements=spec.max_num_elements,
        edge_channels_list=[spec.num_radial_basis, spec.edge_channels, spec.edge_channels],
        use_atom_edge_embedding=spec.use_atom_edge_embedding,
        activation="gate",
        use_attn_renorm=spec.use_attn_renorm,
        use_add_merge=spec.use_add_merge,
        use_rad_l_parametrization=spec.use_rad_l_parametrization,
        softcap=spec.softcap,
        eps=spec.attn_eps,
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
    )

    def uniform_init_linear(module):
        if isinstance(module, torch.nn.Linear):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
            bound = 1.0 / math.sqrt(module.in_features)
            torch.nn.init.uniform_(module.weight, -bound, bound)

    def init_weights(module):
        if isinstance(module, (torch.nn.Linear, modules["so3"].SO3Linear)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, modules["radial"].RadialFunction):
            module.apply(uniform_init_linear)

    force_head.apply(init_weights)
    return force_head, rotation


def _parameter_mapping(spec):
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
    for local_name, official_name in _parameter_mapping(spec).items():
        mapping[local_name.replace("node_modules.", "node_modules.attn_")] = (
            "ga." + official_name
        )
    ffn_mapping = equiformer_v3_feed_forward_program(spec).parameters["parameter_mapping"]
    for local_name, official_name in ffn_mapping.items():
        mapping[local_name.replace("node_modules.", "node_modules.ffn_")] = (
            "ffn." + official_name
        )
    return mapping


def _index(indices, target_size):
    return {"indices": indices, "target_size": target_size}


def test_v3_official_attention_program_is_fully_lowered_and_canonicalization_preserves_parameters():
    _official_modules()
    spec = _spec()
    registry = core_registry()
    program = equiformer_v3_attention_program(spec)
    canonical = canonicalize(program, registry)
    inference = TypeChecker(registry).check(canonical)
    assert len(registry.names()) == 102
    assert len(program.nodes) == 30
    assert all("motif." not in node.op for node in program.nodes)
    assert all("equiformer_v3" not in node.op for node in program.nodes)

    torch.manual_seed(8201)
    original = E3NNGraphBackend(registry, equiformer_v3_root=str(V3_ROOT)).build(
        program,
        TypeChecker(registry).check(program),
    )
    torch.manual_seed(8201)
    normalized = E3NNGraphBackend(registry, equiformer_v3_root=str(V3_ROOT)).build(
        canonical,
        inference,
    )
    original_parameters = list(original.parameters())
    normalized_parameters = list(normalized.parameters())
    assert len(original_parameters) == len(normalized_parameters)
    for original_parameter, normalized_parameter in zip(original_parameters, normalized_parameters):
        torch.testing.assert_close(original_parameter, normalized_parameter, rtol=0.0, atol=0.0)


def test_v3_official_attention_matches_initialization_forward_and_gradients():
    modules = _official_modules()
    spec = _spec()
    registry = core_registry()
    program = equiformer_v3_attention_program(spec)
    inference = TypeChecker(registry).check(program)

    torch.manual_seed(9103)
    expected, rotation = _official_attention(spec, modules)
    torch.manual_seed(9103)
    actual = E3NNGraphBackend(registry, equiformer_v3_root=str(V3_ROOT)).build(
        program,
        inference,
    )

    expected_parameters = dict(expected.named_parameters())
    actual_parameters = dict(actual.named_parameters())
    mapping = _parameter_mapping(spec)
    assert set(actual_parameters) == set(mapping)
    assert set(mapping.values()) == set(expected_parameters)
    for actual_name, expected_name in mapping.items():
        torch.testing.assert_close(
            actual_parameters[actual_name],
            expected_parameters[expected_name],
            rtol=0.0,
            atol=0.0,
        )

    torch.manual_seed(9104)
    node_expected = torch.randn(4, (spec.lmax + 1) ** 2, spec.num_channels, requires_grad=True)
    node_actual_embedding = node_expected.detach().clone().requires_grad_(True)
    radial_expected = torch.randn(6, spec.num_radial_basis, requires_grad=True)
    radial_actual = radial_expected.detach().clone().requires_grad_(True)
    envelope_expected = torch.rand(6, 1, requires_grad=True)
    envelope_actual = envelope_expected.detach().clone().requires_grad_(True)
    edge_vector = torch.tensor(
        [
            [1.0, 0.2, -0.1],
            [0.3, 1.1, 0.4],
            [-0.4, 0.6, 1.2],
            [0.8, -0.5, 0.7],
            [-0.7, -0.2, 0.9],
            [0.2, 0.9, -0.8],
        ],
        dtype=torch.float32,
    )
    source = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    target = torch.tensor([1, 2, 3, 0, 2, 1], dtype=torch.long)
    edge_index = torch.stack((source, target), dim=0)
    atomic_numbers = torch.tensor([1, 6, 8, 14], dtype=torch.long)
    source_species = atomic_numbers.index_select(0, source)
    target_species = atomic_numbers.index_select(0, target)

    rotation_matrix = modules["edge_rot_mat"].init_edge_rot_mat(
        edge_vector,
        use_rotation_mask=not spec.direct_prediction,
    )
    rotation.set_wigner(rotation_matrix)
    expected_output = expected(
        node_expected,
        source_species,
        target_species,
        radial_expected,
        edge_index,
        envelope_expected,
    )
    actual_output = actual(
        {
            "x": embedding_tensor_to_flat(node_actual_embedding, inference.value_types["input:x"].irreps),
            "source_atomic_numbers": source_species,
            "target_atomic_numbers": target_species,
            "edge_radial": radial_actual,
            "edge_vector": edge_vector,
            "edge_envelope": envelope_actual,
            "source_index": _index(source, source.numel()),
            "target_index": _index(target, target.numel()),
            "target_segment": _index(target, atomic_numbers.numel()),
        },
        {},
    )["out"]
    actual_embedding = flat_to_embedding_tensor(actual_output, inference.value_types["projection"].irreps)
    torch.testing.assert_close(actual_embedding, expected_output, rtol=3.0e-5, atol=3.0e-5)

    expected_loss = expected_output.square().sum()
    actual_loss = actual_embedding.square().sum()
    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(node_actual_embedding.grad, node_expected.grad, rtol=5.0e-5, atol=5.0e-5)
    torch.testing.assert_close(radial_actual.grad, radial_expected.grad, rtol=5.0e-5, atol=5.0e-5)
    torch.testing.assert_close(envelope_actual.grad, envelope_expected.grad, rtol=5.0e-5, atol=5.0e-5)
    for actual_name, expected_name in mapping.items():
        torch.testing.assert_close(
            actual_parameters[actual_name].grad,
            expected_parameters[expected_name].grad,
            rtol=7.0e-5,
            atol=7.0e-5,
        )


def test_v3_official_transblock_matches_initialization_forward_and_all_gradients():
    modules = _official_modules()
    spec = _spec()
    registry = core_registry()
    program = equiformer_v3_transblock_program(spec)
    inference = TypeChecker(registry).check(program)

    torch.manual_seed(9301)
    expected, rotation = _official_transblock(spec, modules)
    expected_init_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(9301)
    actual = E3NNGraphBackend(
        registry,
        equiformer_v3_root=str(V3_ROOT),
    ).build(program, inference)
    actual_init_rng = torch.random.get_rng_state().clone()
    assert torch.equal(actual_init_rng, expected_init_rng)

    expected_parameters = dict(expected.named_parameters())
    actual_parameters = dict(actual.named_parameters())
    mapping = _transblock_parameter_mapping(spec)
    assert len(program.nodes) == 61
    assert set(mapping) == set(actual_parameters)
    assert set(mapping.values()) == set(expected_parameters)
    for actual_name, expected_name in mapping.items():
        torch.testing.assert_close(
            actual_parameters[actual_name],
            expected_parameters[expected_name],
            rtol=0.0,
            atol=0.0,
        )

    torch.manual_seed(9302)
    node_expected = torch.randn(
        4,
        (spec.lmax + 1) ** 2,
        spec.num_channels,
        requires_grad=True,
    )
    node_actual_embedding = node_expected.detach().clone().requires_grad_(True)
    radial_expected = torch.randn(6, spec.num_radial_basis, requires_grad=True)
    radial_actual = radial_expected.detach().clone().requires_grad_(True)
    envelope_expected = torch.rand(6, 1, requires_grad=True)
    envelope_actual = envelope_expected.detach().clone().requires_grad_(True)
    edge_vector = torch.tensor(
        [
            [1.0, 0.2, -0.1],
            [0.3, 1.1, 0.4],
            [-0.4, 0.6, 1.2],
            [0.8, -0.5, 0.7],
            [-0.7, -0.2, 0.9],
            [0.2, 0.9, -0.8],
        ],
        dtype=torch.float32,
    )
    source = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    target = torch.tensor([1, 2, 3, 0, 2, 1], dtype=torch.long)
    edge_index = torch.stack((source, target), dim=0)
    atomic_numbers = torch.tensor([1, 6, 8, 14], dtype=torch.long)
    batch = torch.zeros(atomic_numbers.numel(), dtype=torch.long)

    rotation_matrix = modules["edge_rot_mat"].init_edge_rot_mat(
        edge_vector,
        use_rotation_mask=not spec.direct_prediction,
    )
    rotation.set_wigner(rotation_matrix)
    expected_output = expected(
        node_expected,
        atomic_numbers.index_select(0, source),
        atomic_numbers.index_select(0, target),
        radial_expected,
        edge_index,
        envelope_expected,
        batch,
    )
    actual_output = actual(
        {
            "x": embedding_tensor_to_flat(
                node_actual_embedding,
                inference.value_types["input:x"].irreps,
            ),
            "source_atomic_numbers": atomic_numbers.index_select(0, source),
            "target_atomic_numbers": atomic_numbers.index_select(0, target),
            "edge_radial": radial_actual,
            "edge_vector": edge_vector,
            "edge_envelope": envelope_actual,
            "source_index": _index(source, source.numel()),
            "target_index": _index(target, target.numel()),
            "target_segment": _index(target, atomic_numbers.numel()),
            "batch": _index(batch, 1),
        },
        {},
    )["out"]
    actual_embedding = flat_to_embedding_tensor(
        actual_output,
        inference.value_types["ffn_residual"].irreps,
    )
    torch.testing.assert_close(actual_embedding, expected_output, rtol=2.0e-6, atol=2.0e-7)

    expected_output.square().sum().backward()
    actual_embedding.square().sum().backward()
    torch.testing.assert_close(
        node_actual_embedding.grad,
        node_expected.grad,
        rtol=3.0e-5,
        atol=3.0e-7,
    )
    torch.testing.assert_close(radial_actual.grad, radial_expected.grad, rtol=3.0e-5, atol=3.0e-7)
    torch.testing.assert_close(
        envelope_actual.grad,
        envelope_expected.grad,
        rtol=3.0e-5,
        atol=3.0e-7,
    )
    for actual_name, expected_name in mapping.items():
        torch.testing.assert_close(
            actual_parameters[actual_name].grad,
            expected_parameters[expected_name].grad,
            rtol=5.0e-5,
            atol=5.0e-7,
        )


def test_v3_official_direct_force_head_matches_parameters_forward_and_all_gradients():
    modules = _official_modules()
    spec = _spec()
    registry = core_registry()
    program = equiformer_v3_force_head_program(spec)
    inference = TypeChecker(registry).check(program)

    torch.manual_seed(9401)
    expected, rotation = _official_force_head(spec, modules)
    expected_init_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(9401)
    actual = E3NNGraphBackend(
        registry,
        equiformer_v3_root=str(V3_ROOT),
    ).build(program, inference)
    actual_init_rng = torch.random.get_rng_state().clone()
    assert torch.equal(actual_init_rng, expected_init_rng)

    expected_parameters = dict(expected.named_parameters())
    actual_parameters = dict(actual.named_parameters())
    mapping = dict(program.parameters["parameter_mapping"])
    assert len(program.nodes) == 31
    assert set(mapping) == set(actual_parameters)
    assert set(mapping.values()) == set(expected_parameters)
    for actual_name, expected_name in mapping.items():
        torch.testing.assert_close(
            actual_parameters[actual_name],
            expected_parameters[expected_name],
            rtol=0.0,
            atol=0.0,
        )

    torch.manual_seed(9402)
    node_expected = torch.randn(
        4,
        (spec.lmax + 1) ** 2,
        spec.num_channels,
        requires_grad=True,
    )
    node_actual_embedding = node_expected.detach().clone().requires_grad_(True)
    radial_expected = torch.randn(6, spec.num_radial_basis, requires_grad=True)
    radial_actual = radial_expected.detach().clone().requires_grad_(True)
    envelope_expected = torch.rand(6, 1, requires_grad=True)
    envelope_actual = envelope_expected.detach().clone().requires_grad_(True)
    edge_vector = torch.tensor(
        [
            [1.0, 0.2, -0.1],
            [0.3, 1.1, 0.4],
            [-0.4, 0.6, 1.2],
            [0.8, -0.5, 0.7],
            [-0.7, -0.2, 0.9],
            [0.2, 0.9, -0.8],
        ],
        dtype=torch.float32,
    )
    source = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    target = torch.tensor([1, 2, 3, 0, 2, 1], dtype=torch.long)
    edge_index = torch.stack((source, target), dim=0)
    atomic_numbers = torch.tensor([1, 6, 8, 14], dtype=torch.long)

    rotation.set_wigner(
        modules["edge_rot_mat"].init_edge_rot_mat(
            edge_vector,
            use_rotation_mask=not spec.direct_prediction,
        )
    )
    expected_output = expected(
        node_expected,
        atomic_numbers.index_select(0, source),
        atomic_numbers.index_select(0, target),
        radial_expected,
        edge_index,
        envelope_expected,
    ).narrow(1, 1, 3).reshape(-1, 3)
    actual_output = actual(
        {
            "x": embedding_tensor_to_flat(
                node_actual_embedding,
                inference.value_types["input:x"].irreps,
            ),
            "source_atomic_numbers": atomic_numbers.index_select(0, source),
            "target_atomic_numbers": atomic_numbers.index_select(0, target),
            "edge_radial": radial_actual,
            "edge_vector": edge_vector,
            "edge_envelope": envelope_actual,
            "source_index": _index(source, source.numel()),
            "target_index": _index(target, target.numel()),
            "target_segment": _index(target, atomic_numbers.numel()),
        },
        {},
    )["forces"]
    torch.testing.assert_close(actual_output, expected_output, rtol=2.0e-6, atol=2.0e-8)

    expected_output.square().sum().backward()
    actual_output.square().sum().backward()
    torch.testing.assert_close(
        node_actual_embedding.grad,
        node_expected.grad,
        rtol=3.0e-5,
        atol=3.0e-8,
    )
    torch.testing.assert_close(radial_actual.grad, radial_expected.grad, rtol=3.0e-5, atol=3.0e-8)
    torch.testing.assert_close(
        envelope_actual.grad,
        envelope_expected.grad,
        rtol=3.0e-5,
        atol=3.0e-8,
    )
    for actual_name, expected_name in mapping.items():
        torch.testing.assert_close(
            actual_parameters[actual_name].grad,
            expected_parameters[expected_name].grad,
            rtol=5.0e-5,
            atol=5.0e-8,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("use_add_merge", True, "use_add_merge=True"),
        ("use_rad_l_parametrization", False, "degree-shared radial"),
        ("attn_activation", "sep-merge_s2_swiglu", "sep-merge_gates2_swiglu"),
        ("use_envelope", False, "edge envelope"),
    ],
)
def test_v3_attention_program_rejects_unimplemented_official_branches(field, value, message):
    spec = replace(_spec(), **{field: value})
    with pytest.raises(ValueError, match=message):
        equiformer_v3_attention_program(spec)
