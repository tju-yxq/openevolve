import importlib
import sys
import types
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from equivariant_nas.dsl import (  # noqa: E402
    TypeChecker,
    core_registry,
    equiformer_v3_backbone_program,
    equiformer_v3_direct_model_program,
    equiformer_v3_feed_forward_program,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend  # noqa: E402
from equivariant_nas.dsl.backends.equiformer_v3_spec import EquiformerV3Spec  # noqa: E402
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
        avg_num_nodes=5.5,
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
            raise NotImplementedError("the V3 full-model oracle uses dim=0")
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


def _install_torch_scatter_stub():
    try:
        importlib.import_module("torch_scatter")
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("torch_scatter")

    def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
        if dim != 0:
            raise NotImplementedError("the V3 full-model oracle uses dim=0")
        count = int(dim_size) if dim_size is not None else int(index.max().item()) + 1
        output = src.new_zeros((count,) + src.shape[1:])
        output.index_add_(0, index, src)
        if reduce == "sum":
            return output
        if reduce == "mean":
            counts = src.new_zeros(count)
            counts.index_add_(0, index, src.new_ones(index.shape[0]))
            shape = (count,) + (1,) * (src.dim() - 1)
            return output / counts.clamp_min(1).reshape(shape)
        raise NotImplementedError(reduce)

    module.scatter = scatter
    sys.modules["torch_scatter"] = module


def _install_fairchem_import_stubs():
    try:
        importlib.import_module("fairchem.core.common.registry")
        importlib.import_module("fairchem.core.common.utils")
        importlib.import_module("fairchem.core.models.base")
        return
    except ModuleNotFoundError:
        pass

    fairchem = types.ModuleType("fairchem")
    core = types.ModuleType("fairchem.core")
    common = types.ModuleType("fairchem.core.common")
    models = types.ModuleType("fairchem.core.models")
    registry_module = types.ModuleType("fairchem.core.common.registry")
    utils_module = types.ModuleType("fairchem.core.common.utils")
    base_module = types.ModuleType("fairchem.core.models.base")

    class RegistryStub:
        @staticmethod
        def register_model(_name):
            return lambda model_class: model_class

    class GraphModelMixin:
        pass

    def conditional_grad(_context):
        return lambda function: function

    registry_module.registry = RegistryStub()
    utils_module.conditional_grad = conditional_grad
    base_module.GraphModelMixin = GraphModelMixin
    fairchem.core = core
    core.common = common
    core.models = models
    common.registry = registry_module
    common.utils = utils_module
    models.base = base_module
    sys.modules["fairchem"] = fairchem
    sys.modules["fairchem.core"] = core
    sys.modules["fairchem.core.common"] = common
    sys.modules["fairchem.core.common.registry"] = registry_module
    sys.modules["fairchem.core.common.utils"] = utils_module
    sys.modules["fairchem.core.models"] = models
    sys.modules["fairchem.core.models.base"] = base_module


def _official_modules():
    if not equiformer_v3_source_available(str(V3_ROOT)):
        pytest.skip("official Equiformer V3 source is unavailable")
    _install_torch_geometric_oracle_stub()
    _install_torch_scatter_stub()
    _install_fairchem_import_stubs()
    loaded = load_equiformer_v3_modules(str(V3_ROOT))
    package = loaded[0].__package__
    return {
        "model": importlib.import_module(package + ".equiformer_v3"),
        "transformer": importlib.import_module(package + ".transformer_block"),
        "output": importlib.import_module(package + ".output_block"),
    }


def _official_model(spec, modules):
    return modules["model"].EquiformerV3_OC(
        use_pbc=spec.use_pbc,
        use_pbc_single=spec.use_pbc_single,
        otf_graph=spec.otf_graph,
        regress_forces=spec.regress_forces,
        regress_stress=spec.regress_stress,
        direct_prediction=spec.direct_prediction,
        max_neighbors=spec.max_neighbors,
        max_radius=spec.max_radius,
        num_radial_basis=spec.num_radial_basis,
        max_num_elements=spec.max_num_elements,
        num_layers=spec.num_layers,
        num_channels=spec.num_channels,
        attn_hidden_channels=spec.attn_hidden_channels,
        num_heads=spec.num_heads,
        attn_alpha_channels=spec.attn_alpha_channels,
        attn_value_channels=spec.attn_value_channels,
        ffn_hidden_channels=spec.ffn_hidden_channels,
        norm_type=spec.norm_type,
        lmax=spec.lmax,
        mmax=spec.mmax,
        attn_grid_resolution_list=list(spec.attn_grid_resolution),
        ffn_grid_resolution_list=list(spec.ffn_grid_resolution),
        edge_channels=spec.edge_channels,
        use_atom_edge_embedding=spec.use_atom_edge_embedding,
        use_envelope=spec.use_envelope,
        attn_activation=spec.attn_activation,
        use_attn_renorm=spec.use_attn_renorm,
        use_add_merge=spec.use_add_merge,
        use_rad_l_parametrization=spec.use_rad_l_parametrization,
        softcap=spec.softcap,
        attn_eps=spec.attn_eps,
        ffn_activation=spec.ffn_activation,
        use_grid_mlp=spec.use_grid_mlp,
        use_gate_force_head=spec.use_gate_force_head,
        alpha_drop=spec.alpha_drop,
        attn_mask_rate=spec.attn_mask_rate,
        attn_weights_drop=spec.attn_weights_drop,
        value_drop=spec.value_drop,
        drop_path_rate=spec.drop_path_rate,
        proj_drop=spec.proj_drop,
        ffn_drop=spec.ffn_drop,
        gradient_checkpointing_block_list=None,
        avg_num_nodes=spec.avg_num_nodes,
        avg_degree=spec.avg_degree,
        enforce_max_neighbors_strictly=spec.enforce_max_neighbors_strictly,
    )


def _official_forward(model, atomic_numbers, positions, edge_index, batch):
    source = edge_index[0]
    target = edge_index[1]
    edge_vector = positions.index_select(0, source) - positions.index_select(0, target)
    edge_distance = torch.linalg.vector_norm(edge_vector, dim=-1, keepdim=True)
    model.dtype = positions.dtype
    model.device = positions.device
    edge_radial, edge_envelope = model._forward_edge(edge_distance, edge_vector)
    node_features = model._forward_embedding(
        atomic_numbers,
        edge_radial,
        edge_index,
        edge_envelope,
    )
    scalar_features, node_features = model._forward_blocks(
        node_features,
        atomic_numbers.index_select(0, source),
        atomic_numbers.index_select(0, target),
        edge_radial,
        edge_index,
        edge_envelope,
        batch,
    )
    node_energy = model.energy_block(scalar_features)
    graph_count = int(batch.max().item()) + 1
    energy = node_energy.new_zeros(graph_count)
    energy.index_add_(0, batch, node_energy.reshape(-1))
    energy = energy / float(model.avg_num_nodes)
    forces = model.force_block(
        node_features,
        atomic_numbers.index_select(0, source),
        atomic_numbers.index_select(0, target),
        edge_radial,
        edge_index,
        edge_envelope,
    )
    forces = forces.narrow(1, 1, 3).reshape(-1, 3)
    return {"energy": energy, "forces": forces}


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
    for local_name, official_name in equiformer_v3_feed_forward_program(
        spec
    ).parameters["parameter_mapping"].items():
        mapping[local_name.replace("node_modules.", "node_modules.ffn_")] = (
            "ffn." + official_name
        )
    return mapping


def _direct_model_parameter_mapping(spec, program):
    mapping = {}
    backbone = equiformer_v3_backbone_program(spec)
    for local_name, official_name in backbone.parameters["parameter_mapping"][
        "input"
    ].items():
        mapping[local_name.replace("node_modules.", "node_modules.input_")] = (
            official_name
        )
    block_mapping = _transblock_parameter_mapping(spec)
    for index in range(spec.num_layers):
        for local_name, official_name in block_mapping.items():
            mapping[
                local_name.replace(
                    "node_modules.",
                    "node_modules.block{}_".format(index),
                )
            ] = "blocks.{}.{}".format(index, official_name)

    sections = program.parameters["parameter_mapping"]
    for local_name, official_name in sections["shared_final_norm"].items():
        mapping[local_name] = official_name
    for local_name, official_name in sections["energy_head"].items():
        mapping[
            local_name.replace("node_modules.", "node_modules.energy_")
        ] = official_name
    for local_name, official_name in sections["force_head"].items():
        mapping[
            local_name.replace("node_modules.", "node_modules.force_")
        ] = "force_block." + official_name
    return mapping


def _index(indices, target_size):
    return {"indices": indices, "target_size": target_size}


def test_v3_two_layer_direct_model_matches_official_model_end_to_end():
    modules = _official_modules()
    spec = _spec()
    registry = core_registry()
    program = equiformer_v3_direct_model_program(spec)
    inference = TypeChecker(registry).check(program)

    torch.manual_seed(9701)
    expected = _official_model(spec, modules)
    expected_init_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(9701)
    actual = E3NNGraphBackend(
        registry,
        equiformer_v3_root=str(V3_ROOT),
    ).build(program, inference)
    actual_init_rng = torch.random.get_rng_state().clone()
    assert torch.equal(actual_init_rng, expected_init_rng)

    forbidden_types = (
        modules["model"].EquiformerV3_OC,
        modules["transformer"].TransBlockV3,
        modules["transformer"].EquivariantGraphAttention,
        modules["output"].ScalarFeedForwardNetwork,
    )
    assert not any(
        isinstance(module, forbidden_types) for module in actual.modules()
    )

    expected_parameters = dict(expected.named_parameters())
    actual_parameters = dict(actual.named_parameters())
    mapping = _direct_model_parameter_mapping(spec, program)
    assert len(program.nodes) == 186
    assert set(mapping) == set(actual_parameters)
    assert set(mapping.values()) == set(expected_parameters)
    for actual_name, expected_name in mapping.items():
        torch.testing.assert_close(
            actual_parameters[actual_name],
            expected_parameters[expected_name],
            rtol=0.0,
            atol=0.0,
        )

    atomic_numbers = torch.tensor([1, 6, 8, 14, 16], dtype=torch.long)
    positions_expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.2, -0.1],
            [0.3, 1.2, 0.4],
            [-0.4, 0.6, 1.3],
            [0.8, -0.7, 0.5],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    positions_actual = positions_expected.detach().clone().requires_grad_(True)
    source = torch.tensor([0, 1, 2, 3, 4, 3, 4, 2], dtype=torch.long)
    target = torch.tensor([1, 0, 3, 4, 2, 2, 3, 4], dtype=torch.long)
    edge_index = torch.stack((source, target), dim=0)
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

    torch.manual_seed(9702)
    expected_outputs = _official_forward(
        expected,
        atomic_numbers,
        positions_expected,
        edge_index,
        batch,
    )
    expected_forward_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(9702)
    actual_outputs = actual(
        {
            "atomic_numbers": atomic_numbers,
            "positions": positions_actual,
            "source_index": _index(source, source.numel()),
            "target_index": _index(target, target.numel()),
            "target_segment": _index(target, atomic_numbers.numel()),
            "batch": _index(batch, 2),
        },
        {},
    )
    actual_forward_rng = torch.random.get_rng_state().clone()
    assert tuple(actual_outputs["energy"].shape) == (2,)
    assert tuple(actual_outputs["forces"].shape) == (5, 3)
    torch.testing.assert_close(
        actual_outputs["energy"],
        expected_outputs["energy"],
        rtol=3.0e-5,
        atol=3.0e-6,
    )
    torch.testing.assert_close(
        actual_outputs["forces"],
        expected_outputs["forces"],
        rtol=4.0e-5,
        atol=4.0e-7,
    )
    assert torch.equal(actual_forward_rng, expected_forward_rng)

    expected_loss = (
        expected_outputs["energy"].square().sum()
        + expected_outputs["forces"].square().sum()
    )
    actual_loss = (
        actual_outputs["energy"].square().sum()
        + actual_outputs["forces"].square().sum()
    )
    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(
        positions_actual.grad,
        positions_expected.grad,
        rtol=1.0e-4,
        atol=1.0e-6,
    )
    for actual_name, expected_name in mapping.items():
        assert actual_parameters[actual_name].grad is not None
        assert expected_parameters[expected_name].grad is not None
        torch.testing.assert_close(
            actual_parameters[actual_name].grad,
            expected_parameters[expected_name].grad,
            rtol=1.5e-4,
            atol=1.5e-6,
        )
