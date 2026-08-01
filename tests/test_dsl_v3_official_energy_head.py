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
    equiformer_v3_energy_head_program,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend  # noqa: E402
from equivariant_nas.dsl.backends.equiformer_v3_spec import EquiformerV3Spec  # noqa: E402
from equivariant_nas.dsl.backends.v2_runtime import embedding_tensor_to_flat  # noqa: E402
from equivariant_nas.dsl.backends.v3_runtime import (  # noqa: E402
    equiformer_v3_source_available,
    load_equiformer_v3_modules,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = REPOSITORY_ROOT.parent / "equiformer_v3_official"


def _spec():
    return EquiformerV3Spec(
        use_pbc=False,
        regress_forces=True,
        regress_stress=False,
        direct_prediction=True,
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
        edge_channels=5,
        norm_type="merge_layer_norm",
        avg_num_nodes=5.5,
    )


def _install_torch_scatter_stub():
    try:
        importlib.import_module("torch_scatter")
        return
    except ModuleNotFoundError:
        pass
    module = types.ModuleType("torch_scatter")

    def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
        if dim != 0 or reduce != "sum":
            raise NotImplementedError
        count = int(dim_size) if dim_size is not None else int(index.max().item()) + 1
        output = src.new_zeros((count,) + src.shape[1:])
        output.index_add_(0, index, src)
        return output

    module.scatter = scatter
    sys.modules["torch_scatter"] = module


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
            raise NotImplementedError("the V3 energy-head oracle uses dim=0")
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
    _install_torch_scatter_stub()
    _install_torch_geometric_oracle_stub()
    loaded = load_equiformer_v3_modules(str(V3_ROOT))
    package = loaded[0].__package__
    return {
        "so3": loaded[0],
        "radial": importlib.import_module(package + ".radial_function"),
        "layer_norm": importlib.import_module(package + ".layer_norm"),
        "output_block": importlib.import_module(package + ".output_block"),
    }


class _OfficialEnergyHead(torch.nn.Module):
    def __init__(self, spec, modules):
        super().__init__()
        self.spec = spec
        self.modules_by_name = modules
        self.norm = modules["layer_norm"].get_normalization_layer(
            spec.norm_type,
            lmax=spec.lmax,
            num_channels=spec.num_channels,
        )
        self.energy_block = modules["output_block"].ScalarFeedForwardNetwork(
            num_in_channels=spec.num_channels,
            num_hidden_channels=spec.ffn_hidden_channels,
            num_out_channels=1,
            dropout=0.0,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, self.modules_by_name["so3"].SO3Linear)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, self.modules_by_name["radial"].RadialFunction):
            raise AssertionError("the scalar energy head must not contain a RadialFunction")

    def forward(self, node_features, batch):
        normalized = self.norm(node_features)
        scalar = normalized.narrow(1, 0, 1).reshape(
            normalized.shape[0],
            self.spec.num_channels,
        )
        node_energy = self.energy_block(scalar)
        graph_count = int(batch.max().item()) + 1
        energy = node_energy.new_zeros((graph_count, 1))
        energy.index_add_(0, batch, node_energy)
        return energy / float(self.spec.avg_num_nodes)


def _index(indices, target_size):
    return {"indices": indices, "target_size": target_size}


def test_v3_energy_head_matches_official_initialization_forward_and_gradients():
    modules = _official_modules()
    spec = _spec()
    registry = core_registry()
    program = equiformer_v3_energy_head_program(spec)
    inference = TypeChecker(registry).check(program)

    torch.manual_seed(9601)
    expected = _OfficialEnergyHead(spec, modules)
    expected_init_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(9601)
    actual = E3NNGraphBackend(
        registry,
        equiformer_v3_root=str(V3_ROOT),
    ).build(program, inference)
    actual_init_rng = torch.random.get_rng_state().clone()
    assert torch.equal(actual_init_rng, expected_init_rng)

    expected_parameters = dict(expected.named_parameters())
    actual_parameters = dict(actual.named_parameters())
    mapping = dict(program.parameters["parameter_mapping"])
    assert set(mapping) == set(actual_parameters)
    assert set(mapping.values()) == set(expected_parameters)
    for actual_name, expected_name in mapping.items():
        torch.testing.assert_close(
            actual_parameters[actual_name],
            expected_parameters[expected_name],
            rtol=0.0,
            atol=0.0,
        )

    torch.manual_seed(9602)
    node_expected = torch.randn(
        5,
        (spec.lmax + 1) ** 2,
        spec.num_channels,
        requires_grad=True,
    )
    node_actual_embedding = node_expected.detach().clone().requires_grad_(True)
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

    torch.manual_seed(9603)
    expected_energy = expected(node_expected, batch)
    expected_forward_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(9603)
    actual_output = actual(
        {
            "x": embedding_tensor_to_flat(
                node_actual_embedding,
                inference.value_types["input:x"].irreps,
            ),
            "batch": _index(batch, 2),
        },
        {},
    )
    actual_forward_rng = torch.random.get_rng_state().clone()
    actual_energy = actual_output["energy"]
    torch.testing.assert_close(actual_energy, expected_energy, rtol=1.0e-6, atol=1.0e-6)
    assert torch.equal(actual_forward_rng, expected_forward_rng)

    expected_energy.square().sum().backward()
    actual_energy.square().sum().backward()
    torch.testing.assert_close(
        node_actual_embedding.grad,
        node_expected.grad,
        rtol=3.0e-6,
        atol=3.0e-6,
    )
    for actual_name, expected_name in mapping.items():
        torch.testing.assert_close(
            actual_parameters[actual_name].grad,
            expected_parameters[expected_name].grad,
            rtol=3.0e-6,
            atol=3.0e-6,
        )
