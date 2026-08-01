import importlib
import sys
import types
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from equivariant_nas.dsl import (  # noqa: E402
    TypeChecker,
    canonicalize,
    core_registry,
    equiformer_v3_feed_forward_program,
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
        edge_channels=5,
        attn_grid_resolution=(8, 4),
        ffn_grid_resolution=(8, 8),
        attn_weights_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
    )


def _install_torch_geometric_oracle_stub():
    try:
        importlib.import_module("torch_geometric")
        return
    except ModuleNotFoundError:
        pass

    package = types.ModuleType("torch_geometric")
    utils = types.ModuleType("torch_geometric.utils")
    nn = types.ModuleType("torch_geometric.nn")
    num_nodes = types.ModuleType("torch_geometric.utils.num_nodes")

    def maybe_num_nodes(index, requested=None):
        if requested is not None:
            return int(requested)
        return int(index.max().item()) + 1 if index.numel() else 0

    def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
        if dim != 0:
            raise NotImplementedError
        count = maybe_num_nodes(index, dim_size)
        if reduce == "sum":
            output = src.new_zeros((count,) + src.shape[1:])
            output.index_add_(0, index, src)
            return output
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
        raise NotImplementedError

    def segment(*_args, **_kwargs):
        raise NotImplementedError

    utils.scatter = scatter
    utils.segment = segment
    num_nodes.maybe_num_nodes = maybe_num_nodes
    package.utils = utils
    package.nn = nn
    sys.modules["torch_geometric"] = package
    sys.modules["torch_geometric.utils"] = utils
    sys.modules["torch_geometric.nn"] = nn
    sys.modules["torch_geometric.utils.num_nodes"] = num_nodes


def _official_modules():
    if not equiformer_v3_source_available(str(V3_ROOT)):
        pytest.skip("official Equiformer V3 operator source is unavailable")
    _install_torch_geometric_oracle_stub()
    modules = load_equiformer_v3_modules(str(V3_ROOT))
    package = modules[0].__package__
    return {
        "so3": modules[0],
        "transformer": importlib.import_module(package + ".transformer_block"),
    }


def _official_ffn(spec, modules):
    model = modules["transformer"].FeedForwardNetwork(
        num_in_channels=spec.num_channels,
        num_hidden_channels=spec.ffn_hidden_channels,
        num_out_channels=spec.num_channels,
        lmax=spec.lmax,
        mmax=spec.lmax,
        grid_resolution_list=list(spec.ffn_grid_resolution),
        activation=spec.ffn_activation,
        use_grid_mlp=spec.use_grid_mlp,
        dropout=spec.ffn_drop,
    )

    def init_weights(module):
        if isinstance(module, (torch.nn.Linear, modules["so3"].SO3Linear)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    model.apply(init_weights)
    return model


def test_v3_official_ffn_is_fully_lowered_and_canonicalization_preserves_parameters():
    _official_modules()
    spec = _spec()
    registry = core_registry()
    program = equiformer_v3_feed_forward_program(spec)
    canonical = canonicalize(program, registry)

    torch.manual_seed(7301)
    original = E3NNGraphBackend(registry, equiformer_v3_root=str(V3_ROOT)).build(
        program,
        TypeChecker(registry).check(program),
    )
    torch.manual_seed(7301)
    normalized = E3NNGraphBackend(registry, equiformer_v3_root=str(V3_ROOT)).build(
        canonical,
        TypeChecker(registry).check(canonical),
    )
    assert len(program.nodes) == 23
    assert all("motif." not in node.op for node in program.nodes)
    assert len(list(original.parameters())) == len(list(normalized.parameters())) == 10
    for original_parameter, normalized_parameter in zip(
        original.parameters(), normalized.parameters()
    ):
        torch.testing.assert_close(
            original_parameter,
            normalized_parameter,
            rtol=0.0,
            atol=0.0,
        )


def test_v3_official_ffn_matches_initialization_forward_and_all_gradients():
    modules = _official_modules()
    spec = _spec()
    registry = core_registry()
    program = equiformer_v3_feed_forward_program(spec)
    inference = TypeChecker(registry).check(program)

    torch.manual_seed(7302)
    expected = _official_ffn(spec, modules)
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(7302)
    actual = E3NNGraphBackend(
        registry,
        equiformer_v3_root=str(V3_ROOT),
    ).build(program, inference)
    actual_rng = torch.random.get_rng_state().clone()
    assert torch.equal(actual_rng, expected_rng)

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

    torch.manual_seed(7303)
    expected_input = torch.randn(
        5,
        (spec.lmax + 1) ** 2,
        spec.num_channels,
        requires_grad=True,
    )
    actual_embedding = expected_input.detach().clone().requires_grad_(True)
    actual_input = embedding_tensor_to_flat(
        actual_embedding,
        inference.value_types["input:x"].irreps,
    )
    expected_output = expected(expected_input)
    actual_output = actual({"x": actual_input}, {})["out"]
    actual_output = flat_to_embedding_tensor(
        actual_output,
        inference.value_types["so3_linear2"].irreps,
    )
    torch.testing.assert_close(actual_output, expected_output, rtol=0.0, atol=0.0)

    expected_output.square().sum().backward()
    actual_output.square().sum().backward()
    torch.testing.assert_close(
        actual_embedding.grad,
        expected_input.grad,
        rtol=2.0e-6,
        atol=2.0e-8,
    )
    for actual_name, expected_name in mapping.items():
        torch.testing.assert_close(
            actual_parameters[actual_name].grad,
            expected_parameters[expected_name].grad,
            rtol=0.0,
            atol=0.0,
        )
