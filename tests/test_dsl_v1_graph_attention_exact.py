from dataclasses import replace
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
pytest.importorskip("e3nn")
from e3nn import o3

from scripts.audit_equiformer_v1_graph_attention_contract import (
    _load_official_module,
    _segment_reduce,
    _segment_softmax,
)
from equivariant_nas.dsl import (
    ArchitectureProgram,
    AxisSpec,
    Carrier,
    Compiler,
    EquivariantTensorType,
    FeatureRole,
    GroupSpec,
    IndexMapType,
    InputPort,
    InvariantTensorType,
    Irreps,
    Node,
    OutputPort,
    core_registry,
    reference_motif_registry,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.reference_programs import (
    EQUIFORMER_V1_GRAPH_ATTENTION_PARAMETER_MAPPING,
    equiformer_v1_graph_attention_program,
)


def _tp_attrs():
    blocks = (
        (4, "0e"), (2, "0e"), (1, "0e"),
        (4, "1e"), (2, "1e"), (2, "1e"), (2, "1e"), (1, "1e"), (1, "1e"),
        (4, "2e"), (2, "2e"), (2, "2e"), (1, "2e"), (1, "2e"), (1, "2e"),
    )
    indices = (
        (0, 0, 0), (0, 1, 3), (0, 2, 9),
        (1, 0, 4), (1, 1, 1), (1, 1, 5), (1, 1, 10), (1, 2, 6), (1, 2, 11),
        (2, 0, 12), (2, 1, 7), (2, 1, 13), (2, 2, 2), (2, 2, 8), (2, 2, 14),
    )
    return {
        "path_blocks": [{"multiplicity": multiplicity, "irrep": irrep} for multiplicity, irrep in blocks],
        "instructions": [
            {"left": left, "right": right, "out": output, "mode": "uvu", "has_weight": True, "path_weight": 1.0}
            for left, right, output in indices
        ],
    }


def _types(node_count=5, edge_count=8):
    group = GroupSpec.o3()
    node = EquivariantTensorType(
        group, Carrier.NODE, Irreps.parse("4x0e+2x1e+1x2e", group.family), dtype="float64"
    )
    edge_attr = EquivariantTensorType(
        group, Carrier.EDGE, Irreps.parse("1x0e+1x1e+1x2e", group.family), dtype="float64"
    )
    radial = InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("6x0e", group.family),
        axes=("radial_channel",),
        axis_specs=(AxisSpec("radial_channel", 6, FeatureRole.CHANNEL, "independent", 0),),
        dtype="float64",
        feature_role=FeatureRole.CHANNEL,
    )
    source = IndexMapType(group, Carrier.NODE, Carrier.EDGE, "source", target_size=edge_count)
    target = IndexMapType(group, Carrier.NODE, Carrier.EDGE, "target", target_size=edge_count)
    segment = IndexMapType(group, Carrier.EDGE, Carrier.NODE, "segment", target_size=node_count)
    return node, edge_attr, radial, source, target, segment


def _legacy_program_shape_fixture(scales=None):
    node, edge_attr, radial, source, target, segment = _types()
    edge_node = replace(node, carrier=Carrier.EDGE)
    radial_weight = replace(
        radial,
        irreps=Irreps.parse("30x0e", radial.group.family),
        axes=("tp_path",),
        axis_specs=(AxisSpec("tp_path", 30, FeatureRole.TP_PATH, "independent", 0),),
        feature_role=FeatureRole.RADIAL_WEIGHT,
    )
    tp_type = replace(edge_node, irreps=Irreps.parse("7x0e+12x1e+11x2e", node.group.family))
    post_type = replace(edge_node, irreps=Irreps.parse("8x0e+2x1e+2x2e", node.group.family))
    head_type = replace(
        post_type,
        irreps=Irreps.parse("4x0e+1x1e+1x2e", node.group.family),
        axes=("head",),
        axis_specs=(AxisSpec("head", 2, FeatureRole.HEAD, "independent", 0),),
    )
    alpha_channel_type = replace(head_type, irreps=Irreps.parse("2x0e", node.group.family))
    value_type = replace(head_type, irreps=Irreps.parse("2x0e+1x1e+1x2e", node.group.family))
    alpha_type = InvariantTensorType(
        node.group,
        Carrier.EDGE,
        Irreps.parse("2x0e", node.group.family),
        axes=("head",),
        axis_specs=(AxisSpec("head", 2, FeatureRole.HEAD, "independent", 0),),
        dtype="float64",
        feature_role=FeatureRole.ALPHA,
    )
    aggregate_type = replace(value_type, carrier=Carrier.NODE)
    merged_type = replace(node, irreps=Irreps.parse("4x0e+2x1e+2x2e", node.group.family))
    return ArchitectureProgram(
        "2.9.0",
        "official-v1-complete-graph-attention",
        (
            InputPort("node_input", node),
            InputPort("edge_attr", edge_attr),
            InputPort("edge_scalars", radial),
            InputPort("source_index", source),
            InputPort("target_index", target),
            InputPort("segment_index", segment),
        ),
        (
            Node(
                "merge_src",
                "core.irrep_linear@2",
                {"x": ("input:node_input",)},
                {"out_irreps": "4x0e+2x1e+1x2e", "bias": True, "rescale": True},
            ),
            Node(
                "merge_dst",
                "core.irrep_linear@2",
                {"x": ("input:node_input",)},
                {"out_irreps": "4x0e+2x1e+1x2e", "bias": False, "rescale": True},
            ),
            Node("source", "core.endpoint_gather@1", {"x": ("merge_src",), "index": ("input:source_index",)}),
            Node("target", "core.endpoint_gather@1", {"x": ("merge_dst",), "index": ("input:target_index",)}),
            Node("message", "core.residual_add@1", {"left": ("source",), "right": ("target",)}),
            Node(
                "radial",
                "motif.v1_radial_profile@1",
                {"x": ("input:edge_scalars",)},
                {
                    "axis": "radial_channel",
                    "out_axis": "tp_path",
                    "hidden_features": 8,
                    "out_features": 30,
                    "output_scales": list(scales or [1.0] * 30),
                },
            ),
            Node(
                "tp",
                "core.tensor_product@3",
                {"left": ("message",), "right": ("input:edge_attr",), "weight": ("radial",)},
                _tp_attrs(),
            ),
            Node(
                "post_tp",
                "core.irrep_linear@2",
                {"x": ("tp",)},
                {"out_irreps": "8x0e+2x1e+2x2e", "bias": True, "rescale": True},
            ),
            Node("heads", "core.head_split@2", {"x": ("post_tp",)}, {"head_axis": "head", "num_heads": 2}),
            Node(
                "alpha_channels",
                "core.irrep_select@2",
                {"x": ("heads",)},
                {"selections": [{"irrep": "0e", "start": 0, "multiplicity": 2}]},
            ),
            Node(
                "value",
                "core.irrep_select@2",
                {"x": ("heads",)},
                {
                    "selections": [
                        {"irrep": "0e", "start": 2, "multiplicity": 2},
                        {"irrep": "1e", "start": 0, "multiplicity": 1},
                        {"irrep": "2e", "start": 0, "multiplicity": 1},
                    ]
                },
            ),
            Node(
                "alpha_activated",
                "core.scalar_activation@1",
                {"x": ("alpha_channels",)},
                {
                    "activation": "smooth_leaky_relu",
                    "negative_slope": 0.2,
                    "normalization": "second_moment",
                },
            ),
            Node(
                "logits",
                "core.headwise_scalar_contraction@2",
                {"x": ("alpha_activated",)},
                {"head_axis": "head", "bias": False},
            ),
            Node("softmax", "core.segment_softmax@2", {"logits": ("logits",), "index": ("input:segment_index",)}, {}),
            Node("weighted", "core.invariant_scale@1", {"weight": ("softmax",), "value": ("value",)}, {}),
            Node(
                "aggregate",
                "core.segment_reduce@1",
                {"x": ("weighted",), "index": ("input:segment_index",)},
                {"reduce": "sum", "normalization": "none"},
            ),
            Node("merged", "core.head_merge@2", {"x": ("aggregate",)}, {"head_axis": "head"}),
            Node(
                "projection",
                "core.irrep_linear@2",
                {"x": ("merged",)},
                {"out_irreps": "4x0e+2x1e+1x2e", "bias": True, "rescale": True},
            ),
        ),
        (
            OutputPort("merge_src", "merge_src", node),
            OutputPort("merge_dst", "merge_dst", node),
            OutputPort("message", "message", edge_node),
            OutputPort("radial", "radial", radial_weight),
            OutputPort("tp", "tp", tp_type),
            OutputPort("post_tp", "post_tp", post_type),
            OutputPort("heads", "heads", head_type),
            OutputPort("alpha_channels", "alpha_channels", alpha_channel_type),
            OutputPort("value", "value", value_type),
            OutputPort("alpha_activated", "alpha_activated", alpha_channel_type),
            OutputPort("logits", "logits", alpha_type),
            OutputPort("softmax", "softmax", alpha_type),
            OutputPort("weighted", "weighted", value_type),
            OutputPort("aggregate", "aggregate", aggregate_type),
            OutputPort("merged", "merged", merged_type),
            OutputPort("out", "projection", node),
        ),
        program_id="official-v1-complete-graph-attention@1",
    )


def _program(scales=None):
    return equiformer_v1_graph_attention_program(scales or [1.0] * 30)


def _parameter_mapping():
    return dict(EQUIFORMER_V1_GRAPH_ATTENTION_PARAMETER_MAPPING)


def _official_graph_attention():
    root = Path(__file__).resolve().parents[2] / "equiformer"
    module, _source = _load_official_module(root, torch)
    config = {
        "irreps_node_input": o3.Irreps("4x0e+2x1e+1x2e"),
        "irreps_node_attr": o3.Irreps("1x0e"),
        "irreps_edge_attr": o3.Irreps("1x0e+1x1e+1x2e"),
        "irreps_node_output": o3.Irreps("4x0e+2x1e+1x2e"),
        "fc_neurons": [6, 8],
        "irreps_head": o3.Irreps("2x0e+1x1e+1x2e"),
        "num_heads": 2,
        "irreps_pre_attn": None,
        "rescale_degree": False,
        "nonlinear_message": False,
        "alpha_drop": 0.0,
        "proj_drop": 0.0,
    }
    return module.GraphAttention(**config).double().eval()


def _official_forward_intermediates(model, node_input, edge_src, edge_dst, edge_attr, edge_scalars):
    merge_src = model.merge_src(node_input)
    merge_dst = model.merge_dst(node_input)
    message = merge_src.index_select(0, edge_src) + merge_dst.index_select(0, edge_dst)
    radial = model.sep.dtp_rad(edge_scalars)
    tp = model.sep.dtp(message, edge_attr, radial)
    post_tp = model.sep.lin(tp)
    heads = model.vec2heads(post_tp)
    alpha_channels = heads.narrow(2, 0, model.mul_alpha_head)
    value = heads.narrow(2, model.mul_alpha_head, heads.shape[-1] - model.mul_alpha_head)
    alpha_activated = model.alpha_act(alpha_channels)
    logits = torch.einsum("bik,aik->bi", alpha_activated, model.alpha_dot)
    softmax = _segment_softmax(logits, edge_dst, torch)
    weighted = value * softmax.unsqueeze(-1)
    aggregate = _segment_reduce(weighted, edge_dst, node_input.shape[0], torch, reduce="sum")
    merged = model.heads2vec(aggregate)
    out = model.proj(merged)
    return {
        "merge_src": merge_src,
        "merge_dst": merge_dst,
        "message": message,
        "radial": radial,
        "tp": tp,
        "post_tp": post_tp,
        "heads": heads,
        "alpha_channels": alpha_channels,
        "value": value,
        "alpha_activated": alpha_activated,
        "logits": logits,
        "softmax": softmax,
        "weighted": weighted,
        "aggregate": aggregate,
        "merged": merged,
        "out": out,
    }


def test_complete_v1_graph_attention_matches_official_14_parameters_forward_and_all_gradients():
    torch.manual_seed(20260731)
    official = _official_graph_attention()
    scales = torch.ones(30, dtype=torch.float64)
    for output_slice, scale in official.sep.dtp.slices_sqrt_k.values():
        scales[output_slice] *= float(scale)

    registry = core_registry()
    artifact = Compiler(registry, reference_motif_registry()).analyze(_program(scales.tolist()))
    model = E3NNGraphBackend(registry).build(artifact.expanded_program, artifact.inference).double().eval()
    mapping = _parameter_mapping()
    official_parameters = dict(official.named_parameters())
    dsl_parameters = dict(model.named_parameters())
    assert len(mapping) == 14
    assert set(dsl_parameters) == set(mapping)
    for dsl_name, official_name in mapping.items():
        assert tuple(dsl_parameters[dsl_name].shape) == tuple(official_parameters[official_name].shape)
        dsl_parameters[dsl_name].data.copy_(official_parameters[official_name].data)

    node_count = 5
    edge_src = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4], dtype=torch.long)
    edge_dst = torch.tensor([1, 1, 3, 3, 0, 4, 4, 2], dtype=torch.long)
    node_input = torch.randn(node_count, 15, dtype=torch.float64, requires_grad=True)
    edge_attr = torch.randn(edge_src.numel(), 9, dtype=torch.float64, requires_grad=True)
    edge_scalars = torch.randn(edge_src.numel(), 6, dtype=torch.float64, requires_grad=True)
    official_node = node_input.detach().clone().requires_grad_(True)
    official_edge = edge_attr.detach().clone().requires_grad_(True)
    official_radial = edge_scalars.detach().clone().requires_grad_(True)

    actual = model(
        {
            "node_input": node_input,
            "edge_attr": edge_attr,
            "edge_scalars": edge_scalars,
            "source_index": {"indices": edge_src, "target_size": edge_src.numel()},
            "target_index": {"indices": edge_dst, "target_size": edge_dst.numel()},
            "segment_index": {"indices": edge_dst, "target_size": node_count},
        },
        {},
    )
    expected = _official_forward_intermediates(
        official,
        official_node,
        edge_src,
        edge_dst,
        official_edge,
        official_radial,
    )
    for name in expected:
        assert actual[name].shape == expected[name].shape, name
        assert torch.allclose(actual[name], expected[name], atol=1.0e-12, rtol=1.0e-12), (
            name,
            float((actual[name] - expected[name]).detach().abs().max()),
        )

    with torch.no_grad():
        node_irreps = o3.Irreps("4x0e+2x1e+1x2e")
        edge_irreps = o3.Irreps("1x0e+1x1e+1x2e")
        for matrix in (o3.rand_matrix(dtype=torch.float64), -torch.eye(3, dtype=torch.float64)):
            node_action = node_irreps.D_from_matrix(matrix)
            edge_action = edge_irreps.D_from_matrix(matrix)
            tp_action = o3.Irreps("7x0e+12x1e+11x2e").D_from_matrix(matrix)
            post_action = o3.Irreps("8x0e+2x1e+2x2e").D_from_matrix(matrix)
            head_action = o3.Irreps("4x0e+1x1e+1x2e").D_from_matrix(matrix)
            value_action = o3.Irreps("2x0e+1x1e+1x2e").D_from_matrix(matrix)
            merged_action = o3.Irreps("4x0e+2x1e+2x2e").D_from_matrix(matrix)
            transformed = model(
                {
                    "node_input": node_input.detach() @ node_action.transpose(0, 1),
                    "edge_attr": edge_attr.detach() @ edge_action.transpose(0, 1),
                    "edge_scalars": edge_scalars.detach(),
                    "source_index": {"indices": edge_src, "target_size": edge_src.numel()},
                    "target_index": {"indices": edge_dst, "target_size": edge_dst.numel()},
                    "segment_index": {"indices": edge_dst, "target_size": node_count},
                },
                {},
            )
            transformed_expectations = {
                "merge_src": actual["merge_src"] @ node_action.transpose(0, 1),
                "merge_dst": actual["merge_dst"] @ node_action.transpose(0, 1),
                "message": actual["message"] @ node_action.transpose(0, 1),
                "radial": actual["radial"],
                "tp": actual["tp"] @ tp_action.transpose(0, 1),
                "post_tp": actual["post_tp"] @ post_action.transpose(0, 1),
                "heads": actual["heads"] @ head_action.transpose(0, 1),
                "alpha_channels": actual["alpha_channels"],
                "value": actual["value"] @ value_action.transpose(0, 1),
                "alpha_activated": actual["alpha_activated"],
                "logits": actual["logits"],
                "softmax": actual["softmax"],
                "weighted": actual["weighted"] @ value_action.transpose(0, 1),
                "aggregate": actual["aggregate"] @ value_action.transpose(0, 1),
                "merged": actual["merged"] @ merged_action.transpose(0, 1),
                "out": actual["out"] @ node_action.transpose(0, 1),
            }
            for name, transformed_expected in transformed_expectations.items():
                relative_error = (
                    (transformed[name] - transformed_expected).norm()
                    / transformed_expected.norm().clamp_min(1.0e-12)
                )
                assert float(relative_error) < 1.0e-7, (name, float(relative_error))

        permutation = torch.tensor([5, 0, 7, 3, 1, 6, 2, 4], dtype=torch.long)
        permuted_src = edge_src.index_select(0, permutation)
        permuted_dst = edge_dst.index_select(0, permutation)
        permuted = model(
            {
                "node_input": node_input.detach(),
                "edge_attr": edge_attr.detach().index_select(0, permutation),
                "edge_scalars": edge_scalars.detach().index_select(0, permutation),
                "source_index": {"indices": permuted_src, "target_size": edge_src.numel()},
                "target_index": {"indices": permuted_dst, "target_size": edge_dst.numel()},
                "segment_index": {"indices": permuted_dst, "target_size": node_count},
            },
            {},
        )
        assert torch.allclose(
            permuted["softmax"],
            actual["softmax"].index_select(0, permutation),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert torch.allclose(permuted["out"], actual["out"], atol=1.0e-12, rtol=1.0e-12)

    probe = torch.randn_like(actual["out"])
    (actual["out"] * probe).sum().backward()
    (expected["out"] * probe).sum().backward()
    for actual_gradient, expected_gradient in (
        (node_input.grad, official_node.grad),
        (edge_attr.grad, official_edge.grad),
        (edge_scalars.grad, official_radial.grad),
    ):
        assert torch.allclose(actual_gradient, expected_gradient, atol=1.0e-11, rtol=1.0e-11)
    for dsl_name, official_name in mapping.items():
        assert torch.allclose(
            dsl_parameters[dsl_name].grad,
            official_parameters[official_name].grad,
            atol=1.0e-11,
            rtol=1.0e-11,
        )
