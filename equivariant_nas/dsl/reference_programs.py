"""Import trusted reference architectures into the typed graph representation."""

from __future__ import annotations

from dataclasses import replace
from .backends.equiformer_v1_spec import ArchitectureSpec
from .backends.equiformer_v3_spec import EquiformerV3Spec, V3_REFERENCE_COMMIT
from .ast import ArchitectureProgram, InputPort, Node, OutputPort
from .canonicalize import architecture_id
from .groups import GroupSpec
from .irreps import Irreps
from .motifs import expand_motifs
from .reference_motifs import reference_motif_registry
from .registry import core_registry
from .types import (
    AffinePointType,
    AxisSpec,
    Carrier,
    CategoricalTensorType,
    EquivarianceLevel,
    EquivariantTensorType,
    EquivariantType,
    FeatureRole,
    Frame,
    IndexMapType,
    InvariantTensorType,
    LatticeShiftType,
    LatticeType,
)
from .backends.equiformer_v1_constructor import baseline_constructor_parameters


def _as_so3(text: str) -> str:
    """Drop O(3) parity suffixes when importing the rotation-only V1 model."""

    import re

    return re.sub(r"(\d+)[eo]", r"\1", text)


EQUIFORMER_V1_GRAPH_ATTENTION_PARAMETER_MAPPING = {
    "node_modules.merge_src.tp.weight": "merge_src.tp.weight",
    "node_modules.merge_src.bias.0": "merge_src.bias.0",
    "node_modules.merge_dst.tp.weight": "merge_dst.tp.weight",
    "node_modules.radial__linear_in.linear.weight": "sep.dtp_rad.net.0.weight",
    "node_modules.radial__linear_in.linear.bias": "sep.dtp_rad.net.0.bias",
    "node_modules.radial__norm.layer_norm.weight": "sep.dtp_rad.net.1.weight",
    "node_modules.radial__norm.layer_norm.bias": "sep.dtp_rad.net.1.bias",
    "node_modules.radial__linear_out.linear.weight": "sep.dtp_rad.net.3.weight",
    "node_modules.radial__offset.offset": "sep.dtp_rad.offset",
    "node_modules.post_tp.tp.weight": "sep.lin.tp.weight",
    "node_modules.post_tp.bias.0": "sep.lin.bias.0",
    "node_modules.logits.weight": "alpha_dot",
    "node_modules.projection.tp.weight": "proj.tp.weight",
    "node_modules.projection.bias.0": "proj.bias.0",
}


EQUIFORMER_V1_GRAPH_ATTENTION_NONLINEAR_PARAMETER_MAPPING = {
    "node_modules.merge_src.tp.weight": "merge_src.tp.weight",
    "node_modules.merge_src.bias.0": "merge_src.bias.0",
    "node_modules.merge_dst.tp.weight": "merge_dst.tp.weight",
    "node_modules.radial__linear_in.linear.weight": "sep_act.dtp_rad.net.0.weight",
    "node_modules.radial__linear_in.linear.bias": "sep_act.dtp_rad.net.0.bias",
    "node_modules.radial__norm.layer_norm.weight": "sep_act.dtp_rad.net.1.weight",
    "node_modules.radial__norm.layer_norm.bias": "sep_act.dtp_rad.net.1.bias",
    "node_modules.radial__linear_out.linear.weight": "sep_act.dtp_rad.net.3.weight",
    "node_modules.radial__offset.offset": "sep_act.dtp_rad.offset",
    "node_modules.message_linear.tp.weight": "sep_act.lin.tp.weight",
    "node_modules.message_linear.bias.0": "sep_act.lin.bias.0",
    "node_modules.alpha_projection.tp.weight": "sep_alpha.tp.weight",
    "node_modules.alpha_projection.bias.0": "sep_alpha.bias.0",
    "node_modules.value_tp.tp.weight": "sep_value.dtp.tp.weight",
    "node_modules.value_projection.tp.weight": "sep_value.lin.tp.weight",
    "node_modules.value_projection.bias.0": "sep_value.lin.bias.0",
    "node_modules.logits.weight": "alpha_dot",
    "node_modules.projection.tp.weight": "proj.tp.weight",
    "node_modules.projection.bias.0": "proj.bias.0",
}


def _equiformer_v1_depthwise_tp_attrs():
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
        "path_blocks": [
            {"multiplicity": multiplicity, "irrep": irrep}
            for multiplicity, irrep in blocks
        ],
        "instructions": [
            {
                "left": left,
                "right": right,
                "out": output,
                "mode": "uvu",
                "has_weight": True,
                "path_weight": 1.0,
            }
            for left, right, output in indices
        ],
    }


def equiformer_v1_graph_attention_program(
    output_scales,
    *,
    node_count: int = 5,
    edge_count: int = 8,
    rescale_degree: bool = False,
    nonlinear_message: bool = False,
    alpha_drop: float = 0.0,
    proj_drop: float = 0.0,
) -> ArchitectureProgram:
    """Return the exact frozen V1 GraphAttention graph used by the M5 oracle.

    The returned program is built only from typed motifs and core primitives.
    ``output_scales`` records the official DepthwiseTensorProduct slice-sqrt-k
    initialization contract on the radial profile; it is not a TP runtime
    normalization or an official-constructor escape hatch.
    """

    scales = tuple(float(value) for value in output_scales)
    if len(scales) != 30:
        raise ValueError("V1 GraphAttention requires exactly 30 radial output scales")
    if node_count <= 0 or edge_count < 0:
        raise ValueError("node_count must be positive and edge_count must be non-negative")
    alpha_drop = float(alpha_drop)
    proj_drop = float(proj_drop)
    rescale_degree = bool(rescale_degree)
    nonlinear_message = bool(nonlinear_message)
    if not 0.0 <= alpha_drop < 1.0 or not 0.0 <= proj_drop < 1.0:
        raise ValueError("alpha_drop and proj_drop must satisfy 0 <= p < 1")

    group = GroupSpec.o3()
    node = EquivariantTensorType(
        group,
        Carrier.NODE,
        Irreps.parse("4x0e+2x1e+1x2e", group.family),
        dtype="float64",
    )
    edge_attr = EquivariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("1x0e+1x1e+1x2e", group.family),
        dtype="float64",
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
    edge_node = replace(node, carrier=Carrier.EDGE)
    radial_weight = replace(
        radial,
        irreps=Irreps.parse("30x0e", group.family),
        axes=("tp_path",),
        axis_specs=(AxisSpec("tp_path", 30, FeatureRole.TP_PATH, "independent", 0),),
        feature_role=FeatureRole.RADIAL_WEIGHT,
    )
    tp_type = replace(edge_node, irreps=Irreps.parse("7x0e+12x1e+11x2e", group.family))
    post_type = replace(edge_node, irreps=Irreps.parse("8x0e+2x1e+2x2e", group.family))
    head_type = replace(
        post_type,
        irreps=Irreps.parse("4x0e+1x1e+1x2e", group.family),
        axes=("head",),
        axis_specs=(AxisSpec("head", 2, FeatureRole.HEAD, "independent", 0),),
    )
    alpha_channel_type = replace(head_type, irreps=Irreps.parse("2x0e", group.family))
    value_type = replace(head_type, irreps=Irreps.parse("2x0e+1x1e+1x2e", group.family))
    alpha_type = InvariantTensorType(
        group,
        Carrier.EDGE,
        Irreps.parse("2x0e", group.family),
        axes=("head",),
        axis_specs=(AxisSpec("head", 2, FeatureRole.HEAD, "independent", 0),),
        dtype="float64",
        feature_role=FeatureRole.ALPHA,
    )
    aggregate_type = replace(value_type, carrier=Carrier.NODE)
    merged_type = replace(node, irreps=Irreps.parse("4x0e+2x1e+2x2e", group.family))
    message_linear_type = replace(edge_node, irreps=Irreps.parse("7x0e+2x1e+1x2e", group.family))
    message_scalar_type = replace(edge_node, irreps=Irreps.parse("4x0e", group.family))
    message_gate_type = replace(edge_node, irreps=Irreps.parse("3x0e", group.family))
    message_gated_value_type = replace(edge_node, irreps=Irreps.parse("2x1e+1x2e", group.family))
    alpha_projection_type = replace(edge_node, irreps=Irreps.parse("4x0e", group.family))
    value_projection_type = replace(edge_node, irreps=Irreps.parse("4x0e+2x1e+2x2e", group.family))
    alpha_weight_reference = "alpha_dropout" if alpha_drop > 0.0 else "softmax"
    projection_reference = "projection_dropout" if proj_drop > 0.0 else "projection"
    return ArchitectureProgram(
        "2.9.0",
        "official-v1-complete-graph-attention{}".format("-nonlinear-message" if nonlinear_message else ""),
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
            *(
                (
                    Node(
                        "radial",
                        "motif.v1_radial_profile@1",
                        {"x": ("input:edge_scalars",)},
                        {
                            "axis": "radial_channel",
                            "out_axis": "tp_path",
                            "hidden_features": 8,
                            "out_features": 30,
                            "output_scales": list(scales),
                        },
                    ),
                    Node(
                        "tp",
                        "core.tensor_product@3",
                        {"left": ("message",), "right": ("input:edge_attr",), "weight": ("radial",)},
                        _equiformer_v1_depthwise_tp_attrs(),
                    ),
                    Node(
                        "message_linear",
                        "core.irrep_linear@2",
                        {"x": ("tp",)},
                        {"out_irreps": "7x0e+2x1e+1x2e", "bias": True, "rescale": True},
                    ),
                    Node(
                        "message_scalars",
                        "core.irrep_select@2",
                        {"x": ("message_linear",)},
                        {"selections": [{"irrep": "0e", "start": 0, "multiplicity": 4}]},
                    ),
                    Node(
                        "message_gates",
                        "core.irrep_select@2",
                        {"x": ("message_linear",)},
                        {"selections": [{"irrep": "0e", "start": 4, "multiplicity": 3}]},
                    ),
                    Node(
                        "message_gated_values",
                        "core.irrep_select@2",
                        {"x": ("message_linear",)},
                        {
                            "selections": [
                                {"irrep": "1e", "start": 0, "multiplicity": 2},
                                {"irrep": "2e", "start": 0, "multiplicity": 1},
                            ]
                        },
                    ),
                    Node(
                        "message_activated_scalars",
                        "core.scalar_activation@1",
                        {"x": ("message_scalars",)},
                        {"activation": "silu", "normalization": "second_moment"},
                    ),
                    Node(
                        "message_activated_gates",
                        "core.scalar_activation@1",
                        {"x": ("message_gates",)},
                        {"activation": "sigmoid", "normalization": "second_moment"},
                    ),
                    Node(
                        "message_gated",
                        "core.gate@1",
                        {"gates": ("message_activated_gates",), "value": ("message_gated_values",)},
                    ),
                    Node(
                        "message_gate_output",
                        "core.irrep_concat@1",
                        {"xs": ("message_activated_scalars", "message_gated")},
                    ),
                    Node(
                        "alpha_projection",
                        "core.irrep_linear@2",
                        {"x": ("tp",)},
                        {"out_irreps": "4x0e", "bias": True, "rescale": True},
                    ),
                    Node(
                        "alpha_channels",
                        "core.head_split@2",
                        {"x": ("alpha_projection",)},
                        {"head_axis": "head", "num_heads": 2},
                    ),
                    Node(
                        "value_tp",
                        "core.tensor_product@5",
                        {"left": ("message_gate_output",), "right": ("input:edge_attr",)},
                        {**_equiformer_v1_depthwise_tp_attrs(), "bias": False, "rescale": True},
                    ),
                    Node(
                        "value_projection",
                        "core.irrep_linear@2",
                        {"x": ("value_tp",)},
                        {"out_irreps": "4x0e+2x1e+2x2e", "bias": True, "rescale": True},
                    ),
                    Node(
                        "value",
                        "core.head_split@2",
                        {"x": ("value_projection",)},
                        {"head_axis": "head", "num_heads": 2},
                    ),
                )
                if nonlinear_message
                else (
                    Node(
                        "radial",
                        "motif.v1_radial_profile@1",
                        {"x": ("input:edge_scalars",)},
                        {
                            "axis": "radial_channel",
                            "out_axis": "tp_path",
                            "hidden_features": 8,
                            "out_features": 30,
                            "output_scales": list(scales),
                        },
                    ),
                    Node(
                        "tp",
                        "core.tensor_product@3",
                        {"left": ("message",), "right": ("input:edge_attr",), "weight": ("radial",)},
                        _equiformer_v1_depthwise_tp_attrs(),
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
                )
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
            *(
                (
                    Node(
                        "alpha_dropout",
                        "core.scalar_dropout@1",
                        {"x": ("softmax",)},
                        {"p": alpha_drop},
                    ),
                )
                if alpha_drop > 0.0
                else ()
            ),
            Node(
                "weighted",
                "core.invariant_scale@1",
                {"weight": (alpha_weight_reference,), "value": ("value",)},
                {},
            ),
            Node(
                "aggregate",
                "core.segment_reduce@1",
                {"x": ("weighted",), "index": ("input:segment_index",)},
                {
                    "reduce": "sum",
                    "normalization": "target_cardinality" if rescale_degree else "none",
                },
            ),
            Node("merged", "core.head_merge@2", {"x": ("aggregate",)}, {"head_axis": "head"}),
            Node(
                "projection",
                "core.irrep_linear@2",
                {"x": ("merged",)},
                {"out_irreps": "4x0e+2x1e+1x2e", "bias": True, "rescale": True},
            ),
            *(
                (
                    Node(
                        "projection_dropout",
                        "core.equivariant_dropout@1",
                        {"x": ("projection",)},
                        {"p": proj_drop},
                    ),
                )
                if proj_drop > 0.0
                else ()
            ),
        ),
        (
            OutputPort("merge_src", "merge_src", node),
            OutputPort("merge_dst", "merge_dst", node),
            OutputPort("message", "message", edge_node),
            OutputPort("radial", "radial", radial_weight),
            OutputPort("tp", "tp", tp_type),
            *(
                (
                    OutputPort("message_linear", "message_linear", message_linear_type),
                    OutputPort("message_scalars", "message_scalars", message_scalar_type),
                    OutputPort("message_gates", "message_gates", message_gate_type),
                    OutputPort("message_gated_values", "message_gated_values", message_gated_value_type),
                    OutputPort("message_activated_scalars", "message_activated_scalars", message_scalar_type),
                    OutputPort("message_activated_gates", "message_activated_gates", message_gate_type),
                    OutputPort("message_gated", "message_gated", message_gated_value_type),
                    OutputPort("message_gate_output", "message_gate_output", edge_node),
                    OutputPort("alpha_projection", "alpha_projection", alpha_projection_type),
                    OutputPort("value_tp", "value_tp", tp_type),
                    OutputPort("value_projection", "value_projection", value_projection_type),
                )
                if nonlinear_message
                else (
                    OutputPort("post_tp", "post_tp", post_type),
                    OutputPort("heads", "heads", head_type),
                )
            ),
            OutputPort("alpha_channels", "alpha_channels", alpha_channel_type),
            OutputPort("value", "value", value_type),
            OutputPort("alpha_activated", "alpha_activated", alpha_channel_type),
            OutputPort("logits", "logits", alpha_type),
            OutputPort("softmax", "softmax", alpha_type),
            *(
                (OutputPort("alpha_dropout", "alpha_dropout", alpha_type),)
                if alpha_drop > 0.0
                else ()
            ),
            OutputPort("weighted", "weighted", value_type),
            OutputPort("aggregate", "aggregate", aggregate_type),
            OutputPort("merged", "merged", merged_type),
            *(
                (OutputPort("projection", "projection", node),)
                if proj_drop > 0.0
                else ()
            ),
            OutputPort("out", projection_reference, node),
        ),
        program_id="official-v1-complete-graph-attention{}@1".format("-nonlinear-message" if nonlinear_message else ""),
        annotations={
            "reference_family": "EquiformerV1",
            "reference_scope": "exact GraphAttention nonlinear_message={} rescale_degree={} alpha_drop={} proj_drop={}".format(
                nonlinear_message,
                rescale_degree,
                alpha_drop,
                proj_drop,
            ),
            "official_constructor_used_for_execution": False,
        },
    )


EQUIFORMER_V1_TRANSBLOCK_PARAMETER_MAPPING = {
    "node_modules.norm_1.affine_weight": "norm_1.affine_weight",
    "node_modules.norm_1.affine_bias": "norm_1.affine_bias",
    **{
        "node_modules.ga__{}".format(dsl_name.removeprefix("node_modules.")): "ga.{}".format(official_name)
        for dsl_name, official_name in EQUIFORMER_V1_GRAPH_ATTENTION_PARAMETER_MAPPING.items()
    },
    "node_modules.norm_2.affine_weight": "norm_2.affine_weight",
    "node_modules.norm_2.affine_bias": "norm_2.affine_bias",
    "node_modules.ffn__fctp_1.tp.weight": "ffn.fctp_1.tp.weight",
    "node_modules.ffn__fctp_1.bias.0": "ffn.fctp_1.bias.0",
    "node_modules.ffn__fctp_2.tp.weight": "ffn.fctp_2.tp.weight",
    "node_modules.ffn__fctp_2.bias.0": "ffn.fctp_2.bias.0",
}


EQUIFORMER_V1_TRANSBLOCK_NONLINEAR_PARAMETER_MAPPING = {
    "node_modules.norm_1.affine_weight": "norm_1.affine_weight",
    "node_modules.norm_1.affine_bias": "norm_1.affine_bias",
    **{
        "node_modules.ga__{}".format(dsl_name.removeprefix("node_modules.")): "ga.{}".format(official_name)
        for dsl_name, official_name in EQUIFORMER_V1_GRAPH_ATTENTION_NONLINEAR_PARAMETER_MAPPING.items()
    },
    "node_modules.norm_2.affine_weight": "norm_2.affine_weight",
    "node_modules.norm_2.affine_bias": "norm_2.affine_bias",
    "node_modules.ffn__fctp_1.tp.weight": "ffn.fctp_1.tp.weight",
    "node_modules.ffn__fctp_1.bias.0": "ffn.fctp_1.bias.0",
    "node_modules.ffn__fctp_2.tp.weight": "ffn.fctp_2.tp.weight",
    "node_modules.ffn__fctp_2.bias.0": "ffn.fctp_2.bias.0",
}


EQUIFORMER_V1_TRANSBLOCK_SHORTCUT_PARAMETER_MAPPING = {
    **EQUIFORMER_V1_TRANSBLOCK_PARAMETER_MAPPING,
    "node_modules.ffn_shortcut.tp.weight": "ffn_shortcut.tp.weight",
    "node_modules.ffn_shortcut.bias.0": "ffn_shortcut.bias.0",
}


EQUIFORMER_V1_TRANSBLOCK_NONLINEAR_SHORTCUT_PARAMETER_MAPPING = {
    **EQUIFORMER_V1_TRANSBLOCK_NONLINEAR_PARAMETER_MAPPING,
    "node_modules.ffn_shortcut.tp.weight": "ffn_shortcut.tp.weight",
    "node_modules.ffn_shortcut.bias.0": "ffn_shortcut.bias.0",
}


def equiformer_v1_transblock_program(
    output_scales,
    *,
    node_count: int = 5,
    edge_count: int = 8,
    graph_count: int = 2,
    node_output_irreps: str = "4x0e+2x1e+1x2e",
    rescale_degree: bool = False,
    nonlinear_message: bool = False,
    alpha_drop: float = 0.0,
    proj_drop: float = 0.0,
    drop_path: float = 0.0,
) -> ArchitectureProgram:
    """Return the exact V1 TransBlock path for the supported dropout branches."""

    drop_path = float(drop_path)
    if not 0.0 <= drop_path < 1.0:
        raise ValueError("drop_path must satisfy 0 <= p < 1")
    if drop_path > 0.0 and graph_count <= 0:
        raise ValueError("graph_count must be positive when graph stochastic depth is enabled")

    attention = equiformer_v1_graph_attention_program(
        output_scales,
        node_count=node_count,
        edge_count=edge_count,
        rescale_degree=rescale_degree,
        nonlinear_message=nonlinear_message,
        alpha_drop=alpha_drop,
        proj_drop=proj_drop,
    )
    attention_inputs = {port.name: port.value_type for port in attention.inputs}
    node_type = attention_inputs["node_input"]
    node_output_type = replace(
        node_type,
        irreps=Irreps.parse(str(node_output_irreps), node_type.group.family),
    )
    uses_ffn_shortcut = node_output_type.irreps != node_type.irreps
    node_attr_type = EquivariantTensorType(
        node_type.group,
        Carrier.NODE,
        Irreps.parse("1x0e", node_type.group.family),
        dtype=node_type.dtype,
    )
    input_references = {
        "node_input": "norm_1",
        "edge_attr": "input:edge_attr",
        "edge_scalars": "input:edge_scalars",
        "source_index": "input:source_index",
        "target_index": "input:target_index",
        "segment_index": "input:segment_index",
    }
    batch_type = IndexMapType(
        node_type.group,
        Carrier.NODE,
        Carrier.GRAPH,
        "batch",
        target_size=graph_count,
        allows_empty_targets=False,
    )

    def remap_reference(reference: str) -> str:
        if reference.startswith("input:"):
            port = reference.split(":", 1)[1]
            return input_references[port]
        root, separator, suffix = reference.partition(":")
        mapped = "ga__{}".format(root)
        return mapped + (separator + suffix if separator else "")

    attention_nodes = []
    for node in attention.nodes:
        attention_nodes.append(
            replace(
                node,
                id="ga__{}".format(node.id),
                inputs={
                    port: tuple(remap_reference(reference) for reference in references)
                    for port, references in node.inputs.items()
                },
            )
        )
    attention_output_source = next(port.source for port in attention.outputs if port.name == "out")
    attention_reference = remap_reference(attention_output_source)
    attention_residual_reference = "attention_drop_path" if drop_path > 0.0 else attention_reference
    ffn_reference = "ffn_projection_dropout" if float(proj_drop) > 0.0 else "ffn"
    ffn_residual_reference = "ffn_drop_path" if drop_path > 0.0 else ffn_reference
    residual_reference = "ffn_shortcut" if uses_ffn_shortcut else "attention_residual"
    nodes = [
        Node(
            "norm_1",
            "core.irrep_layer_norm@1",
            {"x": ("input:node_input",)},
            {"epsilon": 1.0e-5, "affine": True, "normalization": "component"},
        ),
        *attention_nodes,
        *(
            (
                Node(
                    "attention_drop_path",
                    "core.graph_stochastic_depth@1",
                    {"x": (attention_reference,), "batch": ("input:batch_index",)},
                    {"p": drop_path},
                ),
            )
            if drop_path > 0.0
            else ()
        ),
        Node(
            "attention_residual",
            "core.residual_add@1",
            {"left": ("input:node_input",), "right": (attention_residual_reference,)},
        ),
        Node(
            "norm_2",
            "core.irrep_layer_norm@1",
            {"x": ("attention_residual",)},
            {"epsilon": 1.0e-5, "affine": True, "normalization": "component"},
        ),
        Node(
            "ffn",
            "motif.v1_feed_forward@1",
            {"x": ("norm_2",), "node_attr": ("input:node_attr",)},
            {
                "pre_gate_irreps": "7x0e+2x1e+1x2e",
                "scalar_multiplicity": 4,
                "gate_multiplicity": 3,
                "gated_selections": [
                    {"irrep": "1e", "start": 0, "multiplicity": 2},
                    {"irrep": "2e", "start": 0, "multiplicity": 1},
                ],
                "out_irreps": str(node_output_irreps),
            },
        ),
        *(
            (
                Node(
                    "ffn_projection_dropout",
                    "core.equivariant_dropout@1",
                    {"x": ("ffn",)},
                    {"p": float(proj_drop)},
                ),
            )
            if float(proj_drop) > 0.0
            else ()
        ),
        *(
            (
                Node(
                    "ffn_shortcut",
                    "core.tensor_product@4",
                    {
                        "left": ("attention_residual",),
                        "right": ("input:node_attr",),
                    },
                    {
                        "out_irreps": str(node_output_irreps),
                        "bias": True,
                        "rescale": True,
                    },
                ),
            )
            if uses_ffn_shortcut
            else ()
        ),
        *(
            (
                Node(
                    "ffn_drop_path",
                    "core.graph_stochastic_depth@1",
                    {"x": (ffn_reference,), "batch": ("input:batch_index",)},
                    {"p": drop_path},
                ),
            )
            if drop_path > 0.0
            else ()
        ),
        Node(
            "out",
            "core.residual_add@1",
            {"left": (residual_reference,), "right": (ffn_residual_reference,)},
        ),
    ]
    stochastic = float(alpha_drop) > 0.0 or float(proj_drop) > 0.0 or drop_path > 0.0
    return ArchitectureProgram(
        "2.12.0",
        "official-v1-{}-transblock".format("stochastic" if stochastic else "deterministic"),
        (
            InputPort("node_input", node_type),
            InputPort("node_attr", node_attr_type),
            *(port for port in attention.inputs if port.name != "node_input"),
            *((InputPort("batch_index", batch_type),) if drop_path > 0.0 else ()),
        ),
        tuple(nodes),
        (
            OutputPort("norm_1", "norm_1", node_type),
            OutputPort("attention", attention_reference, node_type),
            *(
                (OutputPort("attention_drop_path", "attention_drop_path", node_type),)
                if drop_path > 0.0
                else ()
            ),
            OutputPort("attention_residual", "attention_residual", node_type),
            OutputPort("norm_2", "norm_2", node_type),
            *(
                (OutputPort("ffn_pre_dropout", "ffn", node_type),)
                if float(proj_drop) > 0.0
                else ()
            ),
            OutputPort("ffn", ffn_reference, node_output_type),
            *(
                (OutputPort("ffn_shortcut", "ffn_shortcut", node_output_type),)
                if uses_ffn_shortcut
                else ()
            ),
            *(
                (OutputPort("ffn_drop_path", "ffn_drop_path", node_type),)
                if drop_path > 0.0
                else ()
            ),
            OutputPort("out", "out", node_output_type),
        ),
        program_id="official-v1-{}-transblock@1".format("stochastic" if stochastic else "deterministic"),
        annotations={
            "reference_family": "EquiformerV1",
            "reference_scope": "exact TransBlock node_output_irreps={} nonlinear_message={} rescale_degree={} alpha_drop={} proj_drop={} drop_path={}".format(
                str(node_output_irreps),
                bool(nonlinear_message),
                bool(rescale_degree),
                float(alpha_drop),
                float(proj_drop),
                drop_path,
            ),
            "official_constructor_used_for_execution": False,
        },
    )


def import_equiformer_v1(spec: ArchitectureSpec, *, task_contract: str = "qm9_alpha") -> ArchitectureProgram:
    """Create a typed representation-flow program for a certified V1 spec.

    This graph makes the representation and aggregation semantics inspectable.
    Numerical lowering remains locked to the unchanged imported graph because
    the official V1 builder is constructor-based rather than node-by-node.
    """

    spec.validate()
    group = GroupSpec.so3()
    input_type = EquivariantType(group, Carrier.NODE, Irreps.parse("5x0", "SO3"))
    sh_type = EquivariantType(
        group,
        Carrier.EDGE,
        Irreps.parse(_as_so3(spec.representation.spherical_harmonics_irreps()), "SO3"),
    )
    hidden = _as_so3(spec.representation.embedding_irreps())
    hidden_type = EquivariantType(group, Carrier.NODE, Irreps.parse(hidden, "SO3"))
    scalar_type = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0", "SO3"))
    graph_scalar = EquivariantType(group, Carrier.GRAPH, Irreps.parse("1x0", "SO3"))
    nodes = []
    previous = "input:node_features"
    for index in range(spec.macro.num_layers):
        node_id = "block{}".format(index)
        op = "motif.v1_initial_message" if index == 0 else "motif.v1_residual_message"
        nodes.append(
            Node(
                node_id,
                op,
                {"x": (previous,), "edge_sh": ("input:edge_sh",)},
                {"hidden_irreps": hidden},
                declared_types={"out": hidden_type},
                annotations={"stage": index, "reference_family": "EquiformerV1"},
            )
        )
        previous = node_id
    nodes.extend(
        (
            Node("scalar_readout", "core.select_scalars", {"x": (previous,)}, {"multiplicity": 1}, declared_types={"out": scalar_type}),
            Node("graph_pool", "core.global_pool", {"x": ("scalar_readout",)}, declared_types={"out": graph_scalar}),
        )
    )
    program = ArchitectureProgram(
        language_version="1.0.0",
        task_contract=task_contract,
        inputs=(InputPort("node_features", input_type), InputPort("edge_sh", sh_type)),
        nodes=tuple(nodes),
        outputs=(OutputPort("prediction", "graph_pool", graph_scalar),),
        parameters=baseline_constructor_parameters(spec),
        program_id="equiformer_v1_{}".format(spec.architecture_id()),
        annotations={
            "reference_backend": "equiformer_v1",
            "equiformer_v1_spec": spec.to_dict(),
            "representation_scope": "type-and-connectivity; numerical backend remains constructor-locked",
        },
    )
    expanded = expand_motifs(program, reference_motif_registry())
    registry = core_registry()
    lock = architecture_id(expanded, registry)
    annotations = dict(program.annotations)
    annotations["reference_lock_architecture_id"] = lock
    return replace(program, annotations=annotations)


def _dense_so3_irreps(channels: int, lmax: int) -> str:
    return "+".join("{}x{}".format(int(channels), degree) for degree in range(int(lmax) + 1))


EQUIFORMER_V3_INPUT_PARAMETER_MAPPING = {
    "node_modules.atom_embedding.weight": "sphere_embedding.weight",
    "node_modules.source_edge_embedding.weight": "edge_degree_embedding.source_embedding.weight",
    "node_modules.target_edge_embedding.weight": "edge_degree_embedding.target_embedding.weight",
    "node_modules.radial_linear0.linear.weight": "edge_degree_embedding.rad_func.net.0.weight",
    "node_modules.radial_linear0.linear.bias": "edge_degree_embedding.rad_func.net.0.bias",
    "node_modules.radial_norm0.layer_norm.weight": "edge_degree_embedding.rad_func.net.1.weight",
    "node_modules.radial_norm0.layer_norm.bias": "edge_degree_embedding.rad_func.net.1.bias",
    "node_modules.radial_linear1.linear.weight": "edge_degree_embedding.rad_func.net.3.weight",
    "node_modules.radial_linear1.linear.bias": "edge_degree_embedding.rad_func.net.3.bias",
    "node_modules.radial_norm1.layer_norm.weight": "edge_degree_embedding.rad_func.net.4.weight",
    "node_modules.radial_norm1.layer_norm.bias": "edge_degree_embedding.rad_func.net.4.bias",
    "node_modules.radial_linear2.linear.weight": "edge_degree_embedding.rad_func.net.6.weight",
    "node_modules.radial_linear2.linear.bias": "edge_degree_embedding.rad_func.net.6.bias",
}


EQUIFORMER_V3_ATTENTION_PARAMETER_MAPPING = {
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


def equiformer_v3_input_program(
    spec: EquiformerV3Spec,
    *,
    dtype: str = "float32",
    length_measure: str = "angstrom",
    frame_cache_id: str = "v3_input_edge",
    task_contract: str = "equiformer_v3_official_input",
) -> ArchitectureProgram:
    """Import the official V3 input and EdgeDegreeEmbedding path as core nodes.

    Graph construction itself remains an explicit typed boundary.  For PBC,
    ``lattice_shift`` follows the official fairchem source-image convention,
    so the edge vector is exactly ``pos[source] - pos[target] + offset``.
    """

    spec.validate()
    if not isinstance(frame_cache_id, str) or not frame_cache_id:
        raise ValueError("the official V3 input lowering requires a nonempty frame_cache_id")
    if dtype != "float32":
        raise ValueError(
            "the first official V3 input lowering is frozen to float32 because the pinned V3 Wigner implementation does not support full-model float64 conversion"
        )
    group = GroupSpec(
        "SO3",
        3,
        periodicity="lattice" if spec.use_pbc else "none",
    )
    species = CategoricalTensorType(
        group=group,
        carrier=Carrier.NODE,
        vocabulary_size=spec.max_num_elements,
        dtype="int64",
    )
    positions = AffinePointType(group, dtype=dtype, measure=length_measure)
    source_index = IndexMapType(group, Carrier.NODE, Carrier.EDGE, "source")
    target_index = IndexMapType(group, Carrier.NODE, Carrier.EDGE, "target")
    target_segment = IndexMapType(group, Carrier.EDGE, Carrier.NODE, "segment")
    vector_type = EquivariantTensorType(
        group=group,
        carrier=Carrier.EDGE,
        irreps=Irreps.parse("1x1", "SO3"),
        frame=Frame("global"),
        dtype=dtype,
        measure=length_measure,
        level=EquivarianceLevel.CONSTRUCTIVE,
    )
    radial_type = InvariantTensorType(
        group=group,
        carrier=Carrier.EDGE,
        irreps=Irreps.parse("{}x0".format(spec.num_radial_basis), "SO3"),
        frame=Frame("invariant"),
        axes=("radial_basis",),
        dtype=dtype,
        measure="dimensionless",
        level=EquivarianceLevel.CONSTRUCTIVE,
        axis_specs=(
            AxisSpec("radial_basis", spec.num_radial_basis, FeatureRole.BASIS, "independent", 0),
        ),
        feature_role=FeatureRole.BASIS,
    )
    envelope_type = InvariantTensorType(
        group=group,
        carrier=Carrier.EDGE,
        irreps=Irreps.parse("1x0", "SO3"),
        frame=Frame("invariant"),
        dtype=dtype,
        measure="dimensionless",
        level=EquivarianceLevel.CONSTRUCTIVE,
    )
    hidden_irreps_text = _dense_so3_irreps(spec.num_channels, spec.lmax)
    hidden_irreps = Irreps.parse(hidden_irreps_text, "SO3")
    node_embedding_type = EquivariantTensorType(
        group=group,
        carrier=Carrier.NODE,
        irreps=hidden_irreps,
        frame=Frame("global"),
        dtype=dtype,
        measure="dimensionless",
        level=EquivarianceLevel.CONSTRUCTIVE,
    )

    inputs = [
        InputPort("atomic_numbers", species),
        InputPort("positions", positions),
        InputPort("source_index", source_index),
        InputPort("target_index", target_index),
        InputPort("target_segment", target_segment),
    ]
    displacement_inputs = {
        "positions": ("input:positions",),
        "source_index": ("input:source_index",),
        "target_index": ("input:target_index",),
    }
    displacement_op = "core.relative_displacement@3"
    if spec.use_pbc:
        lattice = LatticeType(group, "v3_cell", dtype=dtype, measure=length_measure)
        lattice_shift = LatticeShiftType(group, "v3_cell", convention="source_image")
        inputs.extend((InputPort("lattice", lattice), InputPort("lattice_shift", lattice_shift)))
        displacement_inputs.update(
            {
                "lattice": ("input:lattice",),
                "lattice_shift": ("input:lattice_shift",),
            }
        )
        displacement_op = "core.periodic_displacement@2"

    radial_input_axis = "radial_basis"
    radial_input_ref = "distance_expansion"
    final_radial_ref = "radial_linear2"
    nodes = [
        Node(
            "atom_embedding",
            "core.categorical_embedding@1",
            {"x": ("input:atomic_numbers",)},
            {"embedding_dim": spec.num_channels, "axis": "channel", "dtype": dtype},
        ),
        Node(
            "distance_expansion",
            "core.fixed_gaussian_radial_basis@1",
            {"distance": ("distance",)},
            {
                "start": 0.0,
                "stop": spec.max_radius,
                "num_basis": spec.num_radial_basis,
                "basis_width_scalar": 2.0,
                "axis": "radial_basis",
                "construction_dtype": "float32",
            },
        ),
    ]
    if spec.use_envelope:
        nodes.append(
            Node(
                "edge_envelope",
                "core.cutoff_envelope@2",
                {"x": ("distance",)},
                {"cutoff": spec.max_radius, "order": 5},
            )
        )
        final_radial_ref = "radial_envelope_scale"
    nodes.append(
        Node(
            "edge_spherical_lift",
            "core.axisymmetric_spherical_lift@1",
            {"amplitudes": (final_radial_ref,), "direction": ("displacement",)},
            {
                "out_irreps": hidden_irreps_text,
                "mmax": spec.mmax,
                "frame_cache_id": frame_cache_id,
                "use_rotation_mask": not spec.direct_prediction,
            },
        )
    )
    nodes.extend(
        (
            Node(
                "source_species",
                "core.endpoint_gather@2",
                {"x": ("input:atomic_numbers",), "index": ("input:source_index",)},
            ),
            Node(
                "target_species",
                "core.endpoint_gather@2",
                {"x": ("input:atomic_numbers",), "index": ("input:target_index",)},
            ),
        )
    )
    if spec.use_atom_edge_embedding:
        nodes.extend(
            (
                Node(
                    "source_edge_embedding",
                    "core.categorical_embedding@2",
                    {"x": ("source_species",)},
                    {
                        "embedding_dim": spec.edge_channels,
                        "axis": "edge_channel",
                        "dtype": dtype,
                        "init_min": -0.001,
                        "init_max": 0.001,
                    },
                ),
                Node(
                    "target_edge_embedding",
                    "core.categorical_embedding@2",
                    {"x": ("target_species",)},
                    {
                        "embedding_dim": spec.edge_channels,
                        "axis": "edge_channel",
                        "dtype": dtype,
                        "init_min": -0.001,
                        "init_max": 0.001,
                    },
                ),
                Node(
                    "radial_concat",
                    "core.invariant_concat@1",
                    {
                        "xs": (
                            "distance_expansion",
                            "source_edge_embedding",
                            "target_edge_embedding",
                        )
                    },
                    {"axis": "edge_input", "feature_role": "channel"},
                ),
            )
        )
        radial_input_axis = "edge_input"
        radial_input_ref = "radial_concat"
    nodes.extend(
        (
            Node(
                "radial_linear0",
                "core.scalar_linear@3",
                {"x": (radial_input_ref,)},
                {
                    "axis": radial_input_axis,
                    "out_features": spec.edge_channels,
                    "out_axis": "edge_hidden",
                    "output_axis_role": "channel",
                    "output_feature_role": "channel",
                    "bias": True,
                },
            ),
            Node(
                "radial_norm0",
                "core.scalar_layer_norm@1",
                {"x": ("radial_linear0",)},
                {"axis": "edge_hidden", "epsilon": 1.0e-5, "affine": True, "bias": True},
            ),
            Node("radial_act0", "core.scalar_activation@1", {"x": ("radial_norm0",)}, {"activation": "silu"}),
            Node(
                "radial_linear1",
                "core.scalar_linear@3",
                {"x": ("radial_act0",)},
                {"axis": "edge_hidden", "out_features": spec.edge_channels, "bias": True},
            ),
            Node(
                "radial_norm1",
                "core.scalar_layer_norm@1",
                {"x": ("radial_linear1",)},
                {"axis": "edge_hidden", "epsilon": 1.0e-5, "affine": True, "bias": True},
            ),
            Node("radial_act1", "core.scalar_activation@1", {"x": ("radial_norm1",)}, {"activation": "silu"}),
            Node(
                "radial_linear2",
                "core.scalar_linear@3",
                {"x": ("radial_act1",)},
                {
                    "axis": "edge_hidden",
                    "out_features": (spec.lmax + 1) * spec.num_channels,
                    "out_axis": "degree_channel",
                    "output_axis_role": "coefficient",
                    "output_feature_role": "coefficient",
                    "bias": True,
                },
            ),
        )
    )
    if spec.use_envelope:
        nodes.append(
            Node(
                "radial_envelope_scale",
                "core.invariant_scale@1",
                {"weight": ("edge_envelope",), "value": ("radial_linear2",)},
            )
        )
    nodes.extend(
        (
            Node(
                "edge_degree_reduce",
                "core.segment_reduce@1",
                {"x": ("edge_spherical_lift",), "index": ("input:target_segment",)},
                {"reduce": "sum", "normalization": "none"},
            ),
            Node(
                "edge_degree_rescale",
                "core.constant_scale@1",
                {"x": ("edge_degree_reduce",)},
                {"factor": 1.0 / float(spec.avg_degree)},
            ),
            Node("atom_embedding_flat", "core.flatten_invariant_axes@1", {"x": ("atom_embedding",)}),
            Node(
                "atom_embedding_pad",
                "core.irrep_pad@1",
                {"x": ("atom_embedding_flat",)},
                {"out_irreps": hidden_irreps_text},
            ),
            Node(
                "node_embedding",
                "core.residual_add@2",
                {"left": ("atom_embedding_pad",), "right": ("edge_degree_rescale",)},
            ),
            Node("displacement", displacement_op, displacement_inputs),
            Node("distance", "core.distance@2", {"vector": ("displacement",)}),
        )
    )

    outputs = [
        OutputPort("node_embedding", "node_embedding", node_embedding_type),
        OutputPort("source_atomic_numbers", "source_species", replace(species, carrier=Carrier.EDGE)),
        OutputPort("target_atomic_numbers", "target_species", replace(species, carrier=Carrier.EDGE)),
        OutputPort("edge_radial", "distance_expansion", radial_type),
        OutputPort("edge_vector", "displacement", vector_type),
    ]
    if spec.use_envelope:
        outputs.append(OutputPort("edge_envelope", "edge_envelope", envelope_type))
    parameters = {
        "equiformer_v3_spec": spec.to_dict(),
        "lowering_contract": {
            "module_construction_order": [node.id for node in nodes],
            "initializer_schedule": (
                [
                    {
                        "after_node": "target_edge_embedding",
                        "nodes": ["source_edge_embedding", "target_edge_embedding"],
                    }
                ]
                if spec.use_atom_edge_embedding
                else []
            ),
            "deferred_initialization": "post_all_module_construction_in_declared_order",
        },
        "input_adapter": {
            "edge_vector": "pos[source]-pos[target]+official_cell_offset",
            "pbc_lattice_shift": "official_cell_offsets" if spec.use_pbc else "not_applicable",
        },
    }
    return ArchitectureProgram(
        language_version="2.1.0",
        task_contract=task_contract,
        inputs=tuple(inputs),
        nodes=tuple(nodes),
        outputs=tuple(outputs),
        parameters=parameters,
        program_id="equiformer_v3_input_{}".format(spec.architecture_id()),
        annotations={
            "reference_backend": "equiformer_v3_official_input",
            "official_source_commit": V3_REFERENCE_COMMIT,
            "constructor_bypass": False,
            "numerical_scope": "official input embedding and EdgeDegreeEmbedding expressed by core primitives",
            "parameter_mapping": dict(EQUIFORMER_V3_INPUT_PARAMETER_MAPPING),
        },
    )


def equiformer_v3_attention_program(
    spec: EquiformerV3Spec,
    *,
    dtype: str = "float32",
    length_measure: str = "angstrom",
    frame_id: str = "v3_attention_edge",
    frame_cache_id: str | None = None,
    task_contract: str = "equiformer_v3_official_attention",
) -> ArchitectureProgram:
    """Import one official V3 EquivariantGraphAttention as typed core nodes.

    This first exact path covers the official training configuration used by
    the pinned OC20 model: atom-edge embeddings, concat merge, degree-shared
    radial parametrization, ``sep-merge_gates2_swiglu``, envelope-aware
    GraphSoftmax, and direct V3 SO3Linear projection.  Unsupported constructor
    branches are rejected instead of silently approximated.
    """

    spec.validate()
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("the official V3 attention lowering requires a nonempty frame_id")
    if frame_cache_id is None:
        frame_cache_id = frame_id
    if not isinstance(frame_cache_id, str) or not frame_cache_id:
        raise ValueError("the official V3 attention lowering requires a nonempty frame_cache_id")
    if dtype != "float32":
        raise ValueError("the official V3 attention lowering is currently frozen to float32")
    if not spec.use_atom_edge_embedding:
        raise ValueError("the first official V3 attention lowering requires atom-edge embeddings")
    if spec.use_add_merge:
        raise ValueError("the first official V3 attention lowering does not yet cover use_add_merge=True")
    if not spec.use_rad_l_parametrization:
        raise ValueError("the first official V3 attention lowering requires degree-shared radial parametrization")
    if spec.attn_activation != "sep-merge_gates2_swiglu":
        raise ValueError("the first official V3 attention lowering requires sep-merge_gates2_swiglu")
    if not spec.use_envelope:
        raise ValueError("the first official V3 attention lowering requires the official edge envelope")

    group = GroupSpec(
        "SO3",
        3,
        periodicity="lattice" if spec.use_pbc else "none",
    )
    species_node = CategoricalTensorType(
        group=group,
        carrier=Carrier.NODE,
        vocabulary_size=spec.max_num_elements,
        dtype="int64",
    )
    species_edge = replace(species_node, carrier=Carrier.EDGE)
    source_index = IndexMapType(group, Carrier.NODE, Carrier.EDGE, "source")
    target_index = IndexMapType(group, Carrier.NODE, Carrier.EDGE, "target")
    target_segment = IndexMapType(group, Carrier.EDGE, Carrier.NODE, "segment")
    hidden_irreps_text = _dense_so3_irreps(spec.num_channels, spec.lmax)
    hidden_type = EquivariantTensorType(
        group=group,
        carrier=Carrier.NODE,
        irreps=Irreps.parse(hidden_irreps_text, "SO3"),
        frame=Frame("global"),
        dtype=dtype,
        measure="dimensionless",
        level=EquivarianceLevel.CONSTRUCTIVE,
    )
    radial_type = InvariantTensorType(
        group=group,
        carrier=Carrier.EDGE,
        irreps=Irreps.parse("{}x0".format(spec.num_radial_basis), "SO3"),
        frame=Frame("invariant"),
        axes=("radial_basis",),
        dtype=dtype,
        measure="dimensionless",
        level=EquivarianceLevel.CONSTRUCTIVE,
        axis_specs=(AxisSpec("radial_basis", spec.num_radial_basis, FeatureRole.BASIS, "independent", 0),),
        feature_role=FeatureRole.BASIS,
    )
    vector_type = EquivariantTensorType(
        group=group,
        carrier=Carrier.EDGE,
        irreps=Irreps.parse("1x1", "SO3"),
        frame=Frame("global"),
        dtype=dtype,
        measure=length_measure,
        level=EquivarianceLevel.CONSTRUCTIVE,
    )
    envelope_type = InvariantTensorType(
        group=group,
        carrier=Carrier.EDGE,
        irreps=Irreps.parse("1x0", "SO3"),
        frame=Frame("invariant"),
        dtype=dtype,
        measure="dimensionless",
        level=EquivarianceLevel.CONSTRUCTIVE,
    )

    doubled_hidden = 2 * spec.attn_hidden_channels
    attention_pair_irreps = _dense_so3_irreps(doubled_hidden, spec.lmax)
    attention_irreps = _dense_so3_irreps(spec.attn_hidden_channels, spec.lmax)
    value_irreps = _dense_so3_irreps(spec.num_heads * spec.attn_value_channels, spec.lmax)
    alpha_width = spec.num_heads * spec.attn_alpha_channels
    scalar_width = doubled_hidden + spec.attn_hidden_channels

    nodes = [
        Node(
            "source_embedding",
            "core.categorical_embedding@2",
            {"x": ("input:source_atomic_numbers",)},
            {
                "embedding_dim": spec.edge_channels,
                "axis": "edge_channel",
                "dtype": dtype,
                "init_min": -0.001,
                "init_max": 0.001,
            },
        ),
        Node(
            "target_embedding",
            "core.categorical_embedding@2",
            {"x": ("input:target_atomic_numbers",)},
            {
                "embedding_dim": spec.edge_channels,
                "axis": "edge_channel",
                "dtype": dtype,
                "init_min": -0.001,
                "init_max": 0.001,
            },
        ),
        Node(
            "radial_concat",
            "core.invariant_concat@1",
            {"xs": ("input:edge_radial", "source_embedding", "target_embedding")},
            {"axis": "edge_input", "feature_role": "channel"},
        ),
        Node(
            "radial_linear0",
            "core.scalar_linear@3",
            {"x": ("radial_concat",)},
            {"axis": "edge_input", "out_features": spec.edge_channels, "out_axis": "edge_hidden", "bias": True},
        ),
        Node(
            "radial_norm0",
            "core.scalar_layer_norm@1",
            {"x": ("radial_linear0",)},
            {"axis": "edge_hidden", "epsilon": 1.0e-5, "affine": True, "bias": True},
        ),
        Node("radial_act0", "core.scalar_activation@1", {"x": ("radial_norm0",)}, {"activation": "silu"}),
        Node(
            "radial_linear1",
            "core.scalar_linear@3",
            {"x": ("radial_act0",)},
            {"axis": "edge_hidden", "out_features": spec.edge_channels, "bias": True},
        ),
        Node(
            "radial_norm1",
            "core.scalar_layer_norm@1",
            {"x": ("radial_linear1",)},
            {"axis": "edge_hidden", "epsilon": 1.0e-5, "affine": True, "bias": True},
        ),
        Node("radial_act1", "core.scalar_activation@1", {"x": ("radial_norm1",)}, {"activation": "silu"}),
        Node(
            "radial_linear2",
            "core.scalar_linear@3",
            {"x": ("radial_act1",)},
            {
                "axis": "edge_hidden",
                "out_features": (spec.lmax + 1) * 2 * spec.num_channels,
                "out_axis": "degree_channel",
                "output_axis_role": "coefficient",
                "output_feature_role": "radial_weight",
                "bias": True,
            },
        ),
        Node("source_features", "core.endpoint_gather@2", {"x": ("input:x",), "index": ("input:source_index",)}, {}),
        Node("target_features", "core.endpoint_gather@2", {"x": ("input:x",), "index": ("input:target_index",)}, {}),
        Node("message_concat", "core.equivariant_channel_concat@1", {"xs": ("source_features", "target_features")}, {}),
        Node("message_radial", "core.degreewise_invariant_scale@1", {"weight": ("radial_linear2",), "value": ("message_concat",)}, {}),
        Node(
            "message_rotate",
            "core.to_edge_frame@2",
            {"x": ("message_radial",), "direction": ("input:edge_vector",)},
            {
                "mmax": spec.mmax,
                "frame_id": frame_id,
                "frame_cache_id": frame_cache_id,
                "use_rotation_mask": not spec.direct_prediction,
            },
        ),
        Node(
            "so2_linear1",
            "core.so2_linear@2",
            {"x": ("message_rotate",)},
            {
                "out_irreps": attention_pair_irreps,
                "mmax": spec.mmax,
                "extra_m0_channels": alpha_width + scalar_width,
                "zero_bias": True,
            },
            outputs=("out", "extra_m0"),
        ),
        Node(
            "alpha_flat",
            "core.invariant_slice@1",
            {"x": ("so2_linear1:extra_m0",)},
            {"start": 0, "length": alpha_width, "axis": "alpha_flat", "feature_role": "alpha"},
        ),
        Node(
            "activation_scalars",
            "core.invariant_slice@1",
            {"x": ("so2_linear1:extra_m0",)},
            {"start": alpha_width, "length": scalar_width, "axis": "gate_scalar", "feature_role": "gate"},
        ),
        Node(
            "gated_activation",
            "core.s2_gated_swiglu_merge@1",
            {"x": ("so2_linear1:out",), "scalars": ("activation_scalars",)},
            {
                "out_irreps": attention_irreps,
                "mmax": spec.mmax,
                "grid_resolution": list(spec.attn_grid_resolution),
                "normalization": "component",
                "dropout": spec.value_drop,
            },
        ),
        Node(
            "so2_linear2",
            "core.so2_linear@1",
            {"x": ("gated_activation",)},
            {
                "out_irreps": value_irreps,
                "mmax": spec.mmax,
                "m0_prefix_rows": spec.attn_hidden_channels,
                "m0_prefix_scale": 2.0 ** -0.5,
                "zero_bias": True,
            },
        ),
        Node(
            "alpha_heads",
            "core.head_split@1",
            {"x": ("alpha_flat",)},
            {"axis": "alpha_flat", "head_axis": "head", "channel_axis": "alpha_channel", "num_heads": spec.num_heads},
        ),
    ]
    alpha_ref = "alpha_heads"
    if spec.use_attn_renorm:
        nodes.append(
            Node(
                "alpha_norm",
                "core.scalar_layer_norm@1",
                {"x": (alpha_ref,)},
                {"axis": "alpha_channel", "epsilon": 1.0e-5, "affine": True, "bias": True},
            )
        )
        alpha_ref = "alpha_norm"
    nodes.append(
        Node(
            "alpha_activation",
            "core.scalar_activation@1",
            {"x": (alpha_ref,)},
            {"activation": "silu" if spec.alpha_drop != 0.0 else "smooth_leaky_relu", "negative_slope": 0.2},
        )
    )
    alpha_ref = "alpha_activation"
    if spec.alpha_drop != 0.0:
        nodes.append(Node("alpha_dropout", "core.scalar_dropout@1", {"x": (alpha_ref,)}, {"p": spec.alpha_drop}))
        alpha_ref = "alpha_dropout"
    nodes.extend(
        (
            Node(
                "alpha_dot",
                "core.headwise_scalar_contraction@3",
                {"x": (alpha_ref,)},
                {"head_axis": "head", "channel_axis": "alpha_channel", "bias": False},
            ),
            Node(
                "alpha_softmax",
                "core.segment_softmax@3",
                {"logits": ("alpha_dot",), "index": ("input:target_segment",), "exp_rescale": ("input:edge_envelope",)},
                {"epsilon": spec.attn_eps, "exp_dropout": spec.attn_mask_rate, "softcap": spec.softcap},
            ),
            Node("alpha_envelope", "core.invariant_product@1", {"left": ("alpha_softmax",), "right": ("input:edge_envelope",)}, {}),
        )
    )
    alpha_ref = "alpha_envelope"
    if spec.attn_weights_drop != 0.0:
        nodes.append(Node("attention_weight_dropout", "core.scalar_dropout@1", {"x": (alpha_ref,)}, {"p": spec.attn_weights_drop}))
        alpha_ref = "attention_weight_dropout"
    nodes.extend(
        (
            Node("weighted_values", "core.invariant_scale@2", {"weight": (alpha_ref,), "value": ("so2_linear2",)}, {}),
            Node(
                "message_rotate_inv",
                "core.from_edge_frame@2",
                {"x": ("weighted_values",)},
                {"mmax": spec.mmax, "frame_id": frame_id, "use_rotation_mask": not spec.direct_prediction},
            ),
            Node(
                "message_reduce",
                "core.segment_reduce@1",
                {"x": ("message_rotate_inv",), "index": ("input:target_segment",)},
                {"reduce": "sum", "normalization": "none"},
            ),
            Node(
                "projection",
                "core.so3_linear@1",
                {"x": ("message_reduce",)},
                {"out_irreps": hidden_irreps_text, "bias": True},
            ),
        )
    )

    official_constructor_rank = {
        "source_embedding": 10,
        "target_embedding": 11,
        "radial_linear0": 20,
        "radial_norm0": 21,
        "radial_linear1": 22,
        "radial_norm1": 23,
        "radial_linear2": 24,
        "so2_linear1": 30,
        "alpha_norm": 40,
        "alpha_dot": 41,
        "alpha_softmax": 42,
        "gated_activation": 43,
        "so2_linear2": 44,
        "projection": 50,
    }
    construction_order = [
        node.id
        for _index, node in sorted(
            enumerate(nodes),
            key=lambda item: (official_constructor_rank.get(item[1].id, 45), item[0]),
        )
    ]
    parameters = {
        "equiformer_v3_spec": spec.to_dict(),
        "lowering_contract": {
            "module_construction_order": construction_order,
            "initializer_schedule": [
                {
                    "after_node": "target_embedding",
                    "nodes": ["source_embedding", "target_embedding"],
                }
            ],
            "deferred_initialization": "post_all_module_construction_in_declared_order",
        },
    }
    parameter_mapping = dict(EQUIFORMER_V3_ATTENTION_PARAMETER_MAPPING)
    for order in range(1, spec.mmax + 1):
        index = order - 1
        parameter_mapping[
            "node_modules.so2_linear1.so2_m_linear.{}.fc.weight".format(index)
        ] = "so2_linear_1.so2_m_linear.{}.fc.weight".format(index)
        parameter_mapping[
            "node_modules.so2_linear2.so2_m_linear.{}.fc.weight".format(index)
        ] = "so2_linear_2.so2_m_linear.{}.fc.weight".format(index)
    return ArchitectureProgram(
        language_version="2.2.0",
        task_contract=task_contract,
        inputs=(
            InputPort("x", hidden_type),
            InputPort("source_atomic_numbers", species_edge),
            InputPort("target_atomic_numbers", species_edge),
            InputPort("edge_radial", radial_type),
            InputPort("edge_vector", vector_type),
            InputPort("edge_envelope", envelope_type),
            InputPort("source_index", source_index),
            InputPort("target_index", target_index),
            InputPort("target_segment", target_segment),
        ),
        nodes=tuple(nodes),
        outputs=(OutputPort("out", "projection", hidden_type),),
        parameters=parameters,
        program_id="equiformer_v3_attention_{}".format(spec.architecture_id()),
        annotations={
            "reference_backend": "equiformer_v3_official_attention",
            "official_source_commit": V3_REFERENCE_COMMIT,
            "constructor_bypass": False,
            "numerical_scope": "one official EquivariantGraphAttention expressed by core primitives",
            "parameter_mapping": parameter_mapping,
        },
    )


def equiformer_v3_feed_forward_program(
    spec: EquiformerV3Spec,
    *,
    dtype: str = "float32",
    task_contract: str = "equiformer_v3_official_feed_forward",
) -> ArchitectureProgram:
    """Express the official V3 ``FeedForwardNetwork`` by editable core nodes.

    The first implementation covers the official ``use_grid_mlp=True`` and
    ``sep-merge_gates2_swiglu`` branch.  The S2 projection boundary is explicit
    (`grid_project`/`grid_unproject`); all grid MLP operations are separate
    nodes, so a mutation can replace only a channel map, gate, product,
    activation or dropout without introducing a fused V3 activation operator.
    """

    spec.validate()
    if dtype != "float32":
        raise ValueError("the official V3 FFN lowering is currently frozen to float32")
    if not spec.use_grid_mlp or spec.ffn_activation != "sep-merge_gates2_swiglu":
        raise ValueError(
            "the first official V3 FFN lowering requires use_grid_mlp=True and sep-merge_gates2_swiglu"
        )

    group = GroupSpec("SO3", 3, periodicity="lattice" if spec.use_pbc else "none")
    hidden_irreps = _dense_so3_irreps(spec.num_channels, spec.lmax)
    ffn_irreps = _dense_so3_irreps(spec.ffn_hidden_channels, spec.lmax)
    ffn_pair_irreps = _dense_so3_irreps(2 * spec.ffn_hidden_channels, spec.lmax)
    hidden_type = EquivariantTensorType(
        group=group,
        carrier=Carrier.NODE,
        irreps=Irreps.parse(hidden_irreps, "SO3"),
        frame=Frame("global"),
        dtype=dtype,
        measure="dimensionless",
        level=EquivarianceLevel.CONSTRUCTIVE,
    )
    output_type = replace(hidden_type, level=EquivarianceLevel.EMPIRICAL)

    nodes = [
        Node(
            "so3_linear1",
            "core.so3_linear@1",
            {"x": ("input:x",)},
            {"out_irreps": ffn_irreps, "bias": True},
        ),
        Node(
            "scalar_input",
            "core.select_scalars@2",
            {"x": ("input:x",)},
            {"axis": "scalar_channel", "feature_role": "channel"},
        ),
        Node(
            "scalar_linear",
            "core.scalar_linear@4",
            {"x": ("scalar_input",)},
            {
                "axis": "scalar_channel",
                "out_features": 2 * spec.ffn_hidden_channels,
                "out_axis": "scalar_pair",
                "bias": True,
            },
        ),
        Node(
            "scalar_gate",
            "core.invariant_slice@1",
            {"x": ("scalar_linear",)},
            {"start": 0, "length": spec.ffn_hidden_channels, "axis": "scalar_gate", "feature_role": "gate"},
        ),
        Node(
            "scalar_up",
            "core.invariant_slice@1",
            {"x": ("scalar_linear",)},
            {"start": spec.ffn_hidden_channels, "length": spec.ffn_hidden_channels, "axis": "scalar_up", "feature_role": "value"},
        ),
        Node("scalar_gate_act", "core.scalar_activation@1", {"x": ("scalar_gate",)}, {"activation": "silu"}),
        Node("scalar_product", "core.invariant_product@1", {"left": ("scalar_gate_act",), "right": ("scalar_up",)}, {}),
        Node("scalar_dropout", "core.scalar_dropout@1", {"x": ("scalar_product",)}, {"p": spec.ffn_drop}),
        Node(
            "grid_project",
            "core.grid_project@1",
            {"x": ("so3_linear1",)},
            {
                "grid_resolution": list(spec.ffn_grid_resolution),
                "mmax": spec.lmax,
                "normalization": "component",
                "quadrature": "e3nn_s2grid",
                "sampling": "equiangular",
                "use_m_primary": False,
            },
        ),
        Node(
            "gate_input",
            "core.select_scalars@2",
            {"x": ("input:x",)},
            {"axis": "gate_input_channel", "feature_role": "gate"},
        ),
        Node(
            "gate_linear",
            "core.scalar_linear@4",
            {"x": ("gate_input",)},
            {
                "axis": "gate_input_channel",
                "out_features": spec.ffn_hidden_channels,
                "out_axis": "gate_channel",
                "output_axis_role": "gate",
                "output_feature_role": "gate",
                "bias": True,
            },
        ),
        Node("gate_activation", "core.scalar_activation@1", {"x": ("gate_linear",)}, {"activation": "sigmoid"}),
        Node(
            "grid_linear1",
            "core.grid_channel_linear@1",
            {"x": ("grid_project",)},
            {"out_channels": 2 * spec.ffn_hidden_channels, "bias": False},
        ),
        Node(
            "grid_split",
            "core.grid_split@1",
            {"x": ("grid_linear1",)},
            {"left_channels": spec.ffn_hidden_channels},
            outputs=("left", "right"),
        ),
        Node(
            "grid_gate_product",
            "core.grid_pointwise_product@1",
            {"left": ("grid_split:left",), "right": ("gate_activation",)},
            {},
        ),
        Node(
            "grid_value_product",
            "core.grid_pointwise_product@1",
            {"left": ("grid_gate_product",), "right": ("grid_split:right",)},
            {},
        ),
        Node(
            "grid_dropout",
            "core.grid_dropout@1",
            {"x": ("grid_value_product",)},
            {"probability": spec.ffn_drop},
        ),
        Node(
            "grid_linear2",
            "core.grid_channel_linear@1",
            {"x": ("grid_dropout",)},
            {"out_channels": spec.ffn_hidden_channels, "bias": False},
        ),
        Node(
            "grid_unproject",
            "core.grid_unproject@1",
            {"x": ("grid_linear2",)},
            {"out_irreps": ffn_irreps},
        ),
        Node("scalar_flatten", "core.flatten_invariant_axes@1", {"x": ("scalar_dropout",)}, {}),
        Node("scalar_pad", "core.irrep_pad@1", {"x": ("scalar_flatten",)}, {"out_irreps": ffn_irreps}),
        Node("scalar_merge", "core.residual_add@2", {"left": ("grid_unproject",), "right": ("scalar_pad",)}, {}),
        Node(
            "so3_linear2",
            "core.so3_linear@2",
            {"x": ("scalar_merge",)},
            {"out_irreps": _dense_so3_irreps(spec.num_channels, spec.lmax), "bias": True, "l0_weight_scale": 2.0 ** -0.5},
        ),
    ]

    construction_order = [
        "so3_linear1",
        "scalar_input",
        "scalar_linear",
        "scalar_gate",
        "scalar_up",
        "scalar_gate_act",
        "scalar_product",
        "scalar_dropout",
        "grid_project",
        "gate_input",
        "gate_linear",
        "gate_activation",
        "grid_linear1",
        "grid_split",
        "grid_gate_product",
        "grid_value_product",
        "grid_dropout",
        "grid_linear2",
        "grid_unproject",
        "scalar_flatten",
        "scalar_pad",
        "scalar_merge",
        "so3_linear2",
    ]
    parameters = {
        "equiformer_v3_spec": spec.to_dict(),
        "lowering_contract": {
            "module_construction_order": construction_order,
            "initializer_schedule": [{"after_node": "so3_linear2", "nodes": ["scalar_linear", "gate_linear"]}],
            "deferred_initialization": "bias_zero_only_for_scalar_linear_v4; l0_scale_at_so3_linear2_construction",
            "grid_semantics": "e3nn_s2grid_finite_sampling_with_explicit_project_unproject_boundary",
        },
        "parameter_mapping": {
            "node_modules.so3_linear1.weight": "so3_linear_1.weight",
            "node_modules.so3_linear1.bias": "so3_linear_1.bias",
            "node_modules.scalar_linear.linear.weight": "scalar_mlp.0.linear.weight",
            "node_modules.scalar_linear.linear.bias": "scalar_mlp.0.linear.bias",
            "node_modules.gate_linear.linear.weight": "grid_mlp.gating_linear.weight",
            "node_modules.gate_linear.linear.bias": "grid_mlp.gating_linear.bias",
            "node_modules.grid_linear1.weight": "grid_mlp.grid_linear_1.weight",
            "node_modules.grid_linear2.weight": "grid_mlp.grid_linear_2.weight",
            "node_modules.so3_linear2.weight": "so3_linear_2.weight",
            "node_modules.so3_linear2.bias": "so3_linear_2.bias",
        },
    }
    return ArchitectureProgram(
        language_version="2.3.0",
        task_contract=task_contract,
        inputs=(InputPort("x", hidden_type),),
        nodes=tuple(nodes),
        outputs=(OutputPort("out", "so3_linear2", output_type),),
        parameters=parameters,
        program_id="equiformer_v3_feed_forward_{}".format(spec.architecture_id()),
        annotations={
            "reference_backend": "equiformer_v3_official_feed_forward",
            "official_source_commit": V3_REFERENCE_COMMIT,
            "constructor_bypass": False,
            "numerical_scope": "official FeedForwardNetwork use_grid_mlp sep-merge_gates2_swiglu expressed by typed Grid primitives",
            "grid_equivariance_certification": "finite_grid_empirical",
        },
    )


def equiformer_v3_transblock_program(
    spec: EquiformerV3Spec,
    *,
    dtype: str = "float32",
    length_measure: str = "angstrom",
    frame_id: str = "v3_attention_edge",
    frame_cache_id: str | None = None,
    task_contract: str = "equiformer_v3_official_transblock",
) -> ArchitectureProgram:
    """Expand one official V3 ``TransBlockV3`` into typed primitive nodes.

    This is the first block-level composition boundary.  The already audited
    Attention and the explicit Grid FFN programs are prefixed and connected by
    the official pre-norm, residual, graph-drop-path and equivariant-dropout
    sequence.  The implementation intentionally admits only the
    ``merge_layer_norm``/``use_grid_mlp``/``sep-merge_gates2_swiglu`` branch;
    unsupported official norm or activation branches are rejected rather than
    silently represented by an approximation.

    The finite-grid FFN keeps ``EMPIRICAL`` equivariance certification, so the
    block output is also marked empirical until a Torch/e3nn block oracle has
    been completed.
    """

    spec.validate()
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("the official V3 TransBlock lowering requires a nonempty frame_id")
    if frame_cache_id is None:
        frame_cache_id = frame_id
    if not isinstance(frame_cache_id, str) or not frame_cache_id:
        raise ValueError("the official V3 TransBlock lowering requires a nonempty frame_cache_id")
    if dtype != "float32":
        raise ValueError("the official V3 TransBlock lowering is currently frozen to float32")
    if spec.norm_type != "merge_layer_norm":
        raise ValueError(
            "the first official V3 TransBlock lowering requires norm_type=merge_layer_norm"
        )

    attention = equiformer_v3_attention_program(
        spec,
        dtype=dtype,
        length_measure=length_measure,
        frame_id=frame_id,
        frame_cache_id=frame_cache_id,
        task_contract="equiformer_v3_official_attention_subprogram",
    )
    feed_forward = equiformer_v3_feed_forward_program(
        spec,
        dtype=dtype,
        task_contract="equiformer_v3_official_feed_forward_subprogram",
    )

    hidden_type = next(port.value_type for port in attention.inputs if port.name == "x")
    output_type = replace(hidden_type, level=EquivarianceLevel.EMPIRICAL)
    group = hidden_type.group
    batch_type = IndexMapType(
        group,
        Carrier.NODE,
        Carrier.GRAPH,
        endpoint="batch",
        allows_empty_targets=False,
    )

    def _prefixed_nodes(program, prefix: str, input_remap):
        """Copy a subprogram while making its input boundary explicit."""

        copied = []
        for node in program.nodes:
            remapped_inputs = {}
            for port, references in node.inputs.items():
                new_references = []
                for reference in references:
                    if reference.startswith("input:"):
                        input_name = reference[len("input:"):]
                        new_references.append(input_remap.get(input_name, reference))
                    elif ":" in reference:
                        base, output = reference.split(":", 1)
                        new_references.append("{}{}:{}".format(prefix, base, output))
                    else:
                        new_references.append("{}{}".format(prefix, reference))
                remapped_inputs[port] = tuple(new_references)
            copied.append(
                replace(
                    node,
                    id="{}{}".format(prefix, node.id),
                    inputs=remapped_inputs,
                    annotations={
                        **dict(node.annotations),
                        "composed_from": program.program_id,
                        "composition_prefix": prefix,
                    },
                )
            )
        return copied

    inputs = tuple(attention.inputs) + (InputPort("batch", batch_type),)
    nodes = [
        Node(
            "norm1",
            "core.equivariant_merge_norm@1",
            {"x": ("input:x",)},
            {
                "epsilon": 1.0e-5,
                "affine": True,
                "normalization": "component",
                "centering": True,
            },
        )
    ]
    nodes.extend(
        _prefixed_nodes(
            attention,
            "attn_",
            {port.name: "input:{}".format(port.name) for port in attention.inputs if port.name != "x"}
            | {"x": "norm1"},
        )
    )
    nodes.extend(
        (
            Node(
                "attn_drop_path",
                "core.graph_stochastic_depth@1",
                {"x": ("attn_projection",), "batch": ("input:batch",)},
                {"p": spec.drop_path_rate},
            ),
            Node(
                "attn_proj_drop",
                "core.equivariant_dropout@1",
                {"x": ("attn_drop_path",)},
                {"p": spec.proj_drop},
            ),
            Node(
                "attn_residual",
                "core.residual_add@2",
                {"left": ("input:x",), "right": ("attn_proj_drop",)},
            ),
            Node(
                "norm2",
                "core.equivariant_merge_norm@1",
                {"x": ("attn_residual",)},
                {
                    "epsilon": 1.0e-5,
                    "affine": True,
                    "normalization": "component",
                    "centering": True,
                },
            ),
        )
    )
    nodes.extend(_prefixed_nodes(feed_forward, "ffn_", {"x": "norm2"}))
    nodes.extend(
        (
            Node(
                "ffn_drop_path",
                "core.graph_stochastic_depth@1",
                {"x": ("ffn_so3_linear2",), "batch": ("input:batch",)},
                {"p": spec.drop_path_rate},
            ),
            Node(
                "ffn_proj_drop",
                "core.equivariant_dropout@1",
                {"x": ("ffn_drop_path",)},
                {"p": spec.proj_drop},
            ),
            Node(
                "ffn_residual",
                "core.residual_add@2",
                {"left": ("attn_residual",), "right": ("ffn_proj_drop",)},
            ),
        )
    )

    attention_contract = dict(attention.parameters.get("lowering_contract", {}))
    feed_forward_contract = dict(feed_forward.parameters.get("lowering_contract", {}))
    attention_order = [
        "attn_{}".format(node_id)
        for node_id in attention_contract.get(
            "module_construction_order",
            [node.id for node in attention.nodes],
        )
    ]
    feed_forward_order = [
        "ffn_{}".format(node_id)
        for node_id in feed_forward_contract.get(
            "module_construction_order",
            [node.id for node in feed_forward.nodes],
        )
    ]
    construction_order = (
        ["norm1"]
        + attention_order
        + ["attn_drop_path", "attn_proj_drop", "attn_residual", "norm2"]
        + feed_forward_order
        + ["ffn_drop_path", "ffn_proj_drop", "ffn_residual"]
    )

    initializer_schedule = []
    for prefix, contract in (
        ("attn_", attention_contract),
        ("ffn_", feed_forward_contract),
    ):
        for event in contract.get("initializer_schedule", ()):
            initializer_schedule.append(
                {
                    "after_node": "{}{}".format(prefix, event["after_node"]),
                    "nodes": ["{}{}".format(prefix, node_id) for node_id in event["nodes"]],
                }
            )
    parameter_mapping = {
        "norm_1": "norm1",
        "norm_2": "norm2",
        "drop_path": "attn_drop_path/ffn_drop_path",
        "proj_drop": "attn_proj_drop/ffn_proj_drop",
        "attention": dict(attention.annotations.get("parameter_mapping", {})),
        "feed_forward": dict(feed_forward.parameters.get("parameter_mapping", {})),
    }
    parameters = {
        "equiformer_v3_spec": spec.to_dict(),
        "lowering_contract": {
            "module_construction_order": construction_order,
            "initializer_schedule": initializer_schedule,
            "residual_order": [
                "norm1 -> attention -> drop_path -> proj_drop -> residual",
                "norm2 -> feed_forward -> drop_path -> proj_drop -> residual",
            ],
            "batch_contract": "explicit node->graph batch IndexMapType with no empty graph targets",
            "deferred_initialization": "subprogram schedules plus block normalization construction",
        },
        "parameter_mapping": parameter_mapping,
    }
    return ArchitectureProgram(
        language_version="2.4.0",
        task_contract=task_contract,
        inputs=inputs,
        nodes=tuple(nodes),
        outputs=(OutputPort("out", "ffn_residual", output_type),),
        parameters=parameters,
        program_id="equiformer_v3_transblock_{}".format(spec.architecture_id()),
        annotations={
            "reference_backend": "equiformer_v3_official_transblock",
            "official_source_commit": V3_REFERENCE_COMMIT,
            "constructor_bypass": False,
            "numerical_scope": "explicit V3 TransBlock composition from typed Attention and Grid FFN subprograms",
            "grid_equivariance_certification": "finite_grid_empirical",
            "supported_norm_type": "merge_layer_norm",
            "shortcut": "omitted because official V3 block input/output channels are equal",
        },
    )


def equiformer_v3_backbone_program(
    spec: EquiformerV3Spec,
    *,
    dtype: str = "float32",
    length_measure: str = "angstrom",
    frame_cache_id: str = "v3_model_edge",
    task_contract: str = "equiformer_v3_official_backbone",
) -> ArchitectureProgram:
    """Compose the official V3 input path and repeated typed TransBlock stack.

    The program ends at the final node representation and deliberately excludes
    energy/force/stress heads.  It is therefore a backbone mapping boundary,
    not a complete task model.  Every block is expanded into its typed nodes;
    optional stochastic nodes are present exactly when enabled by the spec.
    No official block or network constructor is used by Lowering.
    """

    spec.validate()
    if not isinstance(frame_cache_id, str) or not frame_cache_id:
        raise ValueError("the official V3 backbone lowering requires a nonempty frame_cache_id")
    input_program = equiformer_v3_input_program(
        spec,
        dtype=dtype,
        length_measure=length_measure,
        frame_cache_id=frame_cache_id,
        task_contract="equiformer_v3_official_input_subprogram",
    )
    block_template = equiformer_v3_transblock_program(
        spec,
        dtype=dtype,
        length_measure=length_measure,
        frame_id="v3_attention_edge",
        frame_cache_id=frame_cache_id,
        task_contract="equiformer_v3_official_transblock_subprogram",
    )

    def _copy_nodes(program, prefix: str, input_remap):
        copied = []
        for node in program.nodes:
            remapped_inputs = {}
            for port, references in node.inputs.items():
                mapped = []
                for reference in references:
                    if reference.startswith("input:"):
                        input_name = reference[len("input:"):]
                        mapped.append(input_remap.get(input_name, reference))
                    elif ":" in reference:
                        base, output = reference.split(":", 1)
                        mapped.append("{}{}:{}".format(prefix, base, output))
                    else:
                        mapped.append("{}{}".format(prefix, reference))
                remapped_inputs[port] = tuple(mapped)
            remapped_attrs = dict(node.attrs)
            if "frame_id" in remapped_attrs:
                remapped_attrs["frame_id"] = "{}{}".format(
                    prefix,
                    remapped_attrs["frame_id"],
                )
            copied.append(
                replace(
                    node,
                    id="{}{}".format(prefix, node.id),
                    inputs=remapped_inputs,
                    attrs=remapped_attrs,
                    annotations={
                        **dict(node.annotations),
                        "composed_from": program.program_id,
                        "composition_prefix": prefix,
                    },
                )
            )
        return copied

    def _copy_contract(program, prefix: str):
        contract = dict(program.parameters.get("lowering_contract", {}))
        order = [
            "{}{}".format(prefix, node_id)
            for node_id in contract.get(
                "module_construction_order",
                [node.id for node in program.nodes],
            )
        ]
        schedule = [
            {
                "after_node": "{}{}".format(prefix, event["after_node"]),
                "nodes": ["{}{}".format(prefix, node_id) for node_id in event["nodes"]],
            }
            for event in contract.get("initializer_schedule", ())
        ]
        return order, schedule

    batch_port = next(port for port in block_template.inputs if port.name == "batch")
    inputs = tuple(input_program.inputs) + (batch_port,)
    top_level_input_remap = {
        port.name: "input:{}".format(port.name) for port in input_program.inputs
    }
    nodes = _copy_nodes(input_program, "input_", top_level_input_remap)
    construction_order, initializer_schedule = _copy_contract(input_program, "input_")

    input_outputs = {
        output.name: "input_{}".format(output.source)
        for output in input_program.outputs
    }
    static_block_inputs = {
        "source_atomic_numbers": input_outputs["source_atomic_numbers"],
        "target_atomic_numbers": input_outputs["target_atomic_numbers"],
        "edge_radial": input_outputs["edge_radial"],
        "edge_vector": input_outputs["edge_vector"],
        "edge_envelope": input_outputs["edge_envelope"],
        "source_index": "input:source_index",
        "target_index": "input:target_index",
        "target_segment": "input:target_segment",
        "batch": "input:batch",
    }
    previous = input_outputs["node_embedding"]
    block_mappings = {}
    for index in range(spec.num_layers):
        prefix = "block{}_".format(index)
        block_inputs = {**static_block_inputs, "x": previous}
        nodes.extend(_copy_nodes(block_template, prefix, block_inputs))
        block_order, block_schedule = _copy_contract(block_template, prefix)
        construction_order.extend(block_order)
        initializer_schedule.extend(block_schedule)
        previous = "{}ffn_residual".format(prefix)
        block_mappings["block{}".format(index)] = dict(
            block_template.parameters.get("parameter_mapping", {})
        )

    output_type = block_template.outputs[0].expected_type
    parameters = {
        "equiformer_v3_spec": spec.to_dict(),
        "lowering_contract": {
            "module_construction_order": construction_order,
            "initializer_schedule": initializer_schedule,
            "backbone_order": ["input"] + ["block{}".format(index) for index in range(spec.num_layers)],
            "frame_cache_id": frame_cache_id,
            "frame_cache_semantics": "one official random auxiliary edge frame per model forward",
            "task_heads": "excluded_from_backbone_program",
        },
        "parameter_mapping": {
            "input": dict(input_program.annotations.get("parameter_mapping", {})),
            "blocks": block_mappings,
        },
    }
    return ArchitectureProgram(
        language_version="2.5.0",
        task_contract=task_contract,
        inputs=inputs,
        nodes=tuple(nodes),
        outputs=(OutputPort("node_features", previous, output_type),),
        parameters=parameters,
        program_id="equiformer_v3_backbone_{}".format(spec.architecture_id()),
        annotations={
            "reference_backend": "equiformer_v3_official_backbone",
            "official_source_commit": V3_REFERENCE_COMMIT,
            "constructor_bypass": False,
            "numerical_scope": "official V3 input path plus repeated typed TransBlock stack; task heads excluded",
            "grid_equivariance_certification": "finite_grid_empirical",
            "block_count": spec.num_layers,
            "shared_edge_frame_cache": frame_cache_id,
            "task_heads_complete": False,
        },
    )


def equiformer_v3_energy_head_program(
    spec: EquiformerV3Spec,
    *,
    dtype: str = "float32",
    task_contract: str = "equiformer_v3_official_energy_head",
) -> ArchitectureProgram:
    """Express the official final norm and scalar energy readout by core nodes."""

    spec.validate()
    if dtype != "float32":
        raise ValueError("the official V3 energy-head lowering is currently frozen to float32")
    if spec.norm_type != "merge_layer_norm":
        raise ValueError(
            "the first official V3 energy-head lowering requires norm_type=merge_layer_norm"
        )

    group = GroupSpec("SO3", 3, periodicity="lattice" if spec.use_pbc else "none")
    hidden_type = EquivariantTensorType(
        group=group,
        carrier=Carrier.NODE,
        irreps=Irreps.parse(_dense_so3_irreps(spec.num_channels, spec.lmax), "SO3"),
        frame=Frame("global"),
        dtype=dtype,
        measure="dimensionless",
        level=EquivarianceLevel.EMPIRICAL,
    )
    batch_type = IndexMapType(
        group,
        Carrier.NODE,
        Carrier.GRAPH,
        endpoint="batch",
        allows_empty_targets=False,
    )
    energy_type = InvariantTensorType(
        group=group,
        carrier=Carrier.GRAPH,
        irreps=Irreps.parse("1x0", "SO3"),
        frame=Frame("invariant"),
        axes=("energy_channel",),
        axis_specs=(AxisSpec("energy_channel", 1, FeatureRole.CHANNEL, "independent", 0),),
        dtype=dtype,
        measure="dimensionless",
        level=EquivarianceLevel.EMPIRICAL,
        feature_role=FeatureRole.CHANNEL,
    )
    nodes = (
        Node(
            "final_norm",
            "core.equivariant_merge_norm@1",
            {"x": ("input:x",)},
            {
                "epsilon": 1.0e-5,
                "affine": True,
                "normalization": "component",
                "centering": True,
            },
        ),
        Node(
            "scalar_input",
            "core.select_scalars@2",
            {"x": ("final_norm",)},
            {"axis": "scalar_channel", "feature_role": "channel"},
        ),
        Node(
            "energy_linear1",
            "core.scalar_linear@4",
            {"x": ("scalar_input",)},
            {
                "axis": "scalar_channel",
                "out_features": spec.ffn_hidden_channels,
                "out_axis": "energy_hidden",
                "bias": True,
            },
        ),
        Node(
            "energy_activation",
            "core.scalar_activation@1",
            {"x": ("energy_linear1",)},
            {"activation": "silu"},
        ),
        Node(
            "energy_dropout",
            "core.scalar_dropout@1",
            {"x": ("energy_activation",)},
            {"p": 0.0},
        ),
        Node(
            "energy_linear2",
            "core.scalar_linear@4",
            {"x": ("energy_dropout",)},
            {
                "axis": "energy_hidden",
                "out_features": 1,
                "out_axis": "energy_channel",
                "bias": True,
            },
        ),
        Node(
            "energy_reduce",
            "core.segment_reduce@1",
            {"x": ("energy_linear2",), "index": ("input:batch",)},
            {"reduce": "sum", "normalization": "none"},
        ),
        Node(
            "energy_rescale",
            "core.constant_scale@1",
            {"x": ("energy_reduce",)},
            {"factor": 1.0 / float(spec.avg_num_nodes)},
        ),
    )
    parameters = {
        "equiformer_v3_spec": spec.to_dict(),
        "lowering_contract": {
            "module_construction_order": [node.id for node in nodes],
            "initializer_schedule": [
                {
                    "after_node": "energy_linear2",
                    "nodes": ["energy_linear1", "energy_linear2"],
                }
            ],
            "aggregation": "sum node energies by explicit node->graph batch map, then divide by avg_num_nodes",
        },
        "parameter_mapping": {
            "node_modules.final_norm.norm.affine_weight": "norm.affine_weight",
            "node_modules.final_norm.norm.affine_bias": "norm.affine_bias",
            "node_modules.energy_linear1.linear.weight": "energy_block.linear_1.weight",
            "node_modules.energy_linear1.linear.bias": "energy_block.linear_1.bias",
            "node_modules.energy_linear2.linear.weight": "energy_block.linear_2.weight",
            "node_modules.energy_linear2.linear.bias": "energy_block.linear_2.bias",
        },
    }
    return ArchitectureProgram(
        language_version="2.6.0",
        task_contract=task_contract,
        inputs=(InputPort("x", hidden_type), InputPort("batch", batch_type)),
        nodes=nodes,
        outputs=(OutputPort("energy", "energy_rescale", energy_type),),
        parameters=parameters,
        program_id="equiformer_v3_energy_head_{}".format(spec.architecture_id()),
        annotations={
            "reference_backend": "equiformer_v3_official_energy_head",
            "official_source_commit": V3_REFERENCE_COMMIT,
            "constructor_bypass": False,
            "numerical_scope": "official final merge norm, ScalarFeedForwardNetwork and avg-num-nodes energy aggregation",
            "force_derivative_contract": "outside this direct head; gradient-force task path remains pending",
        },
    )


def equiformer_v3_force_head_program(
    spec: EquiformerV3Spec,
    *,
    dtype: str = "float32",
    length_measure: str = "angstrom",
    frame_id: str = "v3_force_head_edge",
    frame_cache_id: str | None = None,
    task_contract: str = "equiformer_v3_official_force_head",
) -> ArchitectureProgram:
    """Express the official direct-force attention head using typed primitives.

    This first force-head path covers ``use_gate_force_head=True``.  The V3
    m-primary GateActivation is represented by
    ``core.edge_frame_gate_activation@1`` rather than by an official attention
    constructor or a fused force-head operator.
    """

    spec.validate()
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("the official V3 force-head lowering requires a nonempty frame_id")
    if frame_cache_id is None:
        frame_cache_id = frame_id
    if not isinstance(frame_cache_id, str) or not frame_cache_id:
        raise ValueError("the official V3 force-head lowering requires a nonempty frame_cache_id")
    if dtype != "float32":
        raise ValueError("the official V3 force-head lowering is currently frozen to float32")
    if not spec.use_gate_force_head:
        raise ValueError("the first official V3 force-head lowering requires use_gate_force_head=True")
    if spec.norm_type != "merge_layer_norm":
        raise ValueError(
            "the first official V3 force-head lowering requires norm_type=merge_layer_norm"
        )

    common_spec = replace(
        spec,
        attn_activation="sep-merge_gates2_swiglu",
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
    )
    common = equiformer_v3_attention_program(
        common_spec,
        dtype=dtype,
        length_measure=length_measure,
        task_contract="equiformer_v3_force_head_common_prefix",
    )
    prefix_nodes = []
    for node in common.nodes:
        prefix_nodes.append(node)
        if node.id == "message_rotate":
            break

    input_x = next(port.value_type for port in common.inputs if port.name == "x")
    normalized_x = replace(input_x, level=EquivarianceLevel.EMPIRICAL)
    inputs = tuple(
        InputPort(port.name, normalized_x if port.name == "x" else port.value_type)
        for port in common.inputs
    )
    group = normalized_x.group
    attention_irreps = _dense_so3_irreps(spec.attn_hidden_channels, spec.lmax)
    value_irreps = _dense_so3_irreps(spec.num_heads * spec.attn_value_channels, spec.lmax)
    output_irreps = _dense_so3_irreps(1, spec.lmax)
    alpha_width = spec.num_heads * spec.attn_alpha_channels
    gate_width = spec.lmax * spec.attn_hidden_channels

    nodes = list(prefix_nodes)
    nodes[-1] = replace(
        nodes[-1],
        attrs={
            **dict(nodes[-1].attrs),
            "frame_id": frame_id,
            "frame_cache_id": frame_cache_id,
        },
    )
    nodes.extend(
        (
            Node(
                "so2_linear1",
                "core.so2_linear@2",
                {"x": ("message_rotate",)},
                {
                    "out_irreps": attention_irreps,
                    "mmax": spec.mmax,
                    "extra_m0_channels": alpha_width + gate_width,
                    "zero_bias": True,
                },
                outputs=("out", "extra_m0"),
            ),
            Node(
                "alpha_flat",
                "core.invariant_slice@1",
                {"x": ("so2_linear1:extra_m0",)},
                {
                    "start": 0,
                    "length": alpha_width,
                    "axis": "alpha_flat",
                    "feature_role": "alpha",
                },
            ),
            Node(
                "activation_gates",
                "core.invariant_slice@1",
                {"x": ("so2_linear1:extra_m0",)},
                {
                    "start": alpha_width,
                    "length": gate_width,
                    "axis": "degree_gate",
                    "feature_role": "gate",
                },
            ),
            Node(
                "gated_activation",
                "core.edge_frame_gate_activation@1",
                {"x": ("so2_linear1:out",), "scalars": ("activation_gates",)},
                {"mmax": spec.mmax},
            ),
            Node(
                "so2_linear2",
                "core.so2_linear@1",
                {"x": ("gated_activation",)},
                {
                    "out_irreps": value_irreps,
                    "mmax": spec.mmax,
                    "zero_bias": True,
                },
            ),
            Node(
                "alpha_heads",
                "core.head_split@1",
                {"x": ("alpha_flat",)},
                {
                    "axis": "alpha_flat",
                    "head_axis": "head",
                    "channel_axis": "alpha_channel",
                    "num_heads": spec.num_heads,
                },
            ),
        )
    )
    alpha_ref = "alpha_heads"
    if spec.use_attn_renorm:
        nodes.append(
            Node(
                "alpha_norm",
                "core.scalar_layer_norm@1",
                {"x": (alpha_ref,)},
                {"axis": "alpha_channel", "epsilon": 1.0e-5, "affine": True, "bias": True},
            )
        )
        alpha_ref = "alpha_norm"
    nodes.extend(
        (
            Node(
                "alpha_activation",
                "core.scalar_activation@1",
                {"x": (alpha_ref,)},
                {"activation": "smooth_leaky_relu", "negative_slope": 0.2},
            ),
            Node(
                "alpha_dot",
                "core.headwise_scalar_contraction@3",
                {"x": ("alpha_activation",)},
                {"head_axis": "head", "channel_axis": "alpha_channel", "bias": False},
            ),
            Node(
                "alpha_softmax",
                "core.segment_softmax@3",
                {
                    "logits": ("alpha_dot",),
                    "index": ("input:target_segment",),
                    "exp_rescale": ("input:edge_envelope",),
                },
                {"epsilon": spec.attn_eps, "exp_dropout": 0.0, "softcap": spec.softcap},
            ),
            Node(
                "alpha_envelope",
                "core.invariant_product@1",
                {"left": ("alpha_softmax",), "right": ("input:edge_envelope",)},
            ),
            Node(
                "weighted_values",
                "core.invariant_scale@2",
                {"weight": ("alpha_envelope",), "value": ("so2_linear2",)},
            ),
            Node(
                "message_rotate_inv",
                "core.from_edge_frame@2",
                {"x": ("weighted_values",)},
                {
                    "mmax": spec.mmax,
                    "frame_id": frame_id,
                    "use_rotation_mask": not spec.direct_prediction,
                },
            ),
            Node(
                "message_reduce",
                "core.segment_reduce@1",
                {"x": ("message_rotate_inv",), "index": ("input:target_segment",)},
                {"reduce": "sum", "normalization": "none"},
            ),
            Node(
                "projection",
                "core.so3_linear@1",
                {"x": ("message_reduce",)},
                {"out_irreps": output_irreps, "bias": True},
            ),
            Node(
                "force_vector",
                "core.irrep_select@2",
                {"x": ("projection",)},
                {"selections": [{"irrep": "1", "start": 0, "multiplicity": 1}]},
            ),
        )
    )

    force_type = EquivariantTensorType(
        group=group,
        carrier=Carrier.NODE,
        irreps=Irreps.parse("1x1", "SO3"),
        frame=Frame("global"),
        dtype=dtype,
        measure="dimensionless",
        level=EquivarianceLevel.EMPIRICAL,
    )
    rank = {
        "source_embedding": 10,
        "target_embedding": 11,
        "radial_linear0": 20,
        "radial_norm0": 21,
        "radial_linear1": 22,
        "radial_norm1": 23,
        "radial_linear2": 24,
        "so2_linear1": 30,
        "alpha_norm": 40,
        "alpha_dot": 41,
        "alpha_softmax": 42,
        "gated_activation": 43,
        "so2_linear2": 44,
        "projection": 50,
    }
    construction_order = [
        node.id
        for _index, node in sorted(
            enumerate(nodes),
            key=lambda item: (rank.get(item[1].id, 45), item[0]),
        )
    ]
    parameter_mapping = dict(EQUIFORMER_V3_ATTENTION_PARAMETER_MAPPING)
    for order in range(1, spec.mmax + 1):
        index = order - 1
        parameter_mapping[
            "node_modules.so2_linear1.so2_m_linear.{}.fc.weight".format(index)
        ] = "so2_linear_1.so2_m_linear.{}.fc.weight".format(index)
        parameter_mapping[
            "node_modules.so2_linear2.so2_m_linear.{}.fc.weight".format(index)
        ] = "so2_linear_2.so2_m_linear.{}.fc.weight".format(index)
    return ArchitectureProgram(
        language_version="2.8.0",
        task_contract=task_contract,
        inputs=inputs,
        nodes=tuple(nodes),
        outputs=(OutputPort("forces", "force_vector", force_type),),
        parameters={
            "equiformer_v3_spec": spec.to_dict(),
            "lowering_contract": {
                "module_construction_order": construction_order,
                "initializer_schedule": [
                    {
                        "after_node": "target_embedding",
                        "nodes": ["source_embedding", "target_embedding"],
                    }
                ],
                "activation": "official GateActivation in explicit m-primary edge-frame storage",
            },
            "parameter_mapping": parameter_mapping,
        },
        program_id="equiformer_v3_force_head_{}".format(spec.architecture_id()),
        annotations={
            "reference_backend": "equiformer_v3_official_force_head",
            "official_source_commit": V3_REFERENCE_COMMIT,
            "constructor_bypass": False,
            "numerical_scope": "official direct force EquivariantGraphAttention with gate activation and l=1 extraction",
            "supported_force_activation": "gate",
        },
    )


def equiformer_v3_direct_model_program(
    spec: EquiformerV3Spec,
    *,
    dtype: str = "float32",
    length_measure: str = "angstrom",
    frame_cache_id: str = "v3_model_edge",
    task_contract: str = "equiformer_v3_official_direct_energy_force_model",
) -> ArchitectureProgram:
    """Compose the official direct energy+force V3 path without head duplication.

    The official network applies its final equivariant normalization once and
    shares the normalized representation between the scalar energy MLP and the
    direct-force attention head.  This program preserves that aliasing
    contract explicitly; it does not instantiate two copies of ``self.norm``.
    """

    spec.validate()
    if not isinstance(frame_cache_id, str) or not frame_cache_id:
        raise ValueError("the official V3 direct model lowering requires a nonempty frame_cache_id")
    if not spec.direct_prediction:
        raise ValueError("the direct V3 model program requires direct_prediction=True")
    if not spec.regress_forces:
        raise ValueError("the direct V3 model program requires regress_forces=True")
    if spec.regress_stress:
        raise ValueError("the first direct V3 model program excludes the pending stress head")

    backbone = equiformer_v3_backbone_program(
        spec,
        dtype=dtype,
        length_measure=length_measure,
        frame_cache_id=frame_cache_id,
        task_contract="equiformer_v3_official_backbone_subprogram",
    )
    energy_head = equiformer_v3_energy_head_program(
        spec,
        dtype=dtype,
        task_contract="equiformer_v3_official_energy_head_subprogram",
    )
    force_head = equiformer_v3_force_head_program(
        spec,
        dtype=dtype,
        length_measure=length_measure,
        frame_id="v3_force_head_edge",
        frame_cache_id=frame_cache_id,
        task_contract="equiformer_v3_official_force_head_subprogram",
    )

    shared_norm_template = energy_head.nodes[0]
    if shared_norm_template.id != "final_norm":
        raise ValueError("the official energy-head template must begin with final_norm")
    shared_norm = replace(
        shared_norm_template,
        inputs={"x": (backbone.outputs[0].source,)},
        annotations={
            **dict(shared_norm_template.annotations),
            "composed_from": energy_head.program_id,
            "shared_consumers": ["energy", "forces"],
        },
    )

    nodes = list(backbone.nodes)
    nodes.append(shared_norm)

    for node in energy_head.nodes[1:]:
        remapped_inputs = {}
        for port, references in node.inputs.items():
            mapped = []
            for reference in references:
                if reference == "input:x" or reference == "final_norm":
                    mapped.append("final_norm")
                elif reference == "input:batch":
                    mapped.append("input:batch")
                elif reference.startswith("input:"):
                    mapped.append(reference)
                elif ":" in reference:
                    base, output = reference.split(":", 1)
                    mapped.append("energy_{}:{}".format(base, output))
                else:
                    mapped.append("energy_{}".format(reference))
            remapped_inputs[port] = tuple(mapped)
        nodes.append(
            replace(
                node,
                id="energy_{}".format(node.id),
                inputs=remapped_inputs,
                annotations={
                    **dict(node.annotations),
                    "composed_from": energy_head.program_id,
                    "composition_prefix": "energy_",
                },
            )
        )

    force_input_remap = {
        "x": "final_norm",
        "source_atomic_numbers": "input_source_species",
        "target_atomic_numbers": "input_target_species",
        "edge_radial": "input_distance_expansion",
        "edge_vector": "input_displacement",
        "edge_envelope": "input_edge_envelope",
        "source_index": "input:source_index",
        "target_index": "input:target_index",
        "target_segment": "input:target_segment",
    }
    for node in force_head.nodes:
        remapped_inputs = {}
        for port, references in node.inputs.items():
            mapped = []
            for reference in references:
                if reference.startswith("input:"):
                    input_name = reference[len("input:"):]
                    mapped.append(force_input_remap[input_name])
                elif ":" in reference:
                    base, output = reference.split(":", 1)
                    mapped.append("force_{}:{}".format(base, output))
                else:
                    mapped.append("force_{}".format(reference))
            remapped_inputs[port] = tuple(mapped)
        remapped_attrs = dict(node.attrs)
        if "frame_id" in remapped_attrs:
            remapped_attrs["frame_id"] = "force_{}".format(remapped_attrs["frame_id"])
        nodes.append(
            replace(
                node,
                id="force_{}".format(node.id),
                inputs=remapped_inputs,
                attrs=remapped_attrs,
                annotations={
                    **dict(node.annotations),
                    "composed_from": force_head.program_id,
                    "composition_prefix": "force_",
                },
            )
        )

    backbone_contract = dict(backbone.parameters.get("lowering_contract", {}))
    energy_contract = dict(energy_head.parameters.get("lowering_contract", {}))
    force_contract = dict(force_head.parameters.get("lowering_contract", {}))
    construction_order = list(backbone_contract.get("module_construction_order", ()))
    construction_order.append("final_norm")
    construction_order.extend(
        "energy_{}".format(node_id)
        for node_id in energy_contract.get("module_construction_order", ())
        if node_id != "final_norm"
    )
    construction_order.extend(
        "force_{}".format(node_id)
        for node_id in force_contract.get("module_construction_order", ())
    )
    initializer_schedule = list(backbone_contract.get("initializer_schedule", ()))
    initializer_schedule.extend(
        {
            "after_node": "energy_{}".format(event["after_node"]),
            "nodes": ["energy_{}".format(node_id) for node_id in event["nodes"]],
        }
        for event in energy_contract.get("initializer_schedule", ())
    )
    initializer_schedule.extend(
        {
            "after_node": "force_{}".format(event["after_node"]),
            "nodes": ["force_{}".format(node_id) for node_id in event["nodes"]],
        }
        for event in force_contract.get("initializer_schedule", ())
    )

    energy_mapping = dict(energy_head.parameters.get("parameter_mapping", {}))
    final_norm_mapping = {
        key: value for key, value in energy_mapping.items() if ".final_norm." in key
    }
    energy_only_mapping = {
        key: value for key, value in energy_mapping.items() if ".final_norm." not in key
    }
    return ArchitectureProgram(
        language_version="2.9.0",
        task_contract=task_contract,
        inputs=backbone.inputs,
        nodes=tuple(nodes),
        outputs=(
            OutputPort(
                "energy",
                "energy_energy_rescale",
                energy_head.outputs[0].expected_type,
            ),
            OutputPort(
                "forces",
                "force_force_vector",
                force_head.outputs[0].expected_type,
            ),
        ),
        parameters={
            "equiformer_v3_spec": spec.to_dict(),
            "lowering_contract": {
                "module_construction_order": construction_order,
                "initializer_schedule": initializer_schedule,
                "shared_final_norm": {
                    "node": "final_norm",
                    "consumers": ["energy_scalar_input", "force_source_features", "force_target_features"],
                },
                "task_outputs": ["energy", "forces"],
                "global_initializer": "official self.apply(self._init_weights) after complete module construction",
            },
            "parameter_mapping": {
                "backbone": dict(backbone.parameters.get("parameter_mapping", {})),
                "shared_final_norm": final_norm_mapping,
                "energy_head": energy_only_mapping,
                "force_head": dict(force_head.parameters.get("parameter_mapping", {})),
            },
        },
        program_id="equiformer_v3_direct_model_{}".format(spec.architecture_id()),
        annotations={
            "reference_backend": "equiformer_v3_official_direct_energy_force_model",
            "official_source_commit": V3_REFERENCE_COMMIT,
            "constructor_bypass": False,
            "numerical_scope": "official typed backbone with one shared final norm, scalar energy MLP and direct force attention",
            "task_outputs_complete": True,
            "shared_final_norm": True,
            "direct_force_head_complete": True,
            "stress_head_complete": False,
        },
    )


def equiformer_v3_energy_model_program(
    spec: EquiformerV3Spec,
    *,
    dtype: str = "float32",
    length_measure: str = "angstrom",
    task_contract: str = "equiformer_v3_official_energy_model",
) -> ArchitectureProgram:
    """Compose the typed V3 backbone with the official scalar energy head."""

    backbone = equiformer_v3_backbone_program(
        spec,
        dtype=dtype,
        length_measure=length_measure,
        task_contract="equiformer_v3_official_backbone_subprogram",
    )
    energy_head = equiformer_v3_energy_head_program(
        spec,
        dtype=dtype,
        task_contract="equiformer_v3_official_energy_head_subprogram",
    )

    def _copy_head_nodes():
        copied = []
        for node in energy_head.nodes:
            remapped_inputs = {}
            for port, references in node.inputs.items():
                mapped = []
                for reference in references:
                    if reference == "input:x":
                        mapped.append(backbone.outputs[0].source)
                    elif reference == "input:batch":
                        mapped.append("input:batch")
                    elif ":" in reference:
                        base, output = reference.split(":", 1)
                        mapped.append("head_{}:{}".format(base, output))
                    else:
                        mapped.append("head_{}".format(reference))
                remapped_inputs[port] = tuple(mapped)
            copied.append(
                replace(
                    node,
                    id="head_{}".format(node.id),
                    inputs=remapped_inputs,
                    annotations={
                        **dict(node.annotations),
                        "composed_from": energy_head.program_id,
                        "composition_prefix": "head_",
                    },
                )
            )
        return copied

    nodes = tuple(backbone.nodes) + tuple(_copy_head_nodes())
    backbone_contract = dict(backbone.parameters.get("lowering_contract", {}))
    head_contract = dict(energy_head.parameters.get("lowering_contract", {}))
    construction_order = list(backbone_contract["module_construction_order"]) + [
        "head_{}".format(node_id) for node_id in head_contract["module_construction_order"]
    ]
    initializer_schedule = list(backbone_contract.get("initializer_schedule", ())) + [
        {
            "after_node": "head_{}".format(event["after_node"]),
            "nodes": ["head_{}".format(node_id) for node_id in event["nodes"]],
        }
        for event in head_contract.get("initializer_schedule", ())
    ]
    return ArchitectureProgram(
        language_version="2.7.0",
        task_contract=task_contract,
        inputs=backbone.inputs,
        nodes=nodes,
        outputs=(
            OutputPort(
                "energy",
                "head_energy_rescale",
                energy_head.outputs[0].expected_type,
            ),
        ),
        parameters={
            "equiformer_v3_spec": spec.to_dict(),
            "lowering_contract": {
                "module_construction_order": construction_order,
                "initializer_schedule": initializer_schedule,
                "task_outputs": ["energy"],
            },
            "parameter_mapping": {
                "backbone": dict(backbone.parameters.get("parameter_mapping", {})),
                "energy_head": dict(energy_head.parameters.get("parameter_mapping", {})),
            },
        },
        program_id="equiformer_v3_energy_model_{}".format(spec.architecture_id()),
        annotations={
            "reference_backend": "equiformer_v3_official_energy_model",
            "official_source_commit": V3_REFERENCE_COMMIT,
            "constructor_bypass": False,
            "numerical_scope": "official typed backbone plus scalar energy head; force/stress heads excluded",
            "task_outputs_complete": not spec.regress_forces and not spec.regress_stress,
            "direct_force_head_complete": False,
            "stress_head_complete": False,
        },
    )


def import_equiformer_v3(
    spec: EquiformerV3Spec,
    *,
    task_contract: str = "ocp_energy_force_stress",
) -> ArchitectureProgram:
    """Create a canonical, fully expanded-by-primitives V3 representation graph.

    The graph is executable through the generic lowering registry.  It models
    the architecture's typed representation flow and the two V3-specific
    operator semantics, while explicitly not claiming checkpoint-level
    numerical identity with the full fairchem constructor.
    """

    spec.validate_compositional_subset()
    if spec.regress_forces and spec.lmax < 1:
        raise ValueError("force readout requires lmax >= 1")
    if spec.regress_stress and spec.lmax < 2:
        raise ValueError("stress readout requires lmax >= 2")
    group = GroupSpec.so3()
    scalar_irreps = "{}x0".format(spec.num_channels)
    hidden_irreps = _dense_so3_irreps(spec.num_channels, spec.lmax)
    attn_irreps = _dense_so3_irreps(spec.attn_hidden_channels, spec.lmax)
    attn_pair_irreps = _dense_so3_irreps(2 * spec.attn_hidden_channels, spec.lmax)
    ffn_irreps = _dense_so3_irreps(spec.ffn_hidden_channels, spec.lmax)
    ffn_pair_irreps = _dense_so3_irreps(2 * spec.ffn_hidden_channels, spec.lmax)
    input_type = EquivariantType(group, Carrier.NODE, Irreps.parse(scalar_irreps, "SO3"))
    edge_sh_type = EquivariantType(
        group,
        Carrier.EDGE,
        Irreps.parse(_dense_so3_irreps(1, spec.lmax), "SO3"),
    )
    hidden_type = EquivariantType(group, Carrier.NODE, Irreps.parse(hidden_irreps, "SO3"))
    node_scalar = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0", "SO3"))
    graph_scalar = EquivariantType(group, Carrier.GRAPH, Irreps.parse("1x0", "SO3"))
    nodes = [
        Node(
            "embedding",
            "motif.v3_edge_degree_embedding",
            {"x": ("input:node_features",), "edge_sh": ("input:edge_sh",)},
            {"hidden_irreps": hidden_irreps},
            declared_types={"out": hidden_type},
            annotations={"reference_family": "EquiformerV3", "stage": "embedding"},
        )
    ]
    previous = "embedding"
    for index in range(spec.num_layers):
        node_id = "block{}".format(index)
        nodes.append(
            Node(
                node_id,
                "motif.v3_transformer_block",
                {"x": (previous,)},
                {
                    "hidden_irreps": hidden_irreps,
                    "attn_pair_irreps": attn_pair_irreps,
                    "attn_irreps": attn_irreps,
                    "ffn_pair_irreps": ffn_pair_irreps,
                    "ffn_irreps": ffn_irreps,
                    "frame_id": "v3_edge_{}".format(index),
                    "mmax": spec.mmax,
                    "ffn_mmax": spec.lmax,
                    "attn_grid_resolution": list(spec.attn_grid_resolution),
                    "ffn_grid_resolution": list(spec.ffn_grid_resolution),
                },
                declared_types={"out": hidden_type},
                annotations={"reference_family": "EquiformerV3", "stage": index},
            )
        )
        previous = node_id
    nodes.extend(
        (
            Node(
                "energy_scalar",
                "core.select_scalars",
                {"x": (previous,)},
                {"multiplicity": 1},
                declared_types={"out": node_scalar},
            ),
            Node(
                "energy_pool",
                "core.global_pool",
                {"x": ("energy_scalar",)},
                declared_types={"out": graph_scalar},
            ),
        )
    )
    outputs = [OutputPort("energy", "energy_pool", graph_scalar)]
    if spec.regress_forces:
        force_type = EquivariantType(group, Carrier.NODE, Irreps.parse("1x1", "SO3"))
        nodes.append(
            Node(
                "force_head",
                "core.irrep_slice",
                {"x": (previous,)},
                {"irreps": "1x1"},
                declared_types={"out": force_type},
                annotations={"scope": "typed direct vector readout; not official attention-head parameterization"},
            )
        )
        outputs.append(OutputPort("forces", "force_head", force_type))
    if spec.regress_stress:
        graph_hidden = EquivariantType(group, Carrier.GRAPH, Irreps.parse(hidden_irreps, "SO3"))
        stress_type = EquivariantType(group, Carrier.GRAPH, Irreps.parse("1x0+1x2", "SO3"))
        nodes.extend(
            (
                Node(
                    "stress_pool",
                    "core.global_pool",
                    {"x": (previous,)},
                    declared_types={"out": graph_hidden},
                ),
                Node(
                    "stress_head",
                    "core.irrep_slice",
                    {"x": ("stress_pool",)},
                    {"irreps": "1x0+1x2"},
                    declared_types={"out": stress_type},
                    annotations={"scope": "typed l=0 plus l=2 stress representation"},
                ),
            )
        )
        outputs.append(OutputPort("stress", "stress_head", stress_type))
    program = ArchitectureProgram(
        language_version="1.0.0",
        task_contract=task_contract,
        inputs=(InputPort("node_features", input_type), InputPort("edge_sh", edge_sh_type)),
        nodes=tuple(nodes),
        outputs=tuple(outputs),
        parameters=spec.to_dict(),
        program_id="equiformer_v3_{}".format(spec.architecture_id()),
        annotations={
            "reference_backend": "equiformer_v3_compositional",
            "equiformer_v3_spec": spec.to_dict(),
            "official_source_commit": V3_REFERENCE_COMMIT,
            "representation_scope": "typed macro/dataflow plus V3 S2-SwiGLU and merged-normalization semantics",
            "numerical_scope": "generic primitive lowering; not official constructor or checkpoint identity",
            "constructor_bypass": False,
        },
    )
    expanded = expand_motifs(program, reference_motif_registry())
    lock = architecture_id(expanded, core_registry())
    annotations = dict(program.annotations)
    annotations["reference_lock_architecture_id"] = lock
    return replace(program, annotations=annotations)
