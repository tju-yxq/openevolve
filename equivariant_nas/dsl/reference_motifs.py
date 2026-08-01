"""Reference motifs expressed entirely through trusted core primitives."""

from __future__ import annotations

from .ast import Node
from .motifs import MotifDefinition, MotifRegistry


def reference_motif_registry() -> MotifRegistry:
    registry = MotifRegistry()
    registry.register(
        MotifDefinition(
            name="motif.v1_initial_message",
            version=1,
            input_ports=("x", "edge_sh"),
            output_bindings={"out": "project"},
            required_attrs=("hidden_irreps",),
            template_nodes=(
                Node("lift", "core.edge_lift", {"x": ("$input:x",)}),
                Node(
                    "couple",
                    "core.tensor_product",
                    {"left": ("lift",), "right": ("$input:edge_sh",)},
                    {"out_irreps": "$attr:hidden_irreps"},
                ),
                Node("aggregate", "core.segment_sum", {"x": ("couple",)}),
                Node(
                    "project",
                    "core.irrep_linear",
                    {"x": ("aggregate",)},
                    {"out_irreps": "$attr:hidden_irreps"},
                ),
            ),
            provenance={"family": "Equiformer V1", "role": "representation-flow reference"},
            semantic_constraints=(
                "The output irreps equal hidden_irreps.",
                "edge_sh must be an edge-carried spherical-harmonic representation compatible with x.",
            ),
            edit_guidance=(
                "Changing hidden_irreps changes the representation passed to every downstream consumer.",
                "Use an explicit core.irrep_linear or core.change_multiplicity adapter at a width boundary.",
            ),
        )
    )
    registry.register(
        MotifDefinition(
            name="motif.v1_residual_message",
            version=1,
            input_ports=("x", "edge_sh"),
            output_bindings={"out": "residual"},
            required_attrs=("hidden_irreps",),
            template_nodes=(
                Node("lift", "core.edge_lift", {"x": ("$input:x",)}),
                Node(
                    "couple",
                    "core.tensor_product",
                    {"left": ("lift",), "right": ("$input:edge_sh",)},
                    {"out_irreps": "$attr:hidden_irreps"},
                ),
                Node("aggregate", "core.segment_sum", {"x": ("couple",)}),
                Node(
                    "project",
                    "core.irrep_linear",
                    {"x": ("aggregate",)},
                    {"out_irreps": "$attr:hidden_irreps"},
                ),
                Node("residual", "core.residual_add", {"left": ("$input:x",), "right": ("project",)}),
            ),
            provenance={"family": "Equiformer V1", "role": "representation-flow reference"},
            semantic_constraints=(
                "The residual add requires input x irreps to equal hidden_irreps exactly.",
                "The output irreps equal the input x irreps.",
                "Consecutive residual blocks therefore require equal boundary irreps unless an explicit adapter is inserted between them.",
            ),
            edit_guidance=(
                "Do not change hidden_irreps on an isolated residual block.",
                "To create a width transition, insert core.irrep_linear or core.change_multiplicity before the first changed residual block and another adapter before the next unchanged consumer.",
                "Every residual block inside a changed-width region must use the same hidden_irreps as its incoming x value.",
            ),
        )
    )
    registry.register(
        MotifDefinition(
            name="motif.v1_radial_profile",
            version=1,
            input_ports=("x",),
            output_bindings={"out": "offset"},
            required_attrs=("axis", "out_axis", "hidden_features", "out_features", "output_scales"),
            template_nodes=(
                Node(
                    "linear_in",
                    "core.scalar_linear@2",
                    {"x": ("$input:x",)},
                    {
                        "axis": "$attr:axis",
                        "out_features": "$attr:hidden_features",
                        "bias": True,
                    },
                ),
                Node(
                    "norm",
                    "core.scalar_layer_norm@1",
                    {"x": ("linear_in",)},
                    {"axis": "$attr:axis", "epsilon": 1.0e-5, "affine": True, "bias": True},
                ),
                Node(
                    "activation",
                    "core.scalar_activation@1",
                    {"x": ("norm",)},
                    {"activation": "silu"},
                ),
                Node(
                    "linear_out",
                    "core.scalar_linear@2",
                    {"x": ("activation",)},
                    {
                        "axis": "$attr:axis",
                        "out_axis": "$attr:out_axis",
                        "out_features": "$attr:out_features",
                        "bias": False,
                        "output_axis_role": "tp_path",
                        "output_feature_role": "radial_weight",
                        "initializer_scales": "$attr:output_scales",
                    },
                ),
                Node(
                    "offset",
                    "core.scalar_offset@1",
                    {"x": ("linear_out",)},
                    {
                        "axis": "$attr:out_axis",
                        "fan_in": "$attr:hidden_features",
                        "initializer_scales": "$attr:output_scales",
                    },
                ),
            ),
            provenance={
                "family": "Equiformer V1",
                "role": "official RadialProfile parameterization",
                "official_source": "nets/radial_func.py::RadialProfile",
            },
            semantic_constraints=(
                "The expanded graph is Linear(bias) -> LayerNorm -> SiLU -> Linear(no bias) + learned offset.",
                "The output is a dimensionless radial_weight tensor with one tp_path axis.",
                "output_scales modify only final-linear and offset initialization, reproducing TensorProductRescale slice_sqrt_k handling without changing tensor-product runtime.",
            ),
            edit_guidance=(
                "Keep out_features equal to the downstream external tensor-product weight_numel.",
                "Recompute output_scales whenever tensor-product instructions or path blocks change.",
            ),
        )
    )
    registry.register(
        MotifDefinition(
            name="motif.v1_feed_forward",
            version=1,
            input_ports=("x", "node_attr"),
            output_bindings={"out": "fctp_2"},
            required_attrs=(
                "pre_gate_irreps",
                "scalar_multiplicity",
                "gate_multiplicity",
                "gated_selections",
                "out_irreps",
            ),
            template_nodes=(
                Node(
                    "fctp_1",
                    "core.tensor_product@4",
                    {"left": ("$input:x",), "right": ("$input:node_attr",)},
                    {"out_irreps": "$attr:pre_gate_irreps", "bias": True, "rescale": True},
                ),
                Node(
                    "scalars",
                    "core.irrep_select@2",
                    {"x": ("fctp_1",)},
                    {
                        "selections": [
                            {"irrep": "0e", "start": 0, "multiplicity": "$attr:scalar_multiplicity"}
                        ]
                    },
                ),
                Node(
                    "gates",
                    "core.irrep_select@2",
                    {"x": ("fctp_1",)},
                    {
                        "selections": [
                            {
                                "irrep": "0e",
                                "start": "$attr:scalar_multiplicity",
                                "multiplicity": "$attr:gate_multiplicity",
                            }
                        ]
                    },
                ),
                Node(
                    "gated_values",
                    "core.irrep_select@2",
                    {"x": ("fctp_1",)},
                    {"selections": "$attr:gated_selections"},
                ),
                Node(
                    "activated_scalars",
                    "core.scalar_activation@1",
                    {"x": ("scalars",)},
                    {"activation": "silu", "normalization": "second_moment"},
                ),
                Node(
                    "activated_gates",
                    "core.scalar_activation@1",
                    {"x": ("gates",)},
                    {"activation": "sigmoid", "normalization": "second_moment"},
                ),
                Node(
                    "gated",
                    "core.gate@1",
                    {"gates": ("activated_gates",), "value": ("gated_values",)},
                ),
                Node(
                    "gate_output",
                    "core.irrep_concat@1",
                    {"xs": ("activated_scalars", "gated")},
                ),
                Node(
                    "fctp_2",
                    "core.tensor_product@4",
                    {"left": ("gate_output",), "right": ("$input:node_attr",)},
                    {"out_irreps": "$attr:out_irreps", "bias": True, "rescale": True},
                ),
            ),
            provenance={
                "family": "Equiformer V1",
                "role": "official FeedForwardNetwork without projection dropout",
                "official_source": "nets/graph_attention_transformer.py::FeedForwardNetwork",
            },
            semantic_constraints=(
                "fctp_1 is an internal/shared uvw TensorProduct whose output is scalars + scalar gates + gated non-scalars.",
                "Scalar SiLU and gate sigmoid both use deterministic second-moment normalization before complete-irrep gating.",
                "fctp_2 maps the gated hidden representation to out_irreps; projection dropout is deliberately outside this motif.",
            ),
            edit_guidance=(
                "Recompute pre_gate_irreps and gated selections whenever the hidden representation changes.",
                "Keep graph-level stochastic depth and equivariant projection dropout as explicit surrounding nodes.",
            ),
        )
    )
    registry.register(
        MotifDefinition(
            name="motif.v2_so2_residual_message",
            version=1,
            input_ports=("x",),
            output_bindings={"out": "residual"},
            required_attrs=("hidden_irreps", "frame_id"),
            template_nodes=(
                Node("lift", "core.edge_lift", {"x": ("$input:x",)}),
                Node(
                    "to_edge",
                    "core.to_edge_frame",
                    {"x": ("lift",)},
                    {"frame_id": "$attr:frame_id"},
                ),
                Node(
                    "so2",
                    "core.so2_convolution",
                    {"x": ("to_edge",)},
                    {"out_irreps": "$attr:hidden_irreps"},
                ),
                Node("scalars", "core.select_scalars", {"x": ("so2",)}),
                Node("activation", "core.separable_s2_activation", {"scalars": ("scalars",), "x": ("so2",)}),
                Node(
                    "to_global",
                    "core.from_edge_frame",
                    {"x": ("activation",)},
                    {"frame_id": "$attr:frame_id"},
                ),
                Node("aggregate", "core.segment_sum", {"x": ("to_global",)}),
                Node(
                    "project",
                    "core.irrep_linear",
                    {"x": ("aggregate",)},
                    {"out_irreps": "$attr:hidden_irreps"},
                ),
                Node("residual", "core.residual_add", {"left": ("$input:x",), "right": ("project",)}),
            ),
            provenance={"family": "Equiformer V2", "role": "edge-frame SO(2) representation flow"},
            semantic_constraints=(
                "The input is lifted to edges, transformed to frame_id, processed by SO(2) convolution and S2 activation, then returned to the global frame before aggregation.",
                "The residual add requires input x irreps to equal hidden_irreps exactly.",
                "frame_id must be present and the to/from edge-frame operations must use the same value.",
            ),
            edit_guidance=(
                "Do not omit frame_id or split the balanced edge-frame path.",
                "Insert representation adapters around this motif when its hidden_irreps differ from adjacent residual blocks.",
            ),
        )
    )
    registry.register(
        MotifDefinition(
            name="motif.v1_multilevel_readout",
            version=1,
            input_ports=("terminal", "aux"),
            output_bindings={"out": "project"},
            template_nodes=(
                Node("terminal_pool", "core.global_pool", {"x": ("$input:terminal",)}),
                Node("aux_scalars", "core.select_scalars", {"x": ("$input:aux",)}, {"multiplicity": 1}),
                Node("aux_pool", "core.global_pool", {"x": ("aux_scalars",)}),
                Node("concat", "core.irrep_concat", {"xs": ("terminal_pool", "aux_pool")}),
                Node("project", "core.irrep_linear", {"x": ("concat",)}, {"out_irreps": "1x0"}),
            ),
            provenance={
                "family": "Equiformer V1",
                "role": "certified hybrid readout region",
                "backend": "equiformer-v1-readout-hybrid-v1",
            },
            semantic_constraints=(
                "terminal must be the official terminal node-level scalar readout and is pooled inside the motif.",
                "aux must be a node-carried representation from one nonterminal V1 block.",
                "Only scalar irreps are exposed to the auxiliary graph readout.",
                "The output remains one graph-carried scalar.",
                "The certified runtime backend adds a trainable auxiliary head and combiner, so the parameter count increases slightly.",
            ),
            edit_guidance=(
                "Use this motif only as the graph_pool node in the v1_readout region.",
                "Bind terminal to scalar_readout and aux to exactly one of block0 through block4.",
                "Do not edit any attention or embedding block when using this motif.",
            ),
        )
    )
    registry.register(
        MotifDefinition(
            name="motif.v3_edge_degree_embedding",
            version=1,
            input_ports=("x", "edge_sh"),
            output_bindings={"out": "project"},
            required_attrs=("hidden_irreps",),
            template_nodes=(
                Node("lift", "core.edge_lift", {"x": ("$input:x",)}),
                Node(
                    "couple",
                    "core.tensor_product",
                    {"left": ("lift",), "right": ("$input:edge_sh",)},
                    {"out_irreps": "$attr:hidden_irreps"},
                ),
                Node("aggregate", "core.segment_sum", {"x": ("couple",)}),
                Node(
                    "project",
                    "core.irrep_linear",
                    {"x": ("aggregate",)},
                    {"out_irreps": "$attr:hidden_irreps"},
                ),
            ),
            provenance={
                "family": "Equiformer V3",
                "role": "typed edge-degree embedding representation flow",
                "official_source_commit": "a7300c58df683dc99cb48027d5bfd4c887486c48",
            },
            semantic_constraints=(
                "The scalar atom embedding is lifted to edges and coupled with spherical harmonics.",
                "Aggregation returns a node-carried dense SO(3) representation through lmax.",
                "This motif exposes representation semantics; it does not claim parameter identity with the official radial SO(2) embedding block.",
            ),
        )
    )
    registry.register(
        MotifDefinition(
            name="motif.v3_transformer_block",
            version=1,
            input_ports=("x",),
            output_bindings={"out": "ffn_residual"},
            required_attrs=(
                "hidden_irreps",
                "attn_pair_irreps",
                "attn_irreps",
                "ffn_pair_irreps",
                "ffn_irreps",
                "frame_id",
                "mmax",
                "ffn_mmax",
                "attn_grid_resolution",
                "ffn_grid_resolution",
            ),
            template_nodes=(
                Node("attn_norm", "core.equivariant_merge_norm", {"x": ("$input:x",)}),
                Node("lift", "core.edge_lift", {"x": ("attn_norm",)}),
                Node(
                    "to_edge",
                    "core.to_edge_frame",
                    {"x": ("lift",)},
                    {"frame_id": "$attr:frame_id", "mmax": "$attr:mmax"},
                ),
                Node(
                    "attn_expand",
                    "core.so2_convolution",
                    {"x": ("to_edge",)},
                    {"out_irreps": "$attr:attn_pair_irreps", "mmax": "$attr:mmax"},
                ),
                Node(
                    "attn_swiglu",
                    "core.s2_swiglu",
                    {"x": ("attn_expand",)},
                    {
                        "out_irreps": "$attr:attn_irreps",
                        "mmax": "$attr:mmax",
                        "grid_resolution": "$attr:attn_grid_resolution",
                    },
                ),
                Node(
                    "attn_value",
                    "core.so2_convolution",
                    {"x": ("attn_swiglu",)},
                    {"out_irreps": "$attr:hidden_irreps", "mmax": "$attr:mmax"},
                ),
                Node("logits", "core.select_scalars", {"x": ("attn_value",)}, {"multiplicity": 1}),
                Node("weights", "core.segment_softmax", {"logits": ("logits",)}),
                Node("weighted", "core.invariant_weight", {"weight": ("weights",), "value": ("attn_value",)}),
                Node(
                    "to_global",
                    "core.from_edge_frame",
                    {"x": ("weighted",)},
                    {"frame_id": "$attr:frame_id"},
                ),
                Node("aggregate", "core.segment_sum", {"x": ("to_global",)}),
                Node(
                    "attn_project",
                    "core.irrep_linear",
                    {"x": ("aggregate",)},
                    {"out_irreps": "$attr:hidden_irreps"},
                ),
                Node("attn_residual", "core.residual_add", {"left": ("$input:x",), "right": ("attn_project",)}),
                Node("ffn_norm", "core.equivariant_merge_norm", {"x": ("attn_residual",)}),
                Node(
                    "ffn_expand",
                    "core.irrep_linear",
                    {"x": ("ffn_norm",)},
                    {"out_irreps": "$attr:ffn_pair_irreps"},
                ),
                Node(
                    "ffn_swiglu",
                    "core.s2_swiglu",
                    {"x": ("ffn_expand",)},
                    {
                        "out_irreps": "$attr:ffn_irreps",
                        "mmax": "$attr:ffn_mmax",
                        "grid_resolution": "$attr:ffn_grid_resolution",
                    },
                ),
                Node(
                    "ffn_project",
                    "core.irrep_linear",
                    {"x": ("ffn_swiglu",)},
                    {"out_irreps": "$attr:hidden_irreps"},
                ),
                Node("ffn_residual", "core.residual_add", {"left": ("attn_residual",), "right": ("ffn_project",)}),
            ),
            provenance={
                "family": "Equiformer V3",
                "role": "compositional attention and S2-SwiGLU feedforward representation flow",
                "official_source_commit": "a7300c58df683dc99cb48027d5bfd4c887486c48",
            },
            semantic_constraints=(
                "Every local-frame path is balanced by the matching inverse frame operation before aggregation.",
                "S2 SwiGLU receives a paired 2C channel representation and returns C channels for every degree.",
                "Both residual additions require the hidden boundary type exactly.",
                "The first version represents the V3 mathematical dataflow but does not reproduce atom-conditioned radial weights, separate attention alpha channels, or official checkpoint parameters exactly.",
            ),
            edit_guidance=(
                "Change paired and output widths together so each S2 SwiGLU input remains exactly twice its output width.",
                "Keep ffn_mmax equal to lmax for node-global FFN values.",
                "Do not aggregate an edge-frame value before the matching from_edge_frame operation.",
            ),
        )
    )
    return registry
