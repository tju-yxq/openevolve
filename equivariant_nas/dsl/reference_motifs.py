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
    return registry
