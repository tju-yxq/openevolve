"""Trusted Equiformer V1 builder used by certified DSL lowerings."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .equiformer_v1_spec import ArchitectureSpec


def add_equiformer_to_path(equiformer_root: str) -> None:
    root = str(Path(equiformer_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def build_equiformer(
    spec: ArchitectureSpec,
    equiformer_root: str,
    task_mean=None,
    task_std=None,
    atomref=None,
):
    """Build a GraphAttentionTransformer without executing evolved code."""

    spec.validate()
    add_equiformer_to_path(equiformer_root)
    from nets.graph_attention_transformer import GraphAttentionTransformer

    rep = spec.representation
    op = spec.operator
    action = spec.action
    macro = spec.macro
    return GraphAttentionTransformer(
        irreps_in="5x0e",
        irreps_node_embedding=rep.embedding_irreps(),
        num_layers=macro.num_layers,
        irreps_node_attr="1x0e",
        irreps_sh=rep.spherical_harmonics_irreps(),
        max_radius=macro.radius,
        number_of_basis=op.num_basis,
        basis_type=op.basis_type,
        fc_neurons=list(op.radial_hidden),
        irreps_feature="{}x0e".format(rep.feature_channels),
        irreps_head=rep.head_irreps(),
        num_heads=op.num_heads,
        irreps_pre_attn=None,
        rescale_degree=action.rescale_degree,
        nonlinear_message=op.nonlinear_message,
        irreps_mlp_mid=rep.mlp_irreps(),
        norm_layer=action.norm_layer,
        alpha_drop=action.alpha_drop,
        proj_drop=action.projection_drop,
        out_drop=action.output_drop,
        drop_path_rate=action.drop_path,
        mean=task_mean,
        std=task_std,
        scale=None,
        atomref=atomref,
    )


def count_trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
