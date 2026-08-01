"""Official Equiformer V3 checkpoint mapping for direct Typed DSL models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..ast import ArchitectureProgram
from ..checkpoint import (
    CheckpointMappingError,
    CheckpointMappingManifest,
    CheckpointTensorGroup,
    ReconstructedTensorContract,
    export_source_state_dict,
    load_mapped_checkpoint_state_dict,
)
from ..reference_programs import (
    equiformer_v3_backbone_program,
    equiformer_v3_feed_forward_program,
)
from .equiformer_v3_spec import EquiformerV3Spec, V3_REFERENCE_COMMIT


def equiformer_v3_direct_parameter_mapping(
    spec: EquiformerV3Spec,
    program: ArchitectureProgram,
) -> Mapping[str, str]:
    """Flatten the direct-model local-to-official parameter map."""

    spec.validate()
    mapping = {}
    backbone = equiformer_v3_backbone_program(spec)
    for local_name, official_name in backbone.parameters["parameter_mapping"][
        "input"
    ].items():
        mapping[local_name.replace("node_modules.", "node_modules.input_")] = (
            official_name
        )

    attention_mapping = {}
    block_template = backbone.parameters["parameter_mapping"]["blocks"]["block0"]
    for local_name, official_name in block_template["attention"].items():
        attention_mapping[
            local_name.replace("node_modules.", "node_modules.attn_")
        ] = "ga." + official_name
    feed_forward_mapping = {}
    for local_name, official_name in equiformer_v3_feed_forward_program(
        spec
    ).parameters["parameter_mapping"].items():
        feed_forward_mapping[
            local_name.replace("node_modules.", "node_modules.ffn_")
        ] = "ffn." + official_name
    block_mapping = {
        "node_modules.norm1.norm.affine_weight": "norm_1.affine_weight",
        "node_modules.norm1.norm.affine_bias": "norm_1.affine_bias",
        "node_modules.norm2.norm.affine_weight": "norm_2.affine_weight",
        "node_modules.norm2.norm.affine_bias": "norm_2.affine_bias",
        **attention_mapping,
        **feed_forward_mapping,
    }
    for block_index in range(spec.num_layers):
        for local_name, official_name in block_mapping.items():
            mapping[
                local_name.replace(
                    "node_modules.",
                    "node_modules.block{}_".format(block_index),
                )
            ] = "blocks.{}.{}".format(block_index, official_name)

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


def _group(source, target, semantic: str) -> CheckpointTensorGroup:
    source_keys = (source,) if isinstance(source, str) else tuple(source)
    target_keys = (target,) if isinstance(target, str) else tuple(target)
    return CheckpointTensorGroup(source_keys, target_keys, semantic)


def equiformer_v3_direct_checkpoint_manifest(
    spec: EquiformerV3Spec,
    program: ArchitectureProgram,
) -> CheckpointMappingManifest:
    """Build the complete official-state to lowered-state manifest."""

    spec.validate()
    program_spec = program.parameters.get("equiformer_v3_spec")
    if program_spec != spec.to_dict():
        raise CheckpointMappingError(
            "the V3 checkpoint spec does not match the direct-model program"
        )

    groups = []
    for target_name, source_name in equiformer_v3_direct_parameter_mapping(
        spec, program
    ).items():
        groups.append(_group(source_name, target_name, "trainable parameter"))

    groups.append(
        _group(
            "distance_expansion.offset",
            "node_modules.input_distance_expansion.offset",
            "fixed Gaussian radial offsets",
        )
    )

    rotation_suffixes = (
        "wigner_index_to_m_array",
        "wigner_inv_rescale",
    )
    for suffix in rotation_suffixes:
        source_aliases = [
            "so3_rotation.{}".format(suffix),
            "edge_degree_embedding.so3_rotation.{}".format(suffix),
        ]
        source_aliases.extend(
            "blocks.{}.ga.so3_rotation.{}".format(index, suffix)
            for index in range(spec.num_layers)
        )
        source_aliases.append("force_block.so3_rotation.{}".format(suffix))
        target_aliases = [
            "node_modules.input_edge_spherical_lift.rotation.{}".format(suffix)
        ]
        for index in range(spec.num_layers):
            target_aliases.extend(
                (
                    "node_modules.block{}_attn_message_rotate.rotation.{}".format(
                        index, suffix
                    ),
                    "node_modules.block{}_attn_message_rotate_inv.rotation.{}".format(
                        index, suffix
                    ),
                )
            )
        target_aliases.extend(
            (
                "node_modules.force_message_rotate.rotation.{}".format(suffix),
                "node_modules.force_message_rotate_inv.rotation.{}".format(suffix),
            )
        )
        groups.append(
            _group(
                source_aliases,
                target_aliases,
                "shared official Wigner layout buffer broadcast to typed frame entries",
            )
        )

    for index in range(spec.num_layers):
        buffer_pairs = (
            (
                "blocks.{0}.norm_1.expand_index",
                "node_modules.block{0}_norm1.norm.expand_index",
                "first merge-norm degree expansion",
            ),
            (
                "blocks.{0}.norm_1.balance_degree_weight",
                "node_modules.block{0}_norm1.norm.balance_degree_weight",
                "first merge-norm degree balance",
            ),
            (
                "blocks.{0}.ga.act.so3_grid.to_grid_mat",
                "node_modules.block{0}_attn_gated_activation.activation.so3_grid.to_grid_mat",
                "attention S2 forward grid matrix",
            ),
            (
                "blocks.{0}.ga.act.so3_grid.from_grid_mat",
                "node_modules.block{0}_attn_gated_activation.activation.so3_grid.from_grid_mat",
                "attention S2 inverse grid matrix",
            ),
            (
                "blocks.{0}.ga.proj.expand_index",
                "node_modules.block{0}_attn_projection.expand_index",
                "attention projection degree expansion",
            ),
            (
                "blocks.{0}.norm_2.expand_index",
                "node_modules.block{0}_norm2.norm.expand_index",
                "second merge-norm degree expansion",
            ),
            (
                "blocks.{0}.norm_2.balance_degree_weight",
                "node_modules.block{0}_norm2.norm.balance_degree_weight",
                "second merge-norm degree balance",
            ),
            (
                "blocks.{0}.ffn.so3_linear_1.expand_index",
                "node_modules.block{0}_ffn_so3_linear1.expand_index",
                "FFN input linear degree expansion",
            ),
            (
                "blocks.{0}.ffn.so3_grid.to_grid_mat",
                "node_modules.block{0}_ffn_grid_project.grid.to_grid_mat",
                "FFN S2 forward grid matrix",
            ),
            (
                "blocks.{0}.ffn.so3_grid.from_grid_mat",
                "node_modules.block{0}_ffn_grid_project.grid.from_grid_mat",
                "FFN S2 inverse grid matrix",
            ),
            (
                "blocks.{0}.ffn.so3_linear_2.expand_index",
                "node_modules.block{0}_ffn_so3_linear2.expand_index",
                "FFN output linear degree expansion",
            ),
        )
        groups.extend(
            _group(source.format(index), target.format(index), semantic)
            for source, target, semantic in buffer_pairs
        )

    groups.extend(
        (
            _group(
                "norm.expand_index",
                "node_modules.final_norm.norm.expand_index",
                "shared final norm degree expansion",
            ),
            _group(
                "norm.balance_degree_weight",
                "node_modules.final_norm.norm.balance_degree_weight",
                "shared final norm degree balance",
            ),
            _group(
                "force_block.act.expand_index",
                "node_modules.force_gated_activation.activation.expand_index",
                "direct-force gate degree expansion",
            ),
            _group(
                "force_block.proj.expand_index",
                "node_modules.force_projection.expand_index",
                "direct-force projection degree expansion",
            ),
        )
    )

    radial_expand = tuple(
        degree
        for degree in range(spec.lmax + 1)
        for _order in range(2 * degree + 1)
    )
    reconstructed = [
        ReconstructedTensorContract(
            source_key=source_key,
            shape=(len(radial_expand),),
            dtype="int64",
            values=radial_expand,
            semantic=(
                "official RadialFunction degree-to-coefficient expansion is "
                "structurally represented by core.degreewise_invariant_scale@1"
            ),
        )
        for source_key in (
            *(
                "blocks.{}.ga.rad_func.expand_index".format(index)
                for index in range(spec.num_layers)
            ),
            "force_block.rad_func.expand_index",
        )
    ]
    if spec.proj_drop > 0.0:
        reconstructed.extend(
            ReconstructedTensorContract(
                source_key="blocks.{}.proj_drop.expand_index".format(index),
                shape=(len(radial_expand),),
                dtype="int64",
                values=radial_expand,
                semantic=(
                    "official EquivariantDropout degree mask expansion is "
                    "reconstructed from the typed output irreps"
                ),
            )
            for index in range(spec.num_layers)
        )
    return CheckpointMappingManifest(
        source_format="fairchem-equiformer-v3@{}".format(V3_REFERENCE_COMMIT),
        target_format="evoequilang-v3-direct-state-dict",
        architecture_id=spec.architecture_id(),
        tensor_groups=tuple(groups),
        reconstructed_sources=tuple(reconstructed),
    )


def _load_payload(checkpoint, *, map_location="cpu") -> Mapping[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        import torch

        return torch.load(
            Path(checkpoint),
            map_location=map_location,
            weights_only=False,
        )
    if not isinstance(checkpoint, Mapping):
        raise CheckpointMappingError("V3 checkpoint must be a path or mapping")
    return checkpoint


def _validate_checkpoint_config(
    payload: Mapping[str, Any],
    spec: EquiformerV3Spec,
    *,
    require_config: bool,
) -> None:
    config = payload.get("config")
    if config is None:
        if require_config:
            raise CheckpointMappingError(
                "official V3 checkpoint is missing its architecture config"
            )
        return
    try:
        checkpoint_spec, _manifest = EquiformerV3Spec.from_official_config(config)
    except (TypeError, ValueError) as error:
        raise CheckpointMappingError(
            "official V3 checkpoint config is invalid: {}".format(error)
        ) from error
    if checkpoint_spec.architecture_id() != spec.architecture_id():
        raise CheckpointMappingError(
            "official V3 checkpoint architecture {} does not match target {}".format(
                checkpoint_spec.architecture_id(),
                spec.architecture_id(),
            )
        )


def load_equiformer_v3_direct_checkpoint(
    model,
    checkpoint,
    spec: EquiformerV3Spec,
    program: ArchitectureProgram,
    *,
    map_location="cpu",
    require_config: bool = True,
):
    """Load an official Fair-Chem V3 checkpoint into a lowered direct model."""

    payload = _load_payload(checkpoint, map_location=map_location)
    _validate_checkpoint_config(payload, spec, require_config=require_config)
    manifest = equiformer_v3_direct_checkpoint_manifest(spec, program)
    translated = load_mapped_checkpoint_state_dict(model, payload, manifest)
    return manifest, translated


def export_equiformer_v3_direct_checkpoint(
    model,
    spec: EquiformerV3Spec,
    program: ArchitectureProgram,
) -> Mapping[str, Any]:
    """Export a lowered direct model back to the official Fair-Chem namespace."""

    manifest = equiformer_v3_direct_checkpoint_manifest(spec, program)
    model_config = {"name": "equiformer_v3", **spec.official_constructor_kwargs()}
    return {
        "config": {"model": model_config},
        "state_dict": export_source_state_dict(model.state_dict(), manifest),
        "dsl_checkpoint_mapping": manifest.to_dict(),
    }
